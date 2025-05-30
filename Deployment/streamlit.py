import streamlit as st
from google.cloud import bigquery
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import logging
import pandas as pd

# Set up logging (logs will be written to console, not a file)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
try:
    bigquery_client = bigquery.Client()
    storage_client = storage.Client()
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    gen_model = GenerativeModel("gemini-2.0-flash-001")
except Exception as e:
    st.error(f"Failed to initialize clients: {e}")
    logger.error(f"Failed to initialize clients: {e}")
    st.stop()

# BigQuery table reference
project_id = "856598595188"
dataset_id = "mw_llm_poc"
table_id = "embeddings_final_table"
table_ref = f"{project_id}.{dataset_id}.{table_id}"
SUMMARY_COLUMN = "Human_Readable_Summary"

# Function to generate embeddings for the query
def get_embeddings(texts):
    try:
        embeddings = embedding_model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

# Step 3: Retrieve similar summaries from BigQuery
def retrieve_similar_summaries(query_text, top_k=5):
    query_embedding = get_embeddings([query_text])
    if query_embedding is None:
        logger.error(f"Failed to generate embedding for query: {query_text}")
        return []
    query_embedding_str = ', '.join(map(str, query_embedding[0]))
    logger.info(f"Generated query embedding with length: {len(query_embedding[0])}")

    similarity_query = f"""
        SELECT 
            t.id,
            t.{SUMMARY_COLUMN},
            SUM(CAST(t.embedding[offset(i)] AS FLOAT64) * q_emb) AS dot_product
        FROM `{table_ref}` t
        CROSS JOIN UNNEST([{query_embedding_str}]) AS q_emb WITH OFFSET i
        WHERE ARRAY_LENGTH(t.embedding) = {len(query_embedding[0])}
        GROUP BY t.id, t.{SUMMARY_COLUMN}
    """
    
    try:
        logger.info(f"Executing similarity query: {query_text}")
        query_job = bigquery_client.query(similarity_query)
        results = query_job.result()
        retrieved = [{"id": row["id"], "summary": row[SUMMARY_COLUMN]} for row in results]
        logger.info(f"Retrieved {len(retrieved)} summaries")
        return retrieved
    except Exception as e:
        logger.error(f"Error executing similarity query: {str(e)}")
        return []

# Step 4: Generate detailed response using Gemini
def generate_response(query, retrieved_summaries):
    context = "\n".join([s["summary"] for s in retrieved_summaries])
    prompt = f"""
You are a data analyst assistant for broadband subscriber trends in the USA. You have access to detailed information at various>

1. You know the number of broadband subscribers per carrier.
2. You know the percentage market share of each carrier.
3. You can distinguish between two broadband service types: Fixed Broadband and Fixed Wireless Access.
4. You also know, for any two consecutive months, the number of subscribers who moved from one carrier to another.

Use this information to respond to questions by:

- Interpreting location names (e.g., states, cities, ZIP codes) and aggregating across relevant geographies.
- Grouping data based on broadband service types when asked.
- Summarizing or comparing carriers by subscriber count, market share, or churn (i.e., subscriber movement between carriers).
- Clearly stating the date/months when responding to questions about subscriber movement.

Answer in a concise and informative manner, with appropriate numerical detail (subscriber counts, percentages, etc.) where help>

User question: {query}

Context from broadband subscriber RAG data: {context}

Use the broadband subscriber RAG data to answer. Interpret geographic references like 'Texas', and provide carrier-level statis>
"""
    logger.info(f"Generating response for query: {query}")
    try:
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=0.6,  # Slightly lower temperature for more focused and professional output
            top_p=0.9,       # Controls diversity while maintaining relevance
            max_output_tokens=1024  # Allows for detailed responses
        )
        response = gen_model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response. Please try again or refine your query."
# Streamlit app
st.title("MW Final: Broadband Data Query App")
st.write("Enter a query to retrieve and generate detailed insights about broadband data in the USA for July 2024.")

# Input query
query = st.text_input("Enter your query:", value="Which carrier had the most new subscribers in Texas between June and July 202>
# Button to process the query
if st.button("Submit Query"):
    if not query:
        st.error("Please enter a query.")
    else:
        with st.spinner("Retrieving similar summaries..."):
            # Retrieve similar summaries
            top_summaries = retrieve_similar_summaries(query, top_k=5)
            if not top_summaries:
                st.error("No summaries retrieved. Please try a different query or check the BigQuery table.")
                logger.error("No summaries retrieved for the query.")
            else:
                # Display retrieved summaries
                st.subheader("Retrieved Summaries")
                for summary in top_summaries:
                    st.write(f"ID: {summary['id']}, Summary: {summary['summary']}")

                # Generate response
                with st.spinner("Generating detailed response..."):
                    response = generate_response(query, top_summaries)
                    st.subheader("Generated Response")
                    st.write(response)

# Footer
st.write("Powered by Google Cloud Platform, BigQuery, and Vertex AI & Gemini.")
