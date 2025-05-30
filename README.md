Step 1: Data Preprocessing
 
 The Market Share and Flow Share datasets are cleaned and preprocessed to ensure they are structured and in a human-readable format. 
 
Step 2: Vector Embedding and RAG + Gemini Integration

After preprocessing, the cleaned data is passed through the Gecko model to generate vector embeddings. These embeddings are stored in BigQuery and used in a Retrieval-Augmented Generation (RAG) pipeline integrated with Google’s Gemini model. This setup enables efficient similarity-based search and generates accurate, context-aware responses to user queries based on the embedded data

Step 3: Streamlit Web Application
 
 A Streamlit web application was developed for testing and usage. It provides a simple interface for users to interact with the system and query the broadband data.
