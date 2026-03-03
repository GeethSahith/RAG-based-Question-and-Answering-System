# PDF RAG

Upload a PDF, ask questions, and get answers as in the document.

# Working flow
PDF -> Extract Text -> Chunk Text -> Pinecone Auto Embed by multilingual e5 large(Pinecone) -> Store in Pinecone Serverless -> Search & Retrieve Top K Chunks -> Bytez API (GPT4o-mini) -> Answer with Sources

# Setup
1. Create a virtual environment in this folder
    python -m venv venv
    and
    ./venv/Scripts/activate
2. Install dependencies from requirements.txt
    pip install -r requirements.txt
3. Create a .env file with the following API keys in root folder:
    Refer to the .env.example

# Run
Use Streamlit to run app.py
    streamlit run app.py
