# PDF RAG

Upload a PDF document, ask relevant questions, and receive accurate answers extracted from the file.

# Working flow
PDF -> Extract Text -> Chunk Text -> Pinecone Auto Embed by multilingual e5 large(Pinecone) -> Store in Pinecone Serverless -> Search & Retrieve Top K Chunks -> Bytez API (GPT4o-mini) -> Answer with Sources

# Setup
1. Create a virtual environment in this folder<br>
    python -m venv venv<br>
    and<br>
    ./venv/Scripts/activate<br>
2. Install dependencies from requirements.txt <br>
    pip install -r requirements.txt
3. Create a .env file with the following API keys in root folder:<br>
    Refer to the .env.example

# Run
Use Streamlit to run app.py
    streamlit run app.py
