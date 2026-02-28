# PDF RAG

Upload a PDF, ask questions, and get answers as in the document.

# Working flow
PDF -> text -> chunks -> embeddings -> FAISS -> retrieve -> LLM -> answered with sources

# Setup
1. Create a virtual environment in this folder
    python -m venv venv
    and
    ./venv/Scripts/activate
2. Install dependencies from requirements.txt
    pip install -r requirements.txt
3. Create a .env file with OPENAI_API_KEY in root folder.

# Run
Use Streamlit to run app.py
    streamlit run app.py

# Folders
data/uploads holds uploaded PDFs
data/vector_store holds FAISS indexes
rag contains the core logic of loading, chunking and retreiving
