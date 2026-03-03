from __future__ import annotations
from typing import Dict, List
import os
from dotenv import load_dotenv
from pinecone import Pinecone

def _get_pinecone_index():
	load_dotenv()
	api_key = os.getenv("PINECONE_API_KEY", "").strip()
	host = os.getenv("PINECONE_HOST", "").strip()
	index_name = os.getenv("PINECONE_INDEX", "rag-based-qa")
	if not api_key or not host or not index_name:
		raise ValueError("Missing PINECONE_API_KEY, PINECONE_HOST, or PINECONE_INDEX in .env")
	host = host.replace("https://", "").replace("http://", "")
	pc = Pinecone(api_key=api_key)
	return pc.Index(index_name, host=host)

def create_or_load_vector_store(doc_id: str, chunks: List[Dict[str, str]], persist_dir: str = None):
	"""
	Upserted chunks to Pinecone Serverless with integrated embeddings.
	Pinecone automatically generates embeddings using multilingual-e5-large
	"""
	import sys
	index = _get_pinecone_index()
	
	# Prepared records for upsert_records() - metadata fields at same level as text
	# NOT nested - each field becomes a metadata field
	records = []
	for i, chunk in enumerate(chunks):
		chunk_text = (chunk.get("text") or "").strip()
		if not chunk_text:
			continue
		vector_id = f"{doc_id}_chunk_{i}"
		records.append({
			"id": vector_id,
			"text": chunk_text,
			"chunk_text": chunk_text,  # Store text again for retrieval
			"page_num": str(chunk["page_num"]),
			"doc_id": doc_id
		})

	if not records:
		raise ValueError("The document appears to be scanned or image based. Please upload a version that contains selectable text.")
	
	upsert_result = index.upsert_records(records=records, namespace=doc_id)
	return {"index": index, "chunks": chunks, "doc_id": doc_id}
