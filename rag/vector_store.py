from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Generated the embeddings using local model and stored in the FAISS Base
_EMBEDDING_MODEL = None

def _get_embedding_model():
	global _EMBEDDING_MODEL
	if _EMBEDDING_MODEL is None:
		_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
	return _EMBEDDING_MODEL

def _embed_texts(texts: List[str]) -> List[List[float]]:
	model = _get_embedding_model()
	embeddings = model.encode(texts, show_progress_bar=False)
	return embeddings.tolist()


def _save_store(store_dir: Path, index, chunks: List[Dict[str, str]]) -> None:
	store_dir.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(store_dir / "index.faiss"))
	with open(store_dir / "chunks.json", "w", encoding="utf-8") as f:
		json.dump(chunks, f, ensure_ascii=True, indent=2)


def _load_store(store_dir: Path):
	index = faiss.read_index(str(store_dir / "index.faiss"))
	with open(store_dir / "chunks.json", "r", encoding="utf-8") as f:
		chunks = json.load(f)
	return index, chunks

# Created and loaded Faiss store for a document
def create_or_load_vector_store(doc_id: str, chunks: List[Dict[str, str]], persist_dir: str):
	base = Path(persist_dir)
	store_dir = base / doc_id
	if (store_dir / "index.faiss").exists() and (store_dir / "chunks.json").exists():
		index, stored_chunks = _load_store(store_dir)
		return {"index": index, "chunks": stored_chunks}
	texts = [c["text"] for c in chunks]
	embeddings = _embed_texts(texts)
	emb_array = np.array(embeddings, dtype="float32")
	dim = emb_array.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(emb_array)
	_save_store(store_dir, index, chunks)
	return {"index": index, "chunks": chunks}

