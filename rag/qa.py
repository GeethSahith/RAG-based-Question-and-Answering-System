from typing import Dict, List
import os

import numpy as np
from dotenv import load_dotenv
from bytez import Bytez
from sentence_transformers import SentenceTransformer

## Here I loaded the model, built the prompt and used Bytez GPT-4o-mini (free) to get the most relavent chunks and given the response   
SYSTEM_PROMPT = (
	"You are a question-answering assistant. "
    "Answer the question using ONLY the information provided in the context. "
    "Do NOT use any external knowledge or assumptions. "
    "If the answer is not explicitly present in the context, respond with: "
    "\"The document does not contain the information you are asking for.\""
)

_EMBEDDING_MODEL = None

def _get_embedding_model():
	global _EMBEDDING_MODEL
	if _EMBEDDING_MODEL is None:
		_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
	return _EMBEDDING_MODEL

def _get_bytez_model():
	load_dotenv()
	api_key = os.getenv("BYTEZ_API_KEY", "").strip()
	if not api_key:
		raise ValueError("Missing BYTEZ_API_KEY in .env")
	sdk = Bytez(api_key)
	return sdk.model("openai/gpt-4o-mini")

def _embed_query(text: str) -> List[float]:
	model = _get_embedding_model()
	embedding = model.encode(text, show_progress_bar=False)
	return embedding.tolist()

def _build_prompt(question: str, contexts: List[Dict[str, str]]) -> str:
	parts = ["Context:"]
	for ctx in contexts:
		parts.append(f"[Page {ctx['page_num']}] {ctx['text']}")
	parts.append("")
	parts.append(f"Question: {question}")
	parts.append("Answer:")
	return "\n".join(parts)

def answer_question(question: str, vector_store: Dict, top_k: int) -> Dict[str, object]:
	index = vector_store["index"]
	chunks = vector_store["chunks"]
	q_emb = np.array([_embed_query(question)], dtype="float32")
	distances, indices = index.search(q_emb, top_k)
	sources = []
	contexts = []
	for rank, idx in enumerate(indices[0]):
		if idx == -1:
			continue
		chunk = chunks[idx]
		score = float(distances[0][rank])
		sources.append({
			"page_num": chunk["page_num"],
			"text": chunk["text"],
			"score": score,
		})
		contexts.append({
			"page_num": chunk["page_num"],
			"text": chunk["text"],
		})
	prompt = _build_prompt(question, contexts)
	model = _get_bytez_model()
	full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
	results = model.run([
		{"role": "user", "content": full_prompt}
	])
	if isinstance(results, dict):
		if results.get("error"):
			raise RuntimeError(f"Bytez API error: {results.get('error')}")
		answer = results.get("output") or ""
	else:
		if results.error:
			raise RuntimeError(f"Bytez API error: {results.error}")
		answer = results.output or ""
	if isinstance(answer, dict):
		answer_text = answer.get("content", "").strip()
	else:
		answer_text = str(answer).strip() if answer else ""
	return {"answer": answer_text, "sources": sources}

