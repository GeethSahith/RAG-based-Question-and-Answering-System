from typing import Dict, List
import os

from dotenv import load_dotenv
from bytez import Bytez

SYSTEM_PROMPT = (
	"You are a question-answering assistant. "
    "Answer the question using ONLY the information provided in the context. "
    "Do NOT use any external knowledge or assumptions. "
    "If the answer is not explicitly present in the context, respond with: "
    "\"The document does not contain the information you are asking for.\""
)

def _get_bytez_model():
	load_dotenv()
	api_key = os.getenv("BYTEZ_API_KEY", "").strip()
	if not api_key:
		raise ValueError("Missing BYTEZ_API_KEY in .env")
	sdk = Bytez(api_key)
	return sdk.model("openai/gpt-4o-mini")

def _build_prompt(question: str, contexts: List[Dict[str, str]]) -> str:
	parts = ["Context:"]
	for ctx in contexts:
		parts.append(f"[Page {ctx['page_num']}] {ctx['text']}")
	parts.append("")
	parts.append(f"Question: {question}")
	parts.append("Answer:")
	return "\n".join(parts)

def answer_question(question: str, vector_store: Dict, top_k: int) -> Dict[str, object]:
	"""
	Searched Pinecone Serverless with text - Pinecone generates embedding automatically
	Uses search_records() method for integrated embeddings
	"""
	import sys
	index = vector_store["index"]
	doc_id = vector_store.get("doc_id", "default")
	
	# Searched with text integrated embeddings
	results = index.search(
		namespace=doc_id,
		query={
			"inputs": {"text": question},
			"top_k": top_k
		},
		fields=["chunk_text", "page_num", "doc_id"]
	)
	
	sources = []
	contexts = []
	
	# DEBUG: Print search results
	print(f"DEBUG: Question: {question}", file=sys.stderr)
	print(f"DEBUG: Doc ID: {doc_id}", file=sys.stderr)
	print(f"DEBUG: Full search results: {results}", file=sys.stderr)
	
	# Parsed search results
	hits = results.get("result", {}).get("hits", [])
	print(f"DEBUG: Number of hits: {len(hits)}", file=sys.stderr)
	for hit in hits:
		fields = hit.get("fields", {})
		score = hit.get("_score", 0)
		
		# Got text from chunk_text field
		text = fields.get("chunk_text", "")
		page_num = fields.get("page_num", "0")
		
		sources.append({
			"page_num": page_num,
			"text": text,
			"score": float(score),
		})
		contexts.append({
			"page_num": page_num,
			"text": text,
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

