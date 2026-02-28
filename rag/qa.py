from typing import Dict, List
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

## Here I loaded the model, built the prompt and used gpt model to get the most relavent chunks and given the response   
SYSTEM_PROMPT = (
	"You are a question-answering assistant. "
    "Answer the question using ONLY the information provided in the context. "
    "Do NOT use any external knowledge or assumptions. "
    "If the answer is not explicitly present in the context, respond with: "
    "\"The document does not contain the information you are asking for.\""
)
def _get_openai_client() -> OpenAI:
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY", "").strip()
	if not api_key:
		raise ValueError("Missing OPENAI_API_KEY in .env")
	return OpenAI(api_key=api_key)

def _embed_query(text: str) -> List[float]:
	client = _get_openai_client()
	resp = client.embeddings.create(
		model="text-embedding-3-small",
		input=text,
	)
	return resp.data[0].embedding

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
	client = _get_openai_client()
	completion = client.chat.completions.create(
		model="gpt-4o",
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": prompt},
		],
		temperature=0.4,
	)
	answer = completion.choices[0].message.content or ""
	return {"answer": answer.strip(), "sources": sources}

