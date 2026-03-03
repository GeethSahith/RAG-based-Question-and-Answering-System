import hashlib
import os
from pathlib import Path
import streamlit as st
from rag.chunking import load_pdf, chunk_text
from rag.vector_store import create_or_load_vector_store
from rag.qa import answer_question

# imported all the functions from the rag/ folderso that when the 
# upload and index is clicked then the indexing, chunking, storing in the database is done 
# and the uploaded files stored in the data/uploads and indexed 
# and chunked json files are stored in the vector_store 
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vector_store"
def _hash_bytes(data: bytes) -> str:
	return hashlib.md5(data).hexdigest()

def _save_upload(data: bytes, doc_id: str) -> Path:
	UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
	path = UPLOAD_DIR / f"{doc_id}.pdf"
	path.write_bytes(data)
	return path

def main() -> None:
	st.set_page_config(page_title="PDF RAG QA", layout="wide")
	st.title("PDF Question Answering (RAG)")
	st.caption("Upload a PDF, ask questions, and get document-grounded answers.")

	with st.sidebar:
		st.subheader("Settings")
		chunk_size = st.number_input("Chunk size", min_value=200, max_value=1500, value=900)
		chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=400, value=150)
		top_k = st.number_input("Top-k", min_value=1, max_value=10, value=5)

	with st.container():
		st.subheader("Upload PDF")
		uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
		upload_clicked = st.button("Upload and Index")

	if upload_clicked:
		if uploaded is None:
			st.warning("Please choose a PDF before clicking upload.")
			return

		file_bytes = uploaded.getvalue()
		doc_id = _hash_bytes(file_bytes)

		if st.session_state.get("doc_id") != doc_id:
			try:
				pdf_path = _save_upload(file_bytes, doc_id)
				pages = load_pdf(str(pdf_path))
				chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
				import sys
				print(f"DEBUG APP: Loaded {len(pages)} pages, created {len(chunks)} chunks", file=sys.stderr)
				for i, chunk in enumerate(chunks[:3]):
					print(f"DEBUG APP: Chunk {i} - Page {chunk['page_num']}: {chunk['text'][:80]}...", file=sys.stderr)
				vector_store = create_or_load_vector_store(
					doc_id=doc_id,
					chunks=chunks,
					persist_dir=str(VECTOR_DIR),
				)
			except ValueError as exc:
				st.warning(str(exc))
				return
			except Exception as exc:
				st.error(f"Failed to process PDF: {exc}")
				return

			st.session_state["doc_id"] = doc_id
			st.session_state["vector_store"] = vector_store
			st.session_state["num_pages"] = len(pages)
			st.success(f"Indexed {len(pages)} pages and {len(chunks)} chunks.")
		else:
			st.info("This PDF is already indexed.")

	question = st.text_input("To get better results, ask a precise question related to the document. For example, mention a topic, section, keyword or concept from the PDF.")
	ask = st.button("Get answer")

	if ask and question:
		if st.session_state.get("vector_store") is None:
			st.warning("Please upload a PDF first.")
			return
		with st.spinner("Searching and generating answer..."):
			result = answer_question(
				question=question,
				vector_store=st.session_state["vector_store"],
				top_k=top_k,
			)
		st.subheader("Answer")
		st.write(result["answer"])
		st.subheader("Sources")
		for i, src in enumerate(result["sources"], start=1):
			title = f"Source {i} (Page {src['page_num']})"
			with st.expander(title):
				st.write(src["text"])
				st.caption(f"Score: {src['score']:.4f}")


if __name__ == "__main__":
	main()

