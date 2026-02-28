# Loaded the PDF and Chunked the Text
from typing import List, Dict
import re

from pypdf import PdfReader


def _clean_text(text: str) -> str:
	text = text.replace("\x00", " ")
	text = re.sub(r"\s+", " ", text)
	return text.strip()

def load_pdf(path: str) -> List[Dict[str, str]]:
	reader = PdfReader(path)
	pages: List[Dict[str, str]] = []

	for i, page in enumerate(reader.pages):
		raw = page.extract_text() or ""
		pages.append({
			"page_num": i + 1,
			"text": _clean_text(raw),
		})
	return pages

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	if not text:
		return []

	chunks = []
	start = 0
	length = len(text)

	while start < length:
		end = min(start + chunk_size, length)
		chunks.append(text[start:end])
		if end == length:
			break
		start = max(0, end - chunk_overlap)
	return chunks

def chunk_text(pages: List[Dict[str, str]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, str]]:
	chunks: List[Dict[str, str]] = []
	for page in pages:
		page_chunks = _split_text(page["text"], chunk_size, chunk_overlap)
		for idx, chunk in enumerate(page_chunks):
			chunks.append({
				"id": f"p{page['page_num']}_c{idx}",
				"text": chunk,
				"page_num": page["page_num"],
			})
	return chunks

