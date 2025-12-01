import re
from typing import Optional

import fitz  # type: ignore
import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
}


def clean_extracted_text(text: Optional[str]) -> str:
    """Normalize whitespace and remove invisible characters from extracted text."""
    if not text:
        return ""
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF binary and normalize it."""
    if not file_bytes:
        return ""

    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        for page in document:
            pages.append(page.get_text("text"))

    return clean_extracted_text("\n".join(pages))


def extract_text_from_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return extracted text (handles HTML + PDF)."""
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return extract_text_from_pdf(resp.content)

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "svg"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    return clean_extracted_text(text)


__all__ = [
    "clean_extracted_text",
    "extract_text_from_pdf",
    "extract_text_from_url",
]

