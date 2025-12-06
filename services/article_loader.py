"""Article loading and text extraction functions."""
from utils.text_extraction import (
    clean_extracted_text,
    extract_text_from_pdf,
    extract_text_from_url,
)

# Re-export for convenience
__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_url",
    "clean_extracted_text",
]
