"""Functions for adding articles to session vectorstore."""
import numpy as np
import streamlit as st

from core.embeddings import embed_text
from utils.article_import import prepare_article_documents


def add_article_to_session(text, title, source, url=None):
    """Add an uploaded article to the session vectorstore."""
    chunks, meta = prepare_article_documents(text, title, source, url)
    if not chunks:
        return False

    chunk_embs = np.array([embed_text(c) for c in chunks])

    store = st.session_state.session_vectorstore
    store["texts"].extend(chunks)
    store["metadata"].extend(meta)

    if store["embs"] is None:
        store["embs"] = chunk_embs
    else:
        store["embs"] = np.vstack([store["embs"], chunk_embs])

    return True
