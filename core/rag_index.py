"""RAG index loading functions."""
import pickle
import numpy as np
import streamlit as st
from pathlib import Path

from core.embeddings import embed_text


@st.cache_resource
def load_index():
    """Load the document index from disk."""
    index_path = Path("index/index.pkl")
    if not index_path.exists():
        return None, None, None
    
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    
    texts = data.get("texts", [])
    stored_embs = data.get("embs", np.array([]))
    metadata = data.get("metadata", [])
    
    # If embeddings missing or wrong dimension, recompute
    if len(stored_embs) > 0 and stored_embs.shape[1] != 384:
        recomputed = []
        for t in texts:
            recomputed.append(embed_text(t))
        stored_embs = np.array(recomputed)

    return texts, stored_embs, metadata
