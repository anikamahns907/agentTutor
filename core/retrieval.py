"""Context retrieval functions for RAG."""
import numpy as np
import streamlit as st

from core.embeddings import embed_text
from core.rag_index import load_index


def retrieve_context(query, top_k=5, prioritize_article=False):
    """Retrieve relevant context for a query from both permanent index and session vectorstore.
    
    When prioritize_article=True, searches session articles first and only falls back to
    global index if session articles don't provide enough results.
    """
    contexts = []
    q_emb = embed_text(query)
    
    session_store = st.session_state.session_vectorstore
    
    if prioritize_article:
        # Prioritize article: search session articles first
        if session_store["embs"] is not None and len(session_store["embs"]) > 0:
            sims = np.dot(session_store["embs"], q_emb)
            top_idx = np.argsort(sims)[-top_k:][::-1]
            
            for i in top_idx:
                contexts.append({
                    "text": session_store["texts"][i],
                    "score": float(sims[i]),
                    "metadata": session_store["metadata"][i]
                })
            
            # If we have enough high-quality results from article, return them
            # Use a lower threshold to ensure we get article context when available
            if len(contexts) >= min(top_k, 3) and contexts[0]["score"] > 0.2:
                return contexts[:top_k]
        
        # Fall back to global index if article didn't provide enough results
        texts, embs, metadata = load_index()
        if texts is not None and embs is not None and len(embs) > 0:
            sims = np.dot(embs, q_emb)
            top_idx = np.argsort(sims)[-top_k:][::-1]
            for i in top_idx:
                contexts.append({
                    "text": texts[i],
                    "score": float(sims[i]),
                    "metadata": metadata[i]
                })
    else:
        # Normal mode: search both and merge results
        # Search session articles
        if session_store["embs"] is not None and len(session_store["embs"]) > 0:
            sims = np.dot(session_store["embs"], q_emb)
            top_idx = np.argsort(sims)[-top_k:][::-1]
            for i in top_idx:
                contexts.append({
                    "text": session_store["texts"][i],
                    "score": float(sims[i]),
                    "metadata": session_store["metadata"][i]
                })
        
        # Search permanent index
        texts, embs, metadata = load_index()
        if texts is not None and embs is not None and len(embs) > 0:
            sims = np.dot(embs, q_emb)
            top_idx = np.argsort(sims)[-top_k:][::-1]
            for i in top_idx:
                contexts.append({
                    "text": texts[i],
                    "score": float(sims[i]),
                    "metadata": metadata[i]
                })
    
    # Sort by score and return top_k
    contexts.sort(key=lambda x: x["score"], reverse=True)
    return contexts[:top_k]
