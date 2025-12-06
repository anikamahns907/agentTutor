"""Embedding model loading and text embedding functions."""
import numpy as np
import torch
import streamlit as st
from transformers import AutoModel, AutoTokenizer


@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_text(text, tokenizer=None, model=None):
    """Embed a text string using the loaded model."""
    if tokenizer is None or model is None:
        tokenizer, model = load_embedding_model()
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]

        embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = torch.sum(embeddings * mask_expanded, 1)
        counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
        vec = (summed / counts).squeeze().numpy()

    return vec / np.linalg.norm(vec)
