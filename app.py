import os
import pickle
import numpy as np
import torch
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# --- Load environment ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load local embeddings index ---
with open("index/index.pkl", "rb") as f:
    data = pickle.load(f)

texts, embs = data["texts"], data["embs"]

# --- Load embedding model ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# --- Helper functions ---
def embed_query(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return vec / np.linalg.norm(vec)

def answer_question(query, top_k=5):
    q_emb = embed_query(query)
    sims = np.dot(embs, q_emb)
    top = np.argsort(sims)[-top_k:][::-1]
    context = "\n\n".join(texts[i] for i in top)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer clearly based only on the context."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Tutor Agent", page_icon="📘")
st.title("📘 AI Tutor Agent")
st.write("Ask questions directly from your class materials!")

user_input = st.text_input("Ask a question about your course materials:")
if user_input:
    with st.spinner("Thinking..."):
        answer = answer_question(user_input)
        st.markdown("### Answer")
        st.write(answer)
