import os
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load local embeddings index ---
@st.cache_resource
def load_index():
    with open("index/index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["texts"], data["embs"]

texts, embs = load_index()

# --- Load embedding model (lightweight) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Helper: embed a query ---
def embed_query(text):
    vec = model.encode([text])[0]
    return vec / np.linalg.norm(vec)

# --- Helper: answer question using context + GPT ---
def answer_question(query, top_k=5):
    q_emb = embed_query(query)
    sims = np.dot(embs, q_emb)
    top = np.argsort(sims)[-top_k:][::-1]
    context = "\n\n".join(texts[i] for i in top)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely and clearly based only on the context above."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Tutor Agent", page_icon="📘")
st.title("📘 AI Tutor Agent")
st.write("Ask questions directly from your uploaded class materials!")

user_input = st.text_input("Ask a question about your course materials:")

if user_input:
    with st.spinner("Thinking..."):
        answer = answer_question(user_input)
        st.markdown("### Answer")
        st.write(answer)
