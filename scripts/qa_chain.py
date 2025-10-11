import os
import pickle
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# Load your OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your local embeddings index
with open("../index/index.pkl", "rb") as f:
    data = pickle.load(f)

texts, embs = data["texts"], data["embs"]

# Load same embedding model as used in ingest_docs.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_query(text):
    """Create normalized embedding for the user query."""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return vec / np.linalg.norm(vec)

def answer_question(query, top_k=5):
    """Retrieve top context chunks and ask GPT."""
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

if __name__ == "__main__":
    print("📘 Tutor agent ready. Type your question or 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", answer_question(q))
