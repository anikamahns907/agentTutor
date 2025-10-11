import torch
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

load_dotenv()

# Use small model that runs locally
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(texts):
    """Return normalized embeddings using transformers only."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embs = outputs.last_hidden_state.mean(dim=1).numpy()
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs

folders = ["../docs", "../assessments"]
docs = []
for folder in folders:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(".pdf"):
                docs.extend(PyMuPDFLoader(os.path.join(folder, f)).load())

if not docs:
    print("⚠️ No PDFs found.")
    exit()

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(docs)
texts = [c.page_content for c in chunks]

print(f"Loaded {len(texts)} chunks. Generating embeddings…")
embs = embed(texts)

os.makedirs("../index", exist_ok=True)
with open("../index/index.pkl", "wb") as f:
    pickle.dump({"texts": texts, "embs": embs}, f)

print("✅ Index saved to ../index/index.pkl")
