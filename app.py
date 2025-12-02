import os
import pickle
import numpy as np
import streamlit as st
import torch
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import io
import base64

from utils.article_import import (
    get_bruknow_search_url,
    prepare_article_documents,
    search_bruknow_articles,
    search_public_health_articles,
)

from utils.text_extraction import (
    clean_extracted_text,
    extract_text_from_pdf,
    extract_text_from_url,
)

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- Load environment variables ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Inject JS Router (OPTION B3)
# --------------------------
st.markdown("""
<script>
function go(page) {
    const params = new URLSearchParams(window.location.search);
    params.set("page", page);
    window.location.search = params.toString();
}
</script>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar Layout (HTML + JS)
# --------------------------
# This function is not used - keeping for reference
# The actual sidebar_nav() is defined later in the file
# ---------------------------------------------
# Session State Initialization
# ---------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "current_article" not in st.session_state:
    st.session_state.current_article = None
if "assignment_questions" not in st.session_state:
    st.session_state.assignment_questions = []
if "session_vectorstore" not in st.session_state:
    st.session_state.session_vectorstore = {"texts": [], "embs": None, "metadata": []}

# ---------------------------------------------
# Load Embedding Model (cached)
# ---------------------------------------------
@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, embedding_model = load_embedding_model()

# ---------------------------------------------
# Embed Text
# ---------------------------------------------
def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = embedding_model(**inputs)
        attention_mask = inputs["attention_mask"]

        embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = torch.sum(embeddings * mask_expanded, 1)
        counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
        vec = (summed / counts).squeeze().numpy()

    return vec / np.linalg.norm(vec)


# ---------------------------------------------
# Load Index
# ---------------------------------------------
@st.cache_resource
def load_index():
    index_path = Path("index/index.pkl")
    if not index_path.exists():
        return None, None, None
    
        with open(index_path, "rb") as f:
            data = pickle.load(f)
    
    texts = data.get("texts", [])
    stored_embs = data.get("embs", np.array([]))
    metadata = data.get("metadata", [])
    
    # If embeddings missing or wrong dimension, recompute
    if stored_embs.shape[1] != 384:
        recomputed = []
        for t in texts:
            recomputed.append(embed_text(t))
        stored_embs = np.array(recomputed)

    return texts, stored_embs, metadata


try:
    texts, embs, metadata = load_index()
except Exception:
    texts, embs, metadata = None, None, None


# ---------------------------------------------
# Retrieve Context
# ---------------------------------------------
def retrieve_context(query, top_k=5):
    contexts = []
    q_emb = embed_text(query)
    
    # Search permanent index
    if texts is not None and embs is not None:
        sims = np.dot(embs, q_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        for i in top_idx:
            contexts.append({"text": texts[i], "score": float(sims[i]), "metadata": metadata[i]})

    # Search uploaded articles in session
    session_store = st.session_state.session_vectorstore
    if session_store["embs"] is not None:
        sims = np.dot(session_store["embs"], q_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]

        for i in top_idx:
            contexts.append({
                "text": session_store["texts"][i],
                "score": float(sims[i]),
                "metadata": session_store["metadata"][i]
            })

    contexts.sort(key=lambda x: x["score"], reverse=True)
    return contexts[:top_k]


# ---------------------------------------------
# Add Uploaded Article to Session Vectorstore
# ---------------------------------------------
def add_article_to_session(text, title, source, url=None):
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


# ---------------------------------------------
# Build Prompt for AI
# ---------------------------------------------
def build_prompt(user_query, contexts):
    ctx_text = "\n\n".join([f"[Source: {c['metadata'].get('source')}] {c['text']}" for c in contexts])
    
    system = """
You are Isabelle, a helpful biostatistics communication tutor for PHP 1510/2510.
Focus on clarity, conceptual reasoning, plain-language explanations.
DO NOT grade or score.
Guide students, ask clarifying questions, and connect to course concepts.
"""

    return f"{system}\n\nContext:\n{ctx_text}\n\nStudent question:\n{user_query}"


# ---------------------------------------------
# AI Answer Helper
# ---------------------------------------------
def answer_question(query):
    contexts = retrieve_context(query, top_k=5)
    prompt = build_prompt(query, contexts)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are Isabelle."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    ans = response.choices[0].message.content
    if "edstem" not in ans.lower():
        ans += "\n\nFor more help, ask on EdStem or speak with Professor Lipman."
    
    return ans
# ==========================================================
#                    PAGE RENDER FUNCTIONS
# ==========================================================

def render_home_page():
    st.markdown("# Welcome to Isabelle")
    st.markdown('<p class="subtitle">Your AI Tutor for PHP 1510/2510 – Principles of Biostatistics</p>',
                unsafe_allow_html=True)

    st.markdown("""
    Isabelle helps you practice communicating statistical concepts,
    understand research papers, and interpret results using material
    from the course textbook, lectures, and assignments.
    """)

    st.markdown("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='card'>
            <h3>Analyze Articles</h3>
            <p>Answer structured questions and get feedback.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
            <h3>Ask Questions</h3>
            <p>Learn statistical methods and interpretation concepts.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='card'>
            <h3>Course Materials</h3>
            <p>Access lecture notes, homework, and textbook excerpts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Start Analysis", use_container_width=True, type="primary"):
        st.session_state.page = "Analyze"
        st.rerun()


def render_analyze_page():
    st.markdown("## Analyze an Article")

    source = st.radio(
        "Select article source:",
        ["Upload PDF", "Paste URL", "Paste Text"],
        key="src_choice",
    )

    article_text = None
    article_title = None
    article_url = None

    # -----------------------------------------------------
    # Upload PDF
    # -----------------------------------------------------
    if source == "Upload PDF":
        pdf = st.file_uploader("Upload PDF", type=['pdf'])
        if pdf and st.button("Extract Text", key="extract_pdf"):
            try:
                article_text = extract_text_from_pdf(pdf.read())
                article_title = pdf.name
            except Exception as e:
                st.error(f"Error extracting text: {e}")

    # -----------------------------------------------------
    # Paste URL
    # -----------------------------------------------------
    if source == "Paste URL":
        url = st.text_input("Paste article URL")
        if url and st.button("Extract Text", key="extract_url"):
            try:
                article_text = extract_text_from_url(url)
                article_url = url
                article_title = url.split("/")[-1] or "Article"
            except Exception as e:
                st.error(f"Error fetching article: {e}")

    # -----------------------------------------------------
    # Paste Text
    # -----------------------------------------------------
    if source == "Paste Text":
        raw = st.text_area("Paste text", height=200)
        if raw and st.button("Use Text", key="use_text"):
            article_text = clean_extracted_text(raw)
            article_title = "Pasted Article"

    # If article extracted, add it
    if article_text:
        if len(article_text) < 120:
            st.warning("Extracted text too short.")
        else:
            added = add_article_to_session(
                article_text,
                article_title,
                source,
                article_url
            )

            if added:
                st.success(f"Imported: {article_title}")
                st.session_state.current_article = article_title
                st.session_state.assignment_questions = generate_assignment_questions(article_title)
                st.session_state.page = "Analyze"
                st.rerun()

    # -----------------------------------------------------
    # If an article is selected, show questions
    # -----------------------------------------------------
    if st.session_state.current_article:
        st.markdown(f"### Analyzing: {st.session_state.current_article}")

        qs = st.session_state.assignment_questions

        for i, q in enumerate(qs, 1):
            with st.expander(f"Question {i}: {q['question']}"):
                st.caption(f"Focus: {q['focus']} — Hint: {q['hint']}")
                ans = st.text_area("Your answer:", key=f"a{i}", height=120)

                if st.button(f"Get Feedback", key=f"fb{i}", use_container_width=True):
                    if ans.strip():
                        prompt = f"Question: {q['question']}\nStudent answer: {ans}"
                        fb = answer_question(prompt)
                        st.markdown("**Feedback:**")
                        st.markdown(fb)
                    else:
                        st.info("Enter an answer first.")


def render_ask_questions_page():
    st.markdown("## Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about statistical methods or concepts...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = answer_question(user_q)
                st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()


def render_course_materials_page():
    st.markdown("## Course Materials")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
            <h3>Textbook</h3>
            <p>Mathematical Statistics with Resampling and R</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <h3>Assessments</h3>
            <p>Homework, exams, and solution guides.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
            <h3>Lecture Notes</h3>
            <p>Week-by-week slides and handouts.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <h3>Resources</h3>
            <p><a href="https://brugnow.library.brown.edu" target="_blank">BruKnow Library</a></p>
            <p><a href="https://edstem.org/us/courses/80840/discussion" target="_blank">EdStem Discussion</a></p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# Sidebar Navigation (Figma-style)
# ==========================================================

def sidebar_nav():
    # Logo - try local file first, then GitHub URL
    logo_paths = ["assets/logo.png", "assets/logo.jpg", "logo.png", "logo.jpg"]
    logo_html = ""
    logo_found = False
    
    for logo_path in logo_paths:
        if Path(logo_path).exists():
            try:
                logo_bytes = Path(logo_path).read_bytes()
                logo_b64 = base64.b64encode(logo_bytes).decode()
                # Determine MIME type based on file extension
                if logo_path.lower().endswith('.jpg') or logo_path.lower().endswith('.jpeg'):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "image/png"
                logo_html = f'<img src="data:{mime_type};base64,{logo_b64}" class="sidebar-logo" alt="Isabelle Logo"/>'
                logo_found = True
                break
            except Exception as e:
                continue
    
    # Fallback to GitHub URL if local file not found
    if not logo_found:
        logo_html = '<img src="https://raw.githubusercontent.com/anikamahns907/agentTutor/main/assets/logo.png" class="sidebar-logo" alt="Isabelle Logo" onerror="this.style.display=\'none\'"/>'
    
    st.sidebar.markdown(
        f"""
        <div class="sidebar-logo-container">
            {logo_html}
            <div class="sidebar-title">PHP 1510/2510</div>
            <div class="sidebar-subtitle">Biostatistics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Menu
    menu = [
        ("Home", "Home"),
        ("Analyze an Article", "Analyze"),
        ("Ask Questions", "Questions"),
        ("Course Materials", "Materials"),
    ]

    # Add CSS to style buttons as nav items
    st.sidebar.markdown("""
    <style>
        .nav-button-wrapper {
            margin: 4px 0;
        }
        .nav-button-wrapper [data-testid="stButton"] {
            width: 100%;
            margin: 0;
    }
        .nav-button-wrapper [data-testid="stButton"] button {
            width: 100% !important;
            text-align: left !important;
            padding: 10px 14px !important;
            background-color: transparent !important;
            color: #E6E6E6 !important;
            border: none !important;
            border-left: 3px solid transparent !important;
            border-radius: 6px !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            transition: background-color 0.15s ease !important;
            box-shadow: none !important;
            cursor: pointer !important;
    }
        .nav-button-wrapper [data-testid="stButton"] button:hover {
            background-color: #1A1F25 !important;
            color: #E6E6E6 !important;
        }
        .nav-button-wrapper.active [data-testid="stButton"] button {
            background-color: #1A1F25 !important;
            border-left: 3px solid #38b6ff !important;
            padding-left: 11px !important;
            color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

    for label, key in menu:
        active = "active" if st.session_state.page == key else ""
        st.sidebar.markdown(f'<div class="nav-button-wrapper {active}">', unsafe_allow_html=True)
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True, type="secondary"):
            st.session_state.page = key
            st.query_params.page = key
            st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Chat actions
    if st.session_state.messages:
        st.sidebar.markdown("<div class='sidebar-section-label'>Chat</div>",
                            unsafe_allow_html=True)

        if st.sidebar.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()


# ==========================================================
# Apply CSS + Page Config
# ==========================================================

css_path = Path("styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>",
                unsafe_allow_html=True)

st.set_page_config(
    page_title="Isabelle — PHP1510",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Sync Query Params with Session State (BEFORE sidebar)
# ==========================================================

# Read query params and update session state
query = st.query_params.get("page", ["Home"])[0]
if query in ["Home", "Analyze", "Questions", "Materials"]:
    # Map URL keys to display names
    page_map = {
        "Home": "Home",
        "Analyze": "Analyze",
        "Questions": "Questions",
        "Materials": "Materials"
    }
    st.session_state.page = page_map.get(query, "Home")

# ==========================================================
# Main Layout Routing
# ==========================================================

with st.sidebar:
    sidebar_nav()

# Route to appropriate page
if st.session_state.page == "Home":
    render_home_page()
elif st.session_state.page == "Analyze":
    render_analyze_page()
elif st.session_state.page == "Questions":
    render_ask_questions_page()
elif st.session_state.page == "Materials":
    render_course_materials_page()
