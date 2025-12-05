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
if "analyze_chat_started" not in st.session_state:
    st.session_state.analyze_chat_started = False
if "analyze_messages" not in st.session_state:
    st.session_state.analyze_messages = []

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
    if len(stored_embs) > 0 and stored_embs.shape[1] != 384:
        recomputed = []
        for t in texts:
            recomputed.append(embed_text(t))
        stored_embs = np.array(recomputed)

    return texts, stored_embs, metadata


try:
    texts, embs, metadata = load_index()
    if texts is None:
        st.warning("⚠️ No document index found. Please run the ingestion script to create an index.")
except Exception as e:
    texts, embs, metadata = None, None, None
    st.warning(f"⚠️ Error loading index: {e}. Please run the ingestion script.")


# ---------------------------------------------
# Retrieve Context
# ---------------------------------------------
def retrieve_context(query, top_k=5, prioritize_article=False):
    contexts = []
    q_emb = embed_text(query)
    
    # If prioritizing article, search session articles first
    if prioritize_article:
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
    
    # Search permanent index
    if texts is not None and embs is not None:
        sims = np.dot(embs, q_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        for i in top_idx:
            contexts.append({"text": texts[i], "score": float(sims[i]), "metadata": metadata[i]})

    # If not prioritizing article, also search uploaded articles in session
    if not prioritize_article:
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
# Generate Assignment Questions
# ---------------------------------------------
def generate_assignment_questions(article_title: str):
    """Generate the standard 10 questions for article analysis."""
    return [
        {
            "question": "What statistical methods are used in this study?",
            "focus": "Identifying methods",
            "hint": "Look for mentions of tests, confidence intervals, p-values, regression, etc."
        },
        {
            "question": "What is the study design? (e.g., randomized controlled trial, observational study, etc.)",
            "focus": "Study design",
            "hint": "Consider how participants were selected and assigned to groups"
        },
        {
            "question": "How are the results interpreted? What do the statistical findings tell us?",
            "focus": "Interpretation",
            "hint": "Look at confidence intervals, p-values, and what conclusions are drawn"
        },
        {
            "question": "What are the limitations of the statistical analysis?",
            "focus": "Critical thinking",
            "hint": "Consider sample size, assumptions, potential biases, etc."
        },
        {
            "question": "How do the statistical methods used relate to concepts from our course?",
            "focus": "Course connection",
            "hint": "Connect to lecture materials, textbook concepts, and class discussions"
        },
        {
            "question": "How would you explain the methods used to colleagues that don't have any statistics background?",
            "focus": "Communication",
            "hint": "Think about how to translate technical statistical concepts into plain language"
        },
        {
            "question": "What's a 1-2 sentence summary of the main findings? Be as clear/concise as possible while still maintaining technical accuracy.",
            "focus": "Summary",
            "hint": "Focus on the key statistical findings and their practical significance"
        },
        {
            "question": "If you had this data, would there be another analysis you'd perform to gain more insights? Explain it without technical jargon.",
            "focus": "Critical thinking",
            "hint": "Consider what additional questions could be answered or what alternative approaches might be valuable"
        },
        {
            "question": "Would you recommend the authors communicate their findings differently? What changes would improve clarity?",
            "focus": "Communication",
            "hint": "Think about how statistical results are presented and whether they could be more accessible"
        },
        {
            "question": "Pick a specific piece of output (like a p-value or summary statistic). Interpret it.",
            "focus": "Interpretation",
            "hint": "Choose a specific statistical result from the article and explain what it means in practical terms"
        }
    ]


# ---------------------------------------------
# Build Prompt for AI
# ---------------------------------------------
def build_prompt(user_query, contexts, question_focus=None, is_article_feedback=False):
    ctx_text = "\n\n".join([f"[Source: {c['metadata'].get('source')}] {c['text']}" for c in contexts])
    
    system = """You are Isabelle, a helpful biostatistics communication tutor for PHP 1510/2510.

Your role is to help students improve their ability to communicate statistical concepts clearly and effectively to diverse audiences.

Key principles:
- Focus on clarity, conceptual reasoning, and plain-language explanations
- DO NOT grade or score - provide constructive, encouraging feedback
- Guide students with clarifying questions when they struggle
- Connect responses to course concepts from the textbook, lectures, and assignments
- Emphasize translating technical statistical jargon into accessible language
- Help students understand both the "what" and the "why" of statistical methods

When providing feedback:
- Acknowledge what the student got right
- Gently point out areas that need improvement
- Provide specific suggestions for how to improve
- Connect their answer to relevant course materials when possible
- Encourage deeper critical thinking
- Use examples from the course materials to illustrate concepts"""

    # Add question-specific guidance for article analysis
    if is_article_feedback and question_focus:
        focus_guidance = {
            "Identifying methods": "Focus on whether the student correctly identified statistical tests, models, or procedures. Help them understand the purpose of each method.",
            "Study design": "Assess if the student understands how the study was structured. Guide them to think about randomization, control groups, and potential biases.",
            "Interpretation": "Evaluate how well the student interprets statistical results. Help them connect p-values, confidence intervals, and effect sizes to practical meaning.",
            "Critical thinking": "Encourage the student to think beyond surface-level analysis. Help them consider assumptions, limitations, and alternative perspectives.",
            "Course connection": "Assess how well the student links the article's methods to course concepts. Guide them to specific chapters, lectures, or examples from class.",
            "Communication": "Focus on how well the student explains concepts in plain language. Help them avoid jargon while maintaining accuracy.",
            "Summary": "Evaluate clarity and conciseness. Help the student balance technical accuracy with accessibility."
        }
        
        guidance = focus_guidance.get(question_focus, "Provide constructive feedback that helps the student improve their understanding and communication.")
        system += f"\n\nFor this question (Focus: {question_focus}):\n{guidance}"

    return f"{system}\n\nContext from course materials:\n{ctx_text}\n\nStudent question/response:\n{user_query}"


# ---------------------------------------------
# AI Answer Helper
# ---------------------------------------------
def answer_question(query, question_focus=None, is_article_feedback=False, max_tokens=800):
    # Prioritize article context for article analysis feedback
    prioritize_article = is_article_feedback
    contexts = retrieve_context(query, top_k=5, prioritize_article=prioritize_article)
    prompt = build_prompt(query, contexts, question_focus=question_focus, is_article_feedback=is_article_feedback)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are Isabelle."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
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
        <div class='card' style="cursor: pointer;" onclick="go('Analyze')">
            <h3>Analyze Articles</h3>
            <p>Answer structured questions and get feedback.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card' style="cursor: pointer;" onclick="go('Questions')">
            <h3>Ask Questions</h3>
            <p>Learn statistical methods and interpretation concepts.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='card' style="cursor: pointer;" onclick="go('Materials')">
            <h3>Course Materials</h3>
            <p>Access lecture notes, homework, and textbook excerpts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Start Analysis", use_container_width=True, type="primary"):
        st.session_state.page = "Analyze"
        st.rerun()


def render_analyze_page():
    # Initialize session state variables if needed
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "question_answers" not in st.session_state:
        st.session_state.question_answers = {}
    if "pending_article_text" not in st.session_state:
        st.session_state.pending_article_text = None
    if "pending_article_title" not in st.session_state:
        st.session_state.pending_article_title = None
    if "pending_article_url" not in st.session_state:
        st.session_state.pending_article_url = None
    
    # Load logo for avatar
    logo_paths = ["assets/logo.png", "assets/logo.jpg", "logo.png", "logo.jpg"]
    logo_avatar = None
    
    for logo_path in logo_paths:
        if Path(logo_path).exists():
            try:
                logo_avatar = logo_path
                break
            except Exception:
                continue
    
    # If chat hasn't started, show start button
    if not st.session_state.analyze_chat_started:
    st.markdown("## Analyze an Article")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <p style="font-size: 1.2rem; color: #E6E6E6; margin-bottom: 2rem;">
                Start a conversation with Isabelle to analyze research articles and get feedback on your understanding.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start Chat with Isabelle", use_container_width=True, type="primary"):
                st.session_state.analyze_chat_started = True
                # Add welcome message
                welcome_msg = """Hi! I'm Isabelle, your biostatistics communication tutor. 

I'm here to help you analyze research articles and practice explaining statistical concepts clearly. 

**How would you like to share an article with me?**"""
                st.session_state.analyze_messages.append({
                    "role": "assistant",
                    "content": welcome_msg
                })
                st.rerun()
        return

    # Chat interface
    # Display chat history
    prev_role = None
    for idx, msg in enumerate(st.session_state.analyze_messages):
        # Only show avatar if role changed from previous message
        # If same role as previous message, don't show avatar
        if msg["role"] == "assistant":
            # Show avatar ONLY if previous message was NOT from assistant (i.e., from user or first message)
            if prev_role is None or prev_role != "assistant":
                # Show avatar - pass logo
                with st.chat_message(msg["role"], avatar=logo_avatar):
                    st.markdown(msg["content"])
            else:
                # Same role as previous, no avatar - don't pass avatar parameter
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        else:
            # User messages - no avatar
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Update prev_role after displaying message
        prev_role = msg["role"]
        
        # Show buttons if it's the welcome message (only for assistant messages)
        if msg["role"] == "assistant" and "How would you like to share" in msg["content"]:
            # Check if user hasn't already chosen
            if not st.session_state.get("analyze_upload_mode"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Upload PDF", key=f"btn_upload_pdf_{idx}", use_container_width=True):
                        st.session_state.analyze_messages.append({
                            "role": "user",
                            "content": "I'd like to upload a PDF"
                        })
                        st.session_state.analyze_upload_mode = "pdf"
                        st.rerun()
                with col2:
                    if st.button("Paste URL", key=f"btn_paste_url_{idx}", use_container_width=True):
                        st.session_state.analyze_messages.append({
                            "role": "user",
                            "content": "I'd like to paste a URL"
                        })
                        st.session_state.analyze_upload_mode = "url"
                        st.rerun()
                st.markdown("""
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1F232B;">
                    <p style="color: #9BA3AF; font-size: 0.9rem; margin-bottom: 0.5rem;">You can explore and find articles to analyze here:</p>
                    <p style="margin: 0.25rem 0;">
                        <a href="https://bruknow.library.brown.edu/discovery/search?vid=01BU_INST:BROWN" target="_blank" style="color: #99c5ff; text-decoration: none;">BruKnow Library</a> - Search for biostatistics and public health articles
                    </p>
                    <p style="margin: 0.25rem 0;">
                        <a href="https://www.nature.com" target="_blank" style="color: #99c5ff; text-decoration: none;">Nature.com</a> - Public health and medical research
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Handle article upload based on mode
    article_text = None
    article_title = None
    article_url = None
    upload_mode = st.session_state.get("analyze_upload_mode", None)

    # Show upload interface if mode is set and no article yet
    if upload_mode == "pdf" and not st.session_state.current_article:
        # Show assistant message if not already shown
        upload_msg_shown = any("Great! Please upload" in m.get("content", "") for m in st.session_state.analyze_messages if m.get("role") == "assistant")
        if not upload_msg_shown:
            # Check if previous message was from assistant
            last_msg_role = st.session_state.analyze_messages[-1]["role"] if st.session_state.analyze_messages else None
            show_upload_avatar = logo_avatar if last_msg_role != "assistant" else None
            with st.chat_message("assistant", avatar=show_upload_avatar):
                st.markdown("Great! Please upload your PDF file below.")
                st.session_state.analyze_messages.append({
                    "role": "assistant",
                    "content": "Great! Please upload your PDF file below."
                })
            st.rerun()
        
        # Show file uploader
        pdf = st.file_uploader("Upload PDF", type=['pdf'], key="analyze_pdf_upload", label_visibility="collapsed")
        if pdf:
            if st.button("Extract and Analyze", key="extract_pdf_btn", use_container_width=True):
            try:
                article_text = extract_text_from_pdf(pdf.read())
                article_title = pdf.name
                    # Store in session state so it persists across rerun
                    st.session_state.pending_article_text = article_text
                    st.session_state.pending_article_title = article_title
                    st.session_state.pending_article_url = None
                    st.session_state.analyze_messages.append({
                        "role": "user",
                        "content": f"Uploaded: {article_title}"
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error extracting text: {e}")

    elif upload_mode == "url" and not st.session_state.current_article:
        # Show assistant message if not already shown
        url_msg_shown = any("Perfect! Paste the article" in m.get("content", "") for m in st.session_state.analyze_messages if m.get("role") == "assistant")
        if not url_msg_shown:
            # Check if previous message was from assistant
            last_msg_role = st.session_state.analyze_messages[-1]["role"] if st.session_state.analyze_messages else None
            show_url_avatar = logo_avatar if last_msg_role != "assistant" else None
            with st.chat_message("assistant", avatar=show_url_avatar):
                st.markdown("Perfect! Paste the article URL below.")
                st.session_state.analyze_messages.append({
                    "role": "assistant",
                    "content": "Perfect! Paste the article URL below."
                })
            st.rerun()
        
        # Show URL input
        url = st.text_input("Paste article URL", key="analyze_url_input", label_visibility="collapsed", placeholder="https://...")
        if url:
            if st.button("Extract and Analyze", key="extract_url_btn", use_container_width=True):
            try:
                article_text = extract_text_from_url(url)
                article_url = url
                article_title = url.split("/")[-1] or "Article"
                    # Store in session state so it persists across rerun
                    st.session_state.pending_article_text = article_text
                    st.session_state.pending_article_title = article_title
                    st.session_state.pending_article_url = article_url
                    st.session_state.analyze_messages.append({
                        "role": "user",
                        "content": f"Shared URL: {url}"
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error fetching article: {e}")

    # If article extracted, add it and show questions
    # Check session state first, then local variable
    article_text = st.session_state.pending_article_text if st.session_state.pending_article_text else article_text
    if article_text:
        # Use session state values if available
        if st.session_state.pending_article_title:
            article_title = st.session_state.pending_article_title
        if st.session_state.pending_article_url:
            article_url = st.session_state.pending_article_url
        if len(article_text) < 120:
            # Check if previous message was from assistant
            last_msg_role = st.session_state.analyze_messages[-1]["role"] if st.session_state.analyze_messages else None
            show_warning_avatar = logo_avatar if last_msg_role != "assistant" else None
            with st.chat_message("assistant", avatar=show_warning_avatar):
                st.warning("The extracted text seems too short. Please try a different article.")
        else:
            added = add_article_to_session(
                article_text,
                article_title,
                upload_mode or "upload",
                article_url
            )

            if added:
                st.session_state.current_article = article_title
                try:
                st.session_state.assignment_questions = generate_assignment_questions(article_title)
                    st.session_state.current_question_index = 0
                    st.session_state.question_answers = {}
                    
                    success_msg = f"""Successfully imported: **{article_title}**

I've analyzed your article! I'll guide you through 10 questions to help you practice explaining statistical concepts clearly.

Let's start with the first question:"""
                    st.session_state.analyze_messages.append({
                        "role": "assistant",
                        "content": success_msg
                    })
                    
                    # Add the first question to chat
                    if st.session_state.assignment_questions:
                        first_q = st.session_state.assignment_questions[0]
                        question_msg = f"""**Question 1:** {first_q['question']}

**Focus:** {first_q['focus']}
**Hint:** {first_q['hint']}

Please answer this question in the chat below."""
                        st.session_state.analyze_messages.append({
                            "role": "assistant",
                            "content": question_msg
                        })
                except Exception as e:
                    st.error(f"Error generating questions: {e}")
                    st.session_state.assignment_questions = []
                # Clear pending article data
                st.session_state.pending_article_text = None
                st.session_state.pending_article_title = None
                st.session_state.pending_article_url = None
                st.session_state.analyze_upload_mode = None
                st.rerun()

    # If an article is selected, handle conversational question flow
    if st.session_state.current_article:
        qs = st.session_state.assignment_questions
        current_idx = st.session_state.current_question_index
        
        # Check if user has answered the current question and move to next
        if qs and current_idx < len(qs):
            current_q = qs[current_idx]
            question_key = f"q{current_idx + 1}"
            
            # Check if this question was just answered in chat
            last_user_msg = None
            last_assistant_msg = None
            if len(st.session_state.analyze_messages) >= 2:
                last_user_msg = st.session_state.analyze_messages[-1] if st.session_state.analyze_messages[-1].get("role") == "user" else None
                last_assistant_msg = st.session_state.analyze_messages[-2] if len(st.session_state.analyze_messages) >= 2 and st.session_state.analyze_messages[-2].get("role") == "assistant" else None
            
            # If user just answered and we haven't asked next question yet
            if last_user_msg and last_assistant_msg and f"Question {current_idx + 1}:" in last_assistant_msg.get("content", ""):
                # Check if we need to ask next question
                if current_idx + 1 < len(qs):
                    next_q = qs[current_idx + 1]
                    question_msg = f"""**Question {current_idx + 2}:** {next_q['question']}

**Focus:** {next_q['focus']}
**Hint:** {next_q['hint']}

Please answer this question in the chat below."""
                    st.session_state.analyze_messages.append({
                        "role": "assistant",
                        "content": question_msg
                    })
                    st.session_state.current_question_index = current_idx + 1
                    st.rerun()

    # Chat input at bottom
    if st.session_state.analyze_chat_started:
        # Determine placeholder based on context
        if st.session_state.current_article and st.session_state.assignment_questions:
            current_idx = st.session_state.current_question_index
            if current_idx < len(st.session_state.assignment_questions):
                placeholder = f"Answer Question {current_idx + 1}..."
            else:
                placeholder = "Ask Isabelle about the article or analysis..."
        else:
            placeholder = "Ask Isabelle about the article or analysis..."
        
        user_input = st.chat_input(placeholder)
        if user_input:
            # Check previous message role BEFORE adding user message
            last_msg_role = st.session_state.analyze_messages[-1]["role"] if st.session_state.analyze_messages else None
            st.session_state.analyze_messages.append({"role": "user", "content": user_input})
            
            # Check if this is an answer to a current question
            if st.session_state.current_article and st.session_state.assignment_questions:
                current_idx = st.session_state.current_question_index
                if current_idx < len(st.session_state.assignment_questions):
                    current_q = st.session_state.assignment_questions[current_idx]
                    # Store the answer
                    st.session_state.question_answers[current_idx] = user_input
                    
                    # Provide feedback on the answer
                    # Show avatar since previous message was from user (we checked before adding)
                    show_feedback_avatar = logo_avatar
                    
                    with st.chat_message("assistant", avatar=show_feedback_avatar):
                        with st.spinner("Analyzing your answer..."):
                            prompt = f"Question: {current_q['question']}\n\nStudent answer: {user_input}\n\nPlease provide constructive feedback on this answer."
                            feedback = answer_question(
                                prompt, 
                                question_focus=current_q['focus'],
                                is_article_feedback=True,
                                max_tokens=1000
                            )
                            st.markdown(f"**Feedback on Question {current_idx + 1}:**\n\n{feedback}")
                    st.session_state.analyze_messages.append({"role": "assistant", "content": f"**Feedback on Question {current_idx + 1}:**\n\n{feedback}"})
                    
                    # Move to next question if available
                    # Next question won't show avatar since previous message was from assistant (feedback)
                    # This will be handled correctly in the display loop
                    if current_idx + 1 < len(st.session_state.assignment_questions):
                        next_q = st.session_state.assignment_questions[current_idx + 1]
                        question_msg = f"""**Question {current_idx + 2}:** {next_q['question']}

**Focus:** {next_q['focus']}
**Hint:** {next_q['hint']}

Please answer this question in the chat below."""
                        st.session_state.analyze_messages.append({
                            "role": "assistant",
                            "content": question_msg
                        })
                        st.session_state.current_question_index = current_idx + 1
                    else:
                        # All questions answered
                        completion_msg = """Great work! You've completed all 10 analysis questions. 

Would you like to:
- Review your answers
- Ask follow-up questions about the article
- Start analyzing a new article"""
                        st.session_state.analyze_messages.append({
                            "role": "assistant",
                            "content": completion_msg
                        })
                        st.session_state.current_question_index = len(st.session_state.assignment_questions)
            else:
                # General question about the article
                # Previous message was from user (we checked before adding), so show avatar
                show_response_avatar = logo_avatar
                
                with st.chat_message("assistant", avatar=show_response_avatar):
                    with st.spinner("Thinking..."):
                        response = answer_question(user_input, is_article_feedback=bool(st.session_state.current_article))
                        st.markdown(response)
                st.session_state.analyze_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Clear chat button
        if st.button("Start New Analysis", key="clear_analyze_chat"):
            st.session_state.analyze_chat_started = False
            st.session_state.analyze_messages = []
            st.session_state.current_article = None
            st.session_state.assignment_questions = []
            st.session_state.current_question_index = 0
            st.session_state.question_answers = {}
            st.session_state.pending_article_text = None
            st.session_state.pending_article_title = None
            st.session_state.pending_article_url = None
            st.session_state.session_vectorstore = {"texts": [], "embs": None, "metadata": []}
            st.session_state.analyze_upload_mode = None
            st.rerun()


def render_ask_questions_page():
    st.markdown("## Ask Questions")

    # Load logo for avatar
    logo_paths = ["assets/logo.png", "assets/logo.jpg", "logo.png", "logo.jpg"]
    logo_avatar = None
    
    for logo_path in logo_paths:
        if Path(logo_path).exists():
            try:
                logo_avatar = logo_path
                break
            except Exception:
                continue

    for msg in st.session_state.messages:
        avatar = logo_avatar if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about statistical methods or concepts...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar=logo_avatar):
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
            <div class="sidebar-title">Your Biostatistics Peer</div>
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
            border-left: 3px solid #99c5ff !important;
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

# Set favicon/icon
icon_paths = ["assets/logo.png", "logo.png"]
page_icon = None
for icon_path in icon_paths:
    if Path(icon_path).exists():
        page_icon = icon_path
        break

st.set_page_config(
    page_title="Isabelle — PHP1510",
    page_icon=page_icon if page_icon else "📊",
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
