"""
Isabelle - AI Tutor for PHP 1510/2510 - Principles of Biostatistics and Data Analysis

Features:
- Article Analysis: Analyze research articles using course methods
- Article Recommendations: Get suggested articles to analyze
"""

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
    letter = None  # type: ignore
    canvas = None  # type: ignore

# --- Load environment variables ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_article" not in st.session_state:
    st.session_state.current_article = None
if "assignment_questions" not in st.session_state:
    st.session_state.assignment_questions = []
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False
if "session_vectorstore" not in st.session_state:
    st.session_state.session_vectorstore = {"texts": [], "embs": None, "metadata": []}
if "current_article_text" not in st.session_state:
    st.session_state.current_article_text = ""
if "article_metadata" not in st.session_state:
    st.session_state.article_metadata = {}
if "article_source_choice" not in st.session_state:
    st.session_state.article_source_choice = "Upload PDF"
if "article_url_input" not in st.session_state:
    st.session_state.article_url_input = ""
if "article_text_input" not in st.session_state:
    st.session_state.article_text_input = ""
if "pending_article_import" not in st.session_state:
    st.session_state.pending_article_import = None

# --- Load embedding model ---
@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, embedding_model = load_embedding_model()

# --- Helper: embed text ---
def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = embedding_model(**inputs)
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).squeeze().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings)
    return embeddings

# --- Load index ---
@st.cache_resource
def load_index():
    index_path = Path("index/index.pkl")
    if not index_path.exists():
        return None, None, None
    
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None, None
    
    texts = data.get("texts", [])
    stored_embs = data.get("embs", np.array([]))
    metadata = data.get("metadata", [])
    
    if len(texts) == 0 or stored_embs.size == 0:
        return None, None, None
    
    if stored_embs.shape[1] != 384:
        try:
            new_embs = []
            for text in texts:
                vec = embed_text(text)
                new_embs.append(vec)
            embs = np.array(new_embs)
        except Exception as e:
            st.error(f"Error recomputing embeddings: {e}")
            return None, None, None
    else:
        embs = stored_embs
    
    return texts, embs, metadata

try:
    texts, embs, metadata = load_index()
except Exception:
    texts, embs, metadata = None, None, None

# --- Helper: retrieve context ---
def retrieve_context(query, top_k=5, source_filter=None):
    """Retrieve context from both permanent index and session vectorstore."""
    all_contexts = []
    
    # Search permanent index
    if texts is not None and embs is not None and embs.shape[0] > 0:
        try:
            q_emb = embed_text(query)
            sims = np.dot(embs, q_emb)
            top_indices = np.argsort(sims)[-top_k:][::-1]
            
            for idx in top_indices:
                if idx >= len(texts):
                    continue
                if source_filter and metadata and idx < len(metadata):
                    source_type = metadata[idx].get('source_type', '')
                    if source_filter not in source_type:
                        continue
                all_contexts.append({
                    'text': texts[idx],
                    'score': float(sims[idx]),
                    'metadata': metadata[idx] if metadata and idx < len(metadata) else {}
                })
        except Exception as e:
            pass
    
    # Search session vectorstore (uploaded articles)
    session_store = st.session_state.session_vectorstore
    if session_store.get("texts") and len(session_store["texts"]) > 0:
        try:
            q_emb = embed_text(query)
            session_embs = session_store.get("embs")
            if session_embs is not None and session_embs.shape[0] > 0:
                sims = np.dot(session_embs, q_emb)
                top_indices = np.argsort(sims)[-top_k:][::-1]
                
                for idx in top_indices:
                    if idx >= len(session_store["texts"]):
                        continue
                    if source_filter:
                        session_meta = session_store.get("metadata", [])
                        if idx < len(session_meta):
                            source_type = session_meta[idx].get('source_type', '')
                            if source_filter not in source_type:
                                continue
                    all_contexts.append({
                        'text': session_store["texts"][idx],
                        'score': float(sims[idx]),
                        'metadata': session_store.get("metadata", [])[idx] if idx < len(session_store.get("metadata", [])) else {}
                    })
        except Exception as e:
            pass
    
    # Sort all contexts by score and return top_k
    all_contexts.sort(key=lambda x: x['score'], reverse=True)
    return all_contexts[:top_k]

# --- Helper: add article to session vectorstore ---
def add_article_to_session(text: str, article_title: str, source_label: str, source_url: str | None = None):
    """Chunk and embed an article, adding it to the session vectorstore."""
    chunks, metadata_list = prepare_article_documents(text, article_title, source_label, source_url)
    
    if not chunks:
        return False
    
    # Generate embeddings for chunks
    chunk_embs = []
    for chunk in chunks:
        emb = embed_text(chunk)
        chunk_embs.append(emb)
    
    chunk_embs_array = np.array(chunk_embs)
    
    # Add to session vectorstore
    session_store = st.session_state.session_vectorstore
    session_store["texts"].extend(chunks)
    if session_store["embs"] is None:
        session_store["embs"] = chunk_embs_array
    else:
        session_store["embs"] = np.vstack([session_store["embs"], chunk_embs_array])
    session_store["metadata"].extend(metadata_list)
    
    return True

# --- Helper: recommend articles ---
def recommend_articles(topic=None, public_health_focus=True):
    """Recommend research articles for students to analyze."""
    suggestions = [
        {
            "title": "Vaccine Effectiveness Studies",
            "source": "Nature Public Health",
            "keywords": ["vaccines", "effectiveness", "public health"],
            "url": "https://www.nature.com/subjects/public-health",
            "bruknow_url": get_bruknow_search_url("vaccines effectiveness public health statistics"),
            "why": "Great for analyzing confidence intervals, hypothesis testing, and statistical significance in vaccine research"
        },
        {
            "title": "Epidemiological Studies on Disease Outbreaks",
            "source": "BruKnow Library",
            "keywords": ["epidemiology", "outbreaks", "statistical modeling"],
            "search": "biostatistics public health epidemiology",
            "url": get_bruknow_search_url("biostatistics public health epidemiology"),
            "bruknow_url": get_bruknow_search_url("biostatistics public health epidemiology"),
            "why": "Excellent for applying regression analysis, sampling distributions, and resampling methods"
        },
        {
            "title": "Clinical Trial Analysis",
            "source": "Nature Medicine",
            "keywords": ["clinical trials", "RCT", "statistical analysis"],
            "url": "https://www.nature.com/subjects/clinical-trials",
            "bruknow_url": get_bruknow_search_url("clinical trials randomized controlled trial statistics"),
            "why": "Perfect for understanding p-values, hypothesis testing, and interpreting statistical results"
        },
        {
            "title": "Public Health Intervention Studies",
            "source": "BruKnow Library",
            "keywords": ["interventions", "public health", "evaluation"],
            "search": "public health interventions statistical evaluation",
            "url": get_bruknow_search_url("public health interventions statistical evaluation"),
            "bruknow_url": get_bruknow_search_url("public health interventions statistical evaluation"),
            "why": "Ideal for applying course concepts to real-world public health problems"
        },
        {
            "title": "Biostatistics Research Articles",
            "source": "BruKnow Library",
            "keywords": ["biostatistics", "public health", "statistical methods"],
            "url": get_bruknow_search_url("biostatistics public health"),
            "bruknow_url": get_bruknow_search_url("biostatistics public health"),
            "why": "Find articles covering various statistical methods used in public health research"
        },
        {
            "title": "Nature Public Health Articles",
            "source": "Nature & BruKnow",
            "keywords": ["nature", "public health", "research"],
            "url": "https://www.nature.com/subjects/public-health",
            "bruknow_url": get_bruknow_search_url("nature public health"),
            "why": "Access Nature journal articles through BruKnow on public health topics"
        }
    ]
    
    if topic:
        # Filter by topic relevance
        filtered = [s for s in suggestions if topic.lower() in s['why'].lower() or any(topic.lower() in k for k in s['keywords'])]
        return filtered if filtered else suggestions[:2]
    
    return suggestions

# --- Helper: generate assignment questions ---
def generate_assignment_questions(article_title, article_content=None):
    """Generate structured questions for article analysis assignment."""
    
    base_questions = [
        {
            "question": "What statistical methods are used in this study?",
            "focus": "Statistical Methods",
            "hint": "Look for mentions of tests, confidence intervals, p-values, regression, etc."
        },
        {
            "question": "What is the study design? (e.g., randomized controlled trial, observational study, etc.)",
            "focus": "Study Design",
            "hint": "Consider how participants were selected and assigned to groups"
        },
        {
            "question": "How are the results interpreted? What do the statistical findings tell us?",
            "focus": "Interpretation",
            "hint": "Look at confidence intervals, p-values, and what conclusions are drawn"
        },
        {
            "question": "What are the limitations of the statistical analysis?",
            "focus": "Limitations",
            "hint": "Consider sample size, assumptions, potential biases, etc."
        },
        {
            "question": "How do the statistical methods used relate to concepts from our course?",
            "focus": "Course Connection",
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
            "focus": "Critical Thinking",
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
    
    return base_questions

# --- Helper: build prompt ---
def build_prompt(query, contexts):
    context_text = "\n\n".join([f"[Source: {ctx['metadata'].get('source', 'unknown')}]\n{ctx['text']}" 
                                for ctx in contexts[:5]])
    
    system_prompt = """You are Isabelle, a communication coach and article analysis tutor for PHP 1510/2510. Your role is to help students practice communicating statistical concepts and research findings more clearly.

Your core job is to be a communication coach, not a judge. You do NOT grade, score, or use rubrics.

Guidelines:
- Help students identify statistical methods used in articles
- Guide them to evaluate methods using course concepts
- Ask clarifying follow-up questions that improve clarity
- Encourage rephrasings and conceptual simplifications
- Provide definitions and analogies when helpful
- Offer pacing cues ("Want me to go deeper?" "Should I simplify this?")
- Help students break down statistical ideas into clearer explanations
- Guide students through explaining research papers
- Be conversational and encouraging
- Push for deeper thinking but stay supportive
- At the end of your response, if the student needs further clarification, suggest they ask on EdStem (https://edstem.org/us/courses/80840/discussion) or speak to Professor Lipman
"""
    
    prompt = f"""{system_prompt}

Context from course materials (textbook, lectures, examples):
{context_text}

Student is analyzing an article. Their question or response: {query}

Help them analyze the article using concepts from the course. Guide them to think critically about the statistical methods. If they need further clarification beyond what you can provide, remind them to ask on EdStem or speak to Professor Lipman."""
    
    return prompt

# --- Helper: answer question ---
def answer_question(query):
    if texts is None or embs is None:
        return "Please run the ingestion script first to create the index."
    
    # Always use all sources - no filtering
    contexts = retrieve_context(query, top_k=5, source_filter=None)
    
    if not contexts:
        edstem_note = "\n\n💡 **Need more help?** If you need further clarification, please ask on [EdStem](https://edstem.org/us/courses/80840/discussion) or speak to Professor Lipman."
        return "I couldn't find relevant information. Please try rephrasing your question." + edstem_note
    
    prompt = build_prompt(query, contexts)
    
    messages = [
        {"role": "system", "content": "You are Isabelle, a helpful statistics tutor assistant."},
    ]
    
    for msg in st.session_state.messages[-8:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    
    answer = response.choices[0].message.content
    
    # Add EdStem guidance if not already present
    if "edstem" not in answer.lower() and "professor lipman" not in answer.lower():
        answer += "\n\n💡 **Need more help?** If you need further clarification, please ask on [EdStem](https://edstem.org/us/courses/80840/discussion) or speak to Professor Lipman."
    
    return answer

# --- Helper: export chat ---
def export_chat_to_pdf(messages, for_professor=False):
    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        y = height - 50
        pdf.setFont("Helvetica-Bold", 16)
        if for_professor:
            pdf.drawString(50, y, "Isabelle Chat History - For Professor Lipman")
        else:
            pdf.drawString(50, y, "Isabelle Chat History")
        y -= 30
        
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if for_professor:
            y -= 20
            pdf.drawString(50, y, "Course: PHP 1510/2510 - Principles of Biostatistics")
            y -= 20
            pdf.drawString(50, y, "This chat export is intended for Professor Lipman")
        y -= 20
        
        pdf.setFont("Helvetica", 10)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                pdf.setFillColorRGB(0, 0, 0.8)
                pdf.drawString(50, y, "You:")
            else:
                pdf.setFillColorRGB(0, 0.6, 0)
                pdf.drawString(50, y, "Isabelle:")
            
            y -= 15
            pdf.setFillColorRGB(0, 0, 0)
            
            words = content.split()
            line = ""
            for word in words:
                if pdf.stringWidth(line + word, "Helvetica", 10) < width - 100:
                    line += word + " "
                else:
                    if line:
                        pdf.drawString(70, y, line.strip())
                        y -= 15
                    line = word + " "
                    if y < 50:
                        pdf.showPage()
                        y = height - 50
            
            if line:
                pdf.drawString(70, y, line.strip())
                y -= 20
            
            if y < 50:
                pdf.showPage()
                y = height - 50
        
        pdf.save()
        buffer.seek(0)
        return buffer, "application/pdf"
    else:
        if for_professor:
            text_content = f"Isabelle Chat History - For Professor Lipman\n"
            text_content += f"Course: PHP 1510/2510 - Principles of Biostatistics\n"
            text_content += f"This chat export is intended for Professor Lipman\n"
        else:
            text_content = f"Isabelle Chat History\n"
        text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text_content += "=" * 50 + "\n\n"
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                text_content += "You:\n"
            else:
                text_content += "Isabelle:\n"
            
            text_content += f"{content}\n\n"
            text_content += "-" * 50 + "\n\n"
        
        buffer = io.BytesIO(text_content.encode('utf-8'))
        return buffer, "text/plain"

# --- Streamlit UI ---
# Check for logo for page icon
logo_icon_paths = ["assets/logo.png", "assets/logo.jpg", "logo.png", "logo.jpg"]
page_icon = "📊"  # Default
for icon_path in logo_icon_paths:
    if Path(icon_path).exists():
        page_icon = icon_path
        break

st.set_page_config(
    page_title="Isabelle - PHP 1510/2510",
    page_icon=page_icon if isinstance(page_icon, str) and page_icon.endswith(('.png', '.jpg')) else "📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS - black background, white text
st.markdown("""
<style>
    /* Dark theme color scheme */
    :root {
        --primary-color: #3b82f6;
        --primary-dark: #2563eb;
        --text-primary: #ffffff;
        --text-secondary: #d1d5db;
        --bg-dark: #000000;
        --bg-secondary: #1a1a1a;
        --border-color: #333333;
    }
    
    /* Main background - black */
    .main {
        background-color: var(--bg-dark) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--bg-dark) !important;
    }
    
    /* All text - white */
    .main .stMarkdown,
    .main .stMarkdown p,
    .main .stMarkdown li,
    .main .stMarkdown span,
    .main .stMarkdown div,
    .main p,
    .main span,
    .main div,
    .main label,
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6 {
        color: var(--text-primary) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Sidebar - dark background */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
        background-color: var(--bg-secondary) !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        margin: 1rem 0;
    }
    
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: var(--text-primary) !important;
    }
    
    /* Input fields - dark theme */
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
    }
    
    .stTextArea > div > div > textarea {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
    }
    
    /* Select boxes - dark theme */
    .stSelectbox > div > div {
        background-color: var(--bg-secondary) !important;
    }
    
    .stSelectbox label,
    .stRadio label,
    .stTextInput label,
    .stTextArea label {
        color: var(--text-primary) !important;
    }
    
    /* Radio buttons - dark theme */
    .stRadio > div {
        background-color: transparent !important;
    }
    
    .stRadio label {
        color: var(--text-primary) !important;
    }
    
    /* Input placeholders */
    input::placeholder,
    textarea::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Links */
    a {
        color: var(--primary-color) !important;
    }
    
    /* Status indicators - dark theme */
    .stSuccess {
        background-color: #064e3b !important;
        border-left: 4px solid #10b981;
        padding: 0.75rem 1rem;
        border-radius: 4px;
    }
    
    .stSuccess * {
        color: #6ee7b7 !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
    }
    
    .stError * {
        color: #fca5a5 !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        border-left: 4px solid #3b82f6;
    }
    
    .stInfo * {
        color: #93c5fd !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
        color: white !important;
    }
    
    .stButton > button * {
        color: white !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Link button */
    .stLinkButton > a {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Expander - dark theme */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--bg-secondary) !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: var(--bg-secondary) !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* Caption */
    .stCaption {
        color: var(--text-secondary) !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat input */
    .stChatInput > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Simplified layout
with st.sidebar:
    # Logo
    logo_paths = ["assets/logo.png", "assets/logo.jpg", "assets/logo.svg", "logo.png", "logo.jpg", "logo.svg"]
    logo_found = False
    for logo_path in logo_paths:
        if Path(logo_path).exists():
            import base64
            logo_bytes = Path(logo_path).read_bytes()
            logo_b64 = base64.b64encode(logo_bytes).decode()
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; display: block;" />
            </div>
            """, unsafe_allow_html=True)
            logo_found = True
            break
    
    if not logo_found:
        st.markdown("### Isabelle")
    st.markdown("*PHP 1510/2510 - Biostatistics*")
    st.markdown("---")
    
    # Quick Links
    st.markdown("**Quick Links**")
    st.markdown("[📚 BruKnow](https://bruknow.library.brown.edu/discovery/search?vid=01BU_INST:BROWN) | [💬 EdStem](https://edstem.org/us/courses/80840/discussion)")
    
    st.markdown("---")
    
    # Actions - Simplified
    if st.session_state.messages:
        st.markdown("**Chat Actions**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export", use_container_width=True):
                export_buffer, mime_type = export_chat_to_pdf(st.session_state.messages, for_professor=False)
                file_ext = ".pdf" if mime_type == "application/pdf" else ".txt"
                file_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                st.download_button(
                    label="Download",
                    data=export_buffer,
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True
                )
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_article = None
                st.session_state.assignment_questions = []
                st.session_state.session_vectorstore = {"texts": [], "embs": None, "metadata": []}
                st.session_state.current_article_text = ""
                st.session_state.article_metadata = {}
                st.session_state.pending_article_import = None
                st.rerun()

# Main content - Welcome and Header
if not st.session_state.current_article and not st.session_state.messages:
    # Welcome screen
    st.markdown("# 👋 Welcome to Isabelle")
    st.markdown("### Your AI Tutor for PHP 1510/2510 - Principles of Biostatistics")
    st.markdown("---")
    
    st.markdown("""
    **Isabelle** is your intelligent learning assistant designed to help you analyze research articles using statistical methods from your course.
    
    ### What Isabelle can do:
    
    📊 **Article Analysis** - Get guided questions to analyze research articles and apply course concepts
    
    💬 **Ask Questions** - Get help understanding statistical methods, study designs, and data interpretation
    
    📚 **Course Materials** - Access insights from your textbook, lecture slides, and assessments
    
    🔍 **Find Articles** - Get recommendations for articles from BruKnow and Nature on topics like vaccines, public health, and epidemiology
    
    ### How to get started:
    
    1. **Browse article recommendations** below or upload your own PDF
    2. **Click "Analyze"** on any article to see the 10 analysis questions
    3. **Answer questions** and get feedback on your responses
    4. **Ask questions** in the chat to dive deeper into concepts
    
    ---
    """)
    
    # Status indicator
    if texts is not None and embs is not None and len(texts) > 0 and embs.size > 0:
        st.success("✓ Ready - Course materials loaded")
    else:
        st.warning("⚠ Index not found - Some features may be limited")
    
    st.markdown("---")
else:
    # Regular header when working
    st.markdown("## Isabelle - Article Analysis")
    st.caption("PHP 1510/2510 - Principles of Biostatistics")
    
    # Status indicator (compact)
    if texts is not None and embs is not None and len(texts) > 0 and embs.size > 0:
        st.success("✓ Ready")
    else:
        st.error("⚠ Index not found")
    
    st.markdown("---")

# Article import section - NEW
if not st.session_state.current_article:
    st.markdown("### 📄 Upload a Research Article")
    st.markdown("Import any biostatistics/public health research article for analysis.")
    st.markdown("")
    
    # Article source selection
    article_source = st.radio(
        "Choose an Article Source:",
        ["Upload PDF", "Paste URL", "Paste Text", "Search BruKnow", "Search Public Health Articles"],
        key="article_source_choice",
        horizontal=False
    )
    
    article_imported = False
    article_title = ""
    article_text = ""
    article_url = None
    
    if article_source == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'], key="pdf_uploader")
        if uploaded_file:
            if st.button("Extract Text", key="extract_pdf"):
                with st.spinner("Extracting text from PDF..."):
                    try:
                        article_text = extract_text_from_pdf(uploaded_file.read())
                        article_title = uploaded_file.name
                        article_imported = True
                    except Exception as e:
                        st.error(f"Error extracting text: {e}")
    
    elif article_source == "Paste URL":
        url_input = st.text_input("Paste article URL", placeholder="https://...", key="url_input")
        if url_input:
            if st.button("Extract Text", key="extract_url"):
                with st.spinner("Fetching and extracting text from URL..."):
                    try:
                        article_text = extract_text_from_url(url_input)
                        article_title = url_input.split("/")[-1] or "Article from URL"
                        article_url = url_input
                        article_imported = True
                    except Exception as e:
                        st.error(f"Error fetching URL: {e}")
    
    elif article_source == "Paste Text":
        text_input = st.text_area("Paste article text", height=200, key="text_input")
        if text_input:
            if st.button("Use Text", key="use_text"):
                article_text = clean_extracted_text(text_input)
                article_title = "Pasted Article"
                article_imported = True
    
    elif article_source == "Search BruKnow":
        search_query = st.text_input("Search BruKnow articles", placeholder="e.g., vaccine effectiveness", key="bruknow_search")
        if search_query:
            if st.button("Search", key="search_bruknow"):
                with st.spinner("Searching BruKnow..."):
                    results = search_bruknow_articles(search_query, max_results=5)
                    if results:
                        st.markdown("**Search Results:**")
                        for idx, result in enumerate(results):
                            with st.expander(f"{result['title']} - {result['source']}"):
                                st.markdown(f"**Snippet:** {result.get('snippet', '')[:200]}...")
                                if result.get('url'):
                                    st.markdown(f"[🔗 Open Article]({result['url']})")
                                if result.get('bruknow_url'):
                                    st.markdown(f"[📚 BruKnow Link]({result['bruknow_url']})")
                                if st.button(f"Import This Article", key=f"import_bruknow_{idx}"):
                                    with st.spinner("Importing article..."):
                                        try:
                                            article_text = extract_text_from_url(result['url'])
                                            st.session_state.pending_article_import = {
                                                "text": article_text,
                                                "title": result['title'],
                                                "url": result['url'],
                                                "source": "BruKnow"
                                            }
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error importing article: {e}")
                    else:
                        st.info("No results found. Try a different search term.")
    
    elif article_source == "Search Public Health Articles":
        search_query = st.text_input("Search public health articles", placeholder="e.g., epidemiology, disease outbreak", key="ph_search")
        if search_query:
            if st.button("Search", key="search_ph"):
                with st.spinner("Searching public health sources..."):
                    results = search_public_health_articles(search_query, max_results=5)
                    if results:
                        st.markdown("**Search Results:**")
                        for idx, result in enumerate(results):
                            with st.expander(f"{result['title']} - {result['source']}"):
                                st.markdown(f"**Snippet:** {result.get('snippet', '')[:200]}...")
                                if result.get('url'):
                                    st.markdown(f"[🔗 Open Article]({result['url']})")
                                if st.button(f"Import This Article", key=f"import_ph_{idx}"):
                                    with st.spinner("Importing article..."):
                                        try:
                                            article_text = extract_text_from_url(result['url'])
                                            st.session_state.pending_article_import = {
                                                "text": article_text,
                                                "title": result['title'],
                                                "url": result['url'],
                                                "source": result['source']
                                            }
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error importing article: {e}")
                    else:
                        st.info("No results found. Try a different search term.")
    
    # Process pending article import (from search results)
    if st.session_state.pending_article_import:
        pending = st.session_state.pending_article_import
        article_text = pending["text"]
        article_title = pending["title"]
        article_url = pending.get("url")
        source_label = pending.get("source", "Imported Article")
        
        if len(article_text.strip()) > 100:  # Minimum text length
            if add_article_to_session(article_text, article_title, source_label, article_url):
                st.session_state.current_article = article_title
                st.session_state.current_article_text = article_text
                st.session_state.article_metadata = {"source": source_label, "url": article_url}
                st.session_state.assignment_questions = generate_assignment_questions(article_title)
                st.session_state.pending_article_import = None
                st.success(f"✓ Article '{article_title}' imported and ready for analysis!")
                st.rerun()
            else:
                st.error("Failed to process article. Please try again.")
                st.session_state.pending_article_import = None
        else:
            st.warning("Extracted text is too short. Please check the article source.")
            st.session_state.pending_article_import = None
    
    # Process imported article (from direct upload/paste)
    if article_imported and article_text:
        if len(article_text.strip()) > 100:  # Minimum text length
            # Add to session vectorstore
            source_label = article_source.replace("Search ", "").replace("Paste ", "").replace("Upload ", "")
            if add_article_to_session(article_text, article_title, source_label, article_url):
                st.session_state.current_article = article_title
                st.session_state.current_article_text = article_text
                st.session_state.article_metadata = {"source": source_label, "url": article_url}
                st.session_state.assignment_questions = generate_assignment_questions(article_title)
                st.success(f"✓ Article '{article_title}' imported and ready for analysis!")
                st.rerun()
            else:
                st.error("Failed to process article. Please try again.")
        else:
            st.warning("Extracted text is too short. Please check the article source.")

    st.markdown("---")
    
    # Article recommendation section - Clean and Simple
    st.markdown("### 📚 Find Articles to Analyze")
    st.markdown("")
    
    # Simple search
    topic_filter = st.text_input("🔍 Search by topic (optional)", placeholder="e.g., vaccines, epidemiology, public health", key="topic_filter")
    st.markdown("")
    
    articles = recommend_articles(topic=topic_filter if topic_filter else None)
    
    # Show articles in a cleaner format
    articles_to_show = articles[:4] if len(articles) > 4 else articles
    
    if articles_to_show:
        for i, article in enumerate(articles_to_show):
            # Article card with cleaner layout
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{article['title']}**")
                st.caption(f"📖 {article['source']}")
                st.markdown(f"*{article['why']}*")
                
                # Links inline
                links = []
                if article.get('bruknow_url'):
                    links.append(f"[📚 BruKnow]({article['bruknow_url']})")
                if article.get('url') and 'bruknow' not in article.get('url', '').lower():
                    links.append(f"[🔗 Source]({article['url']})")
                if links:
                    st.markdown(" | ".join(links), unsafe_allow_html=True)
            
            with col2:
                if st.button("📊 Analyze", key=f"analyze_{i}", use_container_width=True, type="primary"):
                    st.session_state.current_article = article['title']
                    st.session_state.assignment_questions = generate_assignment_questions(article['title'])
                    st.rerun()
            
            if i < len(articles_to_show) - 1:
                st.markdown("---")

# Assignment questions - Simplified
if st.session_state.current_article:
    st.markdown(f"### 📊 Analyzing: {st.session_state.current_article}")
    st.markdown("")
    
    if not st.session_state.assignment_questions:
        st.session_state.assignment_questions = generate_assignment_questions(st.session_state.current_article)
    
    st.markdown("**Answer these questions about the article:**")
    st.markdown("")
    
    for i, q in enumerate(st.session_state.assignment_questions, 1):
        with st.expander(f"**Question {i}:** {q.get('question', '')}", expanded=False):
            st.caption(f"💡 Focus: {q.get('focus', '')} | 💭 Hint: {q.get('hint', '')}")
            st.markdown("")
            user_input = st.text_area(
                "Your answer:",
                key=f"q_{i}_input",
                height=100,
                placeholder="Type your answer here...",
                label_visibility="visible"
            )
            if st.button(f"Get Feedback", key=f"feedback_{i}", use_container_width=True, type="primary"):
                if user_input:
                    feedback_prompt = f"Question: {q['question']}\n\nStudent's answer: {user_input}\n\nProvide constructive feedback on their answer."
                    with st.spinner("Analyzing your answer..."):
                        feedback = answer_question(feedback_prompt)
                        st.session_state.messages.append({"role": "user", "content": f"Question {i}: {user_input}"})
                        st.session_state.messages.append({"role": "assistant", "content": feedback})
                        st.markdown("**📝 Feedback:**")
                        st.markdown(feedback)
                else:
                    st.info("Please enter an answer before requesting feedback.")
            
# General chat - Clean and Simple
if st.session_state.current_article or st.session_state.messages:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions")
    st.markdown("Ask Isabelle about statistical methods, interpretation, or course concepts.")
    st.markdown("")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about statistical methods, interpretation, or course concepts..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
