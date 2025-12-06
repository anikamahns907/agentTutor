"""Session state initialization and reset functions."""
import streamlit as st


def init_session_state():
    """Initialize all session state variables."""
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


def reset_analyze_session():
    """Reset all analyze page session state (does not affect ask-questions chat)."""
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
    st.session_state.article_processed = False
