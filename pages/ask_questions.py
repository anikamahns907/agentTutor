"""Ask questions page render function."""
import streamlit as st
from ui.css import inject_css
from services.answer_engine import answer_question
from ui.avatars import get_logo_avatar


# Sample questions for quick access
SAMPLE_QUESTIONS = [
    "What is a p-value and how do I interpret it?",
    "What's the difference between Type I and Type II errors?",
    "How do I choose between a t-test and a z-test?",
    "What is the Central Limit Theorem?",
    "Explain confidence intervals in simple terms",
    "What's the purpose of ANOVA?",
]


def render_ask_questions():
    """Render the redesigned ask questions page."""
    # Inject CSS FIRST - this is critical for proper rendering
    inject_css()
    
    # Page header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="margin-bottom: 0.5rem;">Ask Questions</h1>
            <p class="hero-subtitle">Get instant help with statistical concepts and methods</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load logo for avatar
    logo_avatar = get_logo_avatar()

    # Show sample questions if chat is empty
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="card" style="margin-bottom: 2rem;">
                <h3 style="margin-bottom: 1rem;">💡 Sample Questions</h3>
                <p style="color: #9BA3AF; margin-bottom: 1rem;">Click on any question below to get started, or ask your own question:</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display sample questions in a grid
        cols = st.columns(2, gap="medium")
        for idx, question in enumerate(SAMPLE_QUESTIONS):
            col = cols[idx % 2]
            with col:
                if st.button(question, key=f"sample_q_{idx}", use_container_width=True, type="secondary"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Chat container - scrollable area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    prev_role = None
    for msg in st.session_state.messages:
        # Show avatar only for first assistant message or after user message
        avatar = None
        if msg["role"] == "assistant":
            if prev_role is None or prev_role != "assistant":
                avatar = logo_avatar
        
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
        
        prev_role = msg["role"]
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input at the bottom
    user_q = st.chat_input("Ask about statistical methods or concepts...")
    if user_q:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_q})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_q)

        # Generate and display assistant response
        with st.chat_message("assistant", avatar=logo_avatar):
            with st.spinner("Thinking..."):
                ans = answer_question(user_q)
                st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()
    
    # Clear chat button (only show if messages exist)
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Chat", key="clear_questions_chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
