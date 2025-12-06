"""Analyze article page render function."""
import streamlit as st
from ui.css import inject_css

from services.article_loader import extract_text_from_pdf, extract_text_from_url
from services.article_session import add_article_to_session
from services.question_gen import generate_assignment_questions
from services.answer_engine import answer_question
from ui.avatars import get_logo_avatar
from core.state import reset_analyze_session


def _init_analyze_state():
    """Initialize analyze page session state variables."""
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
    if "analyze_upload_mode" not in st.session_state:
        st.session_state.analyze_upload_mode = None
    if "article_processed" not in st.session_state:
        st.session_state.article_processed = False


def render_intro():
    """Render the redesigned intro screen before chat starts."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="margin-bottom: 0.5rem;">Analyze an Article</h1>
            <p class="hero-subtitle">Practice analyzing research articles with personalized feedback</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Intro card
    st.markdown(
        """
        <div class="card" style="max-width: 700px; margin: 2rem auto; text-align: center;">
            <p style="font-size: 1.125rem; color: #9BA3AF; line-height: 1.7; margin-bottom: 2rem;">
                Start a conversation with Isabelle to analyze research articles and get feedback on your understanding of statistical concepts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Chat with Isabelle", use_container_width=True, type="primary"):
            st.session_state.analyze_chat_started = True
            welcome_msg = """Hi! I'm Isabelle, your biostatistics communication tutor. 

I'm here to help you analyze research articles and practice explaining statistical concepts clearly. 

**How would you like to share an article with me?**"""
            st.session_state.analyze_messages.append({
                "role": "assistant",
                "content": welcome_msg
            })
            st.rerun()


def render_upload_options(logo_avatar, idx):
    """Render upload option buttons in the welcome message."""
    if not st.session_state.get("analyze_upload_mode"):
        st.markdown(
            """
            <div class="card" style="margin: 1rem 0;">
                <div style="margin-bottom: 1rem;">
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("📄 Upload PDF", key=f"btn_upload_pdf_{idx}", use_container_width=True, type="primary"):
                st.session_state.analyze_messages.append({
                    "role": "user",
                    "content": "I'd like to upload a PDF"
                })
                st.session_state.analyze_upload_mode = "pdf"
                st.rerun()
        with col2:
            if st.button("🔗 Paste URL", key=f"btn_paste_url_{idx}", use_container_width=True, type="primary"):
                st.session_state.analyze_messages.append({
                    "role": "user",
                    "content": "I'd like to paste a URL"
                })
                st.session_state.analyze_upload_mode = "url"
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #1F232B;">
            <p style="color: #9BA3AF; font-size: 0.875rem; margin-bottom: 0.75rem; font-weight: 500;">Explore articles to analyze:</p>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <a href="https://bruknow.library.brown.edu/discovery/search?vid=01BU_INST:BROWN" target="_blank" style="color: #99c5ff; text-decoration: none; font-size: 0.875rem;">🔍 BruKnow Library</a>
                <a href="https://www.nature.com" target="_blank" style="color: #99c5ff; text-decoration: none; font-size: 0.875rem;">📚 Nature.com</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


def handle_pdf_upload(logo_avatar):
    """Handle PDF file upload and extraction."""
    # Show assistant message if not already shown
    upload_msg_shown = any("Great! Please upload" in m.get("content", "") 
                          for m in st.session_state.analyze_messages 
                          if m.get("role") == "assistant")
    
    if not upload_msg_shown:
        # Add message to session state (will be displayed on rerun)
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": "Great! Please upload your PDF file below."
        })
        st.rerun()
    
    # Show file uploader in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    pdf = st.file_uploader("Upload PDF", type=['pdf'], key="analyze_pdf_upload", label_visibility="collapsed")
    if pdf:
        st.info(f"📄 **{pdf.name}** ({pdf.size / 1024:.1f} KB) ready to analyze")
        if st.button("Extract and Analyze", key="extract_pdf_btn", use_container_width=True, type="primary"):
            try:
                article_text = extract_text_from_pdf(pdf.read())
                if not article_text or len(article_text.strip()) < 50:
                    st.error("The PDF appears to be empty or could not be extracted. Please try a different file.")
                    return
                
                article_title = pdf.name
                st.session_state.pending_article_text = article_text
                st.session_state.pending_article_title = article_title
                st.session_state.pending_article_url = None
                st.session_state.analyze_messages.append({
                    "role": "user",
                    "content": f"Uploaded: {article_title}"
                })
                st.rerun()
            except Exception as e:
                error_msg = f"Error extracting text from PDF: {str(e)}. Please ensure the file is a valid PDF and try again."
                st.error(error_msg)
                st.session_state.analyze_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def handle_url_upload(logo_avatar):
    """Handle URL article extraction."""
    # Show assistant message if not already shown
    url_msg_shown = any("Perfect! Paste the article" in m.get("content", "") 
                       for m in st.session_state.analyze_messages 
                       if m.get("role") == "assistant")
    
    if not url_msg_shown:
        # Add message to session state (will be displayed on rerun)
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": "Perfect! Paste the article URL below."
        })
        st.rerun()
    
    # Show URL input in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    url = st.text_input("Paste article URL", key="analyze_url_input", label_visibility="collapsed", placeholder="https://...")
    if url:
        if st.button("Extract and Analyze", key="extract_url_btn", use_container_width=True, type="primary"):
            try:
                article_text = extract_text_from_url(url)
                if not article_text or len(article_text.strip()) < 50:
                    error_msg = "Could not extract meaningful content from this URL. Please try a different article or upload a PDF instead."
                    st.error(error_msg)
                    st.session_state.analyze_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()
                    return
                
                article_url = url
                # Try to extract a better title from URL
                article_title = url.split("/")[-1].split("?")[0] or url.split("/")[-2] or "Article from URL"
                st.session_state.pending_article_text = article_text
                st.session_state.pending_article_title = article_title
                st.session_state.pending_article_url = article_url
                st.session_state.analyze_messages.append({
                    "role": "user",
                    "content": f"Shared URL: {url}"
                })
                st.rerun()
            except Exception as e:
                error_msg = f"Error fetching article from URL: {str(e)}. Please check the URL and try again, or upload a PDF instead."
                st.error(error_msg)
                st.session_state.analyze_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def process_article(logo_avatar, upload_mode):
    """Process the uploaded article and initialize questions."""
    article_text = st.session_state.pending_article_text
    article_title = st.session_state.pending_article_title
    article_url = st.session_state.pending_article_url
    
    if not article_text:
        return False
    
    # Validate article text length
    if len(article_text.strip()) < 120:
        warning_msg = "The extracted text seems too short. Please try a different article."
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": warning_msg
        })
        # Clear pending data
        st.session_state.pending_article_text = None
        st.session_state.pending_article_title = None
        st.session_state.pending_article_url = None
        st.rerun()
        return False
    
    # Add article to session vectorstore
    added = add_article_to_session(
        article_text,
        article_title,
        upload_mode or "upload",
        article_url
    )
    
    if not added:
        error_msg = "Failed to process article. Please try again."
        st.error(error_msg)
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": error_msg
        })
        st.rerun()
        return False
    
    # Initialize article and questions
    st.session_state.current_article = article_title
    st.session_state.assignment_questions = generate_assignment_questions(article_title)
    st.session_state.current_question_index = 0
    st.session_state.question_answers = {}
    st.session_state.article_processed = True
    
    # Add success message and first question
    success_msg = f"""Successfully imported: **{article_title}**

I've analyzed your article! I'll guide you through 10 questions to help you practice explaining statistical concepts clearly.

Let's start with the first question:"""
    st.session_state.analyze_messages.append({
        "role": "assistant",
        "content": success_msg
    })
    
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
    
    # Clear pending article data
    st.session_state.pending_article_text = None
    st.session_state.pending_article_title = None
    st.session_state.pending_article_url = None
    st.session_state.analyze_upload_mode = None
    
    return True


def display_chat_history(logo_avatar):
    """Display chat message history with proper avatar handling in a scrollable container."""
    # Article info card if article is loaded
    if st.session_state.current_article:
        st.markdown(
            f"""
            <div class="article-card">
                <h3 style="margin-bottom: 0.5rem; font-size: 1.25rem;">📄 Current Article</h3>
                <p style="margin: 0; color: #FFFFFF; font-weight: 500;">{st.session_state.current_article}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Chat container with scrollable area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    prev_role = None
    for idx, msg in enumerate(st.session_state.analyze_messages):
        # Determine if avatar should be shown
        show_avatar = None
        if msg["role"] == "assistant":
            if prev_role is None or prev_role != "assistant":
                show_avatar = logo_avatar
        
        # Display message
        with st.chat_message(msg["role"], avatar=show_avatar):
            st.markdown(msg["content"])
        
        # Update prev_role after displaying
        prev_role = msg["role"]
        
        # Show upload buttons if this is the welcome message
        if msg["role"] == "assistant" and "How would you like to share" in msg["content"]:
            render_upload_options(logo_avatar, idx)
    
    st.markdown('</div>', unsafe_allow_html=True)


def run_question_flow(user_input, logo_avatar):
    """Handle question answering flow and advance to next question.
    
    Returns True if this was handled as a question answer, False otherwise.
    """
    if not st.session_state.current_article or not st.session_state.assignment_questions:
        return False
    
    current_idx = st.session_state.current_question_index
    if current_idx >= len(st.session_state.assignment_questions):
        return False
    
    # Check if user is answering the current question
    # Only process if we're still in question flow (not after completion)
    current_q = st.session_state.assignment_questions[current_idx]
    
    # Store the answer
    st.session_state.question_answers[current_idx] = user_input
    
    # Generate feedback using article-prioritized context
    prompt = f"Question: {current_q['question']}\n\nStudent answer: {user_input}\n\nPlease provide constructive feedback on this answer."
    
    # Generate feedback (this will be displayed in the rerun)
    feedback = answer_question(
        prompt,
        question_focus=current_q['focus'],
        is_article_feedback=True,
        max_tokens=1000
    )
    feedback_msg = f"**Feedback on Question {current_idx + 1}:**\n\n{feedback}"
    
    # Add feedback to messages (will be displayed on rerun)
    st.session_state.analyze_messages.append({
        "role": "assistant",
        "content": feedback_msg
    })
    
    # Advance to next question
    current_idx += 1
    st.session_state.current_question_index = current_idx
    
    if current_idx < len(st.session_state.assignment_questions):
        # Add next question to messages
        next_q = st.session_state.assignment_questions[current_idx]
        question_msg = f"""**Question {current_idx + 1}:** {next_q['question']}

**Focus:** {next_q['focus']}
**Hint:** {next_q['hint']}

Please answer this question in the chat below."""
        
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": question_msg
        })
    else:
        # All questions completed
        completion_msg = """Great work! You've completed all 10 analysis questions. 

Would you like to:
- Review your answers
- Ask follow-up questions about the article
- Start analyzing a new article"""
        
        st.session_state.analyze_messages.append({
            "role": "assistant",
            "content": completion_msg
        })
    
    return True


def render_analyze_page():
    """Render the redesigned analyze article page."""
    # Inject CSS first - MUST be first line for proper rendering
    inject_css()
    
    # Wrap content in container to ensure proper layout
    with st.container():
        # Initialize state
        _init_analyze_state()
        logo_avatar = get_logo_avatar()
        
        # Show intro if chat hasn't started
        if not st.session_state.analyze_chat_started:
            render_intro()
            return
        
        # Display chat history
        display_chat_history(logo_avatar)
        
        # Handle article upload
        upload_mode = st.session_state.get("analyze_upload_mode")
        if upload_mode == "pdf" and not st.session_state.current_article:
            handle_pdf_upload(logo_avatar)
        elif upload_mode == "url" and not st.session_state.current_article:
            handle_url_upload(logo_avatar)
        
        # Process pending article if exists (only once)
        if (st.session_state.pending_article_text 
            and not st.session_state.article_processed 
            and not st.session_state.current_article):
            if process_article(logo_avatar, upload_mode):
                st.rerun()
        
        # Handle chat input
        if st.session_state.analyze_chat_started:
            # Determine placeholder
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
                # Add user message first
                st.session_state.analyze_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Determine if this is a question answer or general question
                is_question_answer = (
                    st.session_state.current_article 
                    and st.session_state.assignment_questions
                    and st.session_state.current_question_index < len(st.session_state.assignment_questions)
                )
                
                if is_question_answer:
                    # Handle as question answer - this will add feedback and next question to messages
                    run_question_flow(user_input, logo_avatar)
                else:
                    # General question about the article
                    response = answer_question(
                        user_input,
                        is_article_feedback=bool(st.session_state.current_article)
                    )
                    
                    st.session_state.analyze_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                st.rerun()
            
            # Clear chat button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Start New Analysis", key="clear_analyze_chat", use_container_width=True, type="secondary"):
                    reset_analyze_session()
                    st.rerun()
