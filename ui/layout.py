"""Layout helper functions and wrappers."""
import base64
import streamlit as st
from pathlib import Path
from core.state import init_session_state


def card_html(title, content, clickable=False, onclick=None, extra_classes=""):
    """Generate HTML for a modern card component."""
    cursor_style = "cursor: pointer;" if clickable else ""
    onclick_attr = f'onclick="{onclick}"' if onclick else ""
    classes = f"card {extra_classes}"
    if clickable:
        classes += " card-clickable"
    
    return f"""
    <div class="{classes}" style="{cursor_style}" {onclick_attr}>
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """


def section_card_html(title, content, extra_classes=""):
    """Generate HTML for a section card."""
    return f"""
    <div class="section-card {extra_classes}">
        <h3>{title}</h3>
        {content}
    </div>
    """


def render_card(title, description, clickable=False, onclick=None):
    """Render a card component using Streamlit markdown."""
    st.markdown(
        card_html(title, description, clickable=clickable, onclick=onclick),
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the redesigned sidebar navigation and return selected page string.
    
    Returns:
        str: One of "Home", "Analyze Article", "Ask Questions", "Course Materials"
    """
    # Initialize session state
    init_session_state()
    
    # Map internal keys to display names
    page_map = {
        "Home": "Home",
        "Analyze": "Analyze Article",
        "Questions": "Ask Questions",
        "Materials": "Course Materials"
    }
    
    # Initialize page if not set
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
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
    
    # Sidebar header with logo and title
    st.sidebar.markdown(
        f"""
        <div class="sidebar-header">
            <div class="sidebar-logo-container">
                {logo_html}
                <div class="sidebar-title">Isabelle</div>
                <div class="sidebar-subtitle">Your Biostatistics Peer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation menu
    menu = [
        ("Home", "Home"),
        ("Analyze Article", "Analyze"),
        ("Ask Questions", "Questions"),
        ("Course Materials", "Materials"),
    ]

    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    
    current_page_key = st.session_state.page
    
    for label, key in menu:
        active = "active" if current_page_key == key else ""
        st.sidebar.markdown(f'<div class="nav-button-wrapper {active}">', unsafe_allow_html=True)
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True, type="secondary"):
            st.session_state.page = key
            st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Collapsible About/Instructions section
    with st.sidebar.expander("ℹ️ About / Instructions", expanded=False):
        st.markdown("""
        **Welcome to Isabelle!**
        
        Isabelle is your AI tutor for PHP 1510/2510 – Principles of Biostatistics.
        
        **Features:**
        - 📄 **Analyze Articles**: Upload PDFs or paste URLs to practice analyzing research articles
        - ❓ **Ask Questions**: Get help with statistical concepts and methods
        - 📚 **Course Materials**: Access lecture notes, homework, and resources
        
        **Tips:**
        - Be specific in your questions for better responses
        - Use the article analysis feature to practice for assignments
        - Explore course materials for additional context
        """)
    
    # Chat actions (if messages exist)
    if st.session_state.messages:
        st.sidebar.markdown("<div class='sidebar-section-label'>Chat</div>",
                            unsafe_allow_html=True)

        if st.sidebar.button("Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
    
    # Professor branding
    st.sidebar.markdown(
        """
        <div class="professor-branding">
            <div class="professor-name">Developed for</div>
            <div class="professor-name">Peter Lipman</div>
            <div class="course-title">DATA 1150</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Return the display name for routing
    return page_map.get(current_page_key, "Home")
