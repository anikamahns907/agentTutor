"""Home page render function."""
import streamlit as st
from ui.css import inject_css


def render_home():
    """Render the redesigned home page with hero section and modern cards."""
    # Inject CSS first
    inject_css()
    
    # Hero section
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <h1 class="hero-title">Welcome to Isabelle</h1>
            <p class="hero-subtitle">Your AI Tutor for PHP 1510/2510 – Principles of Biostatistics</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Introduction text
    st.markdown(
        """
        <div style="text-align: center; max-width: 700px; margin: 0 auto 3rem auto; color: #9BA3AF;">
            <p style="font-size: 1.125rem; line-height: 1.7;">
                Isabelle helps you practice communicating statistical concepts,
                understand research papers, and interpret results using material
                from the course textbook, lectures, and assignments.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Three main feature cards with buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class='card' style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                <h3>Analyze Articles</h3>
                <p>Upload PDFs or paste URLs to practice analyzing research articles with structured questions and personalized feedback.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Go to Analyze", key="nav_analyze_home", use_container_width=True, type="primary"):
            st.session_state.page = "Analyze"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class='card' style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">❓</div>
                <h3>Ask Questions</h3>
                <p>Get instant help with statistical methods, concepts, and interpretation questions tailored to your course material.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Go to Questions", key="nav_questions_home", use_container_width=True, type="primary"):
            st.session_state.page = "Questions"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class='card' style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
                <h3>Course Materials</h3>
                <p>Access lecture notes, homework assignments, solution guides, and textbook excerpts all in one place.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Go to Materials", key="nav_materials_home", use_container_width=True, type="primary"):
            st.session_state.page = "Materials"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Branding section
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid #1F232B;">
            <p style="color: #9BA3AF; font-size: 0.875rem;">
                Developed for <strong style="color: #FFFFFF;">Peter Lipman</strong> / <strong style="color: #FFFFFF;">DATA 1150</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
