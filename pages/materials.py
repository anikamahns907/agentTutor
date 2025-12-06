"""Course materials page render function."""
import streamlit as st
from ui.css import inject_css


def render_materials():
    """Render the redesigned course materials page with card-based layout."""
    # Inject CSS first
    inject_css()
    
    # Page header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="margin-bottom: 0.5rem;">Course Materials</h1>
            <p class="hero-subtitle">Access lecture notes, assignments, and resources</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Materials organized in cards with expanders
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Textbook card
        with st.expander("📖 Textbook", expanded=False):
            st.markdown(
                """
                <div class="section-card">
                    <p><strong>Mathematical Statistics with Resampling and R</strong></p>
                    <p style="color: #9BA3AF; font-size: 0.875rem;">
                        Primary course textbook covering statistical methods and R programming.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Assessments card
        with st.expander("📝 Assessments", expanded=False):
            st.markdown(
                """
                <div class="section-card">
                    <p><strong>Homework, Exams, and Solutions</strong></p>
                    <p style="color: #9BA3AF; font-size: 0.875rem; margin-bottom: 1rem;">
                        Access assignments, exam study guides, and solution manuals.
                    </p>
                    <ul style="color: #E6E6E6; font-size: 0.875rem; padding-left: 1.5rem;">
                        <li>Assignment 1-3</li>
                        <li>Exam #1 Solutions & Study Guide</li>
                        <li>Exam #2 Solutions & Study Guide</li>
                        <li>Final Study Guide</li>
                        <li>Prerequisite Self Assessment</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        # Lecture Notes card
        with st.expander("📚 Lecture Notes", expanded=False):
            st.markdown(
                """
                <div class="section-card">
                    <p><strong>Week-by-Week Materials</strong></p>
                    <p style="color: #9BA3AF; font-size: 0.875rem; margin-bottom: 1rem;">
                        Weekly handouts, slides, and feedback materials.
                    </p>
                    <ul style="color: #E6E6E6; font-size: 0.875rem; padding-left: 1.5rem;">
                        <li>Week 1-4 Materials</li>
                        <li>Week 5-10 Handouts</li>
                        <li>Week 11-14 Materials</li>
                        <li>Lipman Notes (Week 6-7)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Resources card
        with st.expander("🔗 Resources", expanded=False):
            st.markdown(
                """
                <div class="section-card">
                    <p><strong>External Resources</strong></p>
                    <p style="color: #9BA3AF; font-size: 0.875rem; margin-bottom: 1rem;">
                        Useful links and references for the course.
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <a href="https://bruknow.library.brown.edu" target="_blank" style="color: #99c5ff; text-decoration: none; font-size: 0.875rem;">
                            📚 BruKnow Library
                        </a>
                        <a href="https://edstem.org/us/courses/80840/discussion" target="_blank" style="color: #99c5ff; text-decoration: none; font-size: 0.875rem;">
                            💬 EdStem Discussion
                        </a>
                        <a href="https://www.nature.com" target="_blank" style="color: #99c5ff; text-decoration: none; font-size: 0.875rem;">
                            🔬 Nature.com Research
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Additional resources section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="max-width: 800px; margin: 2rem auto; text-align: center;">
            <h3 style="margin-bottom: 1rem;">💡 Practice Problems</h3>
            <p style="color: #9BA3AF; margin-bottom: 0;">
                Access selected practice problems from Chihara textbook (Chapters 3-8) with solutions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
