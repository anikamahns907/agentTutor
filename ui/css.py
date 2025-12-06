"""CSS injection functions."""
from pathlib import Path
import streamlit as st


def inject_css():
    """Inject CSS styles from styles.css file, checking project root first, then ui/ directory."""
    # First, try project root (one level up from ui/)
    css_path = Path(__file__).resolve().parents[1] / "styles.css"
    
    if css_path.exists():
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return
    
    # Fallback: try ui/ directory
    css_path = Path(__file__).parent / "styles.css"
    
    if css_path.exists():
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return
    
    # Final fallback: try current directory
    fallback_path = Path("styles.css")
    if fallback_path.exists():
        css_content = fallback_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return
    
    # If none found, show warning but don't break
    st.warning(f"⚠️ CSS file not found. Checked: {Path(__file__).resolve().parents[1] / 'styles.css'}, {css_path}, {fallback_path}")
