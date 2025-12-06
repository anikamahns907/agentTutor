import streamlit as st
from ui.css import inject_css
from ui.layout import render_sidebar
from pages.home import render_home
from pages.analyze_article import render_analyze_page
from pages.ask_questions import render_ask_questions
from pages.materials import render_materials

st.set_page_config(page_title="Isabelle Curve", layout="wide")

# Load CSS first
inject_css()

# Render sidebar and get selected page
page = render_sidebar()

# Route based on page selection
if page == "Home":
    render_home()
elif page == "Analyze Article":
    render_analyze_page()
elif page == "Ask Questions":
    render_ask_questions()
elif page == "Course Materials":
    render_materials()
