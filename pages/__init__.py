"""Page render functions for Streamlit app."""
from .home import render_home
from .analyze_article import render_analyze_page
from .ask_questions import render_ask_questions
from .materials import render_materials

__all__ = ['render_home', 'render_analyze_page', 'render_ask_questions', 'render_materials']
