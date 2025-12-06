"""Reusable card components for homepage and other pages."""
import streamlit as st


def render_homepage_card(title, description, page_key):
    """Render a clickable card for the homepage."""
    st.markdown(f"""
    <div class='card' style="cursor: pointer;" onclick="go('{page_key}')">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_materials_card(title, description, links=None):
    """Render a card for course materials page."""
    links_html = ""
    if links:
        links_html = "<br/>".join([f'<p><a href="{link["url"]}" target="_blank">{link["text"]}</a></p>' for link in links])
    
    st.markdown(f"""
    <div class='card'>
        <h3>{title}</h3>
        <p>{description}</p>
        {links_html}
    </div>
    """, unsafe_allow_html=True)
