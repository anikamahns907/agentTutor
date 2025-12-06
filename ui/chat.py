"""Chat UI helper functions."""
import streamlit as st
from ui.avatars import get_logo_avatar


def render_chat_message(msg, prev_role=None):
    """Render a chat message with appropriate avatar handling."""
    logo_avatar = get_logo_avatar()
    
    if msg["role"] == "assistant":
        # Show avatar ONLY if previous message was NOT from assistant
        if prev_role is None or prev_role != "assistant":
            with st.chat_message(msg["role"], avatar=logo_avatar):
                st.markdown(msg["content"])
        else:
            # Same role as previous, no avatar
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        # User messages - no avatar
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
