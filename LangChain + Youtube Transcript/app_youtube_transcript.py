# Importing essential libraries for the script
import os
import streamlit as st  # Streamlit library for building web applications
from langchain_community.document_loaders import YoutubeLoader
import requests  # Library for making HTTP requests

# App title
st.subheader("YouTube Transcript Generator")

# Input field for YouTube link
link = st.text_input('🔗 Paste the YouTube video link to transcribe...')

# Language selection dropdown
language_options = {
    "Portuguese": "pt",
    "English": "en",
    "Spanish": "es",
}
language = st.selectbox(
    "🌐 Select the video language:",
    options=list(language_options.keys()),
    index=0  # Default: Portuguese
)

run_button = st.button("Run!")

# When the button is clicked
if run_button and link:
    st.write('Generating transcript from the video...')
    selected_language = language_options[language]
    
    loader = YoutubeLoader.from_youtube_url(
        link,
        add_video_info=False,
        language=[selected_language]
    )
    result = loader.load()

    # Retry if the result is initially empty
    while not result:
        result = loader.load()
        
    # Check if the first item has a 'page_content' attribute
    if result and hasattr(result[0], 'page_content'):
        page_content = result[0].page_content
        with st.expander('Transcript'):
            st.info(page_content)
    else:
        st.warning("Please check if the link is valid.")