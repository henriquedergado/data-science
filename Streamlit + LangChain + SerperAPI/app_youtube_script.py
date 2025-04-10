# Importing essential libraries for the script
import os
import streamlit as st  # Streamlit library for building web applications
from langchain.llms import OpenAI  # OpenAI language model
from langchain.prompts import PromptTemplate  # Class for creating prompt templates
from langchain.chains import LLMChain, SequentialChain  # Classes for building LLM chains
from langchain.memory import ConversationBufferMemory  # Conversational buffer memory
import requests  # Library for making HTTP requests

# Defining a class to search Google using the Serper.dev API
class SerperAPIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key  # Initialize with API key
        self.url = "https://google.serper.dev/search"  # API URL

    # Method to perform the search
    def run(self, query):
        headers = {
            "X-API-KEY": self.api_key,  # API key in the headers
            "Content-Type": "application/json"
        }
        payload = {
            "q": query  # Search query payload
        }
        response = requests.post(self.url, headers=headers, json=payload)  # Perform POST request
        if response.status_code == 200:
            results = response.json()  # Get JSON results
            snippets = [item['snippet'] for item in results.get('organic', [])]  # Extract snippets from organic results
            return "\n".join(snippets)  # Return snippets as a string
        else:
            return "No results found."  # Return error message if request fails

# Setting the Streamlit app title
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Enter your content topic here')  # Input field for user to enter topic

# Prompt templates for YouTube video title and script
title_template = PromptTemplate(
    input_variables=['topic'], 
    template='Write a YouTube video title about... {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'google_research'], 
    template='Write a YouTube video script based on this title: {title} while leveraging this Google research: {google_research}'
)

# Memory setup to store conversation history
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Display results if the user provided a prompt
if prompt:
    # Initialize the language model with temperature 0.9
    llm = OpenAI(temperature=0.9)
    
    # Chain to generate YouTube titles
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    
    # Chain to generate video scripts
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    # Initialize the Serper.dev API wrapper
    google_search = SerperAPIWrapper(api_key=st.secrets["serper_api_key"])

    # Set the OpenAI API key in the environment
    os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

    title = title_chain.run(prompt)  # Generate the video title
    google_research = google_search.run(prompt)  # Perform Google search
    script = script_chain.run(title=title, google_research=google_research)  # Generate the script

    st.write(title)  # Display the generated title
    st.write(script)  # Display the generated script

    with st.expander('Title History'): 
        st.info(title_memory.buffer)  # Display title history

    with st.expander('Script History'): 
        st.info(script_memory.buffer)  # Display script history

    with st.expander('Google Research'): 
        st.info(google_research)  # Display Google search results
