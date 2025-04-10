import os
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from pypdf.errors import PdfReadError
from openai.error import AuthenticationError, InvalidRequestError

# App title
st.subheader("Q&A with AI - NLP using LangChain")

# Interactive components
file_input = st.file_uploader("Upload a file", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png'])
openaikey = st.text_input("Enter your OpenAI API Key", type='password')
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Function to load documents based on file type
def load_document(file_path, file_type):
    if file_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == 'text/plain':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_type == 'text/csv':
        df = pd.read_csv(file_path)
        return [{"page_content": df.to_string()}]
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return [{"page_content": "\n".join(full_text)}]
    elif file_type in ['image/jpeg', 'image/png']:
        text = pytesseract.image_to_string(Image.open(file_path))
        return [{"page_content": text}]
    else:
        st.error("Unsupported file type.")
        return None

# Main Q&A function
def qa(file_path, file_type, query, chain_type, k):
    try:
        documents = load_document(file_path, file_type)
        if not documents:
            return None

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Generate embeddings using SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = [model.encode(text["page_content"]) for text in texts]

        # Create FAISS index
        db = FAISS.from_embeddings(embeddings, texts)

        # Convert index into a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # Create QA chain with retrieval
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"), 
            chain_type=chain_type, 
            retriever=retriever, 
            return_source_documents=True
        )
        result = qa({"query": query})
        return result
    except PdfReadError as e:
        st.error(f"Error reading PDF file: {e}")
        return None
    except AuthenticationError as e:
        st.error(f"Authentication error: {e}")
        return None
    except InvalidRequestError as e:
        st.error(f"Invalid request error: {e}")
        return None

# Function to display the result in Streamlit
def display_result(result):
    if result:
        st.markdown("### Result:")
        st.write(result["result"])
        st.markdown("### Relevant source text:")
        for doc in result["source_documents"]:
            st.markdown("---")
            st.markdown(doc.page_content)

# App execution
if run_button and file_input and openaikey and prompt:
    with st.spinner("Running QA..."):
        # Save uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_input.read())

        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = openaikey

        # Validate the API key
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.encode("test")
        except AuthenticationError as e:
            st.error(f"Invalid OpenAI API Key: {e}")
        else:
            # Run QA function and display results
            result = qa(temp_file_path, file_input.type, prompt, select_chain_type, select_k)
            display_result(result)
