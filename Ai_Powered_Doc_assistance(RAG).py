import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Configure Google Gemini API
genai.configure(api_key="AIzaSyCSCn6EI4whCkTiFqFB2dT06u-et4Flwg4")

# Function to load text from PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to create FAISS index
def create_faiss_index(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# Function to get response from Google Gemini API
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

# Function to get answers from chatbot
def get_legal_answer(vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return get_gemini_response(prompt)

# Custom Background & Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
    }
    .sidebar .sidebar-content {
        background: #1e3c72;
        color: white;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("📜 AI-Powered Legal Document Assistant")

# Sidebar with App Features
st.sidebar.title("🛠 App Features")
st.sidebar.write("- Processes only **PDF** files")
st.sidebar.write("- Uses **FAISS** for fast retrieval")
st.sidebar.write("- Powered by **Gemini API**")
st.sidebar.write("- Embeddings: **Hugging Face**")
st.sidebar.write("- **Model Version 1.0**")
st.sidebar.write("- Ask legal questions and get instant insights")

uploaded_file = st.file_uploader("📂 Upload Legal Document (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analysing Your document... ⏳"):
        text = load_pdf(uploaded_file)
        text_chunks = split_text(text)
        vector_store = create_faiss_index(text_chunks)
    
    st.success("✅ Document processed successfully!")

    query = st.text_input("💬 Ask a legal question:")
    if st.button("Get Answer") and query:
        with st.spinner("Fetching answer... 💡"):
            answer = get_legal_answer(vector_store, query)
        st.write("### ✅ Answer:")
        st.write(answer)

# Footer
st.markdown("""---
#### Developed by Darshanikanta
""")
