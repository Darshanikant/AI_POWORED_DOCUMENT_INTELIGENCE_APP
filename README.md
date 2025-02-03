
# AI-Powered Legal Assistant

## Description
This web app provides an AI-powered legal assistant where users can upload PDF legal documents and ask questions related to the content. The app processes the document using various natural language processing (NLP) techniques and provides relevant answers using the Google Gemini API.

The application utilizes FAISS (Facebook AI Similarity Search) for efficient document retrieval and Hugging Face embeddings for transforming the text data into numerical vectors. Users can interact with the app via a simple web interface built with Streamlit.

## Features
- **PDF Upload**: Allows users to upload legal PDF documents.
- **FAISS**: Uses FAISS for fast similarity search and efficient document retrieval.
- **Gemini API**: Retrieves answers to legal questions using Google’s Gemini generative model.
- **Hugging Face Embeddings**: Utilizes Hugging Face's transformer-based embeddings for text vectorization.
- **Interactive UI**: Provides a clean and responsive user interface using Streamlit.
- **Document Analysis**: Splits large documents into chunks for better processing and retrieval.

## Installation
### Requirements
- Python 3.x
- Streamlit
- Google Gemini API key
- langchain_community
- PyPDF2
- HuggingFace Transformers

### Setup
1. Clone the repository:
  
   git clone https://github.com/yourusername/legal-assistant.git
 
2. Navigate to the project folder:
   
   cd legal-assistant
   
3. Install the dependencies:
   
   pip install -r requirements.txt
   

### Google Gemini API
Make sure to configure your Gemini API key before running the app:

genai.configure(api_key="YOUR_GOOGLE_GEMINI_API_KEY")


## Usage
To run the app, use the following command:

streamlit run app.py


### App Interface
1. Upload a **PDF** document using the file uploader.
2. Once uploaded, the document is analyzed and processed.
3. After processing, you can enter a **legal question** in the text input field.
4. Click the "Get Answer" button to receive an answer based on the uploaded document.


## Try and write you feedback
- App link: https://aipoworeddocumentinteligence-bnyfmsfw4zu765y7j7cwj7.streamlit.app/
- linkedeln: https://www.linkedin.com/in/darshanikanta-behera/
