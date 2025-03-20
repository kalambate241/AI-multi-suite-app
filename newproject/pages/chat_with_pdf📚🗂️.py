import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()

# Get API Key securely
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🚨 API Key not found! Please set 'GOOGLE_API_KEY' in your environment variables.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to generate FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)  # Corrected model name
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function for user input and interaction
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Error loading vector database: {e}")
        return "Error processing your request."

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response.get("output_text", "No response generated.")

# Function to allow users to download processed PDFs
def download_pdf(text):
    output = BytesIO()
    output.write(text.encode('utf-8'))
    output.seek(0)
    st.download_button("📥 Download Processed PDF", output, file_name="processed_pdf.txt", mime="text/plain")

# Main function to set up Streamlit interface
def main():
    st.set_page_config(page_title="Chat PDF with Gemini 💁", layout="wide")
    st.title("📄 Chat with PDF using Gemini AI")
    st.markdown("Upload a PDF, process it, and ask questions to extract information from it.")

    # Upload PDF section
    st.subheader("Upload PDF Files 📄")
    pdf_docs = st.file_uploader("Choose your PDF files", accept_multiple_files=True, type=["pdf"])

    # Input area for asking questions
    user_question = st.text_input("🔎 Ask a question from the PDFs")

    if pdf_docs:
        st.subheader("⏳ Processing Files... Please wait!")
        with st.spinner("Extracting text from your PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ PDF Processing Complete! Ready to Answer Questions 🎉")

    # Answer user queries
    if user_question:
        st.subheader("💬 Your Question")
        st.write(f"**Q:** {user_question}")
        st.subheader("🤖 AI Response 👇")
        answer = user_input(user_question)
        st.write(answer)

        # User rating feature
        rating = st.slider("⭐ Rate the answer:", 1, 5, 3)
        st.write(f"Your Rating: {rating} ⭐")

        # Download processed text
        st.subheader("📥 Download Extracted Text")
        download_pdf(raw_text)

    # Display question history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if user_question:
        st.session_state.query_history.append(user_question)

    if st.session_state.query_history:
        st.subheader("📜 Question History")
        for idx, question in enumerate(st.session_state.query_history):
            st.write(f"{idx+1}. {question}")

if __name__ == "__main__":
    main()
