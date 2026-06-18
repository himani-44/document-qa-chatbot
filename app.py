import os
import streamlit as st
import google.generativeai as genai

from utils.pdf_processor import (
    extract_text_from_pdf,
    chunk_text
)

from utils.embeddings import (
    load_embedding_model,
    create_embeddings
)

from utils.rag_pipeline import (
    create_faiss_index,
    retrieve_relevant_chunks,
    generate_answer
)

# Gemini API Key from Streamlit Secrets
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# Create folders if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

st.title("📄 RAG PDF Question Answering Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embedding_model = load_embedding_model()

    embeddings = create_embeddings(
        embedding_model,
        chunks
    )

    index = create_faiss_index(
        embeddings
    )

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        retrieved_chunks = retrieve_relevant_chunks(
            question,
            chunks,
            index,
            embedding_model
        )

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        st.subheader("Answer")

        st.write(answer)