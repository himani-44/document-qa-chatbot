import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI

# ------------------ SETUP ------------------ #
st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("📄 Document Q&A Chatbot (RAG Lite)")

# 🔑 OPTION 1 (RECOMMENDED): Environment variable
api_key = os.getenv("OPENAI_API_KEY")

# 🔑 OPTION 2 (EASY TEMPORARY WAY) → Uncomment if needed
# api_key = "your_actual_api_key_here"

if not api_key:
    st.error("❌ Please set your OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ INPUT ------------------ #
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the document")

# ------------------ FUNCTIONS ------------------ #

def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def chunk_text(text):
    sentences = text.split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def retrieve_context(chunks, question):
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])

    scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_idx = scores.argmax()

    return chunks[best_idx]


def generate_answer(context, question):
    prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ------------------ MAIN ------------------ #

if uploaded_file and question:
    with st.spinner("Processing..."):
        text = extract_text(uploaded_file)

        if not text:
            st.warning("Could not extract text from PDF.")
        else:
            chunks = chunk_text(text)
            context = retrieve_context(chunks, question)
            answer = generate_answer(context, question)

            st.subheader("📌 Answer:")
            st.write(answer)

            st.subheader("🔍 Retrieved Context:")
            st.info(context)