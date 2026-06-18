# RAG PDF Question Answering Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- PDF Upload
- Text Extraction using pdfplumber
- Chunking Strategy
- MiniLM Embeddings
- FAISS Vector Search
- Gemini 2.5 Flash Answer Generation

## Tech Stack

- Python
- Streamlit
- Sentence Transformers (MiniLM)
- FAISS
- Gemini API
- pdfplumber

## Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → Gemini → Answer

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```