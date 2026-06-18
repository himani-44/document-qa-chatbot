from dotenv import load_dotenv

import os
from xml.parsers.expat import model

from click import prompt
import faiss # store, search find nearest vectors efficiently
import numpy as np # embeddings are numerical vectors 
import google.generativeai as genai # genai is a library for working with Google's generative AI models, such as Gemini. It provides tools for generating text, images, and other content using these models.


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1] 
    # why do we require shape[1]? because FAISS must know hoe many dimensions each vector have?

    # L2 distance is the default metric for FAISS, it measures the straight-line distance between two points in the vector space.
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings)) # stroing vectors inside FAISS index

    return index


def retrieve_relevant_chunks(
    question,
    chunks,
    index,
    embedding_model,    # MiniLM model 
    top_k=3  # asking FAISS to return top 3 most relevant chunks based on the question embedding
):

    question_embedding = embedding_model.encode([question])

    distances, indices = index.search(
        np.array(question_embedding),
        top_k     # retriving information aas best second best to third best match
    )

    retrieved_chunks = [] # storing matching chunks here

    for idx in indices[0]:

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks

def generate_answer(question, retrieved_chunks):

    from dotenv import load_dotenv
    import os

    load_dotenv()

    genai.configure(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}
    """

    model = genai.GenerativeModel(
        "models/gemini-2.5-flash"
    )

    response = model.generate_content(
        prompt
    )

    return response.text





