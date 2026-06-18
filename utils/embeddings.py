from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """
    Load MiniLM embedding model.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


def create_embeddings(model, chunks):
    """
    Convert text chunks into embeddings.
    """

    embeddings = model.encode(chunks)

    return embeddings
