
from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    return model.encode(texts)
