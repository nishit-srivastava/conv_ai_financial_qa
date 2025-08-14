
import streamlit as st
from embeddings import get_embedding_model, embed_texts
from retriever import HybridRetriever
from generator import get_generator, generate_answer
import os, faiss, pandas as pd

st.title("Financial QA RAG System")

if "retriever" not in st.session_state:
    st.session_state.embedding_model = get_embedding_model()
    st.session_state.generator = get_generator()
    # Load data
    if os.path.exists("data/rag_chunks/chunks.csv"):
        df = pd.read_csv("data/rag_chunks/chunks.csv")
        embeddings = embed_texts(st.session_state.embedding_model, df["chunk"].tolist())
        st.session_state.retriever = HybridRetriever(embeddings, df["chunk"].tolist())
    else:
        st.warning("Please prepare chunks first.")

question = st.text_input("Ask a question:")
if question and "retriever" in st.session_state:
    query_vec = embed_texts(st.session_state.embedding_model, [question])
    results = st.session_state.retriever.search(query_vec, question)
    context = "\n".join([r[0] for r in results])
    answer = generate_answer(st.session_state.generator, context, question)
    st.write(answer)
