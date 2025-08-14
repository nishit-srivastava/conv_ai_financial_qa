
import faiss
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, embeddings, texts):
        self.embeddings = embeddings
        self.texts = texts
        self.bm25 = BM25Okapi([t.split() for t in texts])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query_vec, query_text, top_k=5):
        _, dense_idx = self.index.search(query_vec, top_k)
        bm25_scores = self.bm25.get_scores(query_text.split())
        dense_results = [(self.texts[i], 1) for i in dense_idx[0]]
        sparse_results = sorted(zip(self.texts, bm25_scores), key=lambda x: -x[1])[:top_k]
        return list(set(dense_results + sparse_results))
