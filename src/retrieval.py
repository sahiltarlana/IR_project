"""
Retrieval module: BM25 (base paper) and Dense Retrieval (proposed enhancement).
"""
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import time


class BM25Retriever:
    """Sparse retrieval using BM25 (base paper approach)."""

    def __init__(self, corpus):
        self.corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=3):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_idx], [float(scores[i]) for i in top_idx]


class DenseRetriever:
    """Dense retrieval using sentence-transformers (proposed enhancement)."""

    def __init__(self, corpus, model_name="all-MiniLM-L6-v2"):
        self.corpus = corpus
        print(f"  Loading dense retriever model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("  Encoding corpus...")
        self.corpus_embeddings = self.model.encode(corpus, show_progress_bar=False,
                                                    convert_to_numpy=True)

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores = np.dot(self.corpus_embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_idx], [float(scores[i]) for i in top_idx]


def build_corpus_from_data(data_samples):
    """Build a retrieval corpus from dataset contexts."""
    corpus = []
    for s in data_samples:
        ctx = s.get("context", "")
        if ctx and len(ctx) > 50:
            # Split long contexts into chunks
            sentences = ctx.split(". ")
            chunk = ""
            for sent in sentences:
                chunk += sent + ". "
                if len(chunk) > 200:
                    corpus.append(chunk.strip())
                    chunk = ""
            if chunk.strip():
                corpus.append(chunk.strip())
    # Deduplicate
    corpus = list(set(corpus))
    return corpus if corpus else ["No relevant documents found."]
