# vectorstore.py
"""
Book Dude - Vector Store Module
-------------------------------
This module manages storage and retrieval of embeddings.
Supports:
- FAISS (local, in-memory)
- ChromaDB (persistent, disk-based)
"""

import numpy as np
from typing import List, Union

# FAISS
import faiss

# Optional: ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, provider: str = "faiss", dim: int = 384, persist_dir: str = "./chroma_store"):
        """
        Initialize vector store.

        Args:
            provider (str): "faiss" or "chroma"
            dim (int): Dimension of embeddings
            persist_dir (str): Directory for ChromaDB persistence
        """
        self.provider = provider.lower()
        self.dim = dim

        if self.provider == "faiss":
            self.index = faiss.IndexFlatL2(dim)

        elif self.provider == "chroma":
            if not chromadb:
                raise ImportError("ChromaDB not installed. Run `pip install chromadb`.")
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            self.collection = self.client.get_or_create_collection("book_dude")

        else:
            raise ValueError("Provider must be 'faiss' or 'chroma'")

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Add embeddings + chunks to the vector store.

        Args:
            embeddings (np.ndarray): Array of embeddings
            chunks (List[str]): Corresponding text chunks
        """
        if self.provider == "faiss":
            self.index.add(embeddings)

        elif self.provider == "chroma":
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            self.collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[int]:
        """
        Search for top-k nearest neighbors.

        Args:
            query_embedding (np.ndarray): Embedding of query
            top_k (int): Number of results to return

        Returns:
            List[int]: Indices of retrieved chunks (FAISS) or IDs (Chroma)
        """
        if self.provider == "faiss":
            D, I = self.index.search(query_embedding, top_k)
            return I[0].tolist()

        elif self.provider == "chroma":
            results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
            return results["documents"][0]


if __name__ == "__main__":
    # Example usage for local testing
    chunks = [
        "Entropy is a measure of uncertainty in information theory.",
        "Bayes theorem relates conditional probabilities."
    ]

    # Fake embeddings for demo (normally from embeddings.py)
    fake_embeddings = np.random.rand(len(chunks), 384).astype("float32")

    # FAISS example
    store = VectorStore(provider="faiss", dim=384)
    store.add_embeddings(fake_embeddings, chunks)
    query_vec = np.random.rand(1, 384).astype("float32")
    results = store.search(query_vec, top_k=2)
    print("FAISS retrieved indices:", results)
