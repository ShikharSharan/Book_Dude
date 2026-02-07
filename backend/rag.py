# rag.py
"""
Book Dude - RAG Pipeline Module
-------------------------------
This module implements Retrieval-Augmented Generation (RAG).
Steps:
1. Retrieve top-k relevant chunks from vector store
2. Assemble a prompt with context + user query
3. Call LLM to generate answer with citations
"""

from typing import List
import numpy as np

from backend.embeddings import EmbeddingsGenerator
from backend.llm import LLMWrapper


class RAGPipeline:
    def __init__(self, embedder: EmbeddingsGenerator, llm: LLMWrapper, index, chunks: List[str]):
        """
        Initialize the RAG pipeline.

        Args:
            embedder (EmbeddingsGenerator): Embeddings generator instance
            llm (LLMWrapper): LLM wrapper instance
            index (faiss.Index): Vector store index
            chunks (List[str]): List of text chunks
        """
        self.embedder = embedder
        self.llm = llm
        self.index = index
        self.chunks = chunks
        self.embeddings = None

    def build_embeddings(self):
        """Generate embeddings for chunks and store them."""
        self.embeddings = self.embedder.generate(self.chunks)
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k relevant chunks for a query."""
        query_vec = self.embedder.generate([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        return [self.chunks[i] for i in I[0]]

    def assemble_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """Assemble prompt with context and query."""
        context = "\n\n".join(retrieved_chunks)
        prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer (cite sources):"
        )
        return prompt

    def answer(self, query: str, top_k: int = 3) -> str:
        """Generate answer using RAG pipeline."""
        retrieved_chunks = self.retrieve(query, top_k)
        prompt = self.assemble_prompt(query, retrieved_chunks)
        return self.llm.generate(prompt, max_tokens=400)


if __name__ == "__main__":
    # Example usage for local testing
    chunks = [
        "Entropy is a measure of uncertainty in information theory.",
        "Bayes theorem relates conditional probabilities."
    ]

    # Initialize components
    embedder = EmbeddingsGenerator(provider="huggingface")
    import faiss
    dim = embedder.generate([chunks[0]]).shape[1]
    index = faiss.IndexFlatL2(dim)

    llm = LLMWrapper(provider="openai", api_key="YOUR_OPENAI_API_KEY")

    # Build pipeline
    rag = RAGPipeline(embedder, llm, index, chunks)
    rag.build_embeddings()

    # Test query
    answer = rag.answer("What is entropy?")
    print("RAG Answer:", answer)
