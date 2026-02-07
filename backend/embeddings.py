# embeddings.py
"""
Book Dude - Embeddings Module
-----------------------------
This module generates vector embeddings for text chunks.
Supports:
- HuggingFace sentence-transformers (local, free)
- OpenAI embeddings (API-based, paid)
"""

from typing import List, Union
import numpy as np

# Option 1: HuggingFace local model
from sentence_transformers import SentenceTransformer

# Option 2: OpenAI API
import openai


class EmbeddingsGenerator:
    def __init__(self, provider: str = "huggingface", model_name: str = None, api_key: str = None):
        """
        Initialize the embeddings generator.

        Args:
            provider (str): "huggingface" or "openai"
            model_name (str): Model name (default: MiniLM for HuggingFace, text-embedding-ada-002 for OpenAI)
            api_key (str): OpenAI API key if using OpenAI
        """
        self.provider = provider.lower()

        if self.provider == "huggingface":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)

        elif self.provider == "openai":
            self.model_name = model_name or "text-embedding-ada-002"
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            openai.api_key = api_key

        else:
            raise ValueError("Provider must be 'huggingface' or 'openai'")

    def generate(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts (List[str]): List of text strings.

        Returns:
            np.ndarray: Array of embeddings.
        """
        if self.provider == "huggingface":
            return np.array(self.model.encode(texts))

        elif self.provider == "openai":
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [item["embedding"] for item in response["data"]]
            return np.array(embeddings)


if __name__ == "__main__":
    # Example usage for local testing
    chunks = [
        "Entropy is a measure of uncertainty in information theory.",
        "Bayes theorem relates conditional probabilities."
    ]

    # HuggingFace example
    hf_gen = EmbeddingsGenerator(provider="huggingface")
    hf_embeddings = hf_gen.generate(chunks)
    print("HuggingFace embeddings shape:", hf_embeddings.shape)

    # Uncomment for OpenAI example (requires API key)
    # openai_gen = EmbeddingsGenerator(provider="openai", api_key="YOUR_OPENAI_API_KEY")
    # openai_embeddings = openai_gen.generate(chunks)
    # print("OpenAI embeddings shape:", openai_embeddings.shape)
