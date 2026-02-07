# llm.py
"""
Book Dude - LLM Module
----------------------
This module handles interaction with Large Language Models (LLMs).
Supports:
- OpenAI GPT models (API-based)
- HuggingFace transformers (local models)
Provides functions for:
- Summarization
- Q&A with citations
- Concept explanation
- Quiz generation
"""

import openai
from typing import List, Dict, Optional

# Optional: HuggingFace local model
from transformers import pipeline


class LLMWrapper:
    def __init__(self, provider: str = "openai", model_name: str = None, api_key: str = None):
        """
        Initialize the LLM wrapper.

        Args:
            provider (str): "openai" or "huggingface"
            model_name (str): Model name (default: GPT-3.5 for OpenAI, distilgpt2 for HuggingFace)
            api_key (str): OpenAI API key if using OpenAI
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            self.model_name = model_name or "gpt-3.5-turbo"
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI provider")
            openai.api_key = api_key

        elif self.provider == "huggingface":
            self.model_name = model_name or "distilgpt2"
            self.generator = pipeline("text-generation", model=self.model_name)

        else:
            raise ValueError("Provider must be 'openai' or 'huggingface'")

    def _openai_chat(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "You are Book Dude, an AI book explainer."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"].strip()

    def _huggingface_generate(self, prompt: str, max_tokens: int = 200) -> str:
        outputs = self.generator(prompt, max_length=len(prompt.split()) + max_tokens, num_return_sequences=1)
        return outputs[0]["generated_text"]

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        if self.provider == "openai":
            return self._openai_chat(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._huggingface_generate(prompt, max_tokens)

    # -----------------------------
    # Task-specific functions
    # -----------------------------
    def summarize(self, text: str) -> str:
        prompt = f"Summarize the following text clearly and concisely:\n\n{text}"
        return self.generate(prompt, max_tokens=400)

    def explain_concept(self, concept: str, context: Optional[str] = None) -> str:
        prompt = f"Explain the concept '{concept}' in simple terms."
        if context:
            prompt += f"\nContext: {context}"
        return self.generate(prompt, max_tokens=300)

    def answer_question(self, question: str, context: str) -> str:
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.generate(prompt, max_tokens=400)

    def generate_quiz(self, text: str, num_questions: int = 5) -> List[str]:
        prompt = f"Generate {num_questions} multiple-choice questions based on the following text:\n\n{text}"
        quiz = self.generate(prompt, max_tokens=500)
        return quiz.split("\n")


if __name__ == "__main__":
    # Example usage for local testing
    llm = LLMWrapper(provider="openai", api_key="YOUR_OPENAI_API_KEY")

    sample_text = "Entropy is a measure of uncertainty in information theory."
    print("Summary:", llm.summarize(sample_text))
    print("Concept Explanation:", llm.explain_concept("Entropy", sample_text))
    print("Q&A:", llm.answer_question("What is entropy?", sample_text))
    print("Quiz:", llm.generate_quiz(sample_text, num_questions=3))
