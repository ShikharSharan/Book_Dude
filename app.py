# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# -----------------------------
# Config
# -----------------------------
openai.api_key = "YOUR_OPENAI_API_KEY"
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_vector_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_chunks(query, chunks, index, embeddings, top_k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

def ask_llm(query, context):
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].text.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 Book Dude — AI Book Explainer")

uploaded_file = st.file_uploader("Upload a book (PDF)", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")
    text = extract_text_from_pdf(uploaded_file)
    st.write("✅ Extracted text from PDF")

    chunks = chunk_text(text)
    index, embeddings = build_vector_index(chunks)
    st.write(f"✅ Created {len(chunks)} chunks and built vector index")

    st.subheader("💬 Chat with Book")
    user_query = st.text_input("Ask a question about the book:")
    if user_query:
        retrieved = retrieve_chunks(user_query, chunks, index, embeddings)
        context = "\n".join(retrieved)
        answer = ask_llm(user_query, context)
        st.markdown(f"**Answer:** {answer}")

    st.subheader("📖 Chapter Summaries (demo)")
    if st.button("Generate summary of book"):
        summary_prompt = f"Summarize the following text:\n\n{text[:3000]}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=summary_prompt,
            max_tokens=500,
            temperature=0.5
        )
        st.write(response.choices[0].text.strip())
