#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from langchain.schema import Document

# ---------------- Save Documents If Not Already Saved ------------------
if not os.path.exists("documents.pkl"):
    documents = [
        Document(page_content="Type 2 Diabetes is characterized by insulin resistance..."),
        Document(page_content="Symptoms include excessive thirst, frequent urination..."),
        Document(page_content="Blood glucose monitoring is essential in managing diabetes..."),
    ]
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print("✅ Saved 'documents.pkl'")
else:
    print("📄 'documents.pkl' already exists, skipping save.")

# ----------------- Custom CSS Styling ------------------
st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title Styling */
    h1 {
        color: #0A58CA;
        text-align: center;
        padding: 0.5rem 0;
    }

    /* Subheaders */
    .stMarkdown h2 {
        color: #004080;
        border-bottom: 2px solid #004080;
        padding-bottom: 5px;
        margin-top: 2rem;
    }

    /* Input box */
    input[type="text"] {
        border-radius: 10px;
        border: 1px solid #0A58CA;
        padding: 10px;
    }

    /* Button */
    div.stButton > button:first-child {
        background-color: #0A58CA;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #084298;
    }

    /* Answer Box */
    .answer-box {
        background-color: #dbeafe;
        padding: 1rem;
        border-left: 5px solid #0A58CA;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.05rem;
    }

    /* Document box */
    .doc-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Models ------------------
st.cache_resource()
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("./medical")
    model = AutoModelForSeq2SeqLM.from_pretrained("./medical")
    return embedding_model, tokenizer, model

embedding_model, tokenizer, generator_model = load_models()

# ---------------- Load FAISS & Docs ------------------
@st.cache_resource()
def load_index_and_docs():
    faiss_index = faiss.read_index("faiss_index_file.index")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return faiss_index, documents

faiss_index, documents = load_index_and_docs()

# ---------------- Answer Generator ------------------
def generate_answer(context, query, model, tokenizer):
    prompt = f"""Context: {context}
Question: {query}

Instructions:
- Carefully consider negations (words like "not", "never", "negative").
- Ignore any technical symbols like '$cause_1' or '$intermedia_3'. Focus only on medical facts.
- Write a clear, informative answer.
- Do not repeat the same phrase multiple times.
- Focus on summarizing relevant facts from the context.
- If question ends with ?, then explain the answer.
- If information is missing, say "Not enough information."

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- UI ------------------
st.title("🧠 Medical RAG Assistant")

st.markdown("Enter your **clinical query** below and get context-aware medical answers backed by document retrieval.", unsafe_allow_html=True)

query = st.text_input("Enter Clinical Query:", placeholder="e.g., Describe the symptoms of Type 2 Diabetes")

if st.button("Generate Answer"):
    if query.strip() == "":
        st.warning("Please enter a clinical query.")
    else:
        # Encode + Retrieve
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=3)
        top_docs = [documents[i] for i in I[0]]

        # Display documents
        st.markdown("## 📄 Retrieved Documents")
        for idx, doc in enumerate(top_docs, start=1):
            st.markdown(f"""
                <div class="doc-box">
                    <b>Document {idx}:</b><br>
                    {doc.page_content}
                </div>
            """, unsafe_allow_html=True)

        # Generate answer
        context = " ".join([doc.page_content for doc in top_docs])[:1500]
        answer = generate_answer(context, query, generator_model, tokenizer)

        # Display final answer
        st.markdown("## 🤖 Generated Answer")
        st.markdown(f"""<div class="answer-box">{answer}</div>""", unsafe_allow_html=True)

