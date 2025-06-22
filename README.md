# 🤖 ZachzzGPT – AI-Powered News Research Tool

ZachzzGPT is an intelligent research assistant built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Groq's LLM](https://groq.com/). It loads news articles from URLs, processes them, creates a FAISS vector store, and allows users to ask questions with LLM-powered answers and sources.

---

## 🧠 Features

- 🔗 Load multiple article URLs
- 🧱 Chunk and vectorize text using LangChain
- 💾 Save and reuse FAISS vector index
- 🤖 Query documents using Groq's LLaMA 3 or any OpenRouter-compatible LLM
- 🌐 User interface via Streamlit

---

## 📁 File Overview

| File                        | Description |
|-----------------------------|-------------|
| `ZachzzGPT.py`              | Main app — runs the Streamlit interface |
| `FAISS.ipynb`               | Notebook to create FAISS index manually |
| `Langchain_splitter.ipynb` | Notebook for splitting and preprocessing text |
| `Main.ipynb`                | End-to-end demo in Jupyter |
| `vector_index.pkl`         | Pickled FAISS index (for OpenAI) |
| `faiss_store_groq.pkl`     | FAISS index for Groq model |
| `requirements.txt`         | List of required Python packages |
| `nvda_news_1.txt`          | Sample input text file for testing |
| `.env`                     | API key storage (excluded from Git) |
| `movies.csv`, `sample_text.csv` | Sample datasets for testing |
| `.ipynb_checkpoints/`      | Auto-saved Jupyter backup files (can ignore) |

---
