# Consulting AI Assistant (RAG Demo)

A Streamlit demo that lets you upload consulting PDFs (case studies, proposals, client decks) and chat with them using Retrieval-Augmented Generation (RAG).


## Features
- Upload multiple PDFs.
- Chunk + embed using OpenAI or Sentence-Transformers.
- Persistent vector DB (Chroma).
- Conversational memory per session.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

> No OpenAI? It still runs with **HuggingFace sentence-transformers** for embeddings and a dummy fallback for chat.

