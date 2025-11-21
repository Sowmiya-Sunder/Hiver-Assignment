								Mini-RAG for Knowledge Base Answering (Hiver Copilot – Part C)

This project implements a mini Retrieval-Augmented Generation (RAG) system
for answering questions using Hiver-style Knowledge Base (KB) articles.

It is designed to:

- Embed KB articles using SentenceTransformers
- Retrieve the most relevant articles using FAISS
- Generate an answer using retrieved context (LLM-ready, with a simple fallback)
- Return retrieved articles + answer + confidence score for each query

---

Project Structure

Part C/
  app.py               # Flask web app
  rag_engine.py        # RAG engine: embeddings + FAISS + answer generation
  knowledge_base/
    article1.txt       # Automations in Hiver
    article2.txt       # Automation failures / conflicts
    article3.txt       # CSAT not appearing – causes
    article4.txt       # CSAT visibility & integrations
    article5.txt       # Enabling CSAT – Settings path
  templates/
    index.html         # Simple UI to enter query and view results
  static/
    style.css               # (optional) CSS / assets
