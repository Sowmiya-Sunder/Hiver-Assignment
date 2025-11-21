from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KB_FOLDER = "knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 3


@dataclass
class RetrievedDoc:
    title: str
    text: str
    similarity: float   # cosine similarity in [-1, 1]
    confidence: float   # softmax-normalized confidence in [0, 1]


_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_kb_titles: List[str] = []
_kb_texts: List[str] = []


def _load_kb(folder: str = KB_FOLDER) -> Tuple[List[str], List[str]]:
    """Load all .txt KB files from the given folder."""
    titles: List[str] = []
    texts: List[str] = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
            titles.append(fname)

    if not texts:
        raise RuntimeError(f"No .txt KB files found in folder: {folder}")

    return titles, texts


def _build_index(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
) -> tuple[SentenceTransformer, faiss.Index, np.ndarray]:
    """Create embeddings and build a FAISS index (cosine similarity via IP)."""
    # Load embedding model
    model = SentenceTransformer(model_name)

    # Encode documents with L2-normalisation so that inner product = cosine similarity
    doc_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product index (works as cosine here)

    index.add(doc_embeddings)

    return model, index, doc_embeddings


# Initialise on import
_kb_titles, _kb_texts = _load_kb(KB_FOLDER)
_model, _index, _doc_embeddings = _build_index(_kb_texts, EMBEDDING_MODEL_NAME)


def retrieve(query: str, k: int = TOP_K_DEFAULT) -> List[tuple[str, str, float]]:
    """
    Retrieve top-k KB documents for a given query.

    Returns:
        List of tuples: (title, text, confidence)
    """
    if not query or not query.strip():
        return []

    assert _model is not None and _index is not None

    query_vec = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # FAISS expects shape (n_queries, dim)
    sims, idxs = _index.search(query_vec, k)
    sims = sims[0]  # cosine similarities
    idxs = idxs[0]

    # Convert similarities to softmax-based confidence scores
    # Higher similarity -> higher confidence, all in [0, 1] and sum to 1
    exp_scores = np.exp(sims - np.max(sims))
    probs = exp_scores / np.sum(exp_scores)

    retrieved_docs: List[RetrievedDoc] = []
    for sim, idx, prob in zip(sims, idxs, probs):
        if idx < 0:
            continue
        retrieved_docs.append(
            RetrievedDoc(
                title=_kb_titles[idx],
                text=_kb_texts[idx],
                similarity=float(sim),
                confidence=float(round(float(prob), 3)),
            )
        )

    # The Flask app expects a simple tuple format; convert for compatibility
    simple_results: List[tuple[str, str, float]] = []
    for doc in retrieved_docs:
        simple_results.append((doc.title, doc.text, doc.confidence))

    return simple_results


def _build_prompt(query: str, retrieved: List[RetrievedDoc]) -> str:
    """Builds a prompt for an LLM using the retrieved KB docs as context."""
    ctx_parts = []
    for doc in retrieved:
        ctx_parts.append(f"### {doc.title}\n{doc.text}")
    context = "\n\n---\n\n".join(ctx_parts)

    prompt = f"""You are Hiver Copilot, a helpful support assistant.

Use ONLY the information in the knowledge base articles below to answer the
user's question. If the answer is not clearly present, say you are not sure and
suggest what the user can check in Hiver.

Knowledge Base:
{context}

User question:
{query}

Answer in 3–6 sentences, concise and actionable:
"""
    return prompt


def generate_answer(query: str, retrieved_raw: List[tuple[str, str, float]]) -> str:
    """Generate an answer from retrieved docs.

    NOTE:
      - In a real deployment, you should send the built prompt to an LLM
        (OpenAI or an open-source model).
      - Here we keep a simple extractive-style answer so the project runs
        without external dependencies or API keys.
    """
    if not retrieved_raw:
        return "I couldn't find any relevant knowledge base article for this query."

    # Rebuild RetrievedDoc objects from simple tuples
    retrieved: List[RetrievedDoc] = []
    for (title, text, conf) in retrieved_raw:
        # similarity is not used for now in generation; pass 0.0
        retrieved.append(
            RetrievedDoc(
                title=title,
                text=text,
                similarity=0.0,
                confidence=conf,
            )
        )

   
    # Fallback: simple heuristic answer using concatenated snippets
    combined_snippets = "\n\n".join(
        f"[{doc.title}] {doc.text}" for doc in retrieved
    )
    return (
        "Based on the most relevant knowledge base articles, here is a summary:\n\n"
        f"{combined_snippets}"
    )
