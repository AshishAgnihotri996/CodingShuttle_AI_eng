"""
Simple RAG: read a file -> embed each line -> retrieve best matches for a question
Shows both Cosine similarity and L2 (Euclidean) distance for every chunk.

Install:  pip install sentence-transformers numpy
Run:      python simple_rag.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer

FILE_PATH = "knowledge.txt"
TOP_K = 3

# 1. Load the file (one line = one chunk)
with open(FILE_PATH, encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# 2. Embed all chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vecs = model.encode(chunks)  # shape: (num_chunks, 384)


def cosine_similarity(a, b):
    # 1.0 = same direction (very similar), 0 = unrelated
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_distance(a, b):
    # 0 = identical, bigger = more different
    return float(np.linalg.norm(a - b))


def ask(question):
    q_vec = model.encode(question)

    # 3. Score every chunk with both metrics
    results = []
    for text, vec in zip(chunks, chunk_vecs):
        results.append({
            "text": text,
            "cos": cosine_similarity(q_vec, vec),
            "l2": l2_distance(q_vec, vec),
        })

    # 4. Rank by cosine similarity (highest first)
    results.sort(key=lambda r: r["cos"], reverse=True)
    top = results[:TOP_K]

    print(f"\nQuestion: {question}\n")
    print(f"Top {TOP_K} retrieved chunks:")
    print("-" * 80)
    for i, r in enumerate(top, 1):
        print(f"{i}. Cosine: {r['cos']:.4f} | L2: {r['l2']:.4f}")
        print(f"   {r['text']}\n")

    # 5. "Generate" step: here we simply return the best chunk.
    #    To use a real LLM, send this prompt to it instead.
    context = "\n".join(r["text"] for r in top)
    prompt = f"Answer using only this context:\n{context}\n\nQuestion: {question}"
    print("Answer (best match):", top[0]["text"])
    return prompt  # pass to an LLM if you want a written answer


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        ask(q)
