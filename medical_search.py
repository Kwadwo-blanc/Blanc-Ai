"""
Medical Q&A Module -- Step 2: Search the extracted PDF chunks.

Uses TF-IDF + cosine similarity -- a standard, well-understood
information-retrieval technique (no training, no GPU needed, easy to
explain in a report). Given a question, it finds and returns the most
relevant real passages from your PDFs, with their source file and page
number cited. It never generates new text, so it cannot hallucinate a
fact that isn't in your source documents -- it can only ever be wrong
by picking an irrelevant passage, which you can see and judge yourself.

Run after medical_extract.py.

Usage:
  python src/medical_search.py                 # interactive query loop
  python src/medical_search.py "your question"  # one-off query
"""

import json
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = "data/medical/chunks.json"
TOP_K = 3  # how many matching passages to return per query

DISCLAIMER = (
    "NOTE: This tool retrieves passages from the PDFs you provided -- it "
    "does not generate medical advice and is not a substitute for a "
    "licensed healthcare professional. Always verify against the source "
    "document and consult a doctor or pharmacist for real decisions."
)


def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run src/medical_extract.py first."
        )
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def search(query, chunks, vectorizer, matrix, top_k=TOP_K):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    ranked_idx = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked_idx:
        if scores[idx] <= 0:
            continue  # no meaningful match -- don't return noise
        results.append({
            "score": float(scores[idx]),
            "source_file": chunks[idx]["source_file"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
        })
    return results


def print_results(query, results):
    print("=" * 60)
    print(f"QUERY: {query!r}")
    print("=" * 60)
    if not results:
        print("No matching passages found in the PDFs. Try rephrasing, "
              "or this topic may not be covered in your source documents.")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] {r['source_file']} (page {r['page']}) -- relevance score: {r['score']:.3f}")
        print("-" * 60)
        print(r["text"])
    print("\n" + DISCLAIMER)
    print("=" * 60)


def main():
    chunks = load_chunks()
    vectorizer, matrix = build_index(chunks)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}. Index ready.")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = search(query, chunks, vectorizer, matrix)
        print_results(query, results)
        return

    print("\nType a question (e.g. 'symptoms of malaria'), or 'quit' to exit.")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = search(query, chunks, vectorizer, matrix)
        print_results(query, results)


if __name__ == "__main__":
    main()
