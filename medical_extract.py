"""
Medical Q&A Module -- Step 1: Extract and chunk text from PDFs.

This is a separate module from the main from-scratch LLM pipeline. It
powers a retrieval-based Q&A tool: instead of trying to get a small
generative model to "know" and recall medical facts (which it can't do
reliably -- see the team's Phase 7 discussion on hallucination), this
extracts real text from your PDFs and later lets you search it directly.
The model never invents facts; it only ever returns real passages from
your own source documents.

Usage:
1. Put your PDF files in data/medical_pdfs/
2. Run: python src/medical_extract.py
3. This writes data/medical/chunks.json -- a list of text chunks, each
   tagged with which PDF and page it came from.
"""

import json
import os
import re

import pdfplumber

PDF_DIR = "data/medical_pdfs"
OUTPUT_DIR = "data/medical"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "chunks.json")

# How big each searchable chunk is. Smaller chunks = more precise
# retrieval but less surrounding context; bigger chunks = more context
# but less precise matching. ~150-300 words is a reasonable middle
# ground for short illness/treatment entries.
CHUNK_WORD_TARGET = 200
CHUNK_WORD_OVERLAP = 40  # chunks overlap slightly so an answer near a
                          # chunk boundary doesn't get split awkwardly


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, target_words=CHUNK_WORD_TARGET, overlap=CHUNK_WORD_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def extract_pdf(path):
    """Yields (page_number, cleaned_text) for each non-empty page."""
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = clean_text(raw)
            if text:
                yield page_num, text


def main():
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(
            f"{PDF_DIR} not found. Create it and drop your PDF files in there first."
        )

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No .pdf files found in {PDF_DIR}.")

    print("=" * 50)
    print("MEDICAL PDF EXTRACTION")
    print("=" * 50)
    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")

    all_chunks = []
    chunk_id = 0

    for filename in pdf_files:
        path = os.path.join(PDF_DIR, filename)
        print(f"\nProcessing {filename} ...")
        page_count = 0
        chunk_count_for_file = 0

        for page_num, page_text in extract_pdf(path):
            page_count += 1
            for chunk in chunk_text(page_text):
                all_chunks.append({
                    "id": chunk_id,
                    "source_file": filename,
                    "page": page_num,
                    "text": chunk,
                })
                chunk_id += 1
                chunk_count_for_file += 1

        print(f"  {page_count} page(s) with text -> {chunk_count_for_file} chunk(s)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print("-" * 50)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_PATH}")
    print("=" * 50)

    if all_chunks:
        print("\nSample chunk:")
        print(json.dumps(all_chunks[0], indent=2)[:500])


if __name__ == "__main__":
    main()
