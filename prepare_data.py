"""
Phase 2 -- Clean and split the dataset into train/val sets.

Reads a raw text file, reports basic stats, and writes
data/processed/train.txt and data/processed/val.txt.

By default reads data/raw/input.txt (the Tiny Shakespeare starter
corpus from download_data.py). To use a different file -- e.g. your
team's real/custom dataset from Phase 8 -- pass its path:

  python src/prepare_data.py data/raw/your_file.txt
"""

import os
import sys
import unicodedata

PROCESSED_DIR = "data/processed"
VAL_FRACTION = 0.1  # 10% held out for validation
DEFAULT_RAW_PATH = "data/raw/input.txt"


def clean_control_chars(text):
    """Strip non-printable control characters (common artifacts from
    PDF-to-text conversion) while keeping normal whitespace (newline,
    tab, space)."""
    return "".join(
        c for c in text
        if c in ("\n", "\t") or unicodedata.category(c) != "Cc"
    )


def prepare(raw_path):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run download_data.py first, or check the path."
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- basic cleaning ---
    text = clean_control_chars(text)
    text = text.strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    num_chars = len(text)
    num_words = len(text.split())

    print("=" * 50)
    print("DATASET STATS")
    print("=" * 50)
    print(f"Source file: {raw_path}")
    print(f"Total characters: {num_chars:,}")
    print(f"Total words (approx): {num_words:,}")
    print(f"Unique characters (vocab size): {vocab_size}")
    print("=" * 50)

    split_idx = int(len(text) * (1 - VAL_FRACTION))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(os.path.join(PROCESSED_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"Saved {len(train_text):,} chars -> {PROCESSED_DIR}/train.txt")
    print(f"Saved {len(val_text):,} chars -> {PROCESSED_DIR}/val.txt")


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_PATH
    prepare(raw_path)
