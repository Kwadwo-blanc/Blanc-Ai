"""
Phase 2 -- Clean and split the dataset into train/val sets.

Run after download_data.py. Reads data/raw/input.txt, reports basic
stats, and writes data/processed/train.txt and data/processed/val.txt.
"""

import os

RAW_PATH = "data/raw/input.txt"
PROCESSED_DIR = "data/processed"
VAL_FRACTION = 0.1  # 10% held out for validation


def prepare():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run download_data.py first."
        )

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # --- basic cleaning ---
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
    prepare()
