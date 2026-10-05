"""
Phase 3 -- Build a character-level tokenizer and encode the dataset.

A tokenizer turns raw text into numbers the model can learn from, and
back again. We're using the simplest possible kind -- character-level --
where every unique character in the dataset gets its own integer ID.
This keeps vocab size tiny (~65 for Tiny Shakespeare) and is exactly
what the original nanoGPT tutorial uses.

(Word-level or subword/BPE tokenizers -- like the `tiktoken` library
already in requirements.txt -- give a smaller sequence length and more
"real LLM" behavior, but add complexity. We can swap to that later for
the final custom-dataset model if the team wants.)

Run after prepare_data.py.
"""

import os
import json

import torch

PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.txt")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.txt")
META_PATH = os.path.join(PROCESSED_DIR, "meta.json")


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"{TRAIN_PATH} not found. Run prepare_data.py first.")

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_text = f.read()

    # Vocab is built from train+val combined so no character is "unseen"
    full_text = train_text + val_text
    stoi, itos = build_vocab(full_text)
    vocab_size = len(stoi)

    print("=" * 50)
    print("TOKENIZER")
    print("=" * 50)
    print(f"Vocab size: {vocab_size}")
    print(f"Sample chars: {list(stoi.keys())[:20]}")
    print("=" * 50)

    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    torch.save(train_tensor, os.path.join(PROCESSED_DIR, "train.pt"))
    torch.save(val_tensor, os.path.join(PROCESSED_DIR, "val.pt"))

    # json requires string keys, so itos keys become strings on disk --
    # convert back to int with `{int(k): v for k, v in itos.items()}` on load
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    print(f"Saved encoded tensors -> {PROCESSED_DIR}/train.pt, {PROCESSED_DIR}/val.pt")
    print(f"Saved vocab -> {META_PATH}")

    # Quick sanity check: encode then decode a sample string
    sample = "Hello"
    try:
        check = decode(encode(sample, stoi), itos)
        print(f"Sanity check -- encode/decode '{sample}' -> '{check}' (should match)")
    except KeyError:
        print(f"Sanity check skipped -- '{sample}' contains characters not in this vocab")


if __name__ == "__main__":
    main()
