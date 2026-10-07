"""
Phase 6 -- Training & Monitoring: generate text from the trained model.

This is where you actually see what the model learned. Loads the
checkpoint saved in Phase 5, feeds it a starting prompt, and lets it
generate more text one character at a time.

Usage: python src/generate.py
"""

import json
import os

import torch

from model import GPT, GPTConfig

CHECKPOINT_PATH = "checkpoints/model.pt"
META_PATH = "data/processed/meta.json"

PROMPT = "Communication is"  # starting text -- try changing this and re-running
                              # (was "ROMEO:" for the Tiny Shakespeare run; update
                              # this to fit whichever dataset you last trained on)
MAX_NEW_TOKENS = 300       # how many characters to generate
TEMPERATURE = 0.8          # lower = safer/more repetitive, higher = more random

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run src/train.py yourself, or drop "
            "the team's shared checkpoints/model.pt into this folder."
        )
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"{META_PATH} not found. Run src/tokenizer.py first.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}  # json saved keys as strings

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    unknown = [c for c in PROMPT if c not in stoi]
    if unknown:
        print(f"Warning: these characters aren't in the vocab and will be dropped: {unknown}")
    idx = torch.tensor([[stoi[c] for c in PROMPT if c in stoi]], dtype=torch.long, device=device)

    print("=" * 50)
    print(f"PROMPT: {PROMPT!r}")
    print(f"Generating {MAX_NEW_TOKENS} characters at temperature {TEMPERATURE}...")
    print("=" * 50)

    out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    text = "".join(itos[i] for i in out[0].tolist())

    print(text)
    print("=" * 50)


if __name__ == "__main__":
    main()
