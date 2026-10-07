"""
Phase 7 -- Evaluation & Testing.

Goes beyond "the loss number went down" and "the sample text looked okay"
to give the team concrete numbers to compare across runs/datasets:
- Perplexity on train and val sets (the standard LLM quality metric)
- A closer look at the train/val gap (overfitting check)
- A small battery of fixed prompts, generated and saved to a file, so
  outputs are comparable across the team instead of everyone testing
  different random prompts.

Usage: python src/evaluate.py
"""

import json
import math
import os

import torch

from model import GPT, GPTConfig

CHECKPOINT_PATH = "checkpoints/model.pt"
META_PATH = "data/processed/meta.json"
PROCESSED_DIR = "data/processed"
RESULTS_PATH = "eval_results.txt"

EVAL_BATCHES = 100   # more batches = more stable/trustworthy numbers than Phase 5's quick checks
BATCH_SIZE = 64
TEST_PROMPTS = ["Communication is", "The best way to", "When you listen", "A good leader"]
# (these were ["ROMEO:", "JULIET:", "To be, or", "The king"] for the Tiny
# Shakespeare run -- update this list to fit whichever dataset you last
# trained on, so the fixed-prompt comparison stays meaningful)
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def compute_loss_and_perplexity(model, data, block_size, num_batches):
    """Perplexity = e^(average loss). It's the standard way LLMs report
    quality: roughly, "how many equally-likely choices was the model
    choosing between, on average, for each next character?" Lower is
    better. A perplexity near vocab_size means the model is basically
    guessing randomly; a well-trained char-level model on this kind of
    text should land well below that."""
    losses = torch.zeros(num_batches)
    for i in range(num_batches):
        x, y = get_batch(data, block_size, BATCH_SIZE)
        _, loss = model(x, y)
        losses[i] = loss.item()
    avg_loss = losses.mean().item()
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


@torch.no_grad()
def generate_sample(model, stoi, itos, prompt, max_new_tokens, temperature):
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return "".join(itos[i] for i in out[0].tolist())


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run src/train.py, or drop in the "
            "team's shared checkpoint, before running this."
        )

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}

    train_data = torch.load(os.path.join(PROCESSED_DIR, "train.pt"))
    val_data = torch.load(os.path.join(PROCESSED_DIR, "val.pt"))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print("=" * 60)
    print("PHASE 7 -- EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Running {EVAL_BATCHES} eval batches per split for stable numbers...")

    train_loss, train_ppl = compute_loss_and_perplexity(model, train_data, config.block_size, EVAL_BATCHES)
    val_loss, val_ppl = compute_loss_and_perplexity(model, val_data, config.block_size, EVAL_BATCHES)

    gap = val_loss - train_loss
    random_baseline_ppl = config.vocab_size  # an untrained model guessing uniformly at random

    print("-" * 60)
    print(f"Train loss: {train_loss:.4f}  |  Train perplexity: {train_ppl:.2f}")
    print(f"Val loss:   {val_loss:.4f}  |  Val perplexity:   {val_ppl:.2f}")
    print(f"Val - Train loss gap: {gap:.4f}")
    print(f"Random-guessing baseline perplexity: ~{random_baseline_ppl} (vocab_size)")
    print("-" * 60)

    if gap > 0.5:
        verdict = ("Val loss is notably higher than train loss -- likely overfitting. "
                   "Consider: more data, fewer iterations, or a smaller model.")
    elif gap < -0.1:
        verdict = ("Val loss is lower than train loss, which is unusual -- double check "
                   "the train/val split and that eval batches aren't overlapping weirdly.")
    else:
        verdict = "Train and val loss are close -- the model is generalizing reasonably, not just memorizing."
    print(f"Verdict: {verdict}")
    print("=" * 60)

    # --- Fixed prompt battery, saved to file so the whole team compares the same outputs ---
    print(f"\nGenerating {len(TEST_PROMPTS)} fixed test prompts -> {RESULTS_PATH}")
    lines = [
        "PHASE 7 EVALUATION RESULTS",
        "=" * 60,
        f"Train loss: {train_loss:.4f} | Train perplexity: {train_ppl:.2f}",
        f"Val loss:   {val_loss:.4f} | Val perplexity:   {val_ppl:.2f}",
        f"Val - Train gap: {gap:.4f}",
        f"Verdict: {verdict}",
        "=" * 60,
        "",
    ]
    for prompt in TEST_PROMPTS:
        sample = generate_sample(model, stoi, itos, prompt, MAX_NEW_TOKENS, TEMPERATURE)
        lines.append(f"PROMPT: {prompt!r}")
        lines.append("-" * 40)
        lines.append(sample)
        lines.append("")
        print(f"  Generated for prompt {prompt!r}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nDone. Full results saved to {RESULTS_PATH} -- share this file in the team chat.")


if __name__ == "__main__":
    main()
