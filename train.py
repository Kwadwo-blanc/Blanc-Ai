"""
Phase 5 -- Training Loop & Infrastructure.

Trains the GPT model (Phase 4) on the encoded dataset (Phase 3) using
AdamW and random batches of context windows. Run the real, full
training on a Colab/Kaggle GPU -- CPU works too (laptops) but is much
slower, so use CPU only for a quick smoke test with MAX_ITERS turned
way down, not for the actual run.

Usage: python src/train.py
"""

import os
import time

import torch

from model import build_model

# ---------------------------------------------------------------------
# Hyperparameters -- tune these. Defaults finish in a few minutes on a
# free T4 GPU (Colab/Kaggle). For a local CPU smoke test, temporarily
# drop MAX_ITERS to ~50 just to confirm the script runs end-to-end.
# ---------------------------------------------------------------------
BATCH_SIZE = 64
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 3e-4

PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, block_size, batch_size):
    """Sample a random batch of (input, target) sequences from `data`.
    Each target sequence is the input shifted one position to the right
    -- that's what "predict the next token" means in practice."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size):
    """Average loss over several batches for both train and val sets --
    a single batch's loss is noisy, so averaging gives a clean reading.
    A val loss that stays close to the train loss means the model is
    generalizing; a val loss that climbs while train loss falls means
    it's overfitting (memorizing instead of learning patterns)."""
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, BATCH_SIZE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. Fine for a quick smoke test with "
              "MAX_ITERS turned down -- run the real training run on Colab/Kaggle.")

    train_data = torch.load(os.path.join(PROCESSED_DIR, "train.pt"))
    val_data = torch.load(os.path.join(PROCESSED_DIR, "val.pt"))

    model, config = build_model()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    print(f"Training for {MAX_ITERS} iterations, batch size {BATCH_SIZE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_time = time.time()
    best_val_loss = float("inf")

    for step in range(MAX_ITERS + 1):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, config.block_size)
            elapsed = time.time() - start_time

            # Only save when val loss improves -- otherwise a long training
            # run just overwrites a good early checkpoint with a later,
            # more-overfit one. This means checkpoints/model.pt always ends
            # up holding the best-generalizing version seen during this run,
            # not just whatever the final step happened to produce.
            improved = losses["val"] < best_val_loss
            if improved:
                best_val_loss = losses["val"]
                torch.save(
                    {"model_state": model.state_dict(), "config": config.__dict__},
                    CHECKPOINT_PATH,
                )

            marker = " <- best so far, saved" if improved else ""
            print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | {elapsed:.0f}s elapsed{marker}")

        xb, yb = get_batch(train_data, config.block_size, BATCH_SIZE)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
