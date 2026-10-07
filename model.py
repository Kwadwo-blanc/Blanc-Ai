"""
Phase 4 -- Model Architecture: a small GPT-style transformer, built from scratch.

This defines the actual neural network: a decoder-only transformer that
reads a sequence of token IDs and predicts the next one. Same
architecture family as GPT, just far smaller.

Needs: data/processed/meta.json (for vocab_size) from Phase 3.
"""

import json
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------
# Config -- tune these to trade off model size vs training speed.
# Defaults are small on purpose so this trains fast on a free GPU.
# ---------------------------------------------------------------------
class GPTConfig:
    def __init__(self, vocab_size, block_size=256, n_embd=192, n_head=6, n_layer=6, dropout=0.2):
        # dropout raised from 0.1 -> 0.2 after the Communication Book run
        # showed overfitting (val perplexity 17.98 vs train 1.77, gap 2.32)
        # -- small datasets need more regularization to avoid memorizing.
        self.vocab_size = vocab_size   # how many distinct tokens exist (from Phase 3)
        self.block_size = block_size   # max context length the model can look back at
        self.n_embd = n_embd           # embedding / hidden dimension
        self.n_head = n_head           # number of attention heads
        self.n_layer = n_layer         # number of transformer blocks stacked
        self.dropout = dropout         # dropout rate (regularization)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only attend to
    itself and earlier positions -- "causal" means no peeking at the future,
    which is what lets this model generate text one token at a time."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dim

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """Two-layer MLP applied to each position independently. This is where
    most of the model's per-token "thinking" capacity lives."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then feedforward, each wrapped
    in a residual connection and preceded by layer norm (pre-norm style,
    the standard modern setup -- more stable to train than post-norm)."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Given a starting sequence of token IDs, generate more tokens one
        at a time by sampling from the model's predicted next-token distribution."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def build_model():
    """Loads vocab_size from Phase 3's meta.json and builds a GPT model."""
    meta_path = "data/processed/meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. Run src/tokenizer.py first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    config = GPTConfig(vocab_size=meta["vocab_size"])
    model = GPT(config)
    return model, config


if __name__ == "__main__":
    model, config = build_model()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 50)
    print("MODEL ARCHITECTURE")
    print("=" * 50)
    print(f"vocab_size: {config.vocab_size}")
    print(f"block_size (context length): {config.block_size}")
    print(f"n_embd: {config.n_embd}")
    print(f"n_head: {config.n_head}")
    print(f"n_layer: {config.n_layer}")
    print(f"Total parameters: {n_params:,}")
    print("=" * 50)

    # Sanity check -- forward pass on random fake data, no training yet
    dummy_input = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(dummy_input, targets=dummy_input)
    print(f"Sanity check -- output shape: {tuple(logits.shape)}, loss: {loss.item():.4f}")
    print("(loss should be roughly ln(vocab_size) for an untrained model -- "
          f"expected ~{math.log(config.vocab_size):.2f})")
