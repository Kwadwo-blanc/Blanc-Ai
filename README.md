# LLM From Scratch

A team project to build and train a small-scale language model from scratch,
using Python, PyTorch, and free GPU resources (Google Colab / Kaggle).

## Roadmap

1. Environment & Team Setup
2. Dataset Collection & Preparation
3. Tokenizer
4. Model Architecture
5. Training Loop & Infrastructure
6. Training & Monitoring
7. Evaluation & Testing
8. Iteration / Fine-tuning
9. Wrap-up / Demo

We are currently on **Phase 5**.

## Folder structure

```
llm-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── test_setup.py             # Phase 1: confirms everyone's environment works
├── data/                     # gitignored -- generated locally, not committed
│   ├── raw/                  # Phase 2: downloaded source text
│   └── processed/            # Phase 2: cleaned train/val splits
├── src/
│   ├── download_data.py      # Phase 2
│   ├── prepare_data.py       # Phase 2
│   ├── tokenizer.py          # Phase 3 (done)
│   ├── model.py              # Phase 4 (done)
│   └── train.py              # Phase 5 (done)
├── checkpoints/               # Phase 5: saved model weights (gitignored)
└── notebooks/                # Colab/Kaggle notebooks for training runs
```

## Phase 1 setup instructions (everyone does this)

1. Clone the repo:
   ```
   git clone <your-repo-url>
   cd llm-from-scratch
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the test script:
   ```
   python test_setup.py
   ```
5. Confirm you can also open and run the same script in a Google Colab
   notebook and a Kaggle notebook (GPU runtime enabled) -- this is where
   training will actually happen later.
6. Post a screenshot of your `test_setup.py` output in the team chat.

Phase 1 is complete once all 8 people have a clean run.

## Phase 2 instructions (dataset collection & preparation)

1. Pull the latest from the repo (so you have `src/download_data.py` and
   `src/prepare_data.py`).
2. Download the starter dataset:
   ```
   python src/download_data.py
   ```
   This pulls the "Tiny Shakespeare" corpus into `data/raw/input.txt` --
   a small, clean public-domain text used to get the full pipeline
   working end-to-end before you plug in your own data.
3. Clean and split it into train/validation sets:
   ```
   python src/prepare_data.py
   ```
   This prints basic stats (character count, word count, vocab size)
   and writes `data/processed/train.txt` and `data/processed/val.txt`.
4. One person (or pair) should also start scouting a **custom dataset**
   for the real/final model -- something more meaningful than Shakespeare
   (e.g. news articles, a specific book, a domain you care about). Drop
   candidate `.txt` files in `data/raw/` as you find them; we'll decide
   together which one to train on for real once Phase 3 (tokenizer) is
   working.
5. Everyone runs steps 2-3 locally and confirms they see the same stats
   output in the team chat.

Phase 2 is complete once that's confirmed by everyone.

## Phase 3 instructions (tokenizer)

1. Pull the latest from the repo (so you have `src/tokenizer.py`).
2. Make sure `data/processed/train.txt` and `val.txt` exist (from Phase 2).
3. Run:
   ```
   python src/tokenizer.py
   ```
   This builds a character-level vocabulary from your dataset, encodes
   the train/val text into integer tensors, and saves:
   - `data/processed/train.pt` / `val.pt` -- the encoded data the model
     will actually train on
   - `data/processed/meta.json` -- the vocabulary (character <-> integer
     mappings) and vocab size, which Phase 4's model needs to know its
     input/output size
4. Check the printed stats -- vocab size, token counts, and the
   encode/decode sanity check (`'Hello' -> 'Hello'`). If the sanity
   check doesn't match, something's wrong with the data; flag it in
   the team chat.
5. Everyone runs this and confirms the same vocab size in the team chat
   (it should match everyone else's since you're all using the same
   `data/processed/train.txt` + `val.txt`).

**Things to understand for this phase (quick primer):** a tokenizer
maps text to numbers and back. Char-level (what we're using) is the
simplest -- one integer per character, small vocab, longer sequences.
Word-level or subword/BPE tokenizers (like `tiktoken`) use fewer,
longer tokens and are closer to what real LLMs use, at the cost of
more complexity. We're keeping it simple for now; this can be swapped
later if the team wants a more "production" feel for the final model.

Phase 3 is complete once everyone's vocab size matches.

## Phase 4 instructions (model architecture)

1. Pull the latest from the repo (so you have `src/model.py`).
2. Make sure `data/processed/meta.json` exists (from Phase 3).
3. Run:
   ```
   python src/model.py
   ```
   This builds the model (no training yet) and prints its architecture:
   vocab size, context length, embedding size, number of attention
   heads/layers, and total parameter count (should be a couple million
   -- small on purpose so it trains fast). It then runs one forward pass
   on random fake data as a sanity check -- the loss printed should be
   close to `ln(vocab_size)`, which is what a totally untrained model
   should produce by chance.
4. Everyone runs this and confirms the same parameter count and a
   sanity-check loss near the expected value.

**Things to understand for this phase (quick primer):** `src/model.py`
implements a decoder-only transformer -- the same family as GPT:
- **Token + position embeddings** turn each input token ID into a
  vector, and add information about *where* it is in the sequence.
- **Self-attention** (`CausalSelfAttention`) lets each position look
  back at earlier positions and decide which ones matter for predicting
  what comes next. "Causal" means it can't peek ahead -- essential for
  generating text left to right.
- **Feedforward layers** process each position's information further.
- These are stacked into **blocks** (`n_layer` of them), each wrapped
  in residual connections and layer norm for stable training.
- The **config** (`n_embd`, `n_head`, `n_layer`, `block_size`) controls
  model size vs. speed. Current defaults (~2.7M params) are tuned to
  train in minutes on a free Colab/Kaggle GPU. We can scale these up
  later once the pipeline is proven to work.

No new installs needed -- just `torch`, already in `requirements.txt`.

Phase 4 is complete once everyone confirms the same architecture printout.

## Phase 5 instructions (training loop & infrastructure)

1. Pull the latest from the repo (so you have `src/train.py`).
2. **Everyone** runs a quick local smoke test first, just to confirm the
   script works end-to-end:
   - Open `src/train.py` and temporarily change `MAX_ITERS = 3000` to
     `MAX_ITERS = 50`
   - Run `python src/train.py` -- on a laptop CPU this should finish in
     under a minute and print a few loss lines plus a saved checkpoint.
     Don't worry about the loss going down much; this is just confirming
     nothing crashes.
   - Revert the `MAX_ITERS` change afterward (don't commit it).
3. **One real training run as a team**, not 8 separate ones -- running
   the full thing 8 times burns everyone's free GPU quota for no
   benefit, and you want one shared checkpoint to move forward with,
   not 8 different models. Pick one or two people to:
   - Open `src/train.py` in a Colab or Kaggle notebook (GPU runtime on)
   - Run it with the default `MAX_ITERS = 3000` -- should take a few
     minutes on a free T4 GPU
   - Watch train/val loss print every 250 steps
   - Share the resulting `checkpoints/model.pt` file with the team
     (e.g. upload it to the repo via Git LFS, or drop it in shared
     Drive/WhatsApp if it's small -- it's ~15-20MB for this model size)
4. Everyone else reviews the loss curve shared in the team chat: train
   loss should steadily fall; val loss should fall too but may lag
   slightly behind train loss -- that gap is normal and only a problem
   if val loss starts *rising* while train loss keeps falling
   (overfitting).

**Things to understand for this phase (quick primer):**
- **Batching**: instead of training on one example at a time, we sample
  a batch of random chunks from the dataset at once -- faster and gives
  more stable gradient updates.
- **Forward pass -> loss -> backward pass -> optimizer step**: the model
  predicts next-tokens (forward), we measure how wrong it was (loss),
  compute how to adjust every parameter to reduce that error
  (backward/backpropagation), then actually adjust them (optimizer
  step, using AdamW here -- a standard, reliable optimizer).
- **Train vs. val loss**: val data is never trained on, so val loss
  tells you how well the model generalizes vs. just memorizing.
- **Checkpointing**: periodically saving model weights to disk so
  training can be resumed, or the model used later, without starting
  over.

No new installs needed -- just `torch`, already in `requirements.txt`.

Once the team has one shared trained checkpoint with a sensible loss
curve, we move to Phase 6 (training & monitoring -- generating sample
text from the trained model to see what it actually learned).
