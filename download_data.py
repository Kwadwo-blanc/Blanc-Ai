"""
Phase 2 -- Download starter dataset.

Downloads the "Tiny Shakespeare" corpus: the same dataset used in
Andrej Karpathy's original nanoGPT tutorial. It's small (~1MB), clean,
plain text -- perfect for a first full pipeline run.

Once the pipeline works end-to-end, swap this out for your own text
corpus (see the note at the bottom of this file).
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RAW_DIR = "data/raw"
RAW_PATH = os.path.join(RAW_DIR, "input.txt")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"Already downloaded: {RAW_PATH}")
        return RAW_PATH

    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    size_kb = os.path.getsize(RAW_PATH) / 1024
    print(f"Done. Saved to {RAW_PATH} ({size_kb:.1f} KB)")
    return RAW_PATH


if __name__ == "__main__":
    download()

# ---------------------------------------------------------------------
# To use your own data instead:
# 1. Put your .txt file(s) in data/raw/
# 2. Skip this script, or point prepare_data.py at your file directly
# 3. Bigger/messier text? You'll need more cleaning in prepare_data.py
#    (strip HTML, remove duplicate lines, filter non-text junk, etc.)
# ---------------------------------------------------------------------
