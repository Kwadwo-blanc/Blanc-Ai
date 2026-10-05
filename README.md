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

We are currently on **Phase 1**.

## Folder structure

```
llm-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── test_setup.py          # Phase 1: confirms everyone's environment works
├── data/                  # Phase 2: raw and processed text data (gitignored)
├── src/
│   ├── tokenizer.py        # Phase 3
│   ├── model.py            # Phase 4
│   └── train.py             # Phase 5-6
└── notebooks/              # Colab/Kaggle notebooks for training runs
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

Once all 8 people have a clean run, we move to Phase 2 (dataset).
