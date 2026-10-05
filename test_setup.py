"""
Phase 1 -- Environment Setup Test Script
Everyone on the team runs this locally AND in Google Colab / Kaggle
to confirm their environment is ready for Phase 2.

Usage: python test_setup.py
"""

import sys
import platform

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected -- expected on a laptop. Training will happen on Colab/Kaggle.")
except ImportError:
    print("PyTorch NOT installed. Run: pip install torch")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy NOT installed. Run: pip install numpy")

try:
    import tiktoken
    print(f"tiktoken installed OK (will be used in Phase 3)")
except ImportError:
    print("tiktoken NOT installed. Run: pip install tiktoken")

print("=" * 50)
print("If PyTorch imported without errors, this machine is ready for Phase 2.")
print("Everyone on the team should see this same output before we move on.")
print("=" * 50)
