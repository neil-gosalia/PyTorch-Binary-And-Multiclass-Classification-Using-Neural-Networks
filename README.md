# PyTorch Neural Network Classification
 
A self-directed project covering both **binary** and **multiclass** classification using neural networks in PyTorch, with a focus on understanding why non-linearity matters and how to progressively improve a model.
 
---
 
## What This Covers
 
### Part 1 — Binary Classification (Circles Dataset)
 
**Dataset:** 1,000 samples generated using `sklearn.make_circles` — two concentric rings of points that must be separated into class 0 or class 1. A deliberately hard problem for linear models.
 
Four models are built and iterated on:
 
| Model | Architecture | Key Insight |
|---|---|---|
| V0 | Linear → Linear | Baseline — model fails, only guesses |
| V1 | Linear → Linear → Linear (wider + deeper) | More layers/neurons alone don't help — still linear |
| V2 | Linear → **ReLU** → Linear → **ReLU** → Linear | Non-linearity finally solves the problem |
| V3 | Same as V2 but wider (15 hidden units) | Wider network improves decision boundary further |
 
The key discovery demonstrated through V0 and V1 failing: **linear layers alone cannot learn non-linear decision boundaries** — no matter how deep or wide the network. ReLU activation is what unlocks the ability to separate circular data.
 
---
 
### Part 2 — Multiclass Classification (Blobs Dataset)
 
**Dataset:** 1,000 samples across 4 classes generated using `sklearn.make_blobs` with `cluster_std=1.5` (overlapping clusters).
 
**Model: `BlobModel`**
- 3-layer network: Linear → ReLU → Linear → ReLU → Linear
- Output: 4 logits (one per class) converted to probabilities via **Softmax**
- Final prediction via `argmax`
---
 
## Key Concepts Demonstrated
 
| Concept | Implementation |
|---|---|
| Binary classification pipeline | Logits → Sigmoid → Round → Binary label |
| Multiclass classification pipeline | Logits → Softmax → Argmax → Class label |
| Loss functions | `BCEWithLogitsLoss` (binary), `CrossEntropyLoss` (multiclass) |
| Why non-linearity matters | V0/V1 fail on circles; V2 with ReLU succeeds |
| Decision boundary visualisation | `plot_decision_boundary()` for train and test sets |
| Custom activation functions | Manual implementations of ReLU and Sigmoid from scratch |
| Model iteration | Progressively wider/deeper architectures with analysis of what changed |
| Data conversion | NumPy → PyTorch tensors (`torch.from_numpy`) |
| Train/test split | `sklearn.model_selection.train_test_split` |
 
---
 
## Training Setup
 
**Binary Classification (V0–V3)**
- Loss: `BCEWithLogitsLoss`
- Optimizer: SGD (`lr=0.1`)
- Epochs: 100 (V0), 1000 (V1–V3)
**Multiclass Classification (BlobModel)**
- Loss: `CrossEntropyLoss`
- Optimizer: SGD (`lr=0.1`)
- Epochs: 100
---
 
## Tech Stack
 
- Python
- PyTorch
- Scikit-learn
- Matplotlib
- Pandas
- Google Colab
---
 
## How to Run
 
Open the notebook in [Google Colab](https://colab.research.google.com/) and run all cells in order. All datasets are generated synthetically — no downloads required. The notebook fetches `helper_functions.py` from a public GitHub URL automatically for decision boundary plotting.
