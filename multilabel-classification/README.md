# PoC 1.2: Multi-Label & Multi-Class Text Classification

This Proof of Concept (PoC) demonstrates how to fine-tune transformer models (such as `DistilBERT`) for **Multi-Label Text Classification**, where a single text input can simultaneously belong to zero, one, or multiple categories (e.g., `["billing", "urgency"]`).

---

## 🔬 Key Concepts & Differences

| Dimension | Multi-Class Classification (PoC 1.1) | Multi-Label Classification (PoC 1.2) |
| :--- | :--- | :--- |
| **Output Categories** | Mutually exclusive (Exactly 1 label per input) | Non-exclusive (0, 1, or multiple labels per input) |
| **Activation Function** | **Softmax**: $\frac{e^{x_i}}{\sum_j e^{x_j}}$ (sums to 1.0 across classes) | **Sigmoid**: $\frac{1}{1 + e^{-x_i}}$ (independent 0 to 1 score per class) |
| **Loss Function** | `CrossEntropyLoss` | `BCEWithLogitsLoss` (Binary Cross Entropy) |
| **Target Vector Format** | Single integer index `label = 1` | Multi-hot float tensor `labels = [1.0, 0.0, 1.0, 0.0]` |
| **Metrics** | Accuracy, Single-class F1 | Micro-F1, Macro-F1, Hamming Loss |

---

## 🏃 Running the PoC

### 1. Training the Multi-Label Classifier
```powershell
python multilabel-classification/train_multilabel.py --epochs 2 --batch-size 8
```

### 2. Running Multi-Label Inference
```powershell
python multilabel-classification/predict_multilabel.py --text "My invoice was double charged and I need tech support right now!" --threshold 0.4
```

---

## 📊 Metrics Explained
* **Micro F1**: Aggregates total true positives, false positives, and false negatives globally across all instances and classes. Best for evaluating overall system precision/recall.
* **Macro F1**: Calculates F1 per label independently and averages them. Gives equal weight to rare and frequent classes.
* **Hamming Loss**: The fraction of incorrect label predictions (where a label is predicted when it shouldn't be, or missed when it should be). Lower is better.
