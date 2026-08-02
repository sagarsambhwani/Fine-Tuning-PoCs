# PoC 1.3: Tokenization & Custom Vocabulary Extension

This Proof of Concept (PoC) demonstrates how subword tokenization works in transformer models, how to extend a pre-trained tokenizer with **domain-specific custom tokens**, how to safely **resize model token embeddings**, and how to **initialize and fine-tune** a model with custom vocabulary.

> 📖 **Comprehensive Report**: For an in-depth post-mortem, step-by-step issue breakdown, training iterations table, and full raw terminal execution logs, see [POC_1_3_COMPREHENSIVE_REPORT.md](file:///e:/Downloads/Fine-tune/tokenization-custom-vocab/POC_1_3_COMPREHENSIVE_REPORT.md).

---

## 🔬 Core Concepts

### 1. Subword Tokenization (WordPiece / BPE)
Pre-trained models like `DistilBERT` use subword tokenization (WordPiece). When an unknown domain term or special tag (e.g., `[LOG_CRITICAL]` or `[ENV_PROD]`) is passed to a tokenizer:
- Without custom tokens: The tokenizer breaks it down into fragmented subwords (`['[', 'log', '##_c', '##rit', '##ical', ']']`) or replaces it with `[UNK]`.
- With custom tokens: The tokenizer treats `[LOG_CRITICAL]` as a **single, discrete token** (`['[LOG_CRITICAL]']`), retaining explicit semantic boundary and efficiency.

### 2. Extending Tokenizer Vocabulary
We can add new tokens using two methods:
* **`tokenizer.add_tokens(new_tokens)`**: Adds standard words, domain jargon, or custom symbols.
* **`tokenizer.add_special_tokens(special_tokens_dict)`**: Adds special control tokens (e.g., `<pad>`, `[MASK]`, `<bos>`, `<eos>`).

### 3. Resizing Model Token Embeddings
When new tokens are added, the tokenizer's vocabulary size grows ($V_{old} \rightarrow V_{new}$). However, the model's input embedding matrix shape is $(V_{old}, D)$.
We **must** synchronize the model's embedding matrix with the tokenizer:
```python
model.resize_token_embeddings(len(tokenizer))
```
This expands the embedding tensor from $(V_{old}, D)$ to $(V_{new}, D)$ and resizes output head projection layers if tied.

### 4. Embedding Initialization Strategies for New Tokens
By default, PyTorch initializes newly added rows in the embedding matrix with random numbers (often uniform or normal distribution). This can lead to high initial loss spikes or training instability.
* **Mean Initialization Strategy**: Initialize the new token embeddings to the average vector of all existing pretrained token embeddings.
```python
with torch.no_grad():
    embeddings = model.get_input_embeddings()
    mean_embedding = embeddings.weight[:old_vocab_len].mean(dim=0)
    for i in range(old_vocab_len, len(tokenizer)):
        embeddings.weight[i] = mean_embedding
```

---

## 📂 Project Structure

```
tokenization-custom-vocab/
├── README.md               # Detailed concepts & guide
├── demo_tokenization.py    # Exploratory script showing tokenization before/after custom tokens
├── train_custom_vocab.py   # Training script extending vocabulary & fine-tuning DistilBERT
└── predict_custom_vocab.py # Inference CLI predicting with custom tokens
```

---

## 🏃 Quick Start & How to Run

### 1. Run Tokenization & Vocabulary Extension Demo
Inspect token IDs, subword splitting, and embedding shape changes:
```powershell
.venv\Scripts\python.exe tokenization-custom-vocab/demo_tokenization.py
```

### 2. Fine-Tune DistilBERT with Custom Vocabulary
Train a classifier using extended domain tokens (`[LOG_CRITICAL]`, `[ENV_PROD]`, `[SYSTEM_ALERT]`, etc.):
```powershell
.venv\Scripts\python.exe tokenization-custom-vocab/train_custom_vocab.py --epochs 3 --batch-size 8
```

### 3. Test Inference with Custom Vocabulary
Run inference using texts containing domain tags:
```powershell
.venv\Scripts\python.exe tokenization-custom-vocab/predict_custom_vocab.py --text "[LOG_CRITICAL] [ENV_PROD] Database pool connection timeout after 30000ms!"
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Mismatch between Tokenizer & Model Size**: Always call `model.resize_token_embeddings(len(tokenizer))` after adding tokens. Failure to do so will result in `CUDA out of bounds index` or `IndexError` during embedding lookup.
2. **Tokenizer Preservation**: When saving the model after fine-tuning, **always save both model and tokenizer** together (`model.save_pretrained(path)` & `tokenizer.save_pretrained(path)`).
3. **Avoid Over-Adding Tokens**: Adding thousands of unused tokens unnecessarily increases model size and memory footprint without performance benefit.
4. **Local Directory vs. Hub Model ID Collision**: When loading models from Hugging Face Hub (e.g. `distilbert/distilbert-base-uncased`), use the full repo namespace (`distilbert/distilbert-base-uncased`) instead of just `distilbert-base-uncased` if a local directory in the workspace matches the model name. Otherwise, HF `from_pretrained` tries to load from the local folder.
