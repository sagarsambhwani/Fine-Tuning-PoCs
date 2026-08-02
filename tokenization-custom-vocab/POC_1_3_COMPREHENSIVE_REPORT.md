# PoC 1.3: Comprehensive Execution, Debugging, and Learning Report

This document provides an exhaustive post-mortem, technical explanation, training iteration breakdown, issue log, and full execution output for **PoC 1.3: Tokenization & Custom Vocabulary Extension**.

---

## 🎯 1. Overview & Objectives

In PoC 1.3, we explored how subword tokenization functions under the hood when processing specialized domain text (such as system log tags like `[LOG_CRITICAL]` or `[ENV_PROD]`), and how to extend a pre-trained transformer model (`DistilBERT`) with custom vocabulary.

### Primary Objectives:
1. **Analyze Subword Tokenization**: Observe how standard WordPiece tokenizers fragment unknown domain terms into multiple subword tokens.
2. **Extend Tokenizer Vocabulary**: Safely add custom tokens via `tokenizer.add_tokens()`.
3. **Synchronize Model Embeddings**: Expand the model's input token embedding layer using `model.resize_token_embeddings()` to match the expanded tokenizer vocabulary size.
4. **Apply Safe Weight Initialization**: Initialize newly created token embedding vectors using the mean of existing pretrained embeddings to prevent training instability.
5. **Fine-Tune & Validate**: Train the model on a domain-specific log classification dataset, save model & tokenizer together, and verify inference.

---

## 🛠️ 2. Issues Suffered & Technical Solutions

During implementation and execution, we encountered **3 distinct errors/roadblocks**. Here is the detailed post-mortem for each:

### Issue 1: Local Workspace Directory Name Collision with HF Hub
* **Error Encountered**:
  ```text
  ValueError: Couldn't instantiate the backend tokenizer from one of:
  (1) a `tokenizers` library serialization file,
  (2) a slow tokenizer instance to convert or
  (3) an equivalent slow tokenizer class to instantiate and convert.
  ```
* **Root Cause**: The workspace contains a local folder named `e:\Downloads\Fine-tune\distilbert-base-uncased`. When calling `AutoTokenizer.from_pretrained("distilbert-base-uncased")`, Hugging Face `transformers` checks if `os.path.isdir("distilbert-base-uncased")` is `True` relative to the current working directory. Finding the local folder (which only contained workspace Python scripts and documentation), `transformers` attempted to parse it as a model directory rather than fetching `distilbert-base-uncased` from Hugging Face Hub.
* **Resolution**: Updated model ID calls to use the explicit repository namespace:
  `"distilbert/distilbert-base-uncased"`. This bypassed the local folder path check.

---

### Issue 2: Deprecated `evaluation_strategy` Parameter in `TrainingArguments`
* **Error Encountered**:
  ```text
  TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
  ```
* **Root Cause**: Hugging Face `transformers` (v4.41+) deprecated `evaluation_strategy` in favor of `eval_strategy`.
* **Resolution**: Replaced `evaluation_strategy="epoch"` with `eval_strategy="epoch"` in `train_custom_vocab.py`.

---

### Issue 3: Deprecated `tokenizer` Parameter in `Trainer` Initialization
* **Error Encountered**:
  ```text
  TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
  ```
* **Root Cause**: In recent `transformers` releases (v4.46+), the `tokenizer` keyword argument in `Trainer.__init__` was deprecated and renamed to `processing_class`.
* **Resolution**: Replaced `tokenizer=tokenizer` with `processing_class=tokenizer` in `train_custom_vocab.py`.

---

## 📊 3. Training Iterations & Performance Breakdown

### Dataset & Configuration Setup:
- **Base Model**: `distilbert/distilbert-base-uncased`
- **Original Vocab Size**: `30522`
- **Custom Tokens Added**: 6 tokens (`[LOG_CRITICAL]`, `[ENV_PROD]`, `[ENV_STAGING]`, `[SYSTEM_ALERT]`, `[INCIDENT_SECURITY]`, `[INCIDENT_PERFORMANCE]`)
- **New Vocab Size**: `30528`
- **Dataset Size**: 96 training samples, 24 validation samples (80/20 split)
- **Batch Size**: 8
- **Total Training Steps**: **24 steps** (12 iterations/steps per epoch $\times$ 2 epochs)

### Training Iteration Log:

| Epoch | Step | Training Loss | Grad Norm | Learning Rate | Validation Loss | Validation Acc | Validation F1 | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.42** | 5 | `0.6246` | `2.543` | `4.167e-05` | - | - | - | Initial loss stabilization |
| **0.83** | 10 | `0.4394` | `3.159` | `3.125e-05` | - | - | - | Loss steadily decreasing |
| **1.00** | 12 | - | - | - | **`0.2329`** | **`1.0000` (100%)** | **`1.0000` (100%)** | **Epoch 1 Evaluation** |
| **1.25** | 15 | `0.3231` | `2.029` | `2.083e-05` | - | - | - | Fine-tuning custom embeddings |
| **1.67** | 20 | `0.2447` | `2.581` | `1.042e-05` | - | - | - | Convergence |
| **2.00** | 24 | - | - | - | **`0.1584`** | **`1.0000` (100%)** | **`1.0000` (100%)** | **Epoch 2 Evaluation (Final)** |

- **Total Training Runtime**: 31.69 seconds (0.757 steps/sec)
- **Final Metrics**: **100% Accuracy**, **100% F1 Score** on evaluation set.

---

## 🧠 4. Core Conceptual Takeaways & Learnings

1. **Subword Fragmentation Risk**: Standard pre-trained tokenizers will split domain jargon into multiple arbitrary pieces (e.g. `[LOG_CRITICAL]` $\rightarrow$ `['[', 'log', '_', 'critical', ']']`). This inflates sequence lengths and dilutes semantic representations.
2. **Mandatory Embedding Resizing**: Adding tokens to `AutoTokenizer` increases its internal vocabulary count (`len(tokenizer)`). The model's embedding matrix MUST be resized via `model.resize_token_embeddings(len(tokenizer))` before training. Skipping this step causes `CUDA index out of range` runtime crashes.
3. **Weight Initialization Strategy**: PyTorch initializes new embedding rows randomly by default. By explicitly calculating the mean vector of existing pre-trained embeddings (`embeddings.weight[:old_vocab].mean(dim=0)`) and assigning it to the new token IDs, we prevent severe loss spikes at the beginning of fine-tuning.
4. **Co-Dependency in Persistence**: A fine-tuned model with custom embeddings is useless without its corresponding extended tokenizer. Both must always be saved together using `model.save_pretrained()` and `tokenizer.save_pretrained()`.

---

## 📜 5. Full Execution Outputs

Below are the verbatim execution logs from all three PoC 1.3 scripts:

### Execution Result 1: `demo_tokenization.py`
```text
============================================================
PoC 1.3: Tokenization & Custom Vocabulary Extension Demo
============================================================

1. Loading base tokenizer and model: 'distilbert/distilbert-base-uncased'...
   Original Tokenizer Vocab Size: 30522
   Original Model Embedding Shape: torch.Size([30522, 768])

   Target Text: "[LOG_CRITICAL] [ENV_PROD] Connection pool exhausted on [API_GATEWAY]"

2. Tokenization BEFORE adding custom tokens:
   Tokens: ['[', 'log', '_', 'critical', ']', '[', 'en', '##v', '_', 'pro', '##d', ']', 'connection', 'pool', 'exhausted', 'on', '[', 'api', '_', 'gateway', ']']
   Token IDs: [101, 1031, 8833, 1035, 4187, 1033, 1031, 4372, 2615, 1035, 4013, 2094, 1033, 4434, 4770, 9069, 2006, 1031, 17928, 1035, 11909, 1033, 102]
   Notice how '[LOG_CRITICAL]' is fragmented into multiple subwords!

3. Adding 4 custom tokens: ['[LOG_CRITICAL]', '[ENV_PROD]', '[SYSTEM_ALERT]', '[API_GATEWAY]']
   Added 4 new tokens.
   New Tokenizer Vocab Size: 30526

4. Tokenization AFTER adding custom tokens:
   Tokens: ['[log_critical]', '[env_prod]', 'connection', 'pool', 'exhausted', 'on', '[api_gateway]']
   Token IDs: [101, 30522, 30523, 4434, 4770, 9069, 2006, 30525, 102]
   Each custom tag is now treated as a single, discrete token!

5. Resizing Model Embedding Matrix...
   Shape before resize: torch.Size([30522, 768])
   Shape after resize:  torch.Size([30526, 768])

6. Initializing New Token Embedding Weights...
   Successfully initialized new token embeddings (IDs 30522 to 30525)
   Mean vector sample (first 5 dims): [-0.034539, -0.048551, -0.041291, -0.048209, -0.026583]
   New token ID 30522 ('[LOG_CRITICAL]') embedding sample (first 5 dims): [-0.034539, -0.048551, -0.041291, -0.048209, -0.026583]

============================================================
Demo complete! All vocabulary and embedding operations executed smoothly.
============================================================
```

---

### Execution Result 2: `train_custom_vocab.py`
```text
============================================================
PoC 1.3: Custom Vocabulary Fine-Tuning Pipeline
============================================================

1. Loading base tokenizer: distilbert/distilbert-base-uncased
   Original vocabulary size: 30522
   Adding custom tokens: ['[LOG_CRITICAL]', '[ENV_PROD]', '[ENV_STAGING]', '[SYSTEM_ALERT]', '[INCIDENT_SECURITY]', '[INCIDENT_PERFORMANCE]']
   Added 6 tokens. New vocabulary size: 30528

2. Loading base model: distilbert/distilbert-base-uncased
   Resizing model token embeddings...
   Resized embedding shape: torch.Size([30528, 768])
   Initializing new token embeddings with mean vector...

3. Generating domain-specific dataset with custom tokens...
   Train samples: 96, Validation samples: 24

4. Starting Training...
   Epoch 0.42 | Step 5  | Loss: 0.6246
   Epoch 0.83 | Step 10 | Loss: 0.4394
   Epoch 1.00 | Eval Loss: 0.2329 | Eval Accuracy: 1.0000 | Eval F1: 1.0000
   Epoch 1.25 | Step 15 | Loss: 0.3231
   Epoch 1.67 | Step 20 | Loss: 0.2447
   Epoch 2.00 | Eval Loss: 0.1584 | Eval Accuracy: 1.0000 | Eval F1: 1.0000

5. Evaluating Final Model...
   Validation Accuracy: 1.0000
   Validation F1 Score: 1.0000

6. Saving fine-tuned model and custom tokenizer to:
   E:\Downloads\Fine-tune\tokenization-custom-vocab\models\custom_distilbert

============================================================
Training complete! Model and custom tokenizer saved successfully.
============================================================
```

---

### Execution Result 3: `predict_custom_vocab.py`
```text
============================================================
PoC 1.3: Inference with Custom Vocabulary
============================================================

1. Loading model & custom tokenizer from: 'tokenization-custom-vocab\models\custom_distilbert'...

2. Input Text: "[LOG_CRITICAL] [ENV_PROD] Connection pool exhausted on primary database cluster"
   Tokenized Output: ['[log_critical]', '[env_prod]', 'connection', 'pool', 'exhausted', 'on', 'primary', 'database', 'cluster']
   Token IDs:        [101, 30522, 30523, 4434, 4770, 9069, 2006, 3078, 7809, 9324, 102]

3. Classification Results:
   Predicted Class: CRITICAL_INCIDENT
   Confidence:      92.26%

   Probabilities Breakdown:
     - normal_operation    :   7.74%
     - critical_incident   :  92.26%

============================================================
```
