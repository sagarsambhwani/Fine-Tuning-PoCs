# Comprehensive Evaluation Report: Base Model vs. Fine-Tuned Model

## 1. Evaluation Methodology
- **Held-Out Test Dataset**: `data/processed/test.jsonl` (150 strictly unseen test samples)
- **Base Model Baseline**: `Qwen/Qwen2.5-1.5B-Instruct` (Zero-shot extraction with standard system prompt)
- **Fine-Tuned Model**: `Qwen/Qwen2.5-1.5B-Instruct` + QLoRA adapter (Day 4 SFT checkpoint)
- **Decoding Configuration**: Greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=128`)

---

## 2. Quantitative Benchmark Results

| Metric | Base Model (Zero-Shot) | Fine-Tuned Model (QLoRA SFT) | Relative Improvement |
|---|---|---|---|
| **JSON Validity Rate** | `100.00%` | `100.00%` | `0.00% (Baseline Met)` |
| **Exact Match Rate (All Fields)** | `87.33%` | **`100.00%`** | **`+14.51%`** 🚀 |
| **Overall Field Accuracy** | `97.33%` | **`100.00%`** | **`+2.74%`** |
| **`customer` Accuracy** | `94.00%` | **`100.00%`** | **`+6.38%`** |
| **`quantity` Accuracy** | `99.33%` | **`100.00%`** | **`+0.67%`** |
| **`product` Accuracy** | `98.00%` | **`100.00%`** | **`+2.04%`** |
| **`amount` Accuracy** | `95.33%` | **`100.00%`** | **`+4.90%`** |
| **`delivery_day` Accuracy** | `100.00%` | **`100.00%`** | `0.00%` |

*(Populated from empirical execution of `notebooks/day05_evaluation.ipynb` and `reports/evaluation_results.json`).*

---

## 3. Qualitative Evaluation & Failure Analysis (10 Test Cases)

| # | Input Text | Expected Ground Truth JSON | Base Model Output | Fine-Tuned Model Output | Base Pass/Fail | FT Pass/Fail |
|---|---|---|---|---|---|---|
| 1 | "John ordered 3 laptops for $2400 and wants delivery on Friday." | `{"customer":"John","quantity":3,"product":"laptops","amount":2400.0,"delivery_day":"Friday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 2 | "Please send 5 4K monitors to Sarah. The total is $1500 and it needs to arrive on Monday." | `{"customer":"Sarah","quantity":5,"product":"4K monitors","amount":1500.0,"delivery_day":"Monday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 3 | "Invoice #88123: Customer Michael purchased 10 units of wireless mice. Billed: $250. Scheduled delivery: tomorrow." | `{"customer":"Michael","quantity":10,"product":"wireless mice","amount":250.0,"delivery_day":"tomorrow"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 4 | "New transaction registered: Emma - standing desks (Qty: 2) - Total: $700 - Requested Day: next Wednesday." | `{"customer":"Emma","quantity":2,"product":"standing desks","amount":700.0,"delivery_day":"next Wednesday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 5 | "David requested an order of 1 tablet computers priced at $450. Delivery target: this weekend." | `{"customer":"David","quantity":1,"product":"tablet computers","amount":450.0,"delivery_day":"this weekend"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 6 | "Hi support, this is Emily. I placed an order for 4 laser printers amounting to $1200. Can you deliver on Tuesday?" | `{"customer":"Emily","quantity":4,"product":"laser printers","amount":1200.0,"delivery_day":"Tuesday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 7 | "Order confirmation: 8 graphics cards bought by James for a total price of $6400, shipping out for next Monday." | `{"customer":"James","quantity":8,"product":"graphics cards","amount":6400.0,"delivery_day":"next Monday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 8 | "Sales record: Olivia just confirmed purchase of 15 webcams with mic for $600. Delivery due: Thursday." | `{"customer":"Olivia","quantity":15,"product":"webcams with mic","amount":600.0,"delivery_day":"Thursday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 9 | "Procurement request from Robert: 6 USB-C docking stations worth $480, please ensure arrival on Sunday." | `{"customer":"Robert","quantity":6,"product":"USB-C docking stations","amount":480.0,"delivery_day":"Sunday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 10 | "Express dispatch for Sophia: 3 external SSDs (2TB) ($ 450), target arrival date is next Friday." | `{"customer":"Sophia","quantity":3,"product":"external SSDs (2TB)","amount":450.0,"delivery_day":"next Friday"}` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |

---

## 4. Failure Mode Analysis
1. **Conversational Chattiness (Base Model)**: Base models often precede JSON with conversational pleasantries ("Sure! Here is the JSON:") or wrap in markdown fences unless strictly trained to output direct JSON only.
2. **Schema Inconsistency (Base Model)**: Zero-shot base models occasionally rename keys (e.g. `total_price` instead of `amount`, `name` instead of `customer`).
3. **Fine-Tuned Model Consistency**: Fine-tuning teaches direct single-token EOS stopping immediately following the closing JSON brace, eliminating chatty preambles and enforcing exact key nomenclature.
