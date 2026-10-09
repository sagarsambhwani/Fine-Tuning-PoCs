# 🧪 Track 1 Adversarial Stress-Test & Red-Teaming Report
## Discovering the True Failure Boundaries of Fine-Tuned QLoRA (`Qwen-1.5B-Legal-Graph-Adapter`)

**Author & Experimenter**: Sagar Sambhwani  
**Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` + `models/qwen-1.5b-legal-graph-adapter` (Track 1)  
**Execution Runtime**: Google Colab Tesla T4 GPU (Notebook: `03_adversarial_stress_test.ipynb`)  
**Evaluation Scope**: 40 Adversarial Challenge Cases across 4 Stress Dimensions  

---

## 1. Executive Summary: The Real-World Reality Check

In our initial clean benchmark evaluation (Notebook 02), the fine-tuned model achieved a **100.00% Macro F1** on Task A (Triplet Extraction). However, that test evaluated the model in an idealized, predictable setting.

When red-teamed against **40 adversarial edge cases**, the model's true operational boundaries emerged:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Clean Laboratory Benchmark (Notebook 02):    100.00% Macro F1         │
│ Adversarial Stress-Test Scorecard:            61.67% Macro F1          │
│ Real-World Robustness Drop (Δ):              -38.33%                   │
└────────────────────────────────────────────────────────────────────────┘
```

| Challenge Suite | Stress Type | Macro F1 | Failure Rate / Metric | Real-World Operational Diagnosis |
| :--- | :--- | :---: | :---: | :--- |
| **Suite 1: Inverted / Passive Syntax** | Passive voice, fronted clauses | **$73.33\%$** | **$80.0\%$ Directional Inversion** (8/10 cases) | 🚨 **Severe Asymmetry Blindness**: Model relies on token order rather than grammatical roles. |
| **Suite 2: Heavy Distractors & Noise** | Attorneys, dates, \$ sums, docket #s | **$93.33\%$** | **$0.0\%$ Contamination** (0/10 cases) | 🎯 **Noise Resilient**: Filtered out extraneous lawyers and dollar values cleanly. |
| **Suite 3: Dismissed / Negative Cases** | Dismissed with prejudice; zero remedy | **$80.00\%$** | **$100.0\%$ Remedy Hallucination** (10/10 cases) | 🚨 **Hallucination Trap**: Blindly invents remedies because training always included one. |
| **Suite 4: Commercial Partnerships** | Joint ventures, alliances (No lawsuit) | **$0.00\%$** | **$100.0\%$ False Positive Rate** (10/10 cases) | 🚨 **Domain Trigger Bias**: Assumes any two corporate names must be suing each other. |

---

## 2. Deep Dive: The 4 Core Failure Modes Uncovered

### Failure Mode 1: Syntactic Inversion (80% Directional Error in Passive Voice)
* **What Happened**: When presented with passive voice:
  > *"Dragged into arbitration by ApexHoldings, DeltaSystemsInc was accused of PatentInfringement..."*
* **The Error**: In **8 out of 10 cases**, the model output:
  ```json
  {"subject": "DeltaSystemsInc", "predicate": "FILES_CLAIM_AGAINST", "object": "ApexHoldings"}
  ```
* **Root Cause**: The model learned a positional heuristic from the training data (*"First entity seen is the Plaintiff, second entity seen is the Defendant"*). When passive voice puts the Defendant first, the model blindly assigns them as the Plaintiff.

---

### Failure Mode 2: Unconditional Remedy Hallucination (100% Failure Rate on Negative Cases)
* **What Happened**: When a lawsuit was explicitly dismissed with prejudice with zero relief granted:
  > *"...the court granted the motion to dismiss with prejudice, finding zero liability and denying all remedies."*
* **The Error**: In **10 out of 10 cases**, the model invented a `LIABLE_FOR` triple with an un-awarded remedy (e.g. `PermanentInjunction`).
* **Root Cause**: The training dataset contained **zero negative examples**. The model's decoder learned the conditional prior: $P(\text{remedy} \mid \text{case}) \approx 1.0$. It never learned when to output an empty remedy.

---

### Failure Mode 3: False Positive Trigger Bias (100% False Lawsuits on Partnerships)
* **What Happened**: When two companies formed an amicable research alliance:
  > *"ApexHoldings and DeltaSystemsInc announced an expansive multi-year joint venture..."*
* **The Error**: In **10 out of 10 cases**, the model output:
  ```json
  {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"}
  ```
* **Root Cause**: **Task Conditioning Overfit**. The adapter was fine-tuned exclusively on litigation data, giving it a severe confirmation bias: *any pair of corporate entities is assumed to be in litigation*.

---

### Failure Mode 4: Noise Filtering Resilience (The One Bright Spot)
* **What Worked**: In Suite 2, when realistic noise was injected (law firms like *Gibson Dunn*, attorneys like *Sarah Jenkins*, dollar amounts like *\$450 million*, dates, and docket numbers):
  * **0 out of 10 cases** suffered distractor contamination.
  * The model correctly ignored the lawyers and dates, only extracting the true corporate parties and court entities.

---

## 3. Engineering Prescriptions: How to Build a Truly Robust Model

To fix these failure modes before deploying or moving forward:

1. **Passive Voice Augmentation (Fixes Suite 1)**:
   - Augment the training data with syntactically varied templates (passive voice, fronted prepositional clauses, inverted syntax).
2. **Negative Supervision (Fixes Suite 3)**:
   - Include 15–20% dismissed cases where the ground truth assistant target explicitly contains no remedy.
3. **Hard Negative Non-Litigation Data (Fixes Suite 4)**:
   - Include joint ventures and non-litigation articles where the target assistant response is `[]` (empty list). This teaches the model when *not* to extract.

---

## 4. Key Takeaway

Your skepticism was completely justified: **the 100% score was a laboratory illusion**. 

By running this stress test, we identified the exact mathematical limits of pure textual fine-tuning. We now know that our model has **strong entity recognition and noise rejection**, but **severe positional bias and zero negative suppression**.
