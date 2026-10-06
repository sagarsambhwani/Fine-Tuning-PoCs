# Day 9: Reference-Free Alignment (ORPO vs. DPO) Benchmark Report

## 1. Executive Summary
This report benchmarks **ORPO (Odds Ratio Preference Optimization)** and **KTO (Kahneman-Tversky Optimization)** as lightweight, reference-free alternatives to DPO on `Qwen/Qwen2.5-1.5B-Instruct` on a Google Colab T4 GPU (16GB VRAM).

---

## 2. Theoretical Comparison: DPO vs. ORPO vs. KTO

| Dimension | DPO (Direct Preference Optimization) | ORPO (Odds Ratio Preference Optimization) | KTO (Kahneman-Tversky Optimization) |
|---|---|---|---|
| **Data Format Required** | Paired Triplets $(x, y_w, y_l)$ | Paired Triplets $(x, y_w, y_l)$ | **Unpaired Binary** $(x, y, \pm 1)$ |
| **Reference Model $\pi_{\text{ref}}$ Required?** | **YES** (Must keep frozen copy in memory) | **NO** (Reference-free) | **YES** (Reference model used for implicit KL) |
| **Loss Function** | $-\log \sigma \left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)$ | $\mathcal{L}_{\text{SFT}} - \lambda \log \sigma \left(\log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\right)$ | Utility function with Kahneman loss aversion |
| **Stages of Training** | 2 Stages (SFT $\to$ DPO) | **1 Stage** (Monolithic SFT + Alignment) | 2 Stages (SFT $\to$ KTO) |
| **Peak Training VRAM** | High (~1.6x–2.0x SFT) | **Low (~1.0x SFT)** | Moderate (~1.4x SFT) |
| **Hardware Suitability** | High-VRAM GPUs | **Ideal for Consumer GPUs (T4 16GB / RTX 4090)** | Consumer / Enterprise |

---

## 3. Empirical Results (Colab T4 16GB)

| Metric | DPO (Day 8 Verified) | ORPO (Day 9) |
|---|---|---|
| **Peak GPU VRAM during Training** | ~7.2 GB | `NOT RUN` (Est. ~4.1 GB) |
| **Training Steps per Second** | 1.84 steps/sec | `NOT RUN` (Est. ~2.75 steps/sec) |
| **Final Preference Accuracy** | 96.4% | `NOT RUN` |
| **Zero-Preamble JSON Extraction Rate** | 100.0% | `NOT RUN` |
| **Clean Schema Conformance** | 100.0% | `NOT RUN` |

*(Note: Values marked `NOT RUN` will be updated after executing `notebooks_v2/day09_orpo_kto.ipynb` in Colab).*

---

## 4. Key Architectural Takeaways
1. **Single-Stage Efficiency**: ORPO eliminates the operational burden of saving, validating, and reloading intermediate SFT checkpoints before running preference optimization.
2. **Zero Reference Model Overhead**: By expressing the penalty as an odds ratio over the policy model's own output probabilities, ORPO allows fitting larger batch sizes and longer sequence lengths in 16GB VRAM.
3. **KTO for Real-World Feedback**: While ORPO and DPO require synthetic counterfactual pairs, KTO enables fine-tuning directly on raw user telemetry (e.g. upvotes, accepted completions, or retry triggers).
