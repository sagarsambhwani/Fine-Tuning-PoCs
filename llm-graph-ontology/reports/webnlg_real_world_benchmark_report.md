# 🌍 Real-World Public Benchmark Report: WebNLG (Human-Annotated)
## Empirical Evaluation of Base vs. Fine-Tuned LLMs on Out-of-Domain Human Text

**Author & Experimenter**: Sagar Sambhwani  
**Dataset**: WebNLG v1 Official Human-Annotated Challenge Benchmark ($N=100$ Test Samples)  
**Execution Runtime**: Google Colab Tesla T4 GPU (Notebook: `04_webnlg_real_world_benchmark.ipynb`)  
**Models Evaluated**: 
1. `Qwen/Qwen2.5-1.5B-Instruct` (Untuned Base Model)
2. `Qwen-1.5B-Legal-Graph-Adapter` (Domain-Adapted QLoRA Model)

---

## 1. Executive Master 3-Way Benchmark Scorecard

This experiment brings together our entire journey—from laboratory synthetic data to adversarial stress-testing, and finally to **gold-standard human-written data**:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER BENCHMARK PROGRESSION                       │
├─────────────────────────────────────────────────┬───────────┬────────────────────┤
│ Evaluation Setting / Benchmark Dataset          │  Macro F1 │ Operational Context│
├─────────────────────────────────────────────────┼───────────┼────────────────────┤
│ 1. Clean In-Domain Synthetic (Notebook 02)      │  100.00%  │ Laboratory Sandbox │
│ 2. Adversarial Stress-Test (Notebook 03)        │   61.67%  │ Failure Boundaries │
│ 3. WebNLG Real Human Text (Base Zero-Shot)      │    8.00%  │ Real-World Human   │
│ 4. WebNLG Real Human Text (Legal Adapter)       │    2.00%  │ Out-of-Domain OOD  │
└─────────────────────────────────────────────────┴───────────┴────────────────────┘
```

---

## 2. Key Empirical Findings

### Finding 1: The Human Open-World Collapse (Base Model: 8.00% F1)
* **What Happened**: When given real, messy human sentences from Wikipedia (e.g., *"Abilene, Texas is served by the Abilene regional airport"*), the untuned base model achieved only **$8.00\%$ Macro F1**.
* **Why It Happened**:
  * In real-world Knowledge Graphs (DBpedia/Wikidata), relationships use **formal canonical ontology URIs/predicates** (e.g., `cityServed`, `runwayName`, `operatingOrganisation`, `elevationAboveTheSeaLevel`).
  * The untuned base model generated informal, conversational text snippets (e.g., `"is served by"`, `"located in"`, `"has runway of"`).
  * **Conclusion**: Base LLMs possess zero-shot conversational fluency, but **zero-shot ontology grounding on human text is virtually non-existent without domain schema pinning**.

---

### Finding 2: The Double-Edged Sword of Domain Fine-Tuning (Tuned Adapter: 2.00% F1)
* **What Happened**: The fine-tuned legal adapter dropped from **$100\%$ on legal cases** down to **$2.00\%$ on WebNLG**.
* **Why It Happened**:
  * **Domain Specialization / Narrow Tuning**: Our adapter was trained exclusively on the `CorporateLitigationOntology`. It learned to sharply suppress generic relational verbs in favor of legal predicates (`FILES_CLAIM_AGAINST`, `ADJUDICATED_BY`).
  * When fed sentences about airports, monuments, and universities, the model suffered from **domain mismatch**: it attempted to force-fit legal extraction patterns or failed to activate open-domain predicates.
  * **Conclusion**: Parameter-efficient fine-tuning (LoRA) successfully steers a model into a specialized domain, but **a narrow single-domain adapter cannot perform out-of-domain (OOD) open-world graph extraction**.

---

## 3. The Full Research Narrative: The 4 Stages of Truth

Our 4 notebooks tell a complete, scientifically rigorous story:

1. **Phase 1 (`01_legal_zero_training_benchmark.ipynb`)**:
   - Untuned base model failed on legal ontology ($25.07\%$ F1). Proved that base models do not follow closed-world schemas out of the box.
2. **Phase 2 — Laboratory Fine-Tuning (`02_track1_pure_llm_finetuning.ipynb`)**:
   - QLoRA SFT solved the closed-world schema on clean text ($100.00\%$ F1). Proved that LLMs can rapidly internalize strict ontology syntax.
3. **Phase 2 — Adversarial Red-Teaming (`03_adversarial_stress_test.ipynb`)**:
   - Stress-testing dropped F1 to $61.67\%$, exposing positional bias on passive voice and lack of negative suppression. Proved that clean synthetic tests create an illusion of perfection.
4. **Phase 2 — Public Human Benchmark (`04_webnlg_real_world_benchmark.ipynb`)**:
   - Real human text produced $8.00\%$ (base) and $2.00\%$ (adapter). Proved that without domain-specific schema definitions provided in the prompt or multi-domain training, neither base models nor specialized adapters can extract open-world knowledge graphs.

---

## 4. Strategic Next Step: Transition to Track 2 (GNN + LLM Hybrid)

We now have the complete, unassailable baseline for **Track 1 (Pure LLM Fine-Tuning)**.

We know exactly what pure text LLMs can and cannot do:
* ✅ **Can do**: Rapidly learn domain-specific ontology grammar and achieve $100\%$ on clean text.
* ⚠️ **Struggles with**: Passive voice and negative cases without explicit negative supervision.
* ❌ **Fails at**: Topological link prediction ($36\%$ ceiling) and out-of-domain graph extraction ($2\%-8\%$).

This provides the ultimate, peer-review-quality empirical justification for **Track 2: Hybrid Graph Neural Networks (GNN + LLM)**!
