"""
Zero-Training Benchmark Dataset Generator for Graph & Ontology Tasks
Generates standardized benchmark evaluations covering:
- Task A: Triplet Extraction from Text
- Task B: Multi-Hop Deductive Reasoning
- Task C: Link Prediction / Missing Edge Detection
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Sample ontology entities
DRUGS = [
    ("Metformin", "Biguanide antihyperglycemic"),
    ("Lisinopril", "ACE inhibitor"),
    ("Atorvastatin", "HMG-CoA reductase inhibitor"),
    ("Pembrolizumab", "Monoclonal antibody PD-1 inhibitor"),
    ("Imatinib", "Tyrosine kinase inhibitor"),
    ("Omeprazole", "Proton pump inhibitor"),
    ("Losartan", "Angiotensin receptor blocker"),
    ("Aspirin", "NSAID antiplatelet agent")
]

DISEASES = [
    ("Type2Diabetes", "Chronic metabolic disorder"),
    ("Hypertension", "Cardiovascular condition"),
    ("Hypercholesterolemia", "Lipid disorder"),
    ("Melanoma", "Malignant skin cancer"),
    ("ChronicMyeloidLeukemia", "Hematologic cancer"),
    ("GastroesophagealReflux", "Acid reflux disease"),
    ("MyocardialInfarction", "Heart attack"),
    ("Atherosclerosis", "Arterial plaque buildup")
]

PROTEINS = [
    ("AMPK_Enzyme", "Cellular energy sensor"),
    ("ACE_Receptor", "Angiotensin converting enzyme"),
    ("HMGCR_Enzyme", "Rate-limiting enzyme for cholesterol"),
    ("PD1_Protein", "Immune checkpoint surface receptor"),
    ("BCR_ABL_Kinase", "Oncogenic fusion protein"),
    ("ProtonPump_K_ATPase", "Gastric acid pump"),
    ("AT1_Receptor", "Angiotensin II type 1 receptor"),
    ("COX1_Enzyme", "Cyclooxygenase enzyme")
]

GENES = [
    ("PRKAA1", "5'-AMP-activated protein kinase subunit alpha-1"),
    ("ACE", "Angiotensin I converting enzyme gene"),
    ("HMGCR", "3-hydroxy-3-methylglutaryl-CoA reductase gene"),
    ("PDCD1", "Programmed cell death 1 gene"),
    ("BCR_ABL1", "Fusion oncogene"),
    ("ATP4A", "ATPase H+/K+ transporting alpha subunit"),
    ("AGTR1", "Angiotensin II receptor type 1 gene"),
    ("PTGS1", "Prostaglandin-endoperoxide synthase 1 gene")
]

SIDE_EFFECTS = [
    ("LacticAcidosis", "Metabolic complication"),
    ("DryCough", "Persistent non-productive cough"),
    ("Myopathy", "Muscle tissue disorder"),
    ("ImmuneColitis", "Inflammatory bowel complication"),
    ("FluidRetention", "Peripheral edema"),
    ("Hypomagnesemia", "Low magnesium levels"),
    ("Hyperkalemia", "High potassium concentration"),
    ("GastricUlcer", "Gastrointestinal mucosal damage")
]

def generate_benchmark_suite(num_samples: int = 50, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generates standardized benchmark evaluation suites for Tasks A, B, and C."""
    rng = random.Random(seed)
    
    task_a_samples = []
    task_b_samples = []
    task_c_samples = []

    for idx in range(num_samples):
        # Pick consistent chain: Drug -> Protein -> Gene -> Disease
        drug, drug_desc = rng.choice(DRUGS)
        disease, dis_desc = rng.choice(DISEASES)
        protein, prot_desc = rng.choice(PROTEINS)
        gene, gene_desc = rng.choice(GENES)
        side_effect, se_desc = rng.choice(SIDE_EFFECTS)

        # ----------------------------------------------------
        # TASK A: Information Extraction to Ontology Triples
        # ----------------------------------------------------
        text_passage = (
            f"Clinical trials confirm that {drug} effectively treats {disease}. "
            f"However, patients occasionally report {side_effect} as an adverse reaction. "
            f"Biochemical assays demonstrate that {drug} primarily targets the {protein}."
        )
        expected_triples = [
            {"subject": drug, "predicate": "TREATS", "object": disease},
            {"subject": drug, "predicate": "CAUSES_SIDE_EFFECT", "object": side_effect},
            {"subject": drug, "predicate": "TARGETS", "object": protein}
        ]
        task_a_samples.append({
            "id": f"task_a_{idx:03d}",
            "task_type": "triplet_extraction",
            "input_text": text_passage,
            "target_triples": expected_triples
        })

        # ----------------------------------------------------
        # TASK B: Multi-Hop Deductive Reasoning
        # Path: Drug -(TARGETS)-> Protein -(ENCODED_BY)-> Gene -(ASSOCIATED_WITH)-> Disease
        # ----------------------------------------------------
        hop_context = (
            f"Fact 1: {drug} TARGETS {protein}.\n"
            f"Fact 2: {protein} is ENCODED_BY gene {gene}.\n"
            f"Fact 3: Gene {gene} is ASSOCIATED_WITH {disease}."
        )
        hop_query_2 = f"Based on the pathway facts, which Gene is modulated when {drug} targets its primary protein?"
        hop_answer_2 = gene

        hop_query_3 = f"Following the 3-hop biological chain from {drug}, which Disease is causally linked to this target pathway?"
        hop_answer_3 = disease

        task_b_samples.append({
            "id": f"task_b_{idx:03d}",
            "task_type": "multi_hop_deduction",
            "subgraph_context": hop_context,
            "query_2hop": hop_query_2,
            "target_2hop": hop_answer_2,
            "query_3hop": hop_query_3,
            "target_3hop": hop_answer_3
        })

        # ----------------------------------------------------
        # TASK C: Link Prediction / Missing Edge Detection
        # Given Head & Tail, predict valid ontological relation
        # ----------------------------------------------------
        triplet_candidates = [
            (drug, disease, "TREATS"),
            (drug, side_effect, "CAUSES_SIDE_EFFECT"),
            (drug, protein, "TARGETS"),
            (protein, gene, "ENCODED_BY"),
            (gene, disease, "ASSOCIATED_WITH")
        ]
        head, tail, true_relation = rng.choice(triplet_candidates)
        
        task_c_samples.append({
            "id": f"task_c_{idx:03d}",
            "task_type": "link_prediction",
            "head": head,
            "tail": tail,
            "prompt": f"Given entity Head '{head}' and Tail '{tail}', identify the valid ontological relation from: [TREATS, CAUSES_SIDE_EFFECT, TARGETS, ENCODED_BY, ASSOCIATED_WITH].",
            "target_relation": true_relation
        })

    return {
        "task_a_triplets": task_a_samples,
        "task_b_multihop": task_b_samples,
        "task_c_link_pred": task_c_samples
    }

def save_benchmark_suites(out_dir: str = "llm-graph-ontology/data/benchmark"):
    """Saves benchmark JSON files."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    suite = generate_benchmark_suite(num_samples=50, seed=42)
    for name, data in suite.items():
        file_path = path / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} test items to {file_path}")

if __name__ == "__main__":
    save_benchmark_suites()
