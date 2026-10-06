"""
Legal Domain Benchmark Dataset Generator for Graph & Ontology Tasks
Generates standardized benchmark evaluations covering:
- Task A: Triplet Extraction from Legal Case Text
- Task B: Multi-Hop Legal Precedent Reasoning
- Task C: Legal Link Prediction / Missing Edge Detection
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any

PARTIES_PLAINTIFF = [
    ("ApexHoldings", "Private equity investment firm"),
    ("BioGenixLabs", "Pharmaceutical patent holder"),
    ("OmniRetailCorp", "Multinational commercial retailer"),
    ("QuantumTechLLC", "Semiconductor design venture"),
    ("HorizonMediaGroup", "Digital copyright publisher"),
    ("VanguardLogistics", "Freight distribution consortium")
]

PARTIES_DEFENDANT = [
    ("DeltaSystemsInc", "Enterprise cloud software vendor"),
    ("NovaPharmaCorp", "Generic pharmaceutical manufacturer"),
    ("GlobalSupplyChainLtd", "Maritime logistics contractor"),
    ("TitanEnergyLLC", "Industrial infrastructure developer"),
    ("AegisCyberSec", "Network defense service provider"),
    ("NexusRoboticsCorp", "Industrial robotics developer")
]

COURTS = [
    ("DelawareChanceryCourt", "Court of equity with corporate jurisdiction"),
    ("FederalCircuitAppeals", "Appellate court with patent subject-matter jurisdiction"),
    ("SouthernDistrictNewYork", "Federal trial court for commercial disputes"),
    ("NorthernDistrictCalifornia", "Federal trial court for intellectual property"),
    ("SecondCircuitAppeals", "Federal appellate jurisdiction for commercial contracts"),
    ("SupremeCourtDelaware", "State appellate authority on corporate governance")
]

LEGAL_CLAIMS = [
    ("PatentInfringement", "Willful violation of utility patent claims"),
    ("BreachOfContract", "Material non-performance of supply agreement"),
    ("TradeSecretMisappropriation", "Unauthorized acquisition of proprietary source code"),
    ("FiduciaryDutyBreach", "Failure to exercise duty of loyalty in merger"),
    ("SecuritiesFraud", "Misleading material statements regarding quarterly revenue"),
    ("TortiousInterference", "Intentional disruption of existing contractual relationships")
]

PRECEDENTS = [
    ("eBay_v_MercExchange", "Supreme Court standard for permanent injunctive relief"),
    ("Alice_v_CLSBank", "Two-step patent eligibility framework under 35 U.S.C. § 101"),
    ("Revlon_v_MacAndrews", "Fiduciary duty standard in corporate acquisition transactions"),
    ("Basic_v_Levinson", "Fraud-on-the-market presumption of reliance in securities fraud"),
    ("Twombly_v_BellAtlantic", "Plausibility standard for surviving motions to dismiss"),
    ("Pennzoil_v_Texaco", "Landmark precedent defining tortious interference damages")
]

REMEDIES = [
    ("PermanentInjunction", "Court order restraining ongoing commercial infringement"),
    ("TrebleDamages", "Willful violation multiplier awarding triple monetary damages"),
    ("DisgorgementOfProfits", "Equitable remedy stripping wrongful corporate gains"),
    ("RescissionOfAgreement", "Judicial unwinding and cancellation of executed contract"),
    ("StatutoryLiquidatedDamages", "Pre-agreed contractual compensatory sum"),
    ("PreliminaryRestrainingOrder", "Urgent interim relief preserving status quo pending trial")
]

def generate_legal_benchmark_suite(num_samples: int = 50, seed: int = 101) -> Dict[str, List[Dict[str, Any]]]:
    """Generates standardized benchmark evaluation suites for legal Tasks A, B, and C."""
    rng = random.Random(seed)
    
    task_a_samples = []
    task_b_samples = []
    task_c_samples = []

    for idx in range(num_samples):
        plaintiff, p_desc = rng.choice(PARTIES_PLAINTIFF)
        defendant, d_desc = rng.choice(PARTIES_DEFENDANT)
        court, court_desc = rng.choice(COURTS)
        claim, claim_desc = rng.choice(LEGAL_CLAIMS)
        precedent, prec_desc = rng.choice(PRECEDENTS)
        remedy, rem_desc = rng.choice(REMEDIES)

        # ----------------------------------------------------
        # TASK A: Legal Triplet Extraction
        # Text -> Formal legal triples matching schema
        # ----------------------------------------------------
        text_passage = (
            f"In recent corporate litigation, {plaintiff} officially files a claim against {defendant}. "
            f"The underlying claim of {claim} was formally adjudicated by the {court}. "
            f"Following findings of liability, the court ruled that {defendant} is liable for {remedy}."
        )
        expected_triples = [
            {"subject": plaintiff, "predicate": "FILES_CLAIM_AGAINST", "object": defendant},
            {"subject": claim, "predicate": "ADJUDICATED_BY", "object": court},
            {"subject": defendant, "predicate": "LIABLE_FOR", "object": remedy}
        ]
        task_a_samples.append({
            "id": f"legal_task_a_{idx:03d}",
            "task_type": "legal_triplet_extraction",
            "input_text": text_passage,
            "target_triples": expected_triples
        })

        # ----------------------------------------------------
        # TASK B: Multi-Hop Precedent & Remedy Deduction
        # Chain: Court -(APPLIES_PRECEDENT)-> PrecedentCase -(ESTABLISHES_REMEDY)-> Remedy
        # ----------------------------------------------------
        hop_context = (
            f"Fact 1: The lawsuit was heard before {court}.\n"
            f"Fact 2: In its bench opinion, {court} APPLIES_PRECEDENT {precedent}.\n"
            f"Fact 3: Landmark ruling {precedent} ESTABLISHES_REMEDY {remedy}."
        )
        hop_query_2 = f"Based on jurisdictional precedent, which PrecedentCase was applied by {court}?"
        hop_answer_2 = precedent

        hop_query_3 = f"Following the judicial precedent chain from {court}, which LegalRemedy is established under this doctrine?"
        hop_answer_3 = remedy

        task_b_samples.append({
            "id": f"legal_task_b_{idx:03d}",
            "task_type": "legal_multi_hop_deduction",
            "subgraph_context": hop_context,
            "query_2hop": hop_query_2,
            "target_2hop": hop_answer_2,
            "query_3hop": hop_query_3,
            "target_3hop": hop_answer_3
        })

        # ----------------------------------------------------
        # TASK C: Legal Link Prediction
        # Given Head & Tail, predict the correct legal predicate
        # ----------------------------------------------------
        candidate_triples = [
            (plaintiff, defendant, "FILES_CLAIM_AGAINST"),
            (claim, court, "ADJUDICATED_BY"),
            (court, precedent, "APPLIES_PRECEDENT"),
            (precedent, remedy, "ESTABLISHES_REMEDY"),
            (defendant, remedy, "LIABLE_FOR")
        ]
        head, tail, true_rel = rng.choice(candidate_triples)

        task_c_samples.append({
            "id": f"legal_task_c_{idx:03d}",
            "task_type": "legal_link_prediction",
            "head": head,
            "tail": tail,
            "prompt": (
                f"Given legal entity Head '{head}' and Tail '{tail}', identify the valid ontological relation from: "
                f"[FILES_CLAIM_AGAINST, ADJUDICATED_BY, APPLIES_PRECEDENT, ESTABLISHES_REMEDY, LIABLE_FOR]."
            ),
            "target_relation": true_rel
        })

    return {
        "legal_task_a_triplets": task_a_samples,
        "legal_task_b_multihop": task_b_samples,
        "legal_task_c_link_pred": task_c_samples
    }

def save_legal_benchmark_suites(out_dir: str = "llm-graph-ontology/data/benchmark"):
    """Saves legal benchmark JSON files."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    suite = generate_legal_benchmark_suite(num_samples=50, seed=101)
    for name, data in suite.items():
        file_path = path / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} legal test items to {file_path}")

if __name__ == "__main__":
    save_legal_benchmark_suites()
