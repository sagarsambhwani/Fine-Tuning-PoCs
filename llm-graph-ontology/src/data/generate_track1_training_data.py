"""
Track 1 Training and Validation Dataset Generator with Anti-Leakage Guarantee.
Generates balanced, chat-formatted JSONL datasets:
- train.jsonl (1,200 samples)
- val.jsonl (150 samples)
Balanced across:
- Task A: Triplet Extraction (40%)
- Task B: Multi-Hop Reasoning (30%)
- Task C: Link Prediction (30%)
Enforces zero overlap with test benchmark entity pools.
"""
import sys
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Ensure stdout handles UTF-8 on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -------------------------------------------------------------------------
# RESERVED TEST BENCHMARK ENTITY SET (STRICT PROHIBITION LIST)
# -------------------------------------------------------------------------
BENCHMARK_PROHIBITED_ENTITIES = {
    # Test Plaintiffs
    "ApexHoldings", "BioGenixLabs", "OmniRetailCorp", "QuantumTechLLC", "HorizonMediaGroup", "VanguardLogistics",
    # Test Defendants
    "DeltaSystemsInc", "NovaPharmaCorp", "GlobalSupplyChainLtd", "TitanEnergyLLC", "AegisCyberSec", "NexusRoboticsCorp",
    # Test Courts
    "DelawareChanceryCourt", "FederalCircuitAppeals", "SouthernDistrictNewYork", "NorthernDistrictCalifornia", 
    "SecondCircuitAppeals", "SupremeCourtDelaware",
    # Test Claims
    "PatentInfringement", "BreachOfContract", "TradeSecretMisappropriation", "FiduciaryDutyBreach", 
    "SecuritiesFraud", "TortiousInterference",
    # Test Precedents
    "eBay_v_MercExchange", "Alice_v_CLSBank", "Revlon_v_MacAndrews", "Basic_v_Levinson", 
    "Twombly_v_BellAtlantic", "Pennzoil_v_Texaco",
    # Test Remedies
    "PermanentInjunction", "TrebleDamages", "DisgorgementOfProfits", "RescissionOfAgreement", 
    "StatutoryLiquidatedDamages", "PreliminaryRestrainingOrder"
}

# -------------------------------------------------------------------------
# TRAINING & VALIDATION DEDICATED ENTITY POOLS (STRICTLY DISJOINT)
# -------------------------------------------------------------------------
TRAIN_VAL_PLAINTIFFS = [
    ("StarlightVentures", "Venture capital investor in digital assets"),
    ("AeroDynamicsInc", "Commercial aerospace components designer"),
    ("SummitCapitalLLC", "Institutional mezzanine finance group"),
    ("BeaconBioTech", "Genomic diagnostic testing pioneer"),
    ("CrestviewMedia", "Syndicated broadcast and streaming network"),
    ("TerraFirmGlobal", "Agricultural commodity trading conglomerate"),
    ("PinnacleLogistics", "Intermodal freight and distribution service"),
    ("VertexRobotics", "Automated manufacturing robotics developer"),
    ("SolarisRenewables", "Commercial solar farm energy operator"),
    ("FrontierEnterprises", "Natural resource extraction operator"),
    ("SilverlineHoldings", "Private asset management partnership"),
    ("CobaltTechnologies", "Advanced lithium-ion battery developer"),
    ("ParamountIndustries", "Heavy industrial equipment fabricator"),
    ("StratosAviation", "Regional commercial aircraft leasing firm"),
    ("MeridianCapital", "Sovereign wealth co-investment fund"),
    ("TridentMaritime", "Container vessel fleet management company"),
    ("EquinoxEnergy", "Offshore wind power generation utility"),
    ("ValenceMaterials", "Specialty polymers and chemical synthesizer"),
    ("AcuityHealth", "Integrated hospital management network"),
    ("OptimaSystems", "Enterprise database architecture firm"),
    ("CenturionDefense", "Autonomous border monitoring contractor"),
    ("VanguardVentures", "Early-stage deeptech incubator"),
    ("ZephyrNetworks", "Fiber-optic telecommunications provider"),
    ("KeystoneResources", "Rare-earth minerals exploration venture"),
    ("SentrySecurity", "Commercial biometric access developer"),
    ("AetherBio", "CRISPR gene-editing therapeutics startup"),
    ("HighlandLogistics", "Cold-chain pharmaceutical shipping group"),
    ("OlympusCapital", "Real estate mezzanine debt syndicate"),
    ("NautilusMarine", "Subsea cable infrastructure operator"),
    ("IronwoodHoldings", "Commercial timberland investment trust")
]

TRAIN_VAL_DEFENDANTS = [
    ("IroncladSecurities", "Underwriting and investment banking house"),
    ("VortexSoftware", "Cloud-native CRM and workflow provider"),
    ("ZenithChemicals", "Industrial agricultural fertilizer producer"),
    ("AtlasManufacturing", "Precision CNC parts and tooling supplier"),
    ("PalisadesCorp", "Commercial hospitality real estate REIT"),
    ("EchoStreamNetworks", "Distributed content delivery network"),
    ("KryptonIndustrial", "Heavy metallurgical alloy foundry"),
    ("VectorAutonomous", "Self-driving navigation software provider"),
    ("OmegaEnergyPartners", "Petrochemical refining and pipeline operator"),
    ("SterlingMaritime", "Bulk bulk dry cargo shipping contractor"),
    ("HyperionDynamics", "Electric propulsion research firm"),
    ("VigilantSystems", "Enterprise perimeter surveillance provider"),
    ("MonolithRobotics", "Automated warehouse pick-and-pack developer"),
    ("ArcadiaBiofuels", "Second-generation cellulosic ethanol refiner"),
    ("TritonAerospace", "Suborbital satellite launch vehicle developer"),
    ("SolomonFintech", "Algorithmic clearinghouse and payments broker"),
    ("CortexComputing", "Neuromorphic AI processor foundry"),
    ("CerberusSecurity", "Industrial SCADA network defense consultancy"),
    ("HeliosSolar", "Utility-scale photovoltaic module fabricator"),
    ("PrometheusAI", "Foundation language model infrastructure venture"),
    ("StrataMinerals", "Bauxite and copper mining operator"),
    ("AstraLogistics", "Last-mile autonomous delivery fleet manager"),
    ("PhaetonAutomotive", "Electric vehicle battery management maker"),
    ("ViperCyber", "Offensive penetration testing vendor"),
    ("CobaltMining", "Artisanal cobalt refining contractor"),
    ("BorealisChemicals", "Fluoropolymer resin production facility"),
    ("AegisHoldingsInc", "Debt recovery and distressed loan servicer"),
    ("FortressSupply", "Wholesale building materials distributor"),
    ("PolarisHeavyIndustries", "Locomotive engine and chassis builder"),
    ("NemesisNetworks", "Mesh radio communication equipment designer")
]

TRAIN_VAL_COURTS = [
    ("DistrictOfColumbiaCircuit", "Federal appellate authority over administrative law"),
    ("FifthCircuitAppeals", "Federal appellate court for commercial and energy disputes"),
    ("NinthCircuitAppeals", "Federal appellate court for technology and IP litigation"),
    ("SeventhCircuitAppeals", "Appellate authority for antitrust and commercial matters"),
    ("ThirdCircuitAppeals", "Appellate authority over corporate bankruptcy and reorganizations"),
    ("SixthCircuitAppeals", "Federal appellate jurisdiction for industrial disputes"),
    ("EleventhCircuitAppeals", "Federal appellate jurisdiction for financial regulations"),
    ("EasternDistrictTexas", "Premier federal trial venue for patent infringement cases"),
    ("DistrictOfDelaware", "Federal trial court presiding over bankruptcy and IP filings"),
    ("SouthernDistrictFlorida", "Federal trial court for international maritime arbitration"),
    ("NorthernDistrictIllinois", "Federal commercial trial court centered in Chicago"),
    ("WesternDistrictWashington", "Federal trial venue for cloud computing and tech antitrust")
]

TRAIN_VAL_CLAIMS = [
    ("AntitrustConspiracy", "Collusive price-fixing agreement violating Sherman Act § 1"),
    ("FalseAdvertising", "Materially deceptive commercial claims under Lanham Act § 43(a)"),
    ("CopyrightInfringement", "Unauthorized distribution and public display of protected media"),
    ("InsiderTrading", "Trading securities using material non-public insider information"),
    ("UnfairCompetition", "Predatory commercial conduct misleading market consumers"),
    ("ConsumerFraud", "Deceptive sales practices inducing fraudulent subscription billing"),
    ("ProductLiability", "Defective design and failure to warn causing commercial harm"),
    ("DefamationPerSe", "False commercial statements attacking business trade reputation"),
    ("BreachOfWarranty", "Failure of delivered industrial units to meet fitness warranties"),
    ("CivilRICO", "Pattern of racketeering activity through predicate wire fraud acts"),
    ("EmploymentDiscrimination", "Systematic adverse personnel actions violating Title VII"),
    ("TrademarkDilution", "Commercial tarnishment and blurring of a famous trade brand")
]

TRAIN_VAL_PRECEDENTS = [
    ("Daubert_v_MerrellDow", "Federal evidentiary standard for scientific expert testimony"),
    ("Chevron_v_NRDC", "Judicial deference standard to administrative regulatory agencies"),
    ("Celotex_v_Catrett", "Evidentiary burden standard for summary judgment motions"),
    ("Matsushita_v_Zenith", "Plausibility requirement in predatory pricing antitrust claims"),
    ("Ashcroft_v_Iqbal", "Heightened pleading standard for federal civil complaints"),
    ("KSR_v_Teleflex", "Flexible obviousness doctrine under 35 U.S.C. § 103 patent law"),
    ("Graham_v_JohnDeere", "Factual inquiries required to determine patent obviousness"),
    ("Morrison_v_NationalAustralia", "Presumption against extraterritorial application of securities laws"),
    ("Bell_v_Hood", "Federal subject-matter jurisdiction based on federal question claims"),
    ("Marbury_v_Madison", "Constitutional doctrine establishing judicial review power"),
    ("ErieRailroad_v_Tompkins", "Rule requiring federal courts to apply state substantive law"),
    ("InternationalShoe_v_Washington", "Minimum contacts standard for personal jurisdiction")
]

TRAIN_VAL_REMEDIES = [
    ("ConstructiveTrust", "Equitable decree placing wrongful gains into judicial trust"),
    ("RestitutionAward", "Court order requiring return of exact unjustly gained benefits"),
    ("AccountingOfProfits", "Equitable audit stripping defendant of gross net profit margin"),
    ("SpecificPerformance", "Mandatory court order compelling exact contractual execution"),
    ("CivilFines", "Statutory regulatory monetary penalty levied against corporation"),
    ("CorrectiveAdvertisingOrder", "Mandatory court-directed campaign correcting false advertising"),
    ("QuoWarrantoOrder", "Judicial order testing corporate authority or franchise privileges"),
    ("DeclaratoryJudgment", "Binding judicial statement establishing legal rights of parties"),
    ("MonetarySanctions", "Court-ordered attorney fees sanctioning bad-faith litigation conduct"),
    ("AssetFreezeOrder", "Interim judicial restraint preventing offshore transfer of capital"),
    ("CompellingArbitration", "Judicial order enforcing mandatory contractual arbitration forum"),
    ("AttorneysFeesAward", "Statutory fee-shifting reimbursement awarded to prevailing litigant")
]

ONTOLOGY_RELATIONS = [
    "FILES_CLAIM_AGAINST",
    "ADJUDICATED_BY",
    "APPLIES_PRECEDENT",
    "ESTABLISHES_REMEDY",
    "LIABLE_FOR"
]

def assert_anti_leakage():
    """Formally verifies that training/val entity pools have 0% overlap with benchmark test entities."""
    all_train_entities = set()
    for pool in [TRAIN_VAL_PLAINTIFFS, TRAIN_VAL_DEFENDANTS, TRAIN_VAL_COURTS, 
                 TRAIN_VAL_CLAIMS, TRAIN_VAL_PRECEDENTS, TRAIN_VAL_REMEDIES]:
        for entity_name, _ in pool:
            all_train_entities.add(entity_name)
            
    intersection = all_train_entities.intersection(BENCHMARK_PROHIBITED_ENTITIES)
    if intersection:
        raise AssertionError(f"FATAL: Anti-leakage check FAILED! Overlapping entities detected: {intersection}")
    
    print(f"[OK] Anti-Leakage Verified: {len(all_train_entities)} train/val entities have ZERO overlap with {len(BENCHMARK_PROHIBITED_ENTITIES)} test entities.")
    return all_train_entities

def generate_sample(rng: random.Random, sample_id: str, task_type: str) -> Dict[str, Any]:
    """Generates a single balanced sample in chat template format for the specified task."""
    plaintiff, p_desc = rng.choice(TRAIN_VAL_PLAINTIFFS)
    defendant, d_desc = rng.choice(TRAIN_VAL_DEFENDANTS)
    court, court_desc = rng.choice(TRAIN_VAL_COURTS)
    claim, claim_desc = rng.choice(TRAIN_VAL_CLAIMS)
    precedent, prec_desc = rng.choice(TRAIN_VAL_PRECEDENTS)
    remedy, rem_desc = rng.choice(TRAIN_VAL_REMEDIES)

    if task_type == "triplet_extraction":
        # Variations in phrasing to promote robust general extraction
        templates = [
            f"In federal commercial litigation, {plaintiff} officially files a claim against {defendant}. The legal claim of {claim} was formally adjudicated by the {court}. Following liability findings, the court ordered that {defendant} is liable for {remedy}.",
            f"Formal filings record that {plaintiff} files a claim against {defendant} before judicial authorities. The disputed action involving {claim} was adjudicated by the {court}. Upon entering final decree, {court} applies precedent {precedent}, which formally establishes remedy {remedy}.",
            f"During dispute proceedings, plaintiff {plaintiff} files a claim against {defendant}. The {court} adjudicated by procedure the claim of {claim}. Under binding authority, {precedent} establishes remedy {remedy}, and the presiding judge ruled that {defendant} is liable for {remedy}."
        ]
        chosen_template = rng.choice(templates)
        
        # Build ground-truth triples matching the chosen text
        if "applies precedent" in chosen_template:
            triples = [
                {"subject": plaintiff, "predicate": "FILES_CLAIM_AGAINST", "object": defendant},
                {"subject": claim, "predicate": "ADJUDICATED_BY", "object": court},
                {"subject": court, "predicate": "APPLIES_PRECEDENT", "object": precedent},
                {"subject": precedent, "predicate": "ESTABLISHES_REMEDY", "object": remedy}
            ]
        else:
            triples = [
                {"subject": plaintiff, "predicate": "FILES_CLAIM_AGAINST", "object": defendant},
                {"subject": claim, "predicate": "ADJUDICATED_BY", "object": court},
                {"subject": defendant, "predicate": "LIABLE_FOR", "object": remedy}
            ]

        return {
            "id": sample_id,
            "task_type": "triplet_extraction",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a formal corporate litigation knowledge graph extractor. "
                        "Extract factual triples strictly conforming to the legal ontology relations: "
                        "[FILES_CLAIM_AGAINST, ADJUDICATED_BY, APPLIES_PRECEDENT, ESTABLISHES_REMEDY, LIABLE_FOR]. "
                        "Output ONLY a valid JSON list of objects with keys: 'subject', 'predicate', 'object'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract legal triples from this text:\n{chosen_template}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(triples, indent=2)
                }
            ]
        }

    elif task_type == "multihop_reasoning":
        # Multi-hop deduction chain
        context = (
            f"Legal Knowledge Graph Subgraph:\n"
            f"- ({plaintiff}) --[FILES_CLAIM_AGAINST]--> ({defendant})\n"
            f"- ({claim}) --[ADJUDICATED_BY]--> ({court})\n"
            f"- ({court}) --[APPLIES_PRECEDENT]--> ({precedent})\n"
            f"- ({precedent}) --[ESTABLISHES_REMEDY]--> ({remedy})\n"
            f"- ({defendant}) --[LIABLE_FOR]--> ({remedy})"
        )
        
        # 50% 2-hop, 50% 3-hop questions
        if rng.random() < 0.5:
            # 2-Hop Question: Court -> Precedent -> Remedy
            question = f"Which legal remedy is established by the judicial precedent applied by the {court}?"
            answer = remedy
        else:
            # 3-Hop Question: Claim -> Court -> Precedent -> Remedy
            question = f"What legal remedy is established by the judicial precedent applied in the court that adjudicated the {claim} claim?"
            answer = remedy

        return {
            "id": sample_id,
            "task_type": "multihop_reasoning",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a legal knowledge graph deductive reasoning engine. Answer the question using ONLY the facts provided. State the target entity name directly."
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nQuestion: {question}"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }

    elif task_type == "link_prediction":
        # Link Prediction across the 5 relations
        relation_cases = [
            (plaintiff, defendant, "FILES_CLAIM_AGAINST", f"Between legal party '{plaintiff}' and opposing party '{defendant}'."),
            (claim, court, "ADJUDICATED_BY", f"Between dispute matter '{claim}' and judicial forum '{court}'."),
            (court, precedent, "APPLIES_PRECEDENT", f"Between judicial forum '{court}' and historical binding authority '{precedent}'."),
            (precedent, remedy, "ESTABLISHES_REMEDY", f"Between judicial case authority '{precedent}' and relief measure '{remedy}'."),
            (defendant, remedy, "LIABLE_FOR", f"Between responding litigant '{defendant}' and court-sanctioned liability '{remedy}'.")
        ]
        head, tail, target_rel, hint = rng.choice(relation_cases)
        prompt_text = (
            f"Head Entity: '{head}' | Tail Entity: '{tail}'\n"
            f"Available Relations: [FILES_CLAIM_AGAINST, ADJUDICATED_BY, APPLIES_PRECEDENT, ESTABLISHES_REMEDY, LIABLE_FOR]\n"
            f"Context: {hint}\n"
            f"Which relation connects '{head}' to '{tail}'?"
        )

        return {
            "id": sample_id,
            "task_type": "link_prediction",
            "messages": [
                {
                    "role": "system",
                    "content": "Predict the single valid ontology relation connecting Head to Tail. Reply with ONLY the relation name."
                },
                {
                    "role": "user",
                    "content": prompt_text
                },
                {
                    "role": "assistant",
                    "content": target_rel
                }
            ]
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def generate_track1_dataset(total_train: int = 1200, total_val: int = 150, seed: int = 42, out_dir: Path = None):
    """Generates train.jsonl and val.jsonl datasets with strict anti-leakage guarantee."""
    assert_anti_leakage()
    rng = random.Random(seed)

    task_distribution = [
        ("triplet_extraction", 0.40),
        ("multihop_reasoning", 0.30),
        ("link_prediction", 0.30)
    ]

    if out_dir is None:
        try:
            out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        except Exception:
            out_dir = Path("data/processed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate Train split
    train_records = []
    for i in range(total_train):
        # Choose task based on weighted distribution
        r = rng.random()
        if r < 0.40:
            task = "triplet_extraction"
        elif r < 0.70:
            task = "multihop_reasoning"
        else:
            task = "link_prediction"
        sample = generate_sample(rng, f"train_{i:04d}", task)
        train_records.append(sample)

    # Generate Val split (separate seed stream)
    rng_val = random.Random(seed + 999)
    val_records = []
    for i in range(total_val):
        r = rng_val.random()
        if r < 0.40:
            task = "triplet_extraction"
        elif r < 0.70:
            task = "multihop_reasoning"
        else:
            task = "link_prediction"
        sample = generate_sample(rng_val, f"val_{i:04d}", task)
        val_records.append(sample)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    # Counts summary
    def count_tasks(records):
        counts = {}
        for r in records:
            t = r["task_type"]
            counts[t] = counts.get(t, 0) + 1
        return counts

    train_counts = count_tasks(train_records)
    val_counts = count_tasks(val_records)

    manifest = {
        "dataset_name": "track1_pure_llm_legal_graph_sft",
        "total_train_samples": len(train_records),
        "total_val_samples": len(val_records),
        "train_task_breakdown": train_counts,
        "val_task_breakdown": val_counts,
        "anti_leakage_status": "VERIFIED_ZERO_OVERLAP",
        "benchmark_prohibited_entities_count": len(BENCHMARK_PROHIBITED_ENTITIES),
        "train_val_entity_pool_counts": {
            "plaintiffs": len(TRAIN_VAL_PLAINTIFFS),
            "defendants": len(TRAIN_VAL_DEFENDANTS),
            "courts": len(TRAIN_VAL_COURTS),
            "claims": len(TRAIN_VAL_CLAIMS),
            "precedents": len(TRAIN_VAL_PRECEDENTS),
            "remedies": len(TRAIN_VAL_REMEDIES)
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Generated {len(train_records)} train samples -> {train_path}")
    print(f"   Task Breakdown: {train_counts}")
    print(f"[SUCCESS] Generated {len(val_records)} val samples -> {val_path}")
    print(f"   Task Breakdown: {val_counts}")
    print(f"[SAVED] Saved manifest -> {manifest_path}")

if __name__ == "__main__":
    generate_track1_dataset()
