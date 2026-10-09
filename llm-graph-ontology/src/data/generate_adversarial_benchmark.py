"""
Adversarial Stress-Test Benchmark Suite Generator for Legal Knowledge Graph Tasks.
Generates 40 challenging edge cases across 4 failure-testing dimensions:
- Suite 1: Inverted & Passive Syntax (10 cases)
- Suite 2: Heavy Noise & Extraneous Distractors (10 cases)
- Suite 3: Negative / Dismissed Lawsuits with Zero Remedy (10 cases)
- Suite 4: Non-Litigation Commercial Partnerships & Alliances (10 cases)
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def build_adversarial_suite() -> List[Dict[str, Any]]:
    test_cases = []

    # =========================================================================
    # SUITE 1: INVERTED & PASSIVE SYNTAX (10 SAMPLES)
    # Tests whether passive voice flips (Subject, Object) orientation
    # =========================================================================
    suite_1_data = [
        (
            "Dragged into federal court following months of contentious dispute, DeltaSystemsInc was served with an infringement complaint originating from ApexHoldings. Presided over by the NorthernDistrictCalifornia, the underlying matter of PatentInfringement was thoroughly evaluated, culminating in a binding decree holding DeltaSystemsInc accountable for PermanentInjunction.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"},
                {"subject": "DeltaSystemsInc", "predicate": "LIABLE_FOR", "object": "PermanentInjunction"}
            ]
        ),
        (
            "Targeted in ongoing commercial litigation, NovaPharmaCorp faces a material breach action initiated by BioGenixLabs. The disputed BreachOfContract claims were formally heard and adjudicated by the DelawareChanceryCourt, which determined that NovaPharmaCorp must be held liable for RescissionOfAgreement.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"},
                {"subject": "NovaPharmaCorp", "predicate": "LIABLE_FOR", "object": "RescissionOfAgreement"}
            ]
        ),
        (
            "Before the SouthernDistrictNewYork, an emergency petition was lodged by OmniRetailCorp against GlobalSupplyChainLtd. The proceeding, centered on TradeSecretMisappropriation, was adjudicated by the bench, which ultimately found GlobalSupplyChainLtd liable for PreliminaryRestrainingOrder.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"},
                {"subject": "GlobalSupplyChainLtd", "predicate": "LIABLE_FOR", "object": "PreliminaryRestrainingOrder"}
            ]
        ),
        (
            "Confronted with allegations of corporate misconduct, TitanEnergyLLC found itself sued by QuantumTechLLC. The claim of FiduciaryDutyBreach fell under the direct jurisdiction of and was adjudicated by the SupremeCourtDelaware, with the final decree declaring TitanEnergyLLC liable for DisgorgementOfProfits.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "FiduciaryDutyBreach", "predicate": "ADJUDICATED_BY", "object": "SupremeCourtDelaware"},
                {"subject": "TitanEnergyLLC", "predicate": "LIABLE_FOR", "object": "DisgorgementOfProfits"}
            ]
        ),
        (
            "Having endured severe commercial harm, HorizonMediaGroup initiated formal proceedings against AegisCyberSec. In the SecondCircuitAppeals, the complex action involving SecuritiesFraud was formally adjudicated by the appellate panel, concluding that AegisCyberSec remains liable for StatutoryLiquidatedDamages.",
            [
                {"subject": "HorizonMediaGroup", "predicate": "FILES_CLAIM_AGAINST", "object": "AegisCyberSec"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"},
                {"subject": "AegisCyberSec", "predicate": "LIABLE_FOR", "object": "StatutoryLiquidatedDamages"}
            ]
        ),
        (
            "NexusRoboticsCorp was formally haled into court when VanguardLogistics launched an aggressive legal offensive. Adjudicated by the FederalCircuitAppeals, the core claim of TortiousInterference resulted in a finding that NexusRoboticsCorp is legally liable for TrebleDamages.",
            [
                {"subject": "VanguardLogistics", "predicate": "FILES_CLAIM_AGAINST", "object": "NexusRoboticsCorp"},
                {"subject": "TortiousInterference", "predicate": "ADJUDICATED_BY", "object": "FederalCircuitAppeals"},
                {"subject": "NexusRoboticsCorp", "predicate": "LIABLE_FOR", "object": "TrebleDamages"}
            ]
        ),
        (
            "In an unexpected filing, NovaPharmaCorp was accused of wrongdoing by ApexHoldings. Adjudicated by the NorthernDistrictCalifornia, the dispute over PatentInfringement ended with a judicial finding holding NovaPharmaCorp liable for TrebleDamages.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"},
                {"subject": "NovaPharmaCorp", "predicate": "LIABLE_FOR", "object": "TrebleDamages"}
            ]
        ),
        (
            "Subjected to intense judicial scrutiny, DeltaSystemsInc was prosecuted civilly by BioGenixLabs. The DelawareChanceryCourt adjudicated the contested matter of TradeSecretMisappropriation, finding DeltaSystemsInc liable for PermanentInjunction.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"},
                {"subject": "DeltaSystemsInc", "predicate": "LIABLE_FOR", "object": "PermanentInjunction"}
            ]
        ),
        (
            "A contentious lawsuit was leveled against TitanEnergyLLC by OmniRetailCorp. The matter of BreachOfContract was adjudicated by the SouthernDistrictNewYork, with TitanEnergyLLC found directly liable for StatutoryLiquidatedDamages.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"},
                {"subject": "TitanEnergyLLC", "predicate": "LIABLE_FOR", "object": "StatutoryLiquidatedDamages"}
            ]
        ),
        (
            "Under rigorous appellate examination, GlobalSupplyChainLtd faced charges brought by QuantumTechLLC. The claim of SecuritiesFraud was formally adjudicated by the SecondCircuitAppeals, which declared GlobalSupplyChainLtd liable for DisgorgementOfProfits.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"},
                {"subject": "GlobalSupplyChainLtd", "predicate": "LIABLE_FOR", "object": "DisgorgementOfProfits"}
            ]
        )
    ]

    for idx, (text, triples) in enumerate(suite_1_data):
        test_cases.append({
            "id": f"stress_s1_passive_{idx:02d}",
            "stress_suite": "suite_1_inverted_syntax",
            "input_text": text,
            "target_triples": triples,
            "expected_remedy_present": True,
            "is_litigation": True,
            "evaluation_focus": "Check if passive voice causes subject/object directional inversion"
        })

    # =========================================================================
    # SUITE 2: HEAVY DISTRACTORS & NOISE (10 SAMPLES)
    # Tests whether lawyers, dollar sums, dates, docket numbers contaminate triples
    # =========================================================================
    suite_2_data = [
        (
            "On October 14, 2024, lead trial counsel Sarah Jenkins of Latham & Watkins entered an appearance on behalf of ApexHoldings in filing a $450 million commercial complaint against DeltaSystemsInc under Docket No. 24-CV-1082. The disputed cause of action concerning PatentInfringement was formally adjudicated by the NorthernDistrictCalifornia before Senior Judge Henderson. After extensive jury deliberations, the bench ordered that DeltaSystemsInc is liable for PermanentInjunction.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"},
                {"subject": "DeltaSystemsInc", "predicate": "LIABLE_FOR", "object": "PermanentInjunction"}
            ]
        ),
        (
            "In a 140-page filing submitted on June 3, 2023, senior litigation partner David Vance representing BioGenixLabs launched an aggressive $1.2B legal action targeting NovaPharmaCorp. Case Manager Timothy Ross scheduled oral arguments where BreachOfContract was formally adjudicated by the DelawareChanceryCourt. Presiding Chancellor McCormick ruled that defendant NovaPharmaCorp is liable for RescissionOfAgreement.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"},
                {"subject": "NovaPharmaCorp", "predicate": "LIABLE_FOR", "object": "RescissionOfAgreement"}
            ]
        ),
        (
            "Following an exhaustive audit by forensic accountants Ernst & Young, OmniRetailCorp through attorney Marcus Vance initiated civil proceedings against GlobalSupplyChainLtd seeking $85 million. The underlying issue of TradeSecretMisappropriation was adjudicated by the SouthernDistrictNewYork in Courtroom 12B. Magistrate Judge Rivera confirmed that GlobalSupplyChainLtd is liable for PreliminaryRestrainingOrder.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"},
                {"subject": "GlobalSupplyChainLtd", "predicate": "LIABLE_FOR", "object": "PreliminaryRestrainingOrder"}
            ]
        ),
        (
            "On September 9, 2024, corporate counsel Rachel Green filed emergency paperwork on behalf of QuantumTechLLC against TitanEnergyLLC alleging extensive governance breaches valued at $300M. The legal claim of FiduciaryDutyBreach was thoroughly adjudicated by the SupremeCourtDelaware. In a unanimous 5-0 opinion, the court determined that TitanEnergyLLC is liable for DisgorgementOfProfits.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "FiduciaryDutyBreach", "predicate": "ADJUDICATED_BY", "object": "SupremeCourtDelaware"},
                {"subject": "TitanEnergyLLC", "predicate": "LIABLE_FOR", "object": "DisgorgementOfProfits"}
            ]
        ),
        (
            "Represented by Gibson Dunn, plaintiff HorizonMediaGroup brought a multi-district litigation against AegisCyberSec regarding accounting irregularities totaling $750M. Adjudicated by the SecondCircuitAppeals under Docket 23-4011, the allegation of SecuritiesFraud resulted in a published opinion declaring AegisCyberSec liable for StatutoryLiquidatedDamages.",
            [
                {"subject": "HorizonMediaGroup", "predicate": "FILES_CLAIM_AGAINST", "object": "AegisCyberSec"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"},
                {"subject": "AegisCyberSec", "predicate": "LIABLE_FOR", "object": "StatutoryLiquidatedDamages"}
            ]
        ),
        (
            "During third-quarter filings on August 15, VanguardLogistics retaining Quinn Emanuel filed suit against NexusRoboticsCorp for interference claims exceeding $40 million. The matter of TortiousInterference was officially adjudicated by the FederalCircuitAppeals, with Chief Judge Moore writing that NexusRoboticsCorp is liable for TrebleDamages.",
            [
                {"subject": "VanguardLogistics", "predicate": "FILES_CLAIM_AGAINST", "object": "NexusRoboticsCorp"},
                {"subject": "TortiousInterference", "predicate": "ADJUDICATED_BY", "object": "FederalCircuitAppeals"},
                {"subject": "NexusRoboticsCorp", "predicate": "LIABLE_FOR", "object": "TrebleDamages"}
            ]
        ),
        (
            "Before Magistrate Judge Katherine Polk on docket 24-CV-5521, plaintiff ApexHoldings represented by Kirkland & Ellis proceeded against NovaPharmaCorp in an intellectual property battle. The issue of PatentInfringement was formally adjudicated by the NorthernDistrictCalifornia, ruling NovaPharmaCorp liable for TrebleDamages.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"},
                {"subject": "NovaPharmaCorp", "predicate": "LIABLE_FOR", "object": "TrebleDamages"}
            ]
        ),
        (
            "On November 12, 2024, special litigation counsel Arthur Pendelton acting for BioGenixLabs sued DeltaSystemsInc over stolen source algorithms. Adjudicated by the DelawareChanceryCourt with assistance from expert witness Dr. Aris Thorne, the TradeSecretMisappropriation claim concluded with DeltaSystemsInc liable for PermanentInjunction.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"},
                {"subject": "DeltaSystemsInc", "predicate": "LIABLE_FOR", "object": "PermanentInjunction"}
            ]
        ),
        (
            "In an expedited bench proceeding initiated on April 2, OmniRetailCorp filed against TitanEnergyLLC for supply disruptions in the North Sea. Adjudicated by the SouthernDistrictNewYork with amicus briefs filed by the Chamber of Commerce, the BreachOfContract action found TitanEnergyLLC liable for StatutoryLiquidatedDamages.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"},
                {"subject": "TitanEnergyLLC", "predicate": "LIABLE_FOR", "object": "StatutoryLiquidatedDamages"}
            ]
        ),
        (
            "In Manhattan federal court, counsel for QuantumTechLLC served a notice of complaint against GlobalSupplyChainLtd under SEC Rule 10b-5. The claim of SecuritiesFraud was adjudicated by the SecondCircuitAppeals, where a three-judge panel found GlobalSupplyChainLtd liable for DisgorgementOfProfits.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"},
                {"subject": "GlobalSupplyChainLtd", "predicate": "LIABLE_FOR", "object": "DisgorgementOfProfits"}
            ]
        )
    ]

    for idx, (text, triples) in enumerate(suite_2_data):
        test_cases.append({
            "id": f"stress_s2_distractors_{idx:02d}",
            "stress_suite": "suite_2_heavy_distractors",
            "input_text": text,
            "target_triples": triples,
            "expected_remedy_present": True,
            "is_litigation": True,
            "evaluation_focus": "Check if attorney names, monetary sums, or judges leak into extracted triples"
        })

    # =========================================================================
    # SUITE 3: NEGATIVE / DISMISSED CASES (10 SAMPLES)
    # The court dismisses all claims with prejudice; ZERO liability; ZERO remedy!
    # Tests whether model hallucinates a remedy triple because training always had one
    # =========================================================================
    suite_3_data = [
        (
            "ApexHoldings officially files a claim against DeltaSystemsInc alleging PatentInfringement. The claim was adjudicated by the NorthernDistrictCalifornia. However, following summary judgment hearings, the presiding judge granted DeltaSystemsInc's motion to dismiss with prejudice, finding zero liability and denying all remedies.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"}
            ]
        ),
        (
            "BioGenixLabs files a claim against NovaPharmaCorp in Delaware. The matter of BreachOfContract was adjudicated by the DelawareChanceryCourt. Upon reviewing the supply agreement, the Chancellor found the contract terms fully satisfied, exonerated NovaPharmaCorp from all fault, and awarded zero damages or relief.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"}
            ]
        ),
        (
            "OmniRetailCorp officially files a claim against GlobalSupplyChainLtd. The TradeSecretMisappropriation action was formally adjudicated by the SouthernDistrictNewYork. The court held that no proprietary information was ever taken, dismissed the action, and refused to issue any injunction or remedy.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"}
            ]
        ),
        (
            "QuantumTechLLC files a claim against TitanEnergyLLC before judicial authorities. The disputed action involving FiduciaryDutyBreach was adjudicated by the SupremeCourtDelaware. The appellate bench affirmed complete defense immunity for TitanEnergyLLC, striking down all claims with prejudice and granting no remedy.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "FiduciaryDutyBreach", "predicate": "ADJUDICATED_BY", "object": "SupremeCourtDelaware"}
            ]
        ),
        (
            "HorizonMediaGroup officially files a claim against AegisCyberSec alleging SecuritiesFraud. The matter was adjudicated by the SecondCircuitAppeals. The appellate panel affirmed the trial court's dismissal under Rule 12(b)(6), confirming AegisCyberSec bears zero liability and establishing no legal remedy.",
            [
                {"subject": "HorizonMediaGroup", "predicate": "FILES_CLAIM_AGAINST", "object": "AegisCyberSec"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"}
            ]
        ),
        (
            "VanguardLogistics files a claim against NexusRoboticsCorp for TortiousInterference. The commercial dispute was adjudicated by the FederalCircuitAppeals. Finding that the plaintiff failed to establish proximate causation, the court affirmed total dismissal, absolving NexusRoboticsCorp from any relief or sanctions.",
            [
                {"subject": "VanguardLogistics", "predicate": "FILES_CLAIM_AGAINST", "object": "NexusRoboticsCorp"},
                {"subject": "TortiousInterference", "predicate": "ADJUDICATED_BY", "object": "FederalCircuitAppeals"}
            ]
        ),
        (
            "ApexHoldings files a claim against NovaPharmaCorp. The PatentInfringement proceeding was adjudicated by the NorthernDistrictCalifornia. The court invalidated the asserted patent claims under Section 101, dismissing the litigation without imposing any liability or remedy on NovaPharmaCorp.",
            [
                {"subject": "ApexHoldings", "predicate": "FILES_CLAIM_AGAINST", "object": "NovaPharmaCorp"},
                {"subject": "PatentInfringement", "predicate": "ADJUDICATED_BY", "object": "NorthernDistrictCalifornia"}
            ]
        ),
        (
            "BioGenixLabs files a claim against DeltaSystemsInc. The TradeSecretMisappropriation suit was formally adjudicated by the DelawareChanceryCourt. The court determined the statute of limitations had expired, entering a final dismissal order with zero remedies assessed against DeltaSystemsInc.",
            [
                {"subject": "BioGenixLabs", "predicate": "FILES_CLAIM_AGAINST", "object": "DeltaSystemsInc"},
                {"subject": "TradeSecretMisappropriation", "predicate": "ADJUDICATED_BY", "object": "DelawareChanceryCourt"}
            ]
        ),
        (
            "OmniRetailCorp files a claim against TitanEnergyLLC for BreachOfContract. The litigation was adjudicated by the SouthernDistrictNewYork. The parties voluntarily dismissed the complaint following mutual reconciliation, with the court awarding zero damages, sanctions, or remedies.",
            [
                {"subject": "OmniRetailCorp", "predicate": "FILES_CLAIM_AGAINST", "object": "TitanEnergyLLC"},
                {"subject": "BreachOfContract", "predicate": "ADJUDICATED_BY", "object": "SouthernDistrictNewYork"}
            ]
        ),
        (
            "QuantumTechLLC officially files a claim against GlobalSupplyChainLtd. The allegation of SecuritiesFraud was adjudicated by the SecondCircuitAppeals. The court found no evidence of scienter, throwing out the lawsuit completely and ruling that GlobalSupplyChainLtd is liable for nothing.",
            [
                {"subject": "QuantumTechLLC", "predicate": "FILES_CLAIM_AGAINST", "object": "GlobalSupplyChainLtd"},
                {"subject": "SecuritiesFraud", "predicate": "ADJUDICATED_BY", "object": "SecondCircuitAppeals"}
            ]
        )
    ]

    for idx, (text, triples) in enumerate(suite_3_data):
        test_cases.append({
            "id": f"stress_s3_dismissed_{idx:02d}",
            "stress_suite": "suite_3_dismissed_cases",
            "input_text": text,
            "target_triples": triples,
            "expected_remedy_present": False,
            "is_litigation": True,
            "evaluation_focus": "Check if model hallucinates a non-existent remedy or LIABLE_FOR relation"
        })

    # =========================================================================
    # SUITE 4: COMMERCIAL PARTNERSHIPS & ALLIANCES (10 SAMPLES)
    # Non-litigation business agreements. Expected output: EMPTY LIST `[]`!
    # Tests False Positive Rate (does it output FILES_CLAIM_AGAINST on any 2 companies?)
    # =========================================================================
    suite_4_data = [
        "ApexHoldings and DeltaSystemsInc announced an expansive multi-year joint venture to build commercial cloud data centers across North America.",
        "BioGenixLabs entered into an exclusive strategic research alliance with NovaPharmaCorp to co-develop biosimilar medicines for international distribution.",
        "OmniRetailCorp finalized a long-term supply chain master agreement with GlobalSupplyChainLtd to streamline maritime transport operations.",
        "QuantumTechLLC and TitanEnergyLLC executed a cooperative green energy memorandum of understanding to power advanced semiconductor foundries.",
        "HorizonMediaGroup partnered with AegisCyberSec to establish an enterprise cybersecurity monitoring framework protecting digital intellectual property.",
        "VanguardLogistics completed a friendly equity investment in NexusRoboticsCorp to integrate autonomous pick-and-place robotics into regional fulfillment centers.",
        "ApexHoldings and NovaPharmaCorp co-hosted an annual pharmaceutical investment summit in Boston, celebrating joint scientific research achievements.",
        "BioGenixLabs and DeltaSystemsInc signed a mutual nondisclosure agreement and exploring potential licensing opportunities for bioinformatics software.",
        "OmniRetailCorp and TitanEnergyLLC launched a nationwide corporate sustainability initiative aimed at reducing industrial carbon emissions.",
        "QuantumTechLLC joined forces with GlobalSupplyChainLtd to create an open-source tracking consortium for commercial freight transparency."
    ]

    for idx, text in enumerate(suite_4_data):
        test_cases.append({
            "id": f"stress_s4_partnership_{idx:02d}",
            "stress_suite": "suite_4_commercial_partnerships",
            "input_text": text,
            "target_triples": [],
            "expected_remedy_present": False,
            "is_litigation": False,
            "evaluation_focus": "Check False Positive Rate (does it falsely predict FILES_CLAIM_AGAINST?)"
        })

    return test_cases

def main():
    suite = build_adversarial_suite()
    out_dir = Path("llm-graph-ontology/data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "legal_adversarial_stress_test.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)

    suite_counts = {}
    for c in suite:
        s = c["stress_suite"]
        suite_counts[s] = suite_counts.get(s, 0) + 1

    print(f"[SUCCESS] Generated {len(suite)} adversarial test cases -> {out_file}")
    for k, v in suite_counts.items():
        print(f"   - {k}: {v} cases")

if __name__ == "__main__":
    main()
