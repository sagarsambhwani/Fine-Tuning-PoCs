"""
Stanford LegalBench Real-World Litigation Benchmark Downloader.
Extracts 30 real federal securities litigation complaint excerpts (SSLA dataset)
and prepares them for CPU-optimized evaluation.
"""
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import datasets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_complaint_text(raw_text: str, max_chars: int = 1200) -> str:
    # Remove excessive whitespace, line breaks, and page header artifacts
    cleaned = re.sub(r'[\r\n]+', ' ', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Truncate to relevant allegations paragraph
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit('.', 1)[0] + '.'
    return cleaned

def extract_legalbench_samples(num_samples: int = 30) -> List[Dict[str, Any]]:
    print(f"Loading 'ssla_company_defendants' from Stanford LegalBench (nguha/legalbench)...")
    ds = datasets.load_dataset("nguha/legalbench", "ssla_company_defendants", split="test")

    samples = []
    for idx, item in enumerate(ds):
        raw_answer = item.get("answer")
        target = str(raw_answer).strip() if raw_answer is not None else ""
        text = str(item.get("text", "")).strip()

        if not target or len(text) < 100:
            continue

        cleaned_text = clean_complaint_text(text)

        # Extract docket if present
        docket_match = re.search(r'Case\s+[\w\:\-]+', text)
        docket = docket_match.group(0) if docket_match else "Federal Docket"

        samples.append({
            "id": f"legalbench_real_{len(samples):02d}",
            "source_benchmark": "Stanford_LegalBench_SSLA",
            "docket": docket,
            "complaint_excerpt": cleaned_text,
            "target_company_defendant": target
        })

        if len(samples) >= num_samples:
            break

    return samples

def main():
    samples = extract_legalbench_samples(30)
    out_dir = Path("llm-graph-ontology/data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "legalbench_real_litigation_30.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"[SUCCESS] Extracted {len(samples)} real federal litigation complaint samples -> {out_file}")
    print(f"Sample 0 Target Defendant: {samples[0]['target_company_defendant']}")
    print(f"Sample 0 Docket: {samples[0]['docket']}")

if __name__ == "__main__":
    main()
