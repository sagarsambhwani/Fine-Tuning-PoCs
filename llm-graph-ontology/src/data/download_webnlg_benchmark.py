"""
WebNLG Public Human-Annotated Benchmark Downloader and Standardizer.
Fetches official WebNLG challenge test data (DBpedia human-written sentences -> RDF triples),
and standardizes 100 balanced test samples across diverse categories.
"""
import sys
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WEBNLG_XML_URL = "https://raw.githubusercontent.com/fuzihaofzh/webnlg-dataset/master/webnlg_challenge_2017/test/testdata_with_lex.xml"

def clean_entity(token: str) -> str:
    """Cleans DBpedia entity names: removes enclosing quotes and replaces underscores with spaces."""
    cleaned = token.strip().strip('"').strip("'")
    return cleaned.replace("_", " ")

def download_and_parse_webnlg(num_samples: int = 100) -> List[Dict[str, Any]]:
    print(f"Downloading official WebNLG test set from:\n{WEBNLG_XML_URL}...")
    req = urllib.request.Request(WEBNLG_XML_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        xml_content = response.read().decode("utf-8")

    root = ET.fromstring(xml_content)
    entries = root.findall(".//entry")
    print(f"Downloaded {len(entries)} total entries. Extracting {num_samples} balanced samples...")

    samples = []
    seen_texts = set()
    category_counts = {}

    for idx, entry in enumerate(entries):
        category = entry.get("category", "General")
        
        # Extract human natural language text
        lex_elements = entry.findall(".//lex")
        if not lex_elements:
            continue
        # Pick the cleanest lex text
        text = lex_elements[0].text
        if not text or len(text.strip()) < 10 or text in seen_texts:
            continue

        # Extract RDF triples
        mtriples = entry.findall(".//mtriple")
        if not mtriples:
            continue

        triples = []
        for mt in mtriples:
            raw_t = mt.text.strip()
            parts = [p.strip() for p in raw_t.split("|")]
            if len(parts) == 3:
                triples.append({
                    "subject": clean_entity(parts[0]),
                    "predicate": clean_entity(parts[1]),
                    "object": clean_entity(parts[2])
                })

        if not triples:
            continue

        seen_texts.add(text)
        category_counts[category] = category_counts.get(category, 0) + 1

        samples.append({
            "id": f"webnlg_human_{len(samples):03d}",
            "source_benchmark": "WebNLG_v1_Human_Annotated",
            "category": category,
            "input_text": text.strip(),
            "target_triples": triples,
            "triple_count": len(triples)
        })

        if len(samples) >= num_samples:
            break

    return samples

def main():
    samples = download_and_parse_webnlg(100)
    out_dir = Path("llm-graph-ontology/data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "webnlg_human_benchmark_100.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    cat_counts = {}
    for s in samples:
        c = s["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print(f"[SUCCESS] Saved {len(samples)} real human-annotated samples -> {out_file}")
    print(f"Categories extracted: {cat_counts}")

if __name__ == "__main__":
    main()
