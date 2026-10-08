"""
Automated Unit Tests for Track 1 Data Quality and Anti-Leakage Guarantee.
"""
import json
import pytest
from pathlib import Path

DATA_DIR = Path("llm-graph-ontology/data")
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"

def test_dataset_files_exist_and_counts():
    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"
    manifest_file = PROCESSED_DIR / "manifest.json"

    assert train_file.exists(), "train.jsonl does not exist"
    assert val_file.exists(), "val.jsonl does not exist"
    assert manifest_file.exists(), "manifest.json does not exist"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f if line.strip()]

    with open(val_file, "r", encoding="utf-8") as f:
        val_lines = [json.loads(line) for line in f if line.strip()]

    assert len(train_lines) == 1200, f"Expected 1200 train samples, got {len(train_lines)}"
    assert len(val_lines) == 150, f"Expected 150 val samples, got {len(val_lines)}"

def test_sample_structure_and_chat_format():
    for filename in ["train.jsonl", "val.jsonl"]:
        filepath = PROCESSED_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                assert "id" in item, f"Missing 'id' at line {idx} in {filename}"
                assert "task_type" in item, f"Missing 'task_type' at line {idx} in {filename}"
                assert item["task_type"] in ["triplet_extraction", "multihop_reasoning", "link_prediction"]
                assert "messages" in item, f"Missing 'messages' at line {idx} in {filename}"
                assert len(item["messages"]) == 3, f"Expected 3 messages (system, user, assistant), got {len(item['messages'])}"
                assert item["messages"][0]["role"] == "system"
                assert item["messages"][1]["role"] == "user"
                assert item["messages"][2]["role"] == "assistant"
                assert len(item["messages"][2]["content"].strip()) > 0, f"Empty assistant target at line {idx}"

def test_zero_leakage_against_benchmark_test_suites():
    # Load all benchmark test items
    test_entities = set()
    for task_filename in ["legal_task_a_triplets.json", "legal_task_b_multihop.json", "legal_task_c_link_pred.json"]:
        bench_path = BENCHMARK_DIR / task_filename
        assert bench_path.exists(), f"Benchmark file {bench_path} missing"
        with open(bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if "target_triples" in item:
                    for t in item["target_triples"]:
                        test_entities.add(t["subject"].strip())
                        test_entities.add(t["object"].strip())
                if "target_2hop" in item:
                    test_entities.add(item["target_2hop"].strip())
                if "target_3hop" in item:
                    test_entities.add(item["target_3hop"].strip())
                if "head" in item:
                    test_entities.add(item["head"].strip())
                if "tail" in item:
                    test_entities.add(item["tail"].strip())

    assert len(test_entities) > 0, "No test entities extracted from benchmark files"

    # Check that none of the prohibited test entities appear as entity names in train.jsonl or val.jsonl
    for filename in ["train.jsonl", "val.jsonl"]:
        filepath = PROCESSED_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                content = line
                for prohibited in test_entities:
                    # An entity name in our dataset is a discrete alphanumeric token
                    assert prohibited not in content, (
                        f"LEAKAGE DETECTED: Test entity '{prohibited}' leaked into {filename} at line {idx}!"
                    )
