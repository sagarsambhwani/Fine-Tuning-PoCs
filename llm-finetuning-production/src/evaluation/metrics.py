"""
Evaluation Metrics for Structured JSON Extraction
Calculates JSON validity, schema compliance, field-level accuracy, and exact match rates.
"""
from typing import List, Dict, Any, Tuple
from src.data.validator import parse_and_validate_extraction, REQUIRED_SCHEMA_KEYS

def evaluate_single_sample(
    raw_prediction: str,
    ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluates a single raw model string prediction against reference ground truth."""
    is_valid_json, parsed_dict, err = parse_and_validate_extraction(raw_prediction)
    
    result = {
        "is_valid_json": is_valid_json,
        "error": err,
        "exact_match": False,
        "field_matches": {k: False for k in REQUIRED_SCHEMA_KEYS},
        "parsed_output": parsed_dict
    }

    if not is_valid_json or parsed_dict is None:
        return result

    # Check field matches
    all_fields_match = True
    for key in REQUIRED_SCHEMA_KEYS:
        expected = ground_truth.get(key)
        actual = parsed_dict.get(key)

        # Normalize string comparison
        if isinstance(expected, str) and isinstance(actual, str):
            match = expected.strip().lower() == actual.strip().lower()
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            match = abs(float(expected) - float(actual)) < 1e-4
        else:
            match = expected == actual

        result["field_matches"][key] = match
        if not match:
            all_fields_match = False

    result["exact_match"] = all_fields_match
    return result

def compute_extraction_metrics(
    eval_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregates batch of evaluated single samples into overall metrics."""
    total = len(eval_results)
    if total == 0:
        return {"total_samples": 0}

    valid_json_count = sum(1 for r in eval_results if r["is_valid_json"])
    exact_match_count = sum(1 for r in eval_results if r["exact_match"])

    field_counts = {k: 0 for k in REQUIRED_SCHEMA_KEYS}
    for r in eval_results:
        for k in REQUIRED_SCHEMA_KEYS:
            if r["field_matches"].get(k, False):
                field_counts[k] += 1

    field_accuracies = {k: round(v / total, 4) for k, v in field_counts.items()}
    avg_field_accuracy = round(sum(field_accuracies.values()) / len(field_accuracies), 4)

    return {
        "total_samples": total,
        "json_validity_rate": round(valid_json_count / total, 4),
        "exact_match_rate": round(exact_match_count / total, 4),
        "avg_field_accuracy": avg_field_accuracy,
        "field_accuracies": field_accuracies,
    }

def evaluate_predictions(
    raw_predictions: List[str],
    ground_truths: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates parallel lists of model outputs and ground truth dicts."""
    sample_results = []
    for pred, gt in zip(raw_predictions, ground_truths):
        res = evaluate_single_sample(pred, gt)
        sample_results.append(res)
    metrics = compute_extraction_metrics(sample_results)
    return metrics, sample_results
