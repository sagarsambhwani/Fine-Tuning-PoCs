"""
Unit tests for evaluation metrics and inference output processing.
"""
import pytest
from src.evaluation.metrics import evaluate_single_sample, compute_extraction_metrics

def test_evaluate_single_sample_perfect():
    gt = {
        "customer": "John",
        "quantity": 3,
        "product": "laptops",
        "amount": 2400.0,
        "delivery_day": "Friday"
    }
    pred = '{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2400.0, "delivery_day": "Friday"}'
    
    result = evaluate_single_sample(pred, gt)
    assert result["is_valid_json"] is True
    assert result["exact_match"] is True
    assert all(result["field_matches"].values())

def test_evaluate_single_sample_partial_mismatch():
    gt = {
        "customer": "John",
        "quantity": 3,
        "product": "laptops",
        "amount": 2400.0,
        "delivery_day": "Friday"
    }
    # wrong amount (2000 vs 2400)
    pred = '{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2000.0, "delivery_day": "Friday"}'
    
    result = evaluate_single_sample(pred, gt)
    assert result["is_valid_json"] is True
    assert result["exact_match"] is False
    assert result["field_matches"]["customer"] is True
    assert result["field_matches"]["amount"] is False

def test_compute_extraction_metrics_aggregation():
    sample_results = [
        {
            "is_valid_json": True,
            "exact_match": True,
            "field_matches": {"customer": True, "quantity": True, "product": True, "amount": True, "delivery_day": True}
        },
        {
            "is_valid_json": True,
            "exact_match": False,
            "field_matches": {"customer": True, "quantity": True, "product": True, "amount": False, "delivery_day": True}
        },
        {
            "is_valid_json": False,
            "exact_match": False,
            "field_matches": {"customer": False, "quantity": False, "product": False, "amount": False, "delivery_day": False}
        }
    ]

    metrics = compute_extraction_metrics(sample_results)
    assert metrics["total_samples"] == 3
    assert abs(metrics["json_validity_rate"] - 0.6667) < 0.01
    assert abs(metrics["exact_match_rate"] - 0.3333) < 0.01
    assert metrics["field_accuracies"]["customer"] == round(2 / 3, 4)
    assert metrics["field_accuracies"]["amount"] == round(1 / 3, 4)
