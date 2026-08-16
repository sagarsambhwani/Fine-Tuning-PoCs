"""Inference engine and model merging utilities."""
from .engine import StructuredInferenceEngine
from .merger import merge_adapter_to_base

__all__ = ["StructuredInferenceEngine", "merge_adapter_to_base"]
