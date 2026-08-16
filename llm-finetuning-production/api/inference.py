"""
API Model Manager & Inference Lifecycle
Loads weights once at application startup.
"""
import os
from typing import Optional
from src.inference.engine import StructuredInferenceEngine
from src.utils.logger import setup_logger

logger = setup_logger("api_inference_manager")

# Global singleton
_inference_engine: Optional[StructuredInferenceEngine] = None

def get_model_path() -> str:
    """Resolves model path from environment variable or default."""
    # Priority: Merged directory -> Adapter fallback -> Base Model
    merged_path = os.getenv("MODEL_DIR", "models/merged/qwen-1.5b-order-extractor")
    if os.path.exists(merged_path):
        return merged_path
    
    adapter_path = "models/adapters/qwen-1.5b-order-extractor"
    if os.path.exists(adapter_path):
        return "Qwen/Qwen2.5-1.5B-Instruct"
    
    return os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

def initialize_model(model_path: Optional[str] = None) -> StructuredInferenceEngine:
    """Initializes and returns the global inference engine."""
    global _inference_engine
    if _inference_engine is None:
        target_model = model_path or get_model_path()
        adapter_path = "models/adapters/qwen-1.5b-order-extractor" if not os.path.exists("models/merged/qwen-1.5b-order-extractor") and os.path.exists("models/adapters/qwen-1.5b-order-extractor") else None
        device = os.getenv("DEVICE", None)
        
        logger.info(f"Loading model into inference engine: {target_model} (adapter: {adapter_path}, device: {device})")
        _inference_engine = StructuredInferenceEngine(
            model_path_or_id=target_model,
            adapter_path=adapter_path,
            device=device
        )
    return _inference_engine

def get_inference_engine() -> StructuredInferenceEngine:
    """Returns the initialized inference engine."""
    global _inference_engine
    if _inference_engine is None:
        return initialize_model()
    return _inference_engine

def shutdown_model():
    """Releases model memory upon server shutdown."""
    global _inference_engine
    if _inference_engine is not None:
        logger.info("Unloading inference engine...")
        _inference_engine = None
