"""
High-Performance Structured Inference Engine
Loads model once and provides fast, deterministic structured JSON extraction with latency profiling.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from src.data.formatter import format_chat_prompt, apply_template_to_messages
from src.data.validator import parse_and_validate_extraction
from src.utils.logger import setup_logger

logger = setup_logger("inference_engine")

class StructuredInferenceEngine:
    """Manages model lifecycle and JSON extraction generation."""

    def __init__(
        self,
        model_path_or_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        self.model_path_or_id = model_path_or_id
        self.adapter_path = adapter_path
        
        # Resolve device & precision
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        logger.info(f"Initializing Inference Engine on device={self.device}, dtype={self.torch_dtype}")
        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer from {self.model_path_or_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        logger.info(f"Loading weights from {self.model_path_or_id}...")
        base = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            base = base.to("cpu")

        if self.adapter_path and Path(self.adapter_path).exists():
            logger.info(f"Loading PEFT LoRA adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            self.model = base

        self.model.eval()
        logger.info("Model loaded successfully into memory.")

    def extract(
        self,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Extracts structured JSON from input text.
        Returns:
            {
                "data": parsed_json_dict,
                "raw_response": completion_str,
                "is_valid_json": bool,
                "error": Optional[str],
                "latency_ms": float,
                "model_name": str
            }
        """
        start_t = time.perf_counter()
        
        messages = format_chat_prompt(text)
        prompt_text = apply_template_to_messages(messages, self.tokenizer, add_generation_prompt=True)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][input_len:]
        completion = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        is_valid, parsed_dict, err = parse_and_validate_extraction(completion)

        return {
            "data": parsed_dict,
            "raw_response": completion,
            "is_valid_json": is_valid,
            "error": err,
            "latency_ms": elapsed_ms,
            "model_name": str(self.model_path_or_id)
        }
