"""
QLoRA (4-bit NormalFloat) Quantization Configuration
Configures bitsandbytes 4-bit NF4 loading with double quantization and compute dtype.
"""
import torch
from typing import Any, Optional

try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
except ImportError:
    BitsAndBytesConfig = None
    prepare_model_for_kbit_training = None

def get_qlora_bnb_config(
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
    quant_type: str = "nf4"
) -> Any:
    """
    Creates a BitsAndBytesConfig for 4-bit QLoRA.
    - quant_type='nf4': NormalFloat4 for normal distribution of weights.
    - use_double_quant=True: Quantizes the quantization constants, saving ~0.37 bits/param.
    - compute_dtype: torch.float16 for T4, or torch.bfloat16 for Ampere/Ada/Hopper.
    """
    if BitsAndBytesConfig is None:
        raise ImportError("Transformers with bitsandbytes support is required.")

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_compute_dtype = dtype_map.get(compute_dtype.lower(), torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch_compute_dtype
    )

def prepare_qlora_model(
    model: Any,
    use_gradient_checkpointing: bool = True
) -> Any:
    """Prepares 4-bit quantized base model for adapter training with gradient checkpointing."""
    if prepare_model_for_kbit_training is None:
        raise ImportError("PEFT is required to prepare model for k-bit training.")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    return model
