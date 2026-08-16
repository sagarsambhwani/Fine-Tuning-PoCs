"""
PEFT / LoRA Configuration & Parameter Inspector
Implements low-rank adapter configuration for Causal Language Models.
Math: W' = W + (alpha / r) * B @ A
"""
from typing import List, Optional, Tuple, Dict, Any
import torch.nn as nn

try:
    from peft import LoraConfig, TaskType
except ImportError:
    LoraConfig = None
    TaskType = None

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> Any:
    """Instantiates a PEFT LoraConfig object."""
    if LoraConfig is None:
        raise ImportError("PEFT library is required. Install via `pip install peft`.")

    t_modules = target_modules or DEFAULT_TARGET_MODULES
    peft_task = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=t_modules,
        bias=bias,
        task_type=peft_task
    )

def calculate_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculates total parameters, trainable parameters, and trainable ratio percentage."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_percent = 100 * trainable_params / all_param if all_param > 0 else 0.0
    return {
        "trainable_params": trainable_params,
        "all_params": all_param,
        "trainable_percent": trainable_percent,
        "frozen_params": all_param - trainable_params
    }

def print_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Prints a clean summary of trainable vs frozen parameters."""
    stats = calculate_trainable_parameters(model)
    border = "-" * 55
    print(border)
    print(" 📊 PARAMETER EFFICIENCY REPORT (LoRA/PEFT)")
    print(border)
    print(f" Total Parameters      : {stats['all_params']:,}")
    print(f" Trainable Parameters  : {stats['trainable_params']:,}")
    print(f" Frozen Parameters     : {stats['frozen_params']:,}")
    print(f" Trainable Ratio       : {stats['trainable_percent']:.4f}%")
    print(border)
    return stats
