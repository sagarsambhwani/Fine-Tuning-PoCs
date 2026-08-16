"""
Environment & Hardware Detector
Inspects runtime hardware (GPU name, VRAM, CUDA) and package versions.
Works across Google Colab, Linux servers, and local environments.
"""
import sys
import torch

def detect_environment() -> dict:
    """Collects comprehensive hardware and library metadata."""
    env_info = {
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
    }

    # Safe imports for LLM ecosystem packages
    packages = ["transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            env_info[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            env_info[f"{pkg}_version"] = "Not Installed"

    return env_info

def print_environment_info() -> dict:
    """Prints formatted environment report and returns the dictionary."""
    info = detect_environment()
    border = "=" * 60
    print(border)
    print(" 🚀 RUNTIME & HARDWARE ENVIRONMENT REPORT")
    print(border)
    print(f" Python Version        : {info['python_version']}")
    print(f" PyTorch Version       : {info['pytorch_version']}")
    print(f" CUDA Available        : {info['cuda_available']}")
    print(f" CUDA Version          : {info['cuda_version']}")
    print(f" GPU Device Name       : {info['gpu_name']}")
    print(f" GPU Total VRAM        : {info['gpu_memory_gb']} GB")
    print("-" * 60)
    print(f" Transformers Version  : {info.get('transformers_version', 'N/A')}")
    print(f" PEFT Version          : {info.get('peft_version', 'N/A')}")
    print(f" TRL Version           : {info.get('trl_version', 'N/A')}")
    print(f" BitsAndBytes Version  : {info.get('bitsandbytes_version', 'N/A')}")
    print(f" Accelerate Version    : {info.get('accelerate_version', 'N/A')}")
    print(f" Datasets Version      : {info.get('datasets_version', 'N/A')}")
    print(border)

    # Dynamic recommendation
    if info["cuda_available"]:
        vram = info["gpu_memory_gb"]
        if vram >= 14.0:
            print(f"💡 [Recommendation] Detected ~{vram}GB VRAM. Ideal for QLoRA 1.5B/3B or standard LoRA 1.5B with batch size 2-4 and gradient accumulation.")
        else:
            print(f"⚠️ [Recommendation] Detected lower VRAM ({vram}GB). Use QLoRA 4-bit, batch size 1-2, gradient accumulation 8, and gradient checkpointing.")
    else:
        print("⚠️ [Recommendation] No CUDA GPU detected. Running in CPU mode. Suitable for data prep, unit tests, and CPU quantized inference.")
    print(border + "\n")
    return info

if __name__ == "__main__":
    print_environment_info()
