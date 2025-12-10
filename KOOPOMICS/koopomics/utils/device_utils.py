from koopomics.utils import torch, np, wandb

def resolve_device(device: str = "auto") -> str:
    """Resolve a device string like 'auto', 'cpu', or 'cuda:1'."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
