import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "best_ms_resnet18_fast.pth")

def load_model(model_path: str | None = None):
    if model_path is None:
        model_path = DEFAULT_WEIGHTS
    else:
        if not os.path.isabs(model_path):
            model_path = os.path.join(ROOT, model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model
