import torch
from pytorch_lightning import seed_everything

def setup_device():
    if torch.cuda.is_available():    
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")

def set_seed(seed=42):
    seed_everything(seed, workers=True)

