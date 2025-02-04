import torch

def to_tensor(data):
    """
    Converts a list or numpy array to a PyTorch tensor.
    """
    return torch.tensor(data, dtype=torch.float)