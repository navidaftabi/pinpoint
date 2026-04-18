
from typing import Optional
import torch
from torch.utils.data import Dataset

class VectorDataset(Dataset):
    """For single-step models (AE/VAE, classifier inputs already prepared)."""
    def __init__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        self.X = X.float()
        self.Y = Y.long() if Y is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.Y is None:
            return x
        return x, self.Y[idx]

class SequenceDataset(Dataset):
    """For temporal models that predict next latent given a window."""
    def __init__(self, X_seq: torch.Tensor, Y_next: torch.Tensor):
        # X_seq: [N, T, D], Y_next: [N, D]
        self.X = X_seq.float()
        self.Y = Y_next.float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
