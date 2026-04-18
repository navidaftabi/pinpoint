
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.data_utils import VectorDataset, SequenceDataset
from models.trainer import Trainer, TrainConfig
from models.models import MODEL_REGISTRY

def _load_tensor(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        for k in ("tensor", "X", "data", "array"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        raise ValueError(f"Unsupported torch object structure in {path}.")
    elif ext == ".npy":
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .pt/.pth or .npy.")

def build_dataloaders(cfg: dict):
    dcfg = cfg.get("data", {})
    dtype = dcfg.get("type", "vector").lower()
    bs = int(dcfg.get("batch_size",  64))

    train_X = _load_tensor(dcfg["train_X"])
    train_Y = _load_tensor(dcfg["train_Y"]) if "train_Y" in dcfg and dcfg["train_Y"] else None
    val_X   = _load_tensor(dcfg["test_X"])   if "test_X" in dcfg and dcfg["test_X"] else None
    val_Y   = _load_tensor(dcfg["test_Y"])   if "test_Y" in dcfg and dcfg["test_Y"] else None

    if dtype == "vector":
        train_ds = VectorDataset(train_X, train_Y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = None
        if val_X is not None:
            val_ds = VectorDataset(val_X, val_Y)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        return train_loader, val_loader

    elif dtype == "sequence":
        if train_Y is None:
            raise ValueError("Sequence data requires train_Y (next-step targets).")
        train_ds = SequenceDataset(train_X, train_Y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = None
        if val_X is not None and val_Y is not None:
            val_ds = SequenceDataset(val_X, val_Y)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown data.type '{dtype}'.")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to training config file (YAML)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", None)
    if task not in MODEL_REGISTRY:
        raise ValueError(f"Model '{task}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    device = cfg.get("device", "auto")
    if device == "auto":
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    model_kwargs = cfg["model"].get("kwargs", {})
    train_cfg = TrainConfig(
        task = task,
        model_kwargs = model_kwargs,
        epochs = int(cfg.get("train", {}).get("epochs", 100)),
        batch_size = int(cfg.get("train", {}).get("batch_size", 64)),
        lr = float(cfg.get("train", {}).get("lr", 1e-3)),
        weight_decay = float(cfg.get("train", {}).get("weight_decay", 5e-5)),
        device = device,
        ckpt_path = cfg.get("train", {}).get("ckpt_path", None),
        beta_vae = float(cfg.get("train", {}).get("beta_vae", 1.0)),
        grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0)),
        patience = int(cfg.get("train", {}).get("patience", 30)), 
        plot_dir = cfg.get("train", {}).get("plot_dir", "img"),
        plt_name = cfg.get("train", {}).get("plt_name", "learning_curve.png")
    )

    train_loader, val_loader = build_dataloaders(cfg)

    trainer = Trainer(train_cfg)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()