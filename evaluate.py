import os
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.models import MODEL_REGISTRY


class EvalDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert isinstance(X, torch.Tensor), "X must be a torch.Tensor"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
        assert X.shape[0] == y.shape[0], f"Batch mismatch: {X.shape[0]} vs {y.shape[0]}"
        self.X, self.y = X, y.long()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _load_tensor(path: str) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        for k in ("tensor", "X", "y", "data", "array"):
            if isinstance(obj, dict) and k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        raise ValueError(f"Unsupported torch object structure in {path}")
    elif ext == ".npy":
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported file extension: {ext} for {path}")

def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    task = ckpt.get("task", "pinpoint")
    model_kwargs = ckpt.get("model_kwargs", {})
    model = MODEL_REGISTRY[task](**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, task, model_kwargs

@torch.no_grad()
def _reduce_mc(probs_m: torch.Tensor, how: str = "mean") -> torch.Tensor:
    if how == "mean":
        return probs_m.mean(dim=1)
    elif how == "max":
        return probs_m.max(dim=1).values
    else:
        raise ValueError(f"Unsupported mc_reduce: {how}")

@torch.no_grad()
def run_eval(model, loader, device: str, mc_reduce: str = "mean"):
    all_probs, all_preds, all_y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)

        # Case A: PINPOINT-style -> logits [B, M, K_classes]
        if logits.dim() == 3:
            probs_m = F.softmax(logits, dim=-1)      # [B, M, K]
            probs   = _reduce_mc(probs_m, mc_reduce) # [B, K]

        # Case B: AE classifier -> logits [B, K_classes]
        elif logits.dim() == 2:
            probs = F.softmax(logits, dim=-1)        # [B, K]

        else:
            raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}. "
                             f"Expected [B, K] (AE) or [B, M, K] (PINPOINT).")

        preds = probs.argmax(dim=-1)                 # [B]

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_y.append(yb.cpu())

    probs = torch.cat(all_probs, dim=0)              # [N, K]
    preds = torch.cat(all_preds, dim=0)              # [N]
    ytrue = torch.cat(all_y,   dim=0)               # [N]
    return probs, preds, ytrue

def save_confmat(y_true_np, y_pred_np, out_png, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels if labels is not None else None)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="YAML with ckpt path, data paths (X, y), batch_size, and output dir.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = cfg["model"]["ckpt_path"]
    model, task, model_kwargs = _load_model(ckpt_path, device)

    X_path = cfg["data"]["X"]
    y_path = cfg["data"]["y"]
    bs     = int(cfg["data"].get("batch_size", 64))
    X = _load_tensor(X_path)      # AE: [N, C] ; PINPOINT: [N, M, K_res, C] (model handles it)
    y = _load_tensor(y_path)      # [N]
    ds = EvalDataset(X, y)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)

    mc_reduce = cfg.get("out", {}).get("mc_reduce", "mean")   # "mean" or "max"
    probs, preds, ytrue = run_eval(model, dl, device, mc_reduce=mc_reduce)

    out_dir = cfg.get("out", {}).get("dir", "eval_out")
    os.makedirs(out_dir, exist_ok=True)

    num_classes = int(cfg.get("out", {}).get("num_classes", probs.shape[1]))
    labels = list(range(num_classes))
    target_names = cfg.get("out", {}).get("target_names", [str(i) for i in labels])

    torch.save(probs, os.path.join(out_dir, "probs.pt"))     # [N, K]
    torch.save(preds, os.path.join(out_dir, "preds.pt"))     # [N]
    torch.save(ytrue, os.path.join(out_dir, "y_true.pt"))    # [N]

    rep = classification_report(ytrue.numpy(), preds.numpy(),
                                labels=labels, target_names=target_names,
                                digits=6, zero_division=0, output_dict=True)
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    rep_txt = classification_report(ytrue.numpy(), preds.numpy(),
                                    labels=labels, target_names=target_names,
                                    digits=6, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(rep_txt + "\n")

    # Confusion matrix
    cm_title = cfg.get("out", {}).get("cm_title", "Confusion Matrix")
    save_confmat(ytrue.numpy(), preds.numpy(), os.path.join(out_dir, "confusion_matrix.png"),
                 labels=labels, title=cm_title)

    acc = (preds == ytrue).float().mean().item()
    print(f"[OK] Eval done. Acc={acc:.4f}  N={len(ytrue)}  Saved to: {out_dir}")

if __name__ == "__main__":
    main()