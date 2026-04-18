from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.models import MODEL_REGISTRY
from models.earlystopping import EarlyStopping
import matplotlib.pyplot as plt

def plot_learning(mse, val_mse, save_path, plt_name):
    os.makedirs(save_path, exist_ok=True)
    fig, axs = plt.subplots(1, figsize=(6,4))
    axs.plot(mse, color='r', label='Training')
    axs.plot(val_mse, color='b', label='Validation')
    axs.set_title('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, plt_name + "_lc.pdf"), dpi=1200)
    plt.show()

def plot_learning_acc(acc, val_acc, save_path, plt_name):
    os.makedirs(save_path, exist_ok=True)
    fig, axs = plt.subplots(1, figsize=(6,4))
    axs.plot(acc, color='r', label='Training')
    axs.plot(val_acc, color='b', label='Validation')
    axs.set_title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, plt_name + "_acc.pdf"), dpi=1200)
    plt.show()

@dataclass
class TrainConfig:
    task: str 
    model_kwargs: Dict[str, Any]
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-1
    scheduler: bool = True
    weight_decay: float = 5e-5
    device: str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path: Optional[str] = None
    beta_vae: float = 1.0 
    grad_clip: Optional[float] = 1.0
    patience: int = 30 
    plot_dir: str = 'img'
    plt_name: str = 'model'

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.model = MODEL_REGISTRY[cfg.task](**cfg.model_kwargs).to(self.device)
        self._has_acc = hasattr(self.model, "accuracy") and callable(self.model.accuracy)
        if cfg.task in ["pinpoint", "dnn_ae"]:
            self.opt = optim.RMSprop(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sch_lr = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 
                                                           mode='min', 
                                                           factor=0.5, 
                                                           patience=5, 
                                                           min_lr=5e-5,) if cfg.scheduler else None
        self.es = EarlyStopping(patience=cfg.patience, 
                                path=cfg.ckpt_path, verbose=True, 
                                payload={
                                    "task": cfg.task,
                                    "model_kwargs": cfg.model_kwargs,
                                    "train_cfg": asdict(cfg),
                                }
                                ) if cfg.ckpt_path else None

    def _move_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(b.to(self.device) for b in batch)
        return batch.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        mse, val_mse = [], []
        acc, val_acc = [], []
        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_accuracy = self._run_epoch(train_loader, train=True)
            if val_loader is not None:
                val_loss, val_accuracy = self._run_epoch(val_loader, train=False) 
            else:
                val_loss, val_accuracy = tr_loss, tr_accuracy

            if self.sch_lr is not None:
                self.sch_lr.step(val_loss)
            if self._has_acc:
                print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} "
                      f"| train_acc={tr_accuracy:.4f} | val_acc={val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={val_loss:.6f}")
            mse.append(tr_loss)
            val_mse.append(val_loss)
            if self._has_acc:
                acc.append(tr_accuracy)
                val_acc.append(val_accuracy)
            self.es(val_loss=val_loss,
                    model=self.model,
                    optimizer=self.opt,
                    epoch=epoch)
            if self.es.early_stop:
                print("Early stopping!")
                break
        
        if self.cfg.ckpt_path and os.path.isfile(self.cfg.ckpt_path):
            ckpt = torch.load(self.cfg.ckpt_path, map_location=self.device)
            self.model = MODEL_REGISTRY[ckpt["task"]](**ckpt["model_kwargs"]).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])

        plot_learning(mse, val_mse, self.cfg.plot_dir, self.cfg.plt_name)
        if self._has_acc:
            plot_learning_acc(acc, val_acc, self.cfg.plot_dir, self.cfg.plt_name)
        return self.model, mse, val_mse, (acc if self._has_acc else None), (val_acc if self._has_acc else None)

    def _run_epoch(self, loader: Optional[DataLoader], train: bool):
        if loader is None:
            return 0.0
        self.model.train(mode=train)
        total_loss, total_acc, n = 0.0, 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                if train:
                    self.opt.zero_grad()

                batch = self._move_batch(batch)
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x, y = batch, batch

                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    print("[NaNGuard] non-finite in input/target",
                        f"x_finite={torch.isfinite(x).all().item()} y_finite={torch.isfinite(y).all().item()}",
                        f"x_range=({x.min().item():.3e},{x.max().item():.3e})",
                        f"y_range=({y.min().item():.3e},{y.max().item():.3e})")
                    raise RuntimeError("Non-finite data")

                output = self.model(x)

                if isinstance(output, (tuple, list)):
                    tensors = [t for t in output if isinstance(t, torch.Tensor)]
                else:
                    tensors = [output]
                for i, t in enumerate(tensors):
                    if not torch.isfinite(t).all():
                        print(f"[NaNGuard] non-finite in model output tensor #{i}",
                            f"range=({t.min().item():.3e},{t.max().item():.3e})")
                        raise RuntimeError("Non-finite model output")
                    
                loss = self.model.loss_function(output, y, beta=self.cfg.beta_vae)
                if not torch.isfinite(loss):
                    print("[NaNGuard] non-finite loss",
                        f"loss={loss}", f"x_range=({x.min().item():.3e},{x.max().item():.3e})")
                    raise RuntimeError("Non-finite loss")
            
                acc_val = None
                if self._has_acc:
                    acc_val = self.model.accuracy(output, y)
                    if torch.is_tensor(acc_val):
                        acc_val = float(acc_val.detach().cpu().item())
                    else:
                        acc_val = float(acc_val)

                if train:
                    loss.backward()
                    if self.cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.opt.step()

                bs = x.shape[0]
                total_loss += loss.item() * bs
                if acc_val is not None:
                    total_acc += acc_val * bs
                n += bs
        mean_loss = total_loss / max(n, 1)
        mean_acc = (total_acc / max(n, 1)) if self._has_acc else 0.0
        return mean_loss, mean_acc
