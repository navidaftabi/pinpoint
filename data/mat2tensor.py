import os, re, random
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models.models import MODEL_REGISTRY

import yaml
from types import SimpleNamespace

def _load_attack_noise_lib(attack2noise_dir, var_name, C):
    noise = {}
    for filename in os.listdir(attack2noise_dir):
        if not filename.endswith(".mat"):
            continue
        # extract a number from the filename to use as attack type key
        m = re.findall(r"\d+", filename)
        if not m:
            continue
        att_type = int(m[0])
        arr = np.array(loadmat(os.path.join(attack2noise_dir, filename))[var_name])
        assert arr.ndim == 2 and arr.shape[1] == C, f"Noise shape mismatch in {filename}"
        noise[att_type] = arr
    return dict(sorted(noise.items()))

def _contaminate_sensor(X0, noise_percent, noise_lib, norm_operation_window):
    X = X0.copy()
    T, C = X.shape
    noise_size = int(T * noise_percent)
    if noise_size == 0:
        return X

    assert 0 <= norm_operation_window < T, "norm_operation_window out of range"
    noise_time = np.random.choice(np.arange(norm_operation_window, T), size=noise_size, replace=False)

    pick_types = np.random.choice(list(noise_lib.keys()), size=noise_size, replace=True).tolist()
    random.shuffle(pick_types)

    rows = []
    for _type in pick_types:
        lib = noise_lib[_type]
        ridx = random.randrange(min(int(2*T/3), lib.shape[0]))
        rows.append(lib[ridx:ridx+1, :])  # [1, C]
    noises = np.concatenate(rows, axis=0) if rows else None

    if noises is not None:
        assert noises.shape == (noise_size, C)
        X[noise_time, :] = noises
    return X

def mat2spatial(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)

    data = loadmat(cfg.data_path)
    X0 = np.array(data[cfg.var_name])  # [T, C]
    assert X0.ndim == 2, "Expected 2D [T, C] matrix"
    T, C = X0.shape

    noise_lib = _load_attack_noise_lib(cfg.attack2noise_dir, cfg.var_name, C)
    train_size = int(T * cfg.train_percent)

    for noise_percent in cfg.noise_percents:
        X = _contaminate_sensor(X0, noise_percent, noise_lib, cfg.norm_operation_window)
        X_train, X_test = X[:train_size], X[train_size:]
        torch.save(torch.from_numpy(X_train).float(), os.path.join(cfg.out_dir, f'X_train_{int(noise_percent*100)}%.pt'))
        torch.save(torch.from_numpy(X_test).float(),  os.path.join(cfg.out_dir, f'X_test_{int(noise_percent*100)}%.pt'))

def _load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MODEL_REGISTRY[ckpt["task"]](**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

@torch.no_grad()
def _encode_latents(vae, X_np, batch_size=1024, device="cpu"):
    X = torch.from_numpy(X_np).float()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            mu, logvar = vae.encode(xb)
            # z = mu 
            z = vae.reparameterize(mu, logvar)
            outs.append(z.cpu())
    Z = torch.cat(outs, dim=0).numpy()
    return Z

def _make_windows(Z, win):
    T, D = Z.shape
    if win >= T:
        raise ValueError(f"Window size {win} must be < T={T}")
    N = T - win
    X_seq = np.stack([Z[t:t+win, :] for t in range(N)], axis=0)
    y_next = Z[win:, :]  # next latent
    return X_seq, y_next

def mat2temporal(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = getattr(cfg, "device", "cpu")
    win = int(cfg.window)
    bs_enc = int(getattr(cfg, "batch_size_encode", 256))
    train_percent = float(cfg.train_percent)

    base = loadmat(cfg.data_path)
    X_clean = np.array(base[cfg.var_name])  # [T, C]
    assert X_clean.ndim == 2, "Expected [T, C] matrix"
    T, C = X_clean.shape

    for noise_percent in cfg.noise_percents:
        noise_lib = _load_attack_noise_lib(cfg.attack2noise_dir, cfg.var_name, C)
        X_use = _contaminate_sensor(
            X0=X_clean,
            noise_percent=noise_percent,
            noise_lib=noise_lib,
            norm_operation_window=cfg.norm_operation_window,
        )
        vae = _load_model_from_ckpt(os.path.join(cfg.ckpt_dir, f'vae_{int(noise_percent*100)}%.pt'), device)
        Z = _encode_latents(vae, X_use, batch_size=bs_enc, device=device)  # [T, D_latent]

        X_seq, y_next = _make_windows(Z, win)  # [N, win, D], [N, D]
        N = X_seq.shape[0]
        split = int(N * train_percent)
        Xtr, Ytr = X_seq[:split], y_next[:split]
        Xte, Yte = X_seq[split:], y_next[split:]

        # save
        torch.save(torch.from_numpy(Xtr).float(), os.path.join(cfg.out_dir, f"X_train_{int(noise_percent*100)}%.pt"))
        torch.save(torch.from_numpy(Ytr).float(), os.path.join(cfg.out_dir, f"y_train_{int(noise_percent*100)}%.pt"))
        torch.save(torch.from_numpy(Xte).float(), os.path.join(cfg.out_dir, f"X_test_{int(noise_percent*100)}%.pt"))
        torch.save(torch.from_numpy(Yte).float(), os.path.join(cfg.out_dir, f"y_test_{int(noise_percent*100)}%.pt"))

@torch.no_grad()
def _spatial_reconstruct(model, X_np, bs=512, device="cpu"):
    X = torch.from_numpy(X_np).float()
    loader = DataLoader(TensorDataset(X), batch_size=bs, shuffle=False)
    outs, zs = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        if hasattr(model, "encode") and hasattr(model, "decode"):
            enc_out = model.encode(xb)
            if isinstance(enc_out, tuple) and len(enc_out) == 2:
                mu, logvar = enc_out
                z = mu if not hasattr(model, "reparameterize") else model.reparameterize(mu, logvar)
            else:
                z = enc_out
            xrec = model.decode(z)
        else:
            out = model(xb)
            if isinstance(out, tuple):
                xrec, z = out[0], out[1]
            else:
                xrec, z = out, None
        if z is not None:
            zs.append(z.detach().cpu())
        outs.append(xrec.detach().cpu())
    return torch.cat(outs, dim=0).numpy(), torch.cat(zs, dim=0).numpy() if zs else None

@torch.no_grad()
def _txf_predict_next(vae, txf, Z, win, bs=256, device="cpu"):
    X_seq, _ = _make_windows(Z, win)                            # [N, win, D]
    X_seq = torch.from_numpy(X_seq).float().to(device)             # [N, win, D]
    z_next = txf(X_seq)                                            # [N, D]
    y_pred = vae.decode(z_next).cpu().numpy()                        # [N, C]
    return y_pred

def get_data(stacks, labels, train_percent: float):
    X_train, X_test, y_train, y_test = [], [], [], []
    for v, y_i in zip(stacks, labels):
        n_i = v.shape[0]
        split_i = int(n_i * train_percent)
        X_train.append(v[:split_i])
        X_test.append(v[split_i:])
        y_train.append(y_i[:split_i])
        y_test.append(y_i[split_i:])
    X = {
        "train": torch.cat(X_train, dim=0).float(),
        "test":  torch.cat(X_test, dim=0).float(),
    }
    y = {
        "train": torch.cat(y_train, dim=0).long(),
        "test":  torch.cat(y_test, dim=0).long(),
    }
    return X, y

@torch.no_grad()
def mat2pinpoint(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = getattr(cfg, "device", "cpu")
    win = int(cfg.window)
    bs = int(getattr(cfg, "batch_size_encode", 256))
    train_percent = float(cfg.train_percent)
    include_se = bool(getattr(cfg, "include_se", True))

    vae = _load_model_from_ckpt(cfg.ckpt_sp, device)
    txf = _load_model_from_ckpt(cfg.ckpt_tp, device)

    normal_mat = loadmat(cfg.normal_data_path)
    X_norm = np.array(normal_mat[cfg.var_x])       # [T, C]
    Res_norm = np.array(normal_mat[cfg.var_res])   # [T, C]
    Tn, C = X_norm.shape
    assert Res_norm.shape == (Tn, C)

    X_dict, Res_dict = {}, {}
    for filename in os.listdir(cfg.attack_dir):
        if not filename.endswith(".mat"):
            continue
        att_type_m = re.findall(r"\d+", filename)
        if not att_type_m:
            continue
        att = int(att_type_m[0])
        mat = loadmat(os.path.join(cfg.attack_dir, filename))
        X_dict[att] = np.array(mat[cfg.var_x])        # [T, C]
        Res_dict[att] = np.array(mat[cfg.var_res])    # [T, C]
    
    classes = [0] + sorted(X_dict.keys())
    stacks = []
    labels = []
    for k in classes:
        if k == 0:
            X, RES = X_norm, Res_norm
        else:
            X, RES = X_dict[k], Res_dict[k]
        T, C2 = X.shape
        assert C2 == C and RES.shape == (T, C)

        if include_se:
            r1 = RES[win:, :]           # [T-win, C]
        R = []
        for _ in range(cfg.num_samples):
            X_vae, Z = _spatial_reconstruct(vae, X, bs=bs, device=device)          # [T, C]
            r2 = (X_vae - X)[win:, :]                                          # [T-win, C]
            X_txf = _txf_predict_next(vae, txf, Z, win, bs=bs, device=device)  # [T-win, C]
            r3 = (X_txf - X[win:, :])                                          # [T-win, C]
            comps = [r2, r3] if not include_se else [r1, r2, r3]
            R.append(torch.from_numpy(np.stack(comps, axis=1)))           # [T-win, K, C]
        stack = torch.stack(R, dim=1).float()                                  # [T-win, M, K, C]
        stacks.append(stack)
        labels.append(torch.full((stack.shape[0],), k, dtype=torch.long))
    
    X, y = get_data(stacks, labels, train_percent)

    # Save
    torch.save(X["train"], os.path.join(cfg.out_dir, "X_train.pt"))
    torch.save(y['train'], os.path.join(cfg.out_dir, "y_train.pt"))
    torch.save(X['test'],  os.path.join(cfg.out_dir, "X_test.pt"))
    torch.save(y['test'],  os.path.join(cfg.out_dir, "y_test.pt"))

@torch.no_grad()
def mat2pinpoint_eval(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = getattr(cfg, "device", "cpu")
    win = int(cfg.window)
    bs = int(getattr(cfg, "batch_size_encode", 256))
    include_se = bool(getattr(cfg, "include_se", True))

    severity = int(cfg.severity) 
    noise_level = int(cfg.noise_level)

    vae = _load_model_from_ckpt(cfg.ckpt_sp, device)
    txf = _load_model_from_ckpt(cfg.ckpt_tp, device)

    normal_mat = loadmat(cfg.normal_data_path)
    X_norm = np.array(normal_mat[cfg.var_x])       # [T, C]
    Res_norm = np.array(normal_mat[cfg.var_res])   # [T, C]
    Tn, C = X_norm.shape
    assert Res_norm.shape == (Tn, C)

    X_dict, Res_dict = {}, {}
    for filename in os.listdir(cfg.attack_dir):
        if not filename.endswith(".mat"):
            continue
        m = re.findall(r"\d+", filename)
        if not m:
            continue
        att = int(m[0])
        mat = loadmat(os.path.join(cfg.attack_dir, filename))
        X_dict[att]  = np.array(mat[cfg.var_x])     # [T, C]
        Res_dict[att]= np.array(mat[cfg.var_res])   # [T, C]

    classes = [0] + sorted(X_dict.keys())

    stacks = []
    labels = []
    for k in classes:
        if k == 0:
            X, RES = X_norm, Res_norm
        else:
            X, RES = X_dict[k], Res_dict[k]
        T, C2 = X.shape
        assert C2 == C and RES.shape == (T, C)

        if include_se:
            r1 = RES[win:, :]  # [T-win, C]

        R = []
        for _ in range(cfg.num_samples):
            X_vae, Z = _spatial_reconstruct(vae, X, bs=bs, device=device)  # [T,C], [T,H]
            r2 = (X_vae - X)[win:, :]                                  # [T-win, C]
            X_txf = _txf_predict_next(vae, txf, Z, win, bs=bs, device=device)  # [T-win, C]
            r3 = (X_txf - X[win:, :])                                   # [T-win, C]
            comps = [r2, r3] if not include_se else [r1, r2, r3]
            R.append(torch.from_numpy(np.stack(comps, axis=1)))  # [T-win, K, C]

        stack = torch.stack(R, dim=1).float()  # [T-win, M, K, C]
        stacks.append(stack)
        labels.append(torch.full((stack.shape[0],), k, dtype=torch.long))

    X = torch.cat(stacks, dim=0).float()  # [sum_i (T_i-win), M, K, C]
    y = torch.cat(labels, dim=0).long()   # [sum_i (T_i-win)]

    torch.save(X, os.path.join(cfg.out_dir, f"X_I{severity}_nl{noise_level}.pt"))
    torch.save(y, os.path.join(cfg.out_dir, f"y_I{severity}_nl{noise_level}.pt"))

@torch.no_grad()
def mat2ae(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = getattr(cfg, "device", "cpu")
    bs = int(getattr(cfg, "batch_size_encode", 256))
    train_percent = float(cfg.train_percent)

    ae_ckpt = getattr(cfg, "ckpt_ae", None)
    assert ae_ckpt is not None, "Provide cfg.ckpt_ae (or reuse cfg.ckpt_sp) for AE baseline."
    ae = _load_model_from_ckpt(ae_ckpt, device)

    normal_mat = loadmat(cfg.normal_data_path)
    X_norm = np.array(normal_mat[cfg.var_x])       # [T, C]
    _, C = X_norm.shape

    X_dict = {}
    for filename in os.listdir(cfg.attack_dir):
        if not filename.endswith(".mat"):
            continue
        m = re.findall(r"\d+", filename)
        if not m:
            continue
        att = int(m[0])
        mat = loadmat(os.path.join(cfg.attack_dir, filename))
        X_dict[att]  = np.array(mat[cfg.var_x])     # [T, C]

    classes = [0] + sorted(X_dict.keys())
    stacks, labels = [], []
    for k in classes:
        X = X_norm if k == 0 else X_dict[k]
        T, C2 = X.shape
        assert C2 == C

        X_ae, _ = _spatial_reconstruct(ae, X, bs=bs, device=device)  # [T, C]
        R = X_ae - X                                                 # [T, C]
        stacks.append(torch.from_numpy(R).float())
        labels.append(torch.full((R.shape[0],), k, dtype=torch.long))

    X, y = get_data(stacks, labels, train_percent)
    torch.save(X["train"], os.path.join(cfg.out_dir, "X_train.pt"))
    torch.save(y['train'], os.path.join(cfg.out_dir, "y_train.pt"))
    torch.save(X['test'],  os.path.join(cfg.out_dir, "X_test.pt"))
    torch.save(y['test'],  os.path.join(cfg.out_dir, "y_test.pt"))


@torch.no_grad()
def mat2ae_eval(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = getattr(cfg, "device", "cpu")
    bs = int(getattr(cfg, "batch_size_encode", 256))

    severity = int(cfg.severity)
    noise_level = int(cfg.noise_level)

    ae_ckpt = getattr(cfg, "ckpt_ae", None)
    assert ae_ckpt is not None, "Provide cfg.ckpt_ae (or reuse cfg.ckpt_sp) for AE baseline."
    ae = _load_model_from_ckpt(ae_ckpt, device)

    normal_mat = loadmat(cfg.normal_data_path)
    X_norm = np.array(normal_mat[cfg.var_x])       # [T, C]
    _, C = X_norm.shape

    X_dict = {}
    for filename in os.listdir(cfg.attack_dir):
        if not filename.endswith(".mat"):
            continue
        m = re.findall(r"\d+", filename)
        if not m:
            continue
        att = int(m[0])
        mat = loadmat(os.path.join(cfg.attack_dir, filename))
        X_dict[att]  = np.array(mat[cfg.var_x])     # [T, C]

    classes = [0] + sorted(X_dict.keys())
    stacks, labels = [], []
    for k in classes:
        X = X_norm if k == 0 else X_dict[k]
        T, C2 = X.shape
        assert C2 == C

        X_ae, _ = _spatial_reconstruct(ae, X, bs=bs, device=device)  # [T, C]
        R = X_ae - X
        stacks.append(torch.from_numpy(R).float())
        labels.append(torch.full((R.shape[0],), k, dtype=torch.long))

    X = torch.cat(stacks, dim=0).float()  # [sum_i (T_i - win), M, 1, C]
    y = torch.cat(labels, dim=0).long()
    torch.save(X, os.path.join(cfg.out_dir, f"X_I{severity}_nl{noise_level}.pt"))
    torch.save(y, os.path.join(cfg.out_dir, f"y_I{severity}_nl{noise_level}.pt"))


def _dict_to_ns(d):
    return SimpleNamespace(**d)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spatial", "temporal", "pinpoint", "pinpoint_eval", "ae", "ae_eval"], required=True)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = _dict_to_ns(cfg_dict)

    if args.mode == "spatial":
        mat2spatial(cfg)
    elif args.mode == "temporal":
        mat2temporal(cfg)
    elif args.mode == "pinpoint":
        mat2pinpoint(cfg)
    elif args.mode == "pinpoint_eval":
        mat2pinpoint_eval(cfg)
    elif args.mode == "ae":
        mat2ae(cfg)
    elif args.mode == "ae_eval":
        mat2ae_eval(cfg)
    else:
        raise ValueError("data can be constructed for `spatial`, `temporal`, `pinpoint`, `pinpoint_eval`, `ae`, or `ae_eval`.")

if __name__ == "__main__":
    main()
