# Data preparation and tensor generation

This document describes how raw MATLAB files are converted into the PyTorch tensors used by PINPOINT and its baselines.

## Overview

The repository includes raw MATLAB `.mat` files under:

```text
data/matlab/
├── data.mat
├── Normal1.mat
├── Normal2.mat
├── I5/             # severity alpha=0.5 in paper
├── I6/             # severity alpha=0.4 in paper
├── I7/             # severity alpha=0.3 in paper
├── I8/             # severity alpha=0.2 in paper
└── I9/             # severity alpha=0.1 in paper
```

The severity folders (`I5`–`I9`) contain three attack classes each, with filenames such as:

- `Attack1-5.mat`
- `Attack2-5.mat`
- `Attack3-5.mat`

The tensor-construction pipeline is implemented in:

```bash
python -m data.mat2tensor --mode <mode> --config <config.yaml>
```

Supported modes are:

- `spatial`
- `temporal`
- `pinpoint`
- `pinpoint_eval`
- `ae`
- `ae_eval`

## Expected MATLAB variables

The YAML configs refer to these variable names inside the `.mat` files:

- `Z`: measurement matrix
- `Res`: state-estimation residual matrix

Which variables are used depends on the generation mode.

## Mode 1: spatial

This mode constructs datasets for training the **AE** and **VAE** on measurement snapshots.

### Example config

`data/configs/spatial.yaml`

```yaml
data_path: "data/matlab/data.mat"
var_name: "Z"
attack2noise_dir: "data/matlab/I7"
out_dir: "data/spatial"
train_percent: 0.8
noise_percents: [0.0, 0.05, 0.10, 0.20, 0.50] # corruption level
norm_operation_window: 50                     # after `t=50` noise will be added
```

### Output

This creates files like:

```text
data/spatial/
├── X_train_0%.pt
├── X_test_0%.pt
├── X_train_5%.pt
├── X_test_5%.pt
...
└── X_train_50%.pt / X_test_50%.pt
```

### Command

```bash
python -m data.mat2tensor --mode spatial --config data/configs/spatial.yaml
```

## Mode 2: temporal

This mode constructs latent temporal datasets for training the **RNN** and **Transformer**. It uses a trained VAE checkpoint to encode measurement snapshots into latent representations, then forms fixed-length input/output temporal pairs.

### Example config

`data/configs/temporal.yaml`

```yaml
data_path: "data/matlab/Normal1.mat"
var_name: "Z"
attack2noise_dir: "data/matlab/I7"
out_dir: "data/temporal"
train_percent: 0.85
window: 10
batch_size_encode: 256
device: "mps"
norm_operation_window: 50
noise_percents: [0.0, 0.05, 0.10, 0.20, 0.50]
ckpt_dir: "ckpt/vae"
```

### Output

This creates files like:

```text
data/temporal/
├── X_train_0%.pt
├── y_train_0%.pt
├── X_test_0%.pt
├── y_test_0%.pt
...
└── X_train_50%.pt / y_train_50%.pt / X_test_50%.pt / y_test_50%.pt
```

### Command

```bash
python -m data.mat2tensor --mode temporal --config data/configs/temporal.yaml
```

## Mode 3: pinpoint

This mode constructs the **PINPOINT classifier training dataset**. It uses:

- raw measurements from `Z`,
- SE residuals from `Res`,
- a trained VAE checkpoint with 0% corruption for spatial reconstruction,
- a trained RNN checkpoint with 0% corruption for temporal prediction.

The stacked residual representation is then split into train/test tensors for classifier-head training.

### Example config

`data/configs/pinpoint/pinpoint.yaml`

```yaml
normal_data_path: "data/matlab/Normal2.mat"
var_x: "Z"
var_res: "Res"
attack_dir: "data/matlab/I9"
out_dir: "data/pinpoint"
train_percent: 0.85
window: 10
batch_size_encode: 256
device: "cpu"
num_samples: 30
include_se: true
ckpt_sp: "ckpt/vae/vae_0%.pt"
ckpt_tp: "ckpt/rnn/rnn_0%.pt"
```

### Output

```text
data/pinpoint/
├── X_train.pt
├── y_train.pt
├── X_test.pt
└── y_test.pt
```

### Command

```bash
python -m data.mat2tensor --mode pinpoint --config data/configs/pinpoint/pinpoint.yaml
```

## Mode 4: pinpoint_eval

This mode constructs **evaluation-only PINPOINT tensors** for a specific attack severity and noise setting.

### Example config

`data/configs/pinpoint/eval/I5/n0.yaml`

```yaml
mode: pinpoint_eval
device: "cpu"
window: 10
num_samples: 30
batch_size_encode: 256
severity: 5
noise_level: 0
ckpt_sp: ckpt/vae/vae_0%.pt
ckpt_tp: ckpt/rnn/rnn_0%.pt
normal_data_path: data/matlab/Normal2.mat
attack_dir: data/matlab/I5
var_x: Z
var_res: Res
include_se: true
out_dir: data/pinpoint/eval/I5/n0
```

### Output

```text
data/pinpoint/eval/I5/n0/
├── X_I5_nl0.pt
└── y_I5_nl0.pt
```

### Single-run command

```bash
python -m data.mat2tensor --mode pinpoint_eval --config data/configs/pinpoint/eval/I5/n0.yaml
```

### Sweep command

If your directory layout matches the provided shell script, you can sweep all severities and noise levels using:

```bash
bash data/configs/pinpoint/eval/generate_eval_data.sh
```

If you move config files, update the paths in the shell script accordingly.

## Mode 5: ae

This mode constructs the **AE baseline classifier training dataset** using only spatial reconstruction residuals.

### Output

```text
data/baseline/ae/
├── X_train.pt
├── y_train.pt
├── X_test.pt
└── y_test.pt
```

### Command pattern

```bash
python -m data.mat2tensor --mode ae --config <ae_training_data_config.yaml>
```

## Mode 6: ae_eval

This mode constructs **evaluation-only AE baseline tensors** for a specific severity/noise setting.

### Example config

`data/configs/baseline/ae/I5/n0.yaml`

```yaml
mode: ae_eval
device: "mps"
batch_size_encode: 256
severity: 5
noise_level: 0
ckpt_ae: ckpt/ae/ae_0%.pt
normal_data_path: data/matlab/Normal2.mat
attack_dir: data/matlab/I5
var_x: Z
var_res: Res
out_dir: data/baseline/ae/eval/I5/n0
```

### Output

```text
data/baseline/ae/eval/I5/n0/
├── X_I5_nl0.pt
└── y_I5_nl0.pt
```

### Command

```bash
python -m data.mat2tensor --mode ae_eval --config data/configs/baseline/ae/I5/n0.yaml
```

## Baseline evaluation data: VAE+RNN and VAE+Transformer

The VAE+RNN and VAE+Transformer baselines reuse the `pinpoint_eval` path in `mat2tensor.py`, but differ by:

- the temporal checkpoint (`ckpt_tp`), and
- whether the SE residual is included.

### VAE+RNN example

`data/configs/baseline/vae_rnn/I5/n0.yaml`

```yaml
mode: pinpoint_eval
device: "cpu"
window: 10
num_samples: 30
batch_size_encode: 256
severity: 5
noise_level: 0
ckpt_sp: ckpt/vae/vae_0%.pt
ckpt_tp: ckpt/rnn/rnn_0%.pt
normal_data_path: data/matlab/Normal2.mat
attack_dir: data/matlab/I5
var_x: Z
var_res: Res
include_se: false
out_dir: data/baseline/vae_rnn/eval/I5/n0
```

### Transformer example

`data/configs/baseline/transformer/I5/n0.yaml`

```yaml
mode: pinpoint_eval
device: "mps"
window: 10
num_samples: 30
batch_size_encode: 256
severity: 5
noise_level: 0
ckpt_sp: ckpt/vae/vae_0%.pt
ckpt_tp: ckpt/transformer/transformer_0%.pt
normal_data_path: data/matlab/Normal2.mat
attack_dir: data/matlab/I5
var_x: Z
var_res: Res
include_se: true
out_dir: data/baseline/transformer/eval/I5/n0
```

### Sweep command for baselines

The provided shell script supports sweeping one baseline at a time by setting the `<baseline_name>`:

```bash
bash data/configs/baseline/generate_eval_data.sh <baseline_name>
```

## Evaluation tensors and downstream evaluation

Once evaluation tensors are created, they can be consumed by `evaluate.py` using configs under `configs/eval/`.

For example:

```bash
python evaluate.py --config configs/eval/pinpoint/I5/n0.yaml
python evaluate.py --config configs/eval/baseline/ae/I5/n0.yaml
python evaluate.py --config configs/eval/baseline/vae_rnn/I5/n0.yaml
python evaluate.py --config configs/eval/baseline/transformer/I5/n0.yaml
```

The outputs are written under `eval/...` and include:

- `classification_report.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `preds.pt`
- `probs.pt`
- `y_true.pt`

## Practical notes

- `device` may be set to `cpu`, `cuda`, `mps`, or `auto` depending on the script/config.
- The temporal pipeline depends on VAE checkpoints already being available.
- The PINPOINT and baseline residual-fusion evaluation pipelines depend on both pretrained generative modules and the raw MATLAB files.
- Severity labels are encoded through folders such as `I5`, `I6`, ..., `I9`.
- The class labels used throughout the repository are:
  - `0`: Normal
  - `1`: Location1
  - `2`: Location2
  - `3`: Location3
