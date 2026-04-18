# PINPOINT

This repository is the implementation of the paper "**PINPOINT: <ins>P</ins>robabilistic Res<ins>i</ins>dual Fusio<ins>n</ins> for Cyberattack Detecti<ins>o</ins>n and Local<ins>i</ins>zatio<ins>n</ins> in <ins>P</ins>ower Sys<ins>t</ins>ems**''.

PINPOINT is a probabilistic residual-fusion framework for covert cyberattack detection and localization in transmission-level power systems. Multiple stochastic latent samples are aggregated through a voting-based classifier head. The method combines three complementary residual signals:

- a **state-estimation (SE) residual**,
- a **spatial reconstruction residual** from a VAE,
- a **temporal prediction residual** from an RNN.

<p align=center>
    <img src="img/Graphics.png" width="900"/>
</p>

The repository also includes three architectural baselines used in the paper:

- **AE**
- **VAE+RNN**
- **VAE+Transformer**

**Paper:** `[PAPER_HERE]` 
- Link: `[LINK_HERE]`

#
This repo is organized around a *config-driven workflow* for:

1. constructing tensor datasets from MATLAB `.mat` files,
2. training generative modules,
3. training classifier heads,
4. evaluating trained models and saving reports.

## Repository layout

```text
.
├── configs/                 # training and evaluation configs
├── data/
│   ├── matlab/              # included raw .mat files
│   ├── configs/             # data-generation configs and shell scripts
│   ├── spatial/             # generated VAE/AE training tensors
│   ├── temporal/            # generated RNN/Transformer training tensors
│   ├── pinpoint/            # generated PINPOINT classifier tensors
│   └── baseline/            # generated baseline classifier tensors
├── ckpt/                    # saved checkpoints
├── eval/                    # saved evaluation outputs
├── img/                     # saved learning curves and figures
├── models/
│   ├── models.py            # model definitions + registry
│   ├── trainer.py           # training loop
│   ├── data_utils.py        # dataset wrappers/utilities
│   └── earlystopping.py     # early stopping + checkpointing
├── train.py                 # main training entry point
└── evaluate.py              # main evaluation entry point
```

## Included artifacts

The repository includes:

- raw MATLAB `.mat` files under `data/matlab/`, including:
  - streaming measurement signals
  - SE residuals
- training and evaluation configs,
- saved checkpoints under `ckpt/`,
- saved evaluation outputs under `eval/`, including:
  - `classification_report.json`
  - `classification_report.txt`
  - `confusion_matrix.png`
  - `preds.pt`
  - `probs.pt`
  - `y_true.pt`

The generated `.pt` tensors are intermediate artifacts produced by `data.mat2tensor`. These may be regenerated from the included `.mat` files and configs.

## Environment setup

An example Conda environment file is included as `environment.yml`.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate pinpoint
```

### Notes

- The provided `environment.yml` uses Python 3.13 and includes PyTorch, torchvision, NumPy, SciPy, scikit-learn, matplotlib, pandas, tqdm, PyYAML, and torchinfo.
- Device selection in the scripts supports `auto`, `cpu`, `cuda`, and `mps` depending on the config and hardware.
- In the training/evaluation entry points, `device: auto` selects `mps` first when available, then `cuda`, then `cpu`.

## Main entry points

### `train.py`
Trains a model from a YAML config.

```bash
python train.py --config <path/to/config.yaml>
```

Supported model tasks are registered in `models/models.py` and include:

- `ae`
- `vae`
- `rnn`
- `transformer`
- `pinpoint`
- `dnn_ae`

### `evaluate.py`
Loads a saved checkpoint and an evaluation tensor pair `(X, y)`, then writes predictions, probabilities, classification reports, and a confusion matrix image.

```bash
python evaluate.py --config <path/to/eval_config.yaml>
```

### `data.mat2tensor`
Constructs PyTorch tensors from MATLAB `.mat` files.

```bash
python -m data.mat2tensor --mode <mode> --config <path/to/config.yaml>
```

Supported modes:

- `spatial`
- `temporal`
- `pinpoint`
- `pinpoint_eval`
- `ae`
- `ae_eval`

All details related to data construction are documented in [`data/README.md`](data/README.md).

## Training workflow

A typical end-to-end workflow is:

1. Generate spatial tensors for AE/VAE training.
2. Train AE and/or VAE.
3. Generate temporal tensors using the trained VAE.
4. Train RNN and/or Transformer.
5. Generate classifier tensors for PINPOINT or baselines.
6. Train classifier heads.
7. Generate evaluation tensors for each severity/noise setting.
8. Run `evaluate.py` over the saved checkpoints.

## Example training commands

### 1) Train AE

```bash
python train.py --config configs/ae.yaml
```

### 2) Train VAE

```bash
python train.py --config configs/vae.yaml
```

### 3) Train RNN

```bash
python train.py --config configs/rnn.yaml
```

### 4) Train Transformer

```bash
python train.py --config configs/transformer.yaml
```

### 5) Train baseline classifier heads

```bash
python train.py --config configs/ae_clf.yaml
python train.py --config configs/vae_rnn_clf.yaml
python train.py --config configs/vae_transformer_clf.yaml
```

### 6) Train PINPOINT classifier head

```bash
python train.py --config configs/pinpoint.yaml
```

## Example evaluation commands

### PINPOINT

```bash
python evaluate.py --config configs/eval/pinpoint/I5/n0.yaml
```

or sweep all included evaluation settings:

```bash
bash configs/eval/evaluate_pinpoint.sh
```

### Baselines

```bash
python evaluate.py --config configs/eval/baseline/ae/I5/n0.yaml
python evaluate.py --config configs/eval/baseline/vae_rnn/I5/n0.yaml
python evaluate.py --config configs/eval/baseline/transformer/I5/n0.yaml
```

or sweep all settings for one baseline by choosing `ae`, `vae_rnn`, or `transformer` as `<baseline_name>` in:

```bash
bash configs/eval/evaluate_baseline.sh <baseline_name>
```

## Config conventions

The repository uses YAML files throughout.

### Training configs
Training configs specify:

- `task`
- `device`
- model keyword arguments
- optimizer/training hyperparameters
- checkpoint path
- plot output path/name
- tensor dataset paths

Examples:

- `configs/ae.yaml`
- `configs/vae.yaml`
- `configs/rnn.yaml`
- `configs/transformer.yaml`
- `configs/ae_clf.yaml`
- `configs/vae_rnn_clf.yaml`
- `configs/vae_transformer_clf.yaml`
- `configs/pinpoint.yaml`

### Evaluation configs
Evaluation configs specify:

- checkpoint path
- evaluation tensor paths (`X`, `y`)
- batch size
- output directory
- class labels / target names
- confusion matrix title

## Notes on reproducibility

- The raw `.mat` files required for reconstruction of the generated datasets are included in the repository.
- The repository also includes evaluation outputs under `eval/`, so users can inspect the saved predictions and reports directly even without rerunning the full pipeline.
- Generated `.pt` tensors are intermediate products of the preprocessing pipeline.

## Notes on included data

The included MATLAB files are organized by normal operation and attack severity folders. The training/evaluation pipeline expects variables such as `Z` (measurements) and, when needed, `Res` (SE residuals) as configured in the YAML files.

See [`data/README.md`](data/README.md) for exact data-generation details.

---

## Citation
If you use this code, please cite the paper:

```bibtex
@misc{pinpoint2026,
  title   = {PINPOINT: Probabilistic Residual Fusion for Cyberattack Detection and Localization in Power Systems},
  author  = {Author(s)},
  note    = {Note},
  year    = {2026},
  url     = {LINK_HERE}
}
```

---

## Contact
For questions/issues: open a GitHub Issue or contact the authors.

