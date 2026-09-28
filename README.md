# Synopsys ADHD Classifier

Python utilities and the original Colab notebook for experimenting with an ADHD classifier from precomputed EEG feature data.

## Contents

- `notebooks/Synopsys_ADHD_Classifier.ipynb` - cleaned copy of the original Colab notebook with saved outputs removed.
- `src/synopsys_adhd_classifier/train.py` - command-line training and evaluation workflow based on the notebook's classifier cells.
- `requirements.txt` / `pyproject.toml` - dependencies and package metadata.

## Data

The notebook expects a `features_df.csv` file with one row per channel/segment and at least these columns:

- `group` - target label
- `file_id`, `band`, `channel_id` - metadata columns excluded from model features
- numeric feature columns such as `std_dev`, `rms`, `skewness`, `kurtosis`, `spectral_entropy`, `band_power`, `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`, and `shannons_entropy`

The data file is not included in this repository. Put it at `data/features_df.csv` or pass a path with `--features-csv`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m synopsys_adhd_classifier.train --features-csv data/features_df.csv
```

You can also run the installed command:

```bash
synopsys-adhd-train --features-csv data/features_df.csv
```

## Notes From The Notebook Conversion

The original Colab mounted Google Drive and loaded `/content/drive/MyDrive/features_df.csv`. The repo version replaces that with a local CSV path.

The notebook also contains an optional feature-extraction cell that references variables/functions not defined in the notebook: `segments_by_band_with_meta`, `calculate_hjorth_parameters`, and `calculate_shannons_entropy`. This repository keeps that cell in the notebook for provenance, while the CLI starts from the already-created `features_df.csv`.

The SVM workflow in `src/` uses a scikit-learn pipeline with `StandardScaler` and `SVC`, then reports validation/test accuracy plus cross-validation scores.
