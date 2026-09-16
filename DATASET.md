# Dataset instructions

## Selected dataset

This project targets the `Cardiomegaly` observation in **CheXpert-v1.0-small**, released by the Stanford Machine Learning Group. The official dataset contains chest radiographs and labels derived from radiology reports, including positive, negative, uncertain, and unmentioned states.

Official page: https://stanfordmlgroup.github.io/competitions/chexpert/

## Access and use

1. Read the current Stanford terms and complete the required registration or agreement.
2. Download `CheXpert-v1.0-small` directly from the official source.
3. Do not commit or redistribute the images.
4. Place the extracted directory at `02_Code/datasets/CheXpert-v1.0-small` or pass another location with `--dataset-dir`.
5. Preserve the original CSV files and do not edit labels manually.

The repository ignores `datasets/` to prevent accidental publication of medical images.

## Label policy

- Task: binary radiographic cardiomegaly detection.
- Positive label: `Cardiomegaly == 1`.
- Negative label: `Cardiomegaly == 0`.
- Missing and uncertain (`-1`) labels: excluded in Version 0.2.
- Views: frontal images only.
- Splitting: group-separated by patient ID.

This policy must be stated in the report because changing the treatment of uncertain labels can change performance.
