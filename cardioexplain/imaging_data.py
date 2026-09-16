"""CheXpert metadata validation, patient-level splitting, and image loading."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

LABEL_COLUMN = "Cardiomegaly"
PATH_COLUMN = "Path"
PATIENT_PATTERN = re.compile(r"patient\d+", re.IGNORECASE)


def patient_id_from_path(path: str) -> str:
    """Extract a CheXpert patient identifier from an image path."""
    match = PATIENT_PATTERN.search(str(path).replace("\\", "/"))
    if not match:
        raise ValueError(f"No patient ID found in image path: {path}")
    return match.group(0).lower()


def resolve_image_path(dataset_dir: Path, csv_path: str) -> Path:
    """Resolve paths whether the CSV includes the dataset directory prefix or not."""
    csv_path = str(csv_path).replace("\\", "/")
    candidate = Path(csv_path)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0].lower() == dataset_dir.name.lower():
        candidate = Path(*parts[1:])
    return dataset_dir / candidate


def load_chexpert_metadata(dataset_dir: str | Path, csv_name: str = "train.csv") -> pd.DataFrame:
    """Load usable frontal studies with certain binary cardiomegaly labels."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    csv_file = dataset_dir / csv_name
    if not csv_file.exists():
        raise FileNotFoundError(f"CheXpert metadata not found: {csv_file}")
    frame = pd.read_csv(csv_file)
    required = {PATH_COLUMN, LABEL_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if "Frontal/Lateral" in frame.columns:
        frame = frame[frame["Frontal/Lateral"].eq("Frontal")]
    frame = frame[frame[LABEL_COLUMN].isin([0.0, 1.0])].copy()
    frame["patient_id"] = frame[PATH_COLUMN].map(patient_id_from_path)
    frame["image_path"] = frame[PATH_COLUMN].map(lambda p: resolve_image_path(dataset_dir, p))
    frame["target"] = frame[LABEL_COLUMN].astype("float32")
    if frame.empty:
        raise ValueError("No frontal images with certain cardiomegaly labels were found")
    return frame.reset_index(drop=True)


def patient_level_split(frame: pd.DataFrame, validation_fraction: float = 0.15,
                        test_fraction: float = 0.15,
                        random_state: int = 42) -> dict[str, pd.DataFrame]:
    """Create mutually exclusive train, validation, and test patient groups."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    patients = frame.groupby("patient_id", as_index=False)["target"].max()
    train_patients, holdout_patients = train_test_split(
        patients, test_size=validation_fraction + test_fraction,
        stratify=patients["target"], random_state=random_state)
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation_patients, test_patients = train_test_split(
        holdout_patients, test_size=relative_test,
        stratify=holdout_patients["target"], random_state=random_state + 1)
    splits = {
        "train": frame[frame["patient_id"].isin(train_patients["patient_id"])].reset_index(drop=True),
        "validation": frame[frame["patient_id"].isin(validation_patients["patient_id"])].reset_index(drop=True),
        "test": frame[frame["patient_id"].isin(test_patients["patient_id"])].reset_index(drop=True),
    }
    groups = {name: set(part["patient_id"]) for name, part in splits.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise RuntimeError("Patient leakage detected across data splits")
    return splits


def limit_by_patient(frame: pd.DataFrame, max_samples: int | None,
                     random_state: int = 42) -> pd.DataFrame:
    """Limit rows for a smoke run while keeping chosen-patient studies together."""
    if not max_samples or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    patients = frame[["patient_id"]].drop_duplicates().sample(frac=1, random_state=random_state)
    selected: list[str] = []
    rows = 0
    for patient_id in patients["patient_id"]:
        selected.append(patient_id)
        rows += int(frame["patient_id"].eq(patient_id).sum())
        if rows >= max_samples:
            break
    return frame[frame["patient_id"].isin(selected)].reset_index(drop=True)


class CheXpertCardiomegalyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable | None = None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, target
