from pathlib import Path

import pandas as pd
from PIL import Image

from cardioexplain.imaging_data import (CheXpertCardiomegalyDataset,
                                        load_chexpert_metadata, patient_level_split)
from cardioexplain.vision_model import build_transforms


def make_dataset(root: Path, patients: int = 20) -> Path:
    dataset_dir = root / "CheXpert-v1.0-small"
    rows = []
    for number in range(1, patients + 1):
        relative = Path("train") / f"patient{number:05d}" / "study1" / "view1_frontal.jpg"
        image_path = dataset_dir / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (160, 160), color=30 + number).save(image_path)
        rows.append({"Path": f"CheXpert-v1.0-small/{relative.as_posix()}",
                     "Frontal/Lateral": "Frontal", "Cardiomegaly": float(number % 2)})
    pd.DataFrame(rows).to_csv(dataset_dir / "train.csv", index=False)
    return dataset_dir


def test_metadata_and_patient_split_have_no_leakage(tmp_path):
    frame = load_chexpert_metadata(make_dataset(tmp_path))
    splits = patient_level_split(frame)
    groups = {name: set(part["patient_id"]) for name, part in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert sum(map(len, splits.values())) == len(frame)


def test_image_dataset_returns_normalized_tensor(tmp_path):
    frame = load_chexpert_metadata(make_dataset(tmp_path))
    dataset = CheXpertCardiomegalyDataset(frame.iloc[:1], build_transforms(False))
    image, target = dataset[0]
    assert image.shape == (3, 224, 224)
    assert target.item() in (0.0, 1.0)
