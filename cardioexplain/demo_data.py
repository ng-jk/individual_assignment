"""Generate non-medical image data for reproducible pipeline demonstrations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def create_synthetic_chexpert(output_dir: str | Path, patients: int = 60,
                              seed: int = 42) -> Path:
    """Create a balanced CheXpert-shaped dataset with deliberately simple patterns.

    The generated images are only suitable for software verification. They contain
    no anatomy and must never be used to make medical-performance claims.
    """
    if patients < 20:
        raise ValueError("patients must be at least 20 so every split contains both classes")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    rng = np.random.default_rng(seed)

    for number in range(1, patients + 1):
        positive = number % 2
        relative = Path("train") / f"patient{number:05d}" / "study1" / "view1_frontal.png"
        image_path = output_dir / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.normal(70 + positive * 40, 12, size=(160, 160)).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L")
        if positive:
            ImageDraw.Draw(image).ellipse((45, 35, 120, 130), outline=190, width=5)
        image.save(image_path)
        rows.append({
            "Path": f"{output_dir.name}/{relative.as_posix()}",
            "Frontal/Lateral": "Frontal",
            "Cardiomegaly": float(positive),
        })

    pd.DataFrame(rows).to_csv(output_dir / "train.csv", index=False)
    return output_dir
