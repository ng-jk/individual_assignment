"""Shared image validation and explainable inference helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from cardioexplain.vision_model import GradCAM, overlay_heatmap, prepare_image


DISCLAIMER = (
    "Research and education only. This model is not a medical device and its "
    "output must not be used for diagnosis, treatment, or patient-care decisions."
)


@dataclass(frozen=True)
class PredictionResult:
    """Serializable result returned by the web and command-line interfaces."""

    label: str
    probability: float
    threshold: float
    classification: str
    device: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def open_input_image(path: str | Path, minimum_size: int = 128) -> Image.Image:
    """Open a supported image and reject unreadable or unreasonably small input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Input is not a readable PNG or JPEG image: {path}") from exc
    if min(image.size) < minimum_size:
        raise ValueError(
            f"Input image is too small ({image.width} x {image.height}); "
            f"minimum dimensions are {minimum_size} x {minimum_size} pixels."
        )
    return image


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    metadata: dict,
    device: torch.device | str = "cpu",
) -> tuple[PredictionResult, Image.Image]:
    """Return a thresholded prediction and a Grad-CAM explanation image."""
    device = torch.device(device)
    threshold = float(metadata.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Checkpoint threshold must be between 0 and 1, got {threshold}")
    model = model.to(device).eval()
    tensor = prepare_image(image, int(metadata.get("image_size", 224))).to(device)
    explainer = GradCAM(model)
    try:
        probability, heatmap = explainer.generate(tensor)
    finally:
        explainer.close()
    classification = "above_threshold" if probability >= threshold else "below_threshold"
    result = PredictionResult(
        label=str(metadata.get("label", "Cardiomegaly")),
        probability=float(probability),
        threshold=threshold,
        classification=classification,
        device=str(device),
    )
    return result, overlay_heatmap(image, heatmap)
