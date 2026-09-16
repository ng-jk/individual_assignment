"""ResNet18 model, preprocessing, checkpoint handling, and Grad-CAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(training: bool, image_size: int = IMAGE_SIZE):
    operations = [transforms.Resize((image_size + 32, image_size + 32),
                                    interpolation=InterpolationMode.BILINEAR)]
    if training:
        operations.extend([transforms.RandomCrop(image_size), transforms.RandomRotation(5)])
    else:
        operations.append(transforms.CenterCrop(image_size))
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


def build_model(pretrained: bool = True, train_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if not train_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model


def save_checkpoint(path: str | Path, model: nn.Module, threshold: float, metrics: dict,
                    history: list[dict], dataset_summary: dict,
                    data_provenance: dict | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "resnet18", "state_dict": model.state_dict(),
                "threshold": float(threshold), "image_size": IMAGE_SIZE,
                "label": "Cardiomegaly", "metrics": metrics, "history": history,
                "dataset_summary": dataset_summary,
                "data_provenance": data_provenance or {}}, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Unsupported checkpoint architecture")
    model = build_model(pretrained=False, train_backbone=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


class GradCAM:
    """Minimal Grad-CAM implementation for the final ResNet convolutional block."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = model.layer4[-1]
        self.forward_handle = layer.register_forward_hook(self._capture_activations)
        self.backward_handle = layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _capture_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(tensor).reshape(-1)[0]
        logit.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(heatmap, size=tensor.shape[-2:],
                                                  mode="bilinear", align_corners=False)[0, 0]
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap /= maximum
        return torch.sigmoid(logit).item(), heatmap.cpu().numpy()

    def close(self):
        self.forward_handle.remove(); self.backward_handle.remove()


def prepare_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    return build_transforms(training=False, image_size=image_size)(image.convert("RGB")).unsqueeze(0)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])), dtype=np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255 * heatmap
    color[..., 1] = 80 * heatmap
    strength = alpha * heatmap[..., None]
    mixed = np.clip(base * (1 - strength) + color * strength, 0, 255)
    return Image.fromarray(mixed.astype(np.uint8))
