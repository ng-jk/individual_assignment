import numpy as np
import torch
from PIL import Image

from cardioexplain.evaluation import choose_threshold, classification_metrics
from cardioexplain.vision_model import (GradCAM, build_model, overlay_heatmap,
                                        prepare_image)


def test_resnet_forward_and_gradcam():
    model = build_model(pretrained=False, train_backbone=True).eval()
    tensor = torch.rand(1, 3, 224, 224)
    explainer = GradCAM(model)
    probability, heatmap = explainer.generate(tensor)
    explainer.close()
    assert 0 <= probability <= 1
    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()


def test_image_preparation_and_overlay():
    image = Image.new("L", (300, 250), color=100)
    tensor = prepare_image(image)
    overlay = overlay_heatmap(image, np.ones((224, 224), dtype=np.float32) * 0.5)
    assert tensor.shape == (1, 3, 224, 224)
    assert overlay.size == (224, 224)


def test_metrics_and_threshold():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = choose_threshold(targets, probabilities)
    metrics = classification_metrics(targets, probabilities, threshold)
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
