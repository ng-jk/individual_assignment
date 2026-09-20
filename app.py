"""Streamlit interface for CardioExplain Vision."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from cardioexplain.inference import predict_image
from cardioexplain.vision_model import load_checkpoint

st.set_page_config(page_title="CardioExplain Vision", page_icon="🫀", layout="wide")
st.title("CardioExplain Vision")
st.write("Explainable cardiomegaly screening from frontal chest X-rays")

checkpoint_path = Path(os.getenv("CARDIOEXPLAIN_CHECKPOINT", "artifacts/vision/best_model.pt"))
if not checkpoint_path.exists():
    st.warning(f"No trained image checkpoint found at `{checkpoint_path}`.")
    st.code("python train.py --dataset-dir datasets/CheXpert-v1.0-small", language="bash")
    st.stop()


@st.cache_resource
def get_model(path: str):
    return load_checkpoint(path, device="cpu")


try:
    model, metadata = get_model(str(checkpoint_path))
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

threshold = float(metadata.get("threshold", 0.5))
data_provenance = metadata.get("data_provenance", {})
if data_provenance.get("kind") == "synthetic_demo":
    st.error("DEMONSTRATION CHECKPOINT: this model was trained on generated patterns, not real chest X-rays. Its output has no medical meaning.")
uploaded = st.file_uploader("Upload a de-identified frontal chest X-ray",
                            type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a readable PNG or JPEG image.")
        st.stop()
    if min(image.size) < 128:
        st.error("The image is too small. Upload an image at least 128 x 128 pixels.")
        st.stop()
    st.image(image, caption="Uploaded image", width=480)

    if st.button("Run educational screening", type="primary"):
        result, overlay = predict_image(image, model, metadata, device="cpu")
        label = "Above the model threshold" if result.classification == "above_threshold" else "Below the model threshold"
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated cardiomegaly probability", f"{result.probability:.1%}")
            st.metric("Decision threshold", f"{threshold:.1%}")
            st.write(f"**Model classification:** {label}")
        with col2:
            st.image(overlay, caption="Grad-CAM influence map (red indicates greater model influence)",
                     use_container_width=True)
        st.warning("The heatmap describes model sensitivity, not disease location or medical causation. A qualified clinician must interpret medical images.")

with st.expander("Model information"):
    st.json({"architecture": metadata.get("architecture"),
             "label": metadata.get("label"),
             "threshold": threshold,
             "data_provenance": data_provenance,
             "test_metrics": metadata.get("metrics", {}),
             "dataset_summary": metadata.get("dataset_summary", {})})
