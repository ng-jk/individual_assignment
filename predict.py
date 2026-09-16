"""Run CardioExplain Vision on one chest X-ray and return JSON output."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from cardioexplain.inference import open_input_image, predict_image
from cardioexplain.vision_model import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to a de-identified frontal chest X-ray (PNG/JPG/JPEG)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/vision/best_model.pt"),
                        help="Trained CardioExplain checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"),
                        help="Directory for JSON and Grad-CAM output")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}. Train the model first or pass "
                "--checkpoint with the path to a trained .pt file."
            )
        device = resolve_device(args.device)
        image = open_input_image(args.image)
        model, metadata = load_checkpoint(args.checkpoint, device=device)
        result, explanation = predict_image(image, model, metadata, device=device)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(character if character.isalnum() or character in "-_" else "_"
                            for character in args.image.stem) or "image"
        path_tag = hashlib.sha256(str(args.image.resolve()).encode("utf-8")).hexdigest()[:8]
        output_stem = f"{safe_stem}_{path_tag}"
        explanation_path = (args.output_dir / f"{output_stem}_gradcam.png").resolve()
        json_path = (args.output_dir / f"{output_stem}_prediction.json").resolve()
        explanation.save(explanation_path)
        payload = result.to_dict() | {
            "input_image": str(args.image.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "explanation_image": str(explanation_path),
            "model_test_metrics": metadata.get("metrics", {}),
            "checkpoint_data_provenance": metadata.get("data_provenance", {}),
        }
        json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        payload["result_json"] = str(json_path)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
