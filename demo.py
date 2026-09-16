"""Run a complete non-medical CardioExplain image demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import predict
import train
from cardioexplain.demo_data import create_synthetic_chexpert
from cardioexplain.imaging_data import resolve_image_path


WARNING = (
    "DEMONSTRATION ONLY: generated images contain no anatomy and all resulting "
    "metrics and predictions have no medical meaning."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("datasets/synthetic-chexpert-demo"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/demo_prediction"))
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--retrain", action="store_true",
                        help="Train again even when the demonstration checkpoint exists")
    return parser.parse_args(argv)


def first_input_image(dataset_dir: Path) -> Path:
    frame = pd.read_csv(dataset_dir / "train.csv")
    if frame.empty or "Path" not in frame.columns:
        raise ValueError("Generated demonstration metadata contains no usable image path")
    return resolve_image_path(dataset_dir.resolve(), str(frame.iloc[0]["Path"]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(WARNING)

    metadata_path = args.dataset_dir / "train.csv"
    if not metadata_path.is_file():
        created = create_synthetic_chexpert(args.dataset_dir, patients=args.patients)
        print(f"Created generated demonstration dataset: {created}")
    else:
        print(f"Using generated demonstration dataset: {args.dataset_dir.resolve()}")

    checkpoint = args.model_dir / "best_model.pt"
    if args.retrain or not checkpoint.is_file():
        train.main([
            "--dataset-dir", str(args.dataset_dir),
            "--output-dir", str(args.model_dir),
            "--epochs", "1",
            "--batch-size", "8",
            "--workers", "0",
            "--no-pretrained",
            "--dataset-kind", "synthetic_demo",
        ])
    else:
        print(f"Using generated-data checkpoint: {checkpoint.resolve()}")

    return predict.main([
        "--image", str(first_input_image(args.dataset_dir)),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(args.output_dir),
        "--device", "cpu",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
