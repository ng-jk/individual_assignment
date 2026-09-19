"""Train and evaluate the CardioExplain Vision cardiomegaly classifier."""
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cardioexplain.evaluation import (choose_threshold, classification_metrics,
                                      save_evaluation_plots, save_json)
from cardioexplain.imaging_data import (CheXpertCardiomegalyDataset,
                                        limit_by_patient, load_chexpert_metadata,
                                        patient_level_split)
from cardioexplain.vision_model import build_model, build_transforms, save_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Path to an extracted CheXpert v1.0 directory (full or small)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vision"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Patient-safe approximate row limit for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip ImageNet weights; useful only for offline pipeline tests")
    parser.add_argument("--train-full-backbone", action="store_true")
    parser.add_argument("--dataset-kind", choices=["chexpert", "synthetic_demo"],
                        default="chexpert",
                        help="Record whether this run uses real CheXpert or generated demo data")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); targets: list[float] = []; probabilities: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).reshape(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(targets, dtype=int), np.asarray(probabilities, dtype=float)


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer,
                device: torch.device) -> float:
    model.train(); loss_total = 0.0
    for images, labels in loader:
        images = images.to(device); labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images).reshape(-1), labels)
        loss.backward(); optimizer.step()
        loss_total += loss.item() * len(labels)
    return loss_total / len(loader.dataset)


def split_summary(splits: dict) -> dict:
    return {name: {"images": int(len(frame)),
                   "patients": int(frame["patient_id"].nunique()),
                   "positive_images": int(frame["target"].sum()),
                   "positive_rate": float(frame["target"].mean())}
            for name, frame in splits.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: CPU training is intended only for small smoke runs. Use a GPU for final training.")

    frame = load_chexpert_metadata(args.dataset_dir)
    frame = limit_by_patient(frame, args.max_samples, args.seed)
    missing_images = [str(path) for path in frame["image_path"] if not Path(path).exists()]
    if missing_images:
        preview = "\n".join(missing_images[:3])
        raise FileNotFoundError(f"{len(missing_images)} image files are missing. Examples:\n{preview}")
    splits = patient_level_split(frame, random_state=args.seed)
    summary = split_summary(splits)
    save_json(args.output_dir / "split_summary.json", summary)
    print(f"Split summary: {summary}")

    training_data = CheXpertCardiomegalyDataset(splits["train"], build_transforms(True))
    validation_data = CheXpertCardiomegalyDataset(splits["validation"], build_transforms(False))
    test_data = CheXpertCardiomegalyDataset(splits["test"], build_transforms(False))
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(training_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    model = build_model(pretrained=not args.no_pretrained,
                        train_backbone=args.train_full_backbone).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    positives = float(splits["train"]["target"].sum())
    negatives = float(len(splits["train"]) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both positive and negative examples")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))

    best_state = None; best_f1 = -1.0; best_threshold = 0.5; stale_epochs = 0; history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_targets, validation_probabilities = predict_loader(model, validation_loader, device)
        threshold = choose_threshold(validation_targets, validation_probabilities)
        score = f1_score(validation_targets, validation_probabilities >= threshold, zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(loss),
                        "validation_f1": float(score), "threshold": float(threshold)})
        print(f"Epoch {epoch:02d}: loss={loss:.4f}, validation_f1={score:.4f}, threshold={threshold:.2f}")
        if score > best_f1:
            best_f1 = float(score); best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict()); stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_state)
    test_targets, test_probabilities = predict_loader(model, test_loader, device)
    metrics = classification_metrics(test_targets, test_probabilities, best_threshold)
    metrics["training_seconds"] = float(time.perf_counter() - started)
    metrics["device"] = str(device)
    save_json(args.output_dir / "test_metrics.json", metrics)
    save_json(args.output_dir / "training_history.json", history)
    save_evaluation_plots(test_targets, test_probabilities, best_threshold, args.output_dir)
    data_provenance = {
        "kind": args.dataset_kind,
        "dataset_directory": str(args.dataset_dir.resolve()),
        "pretrained_imagenet": not args.no_pretrained,
    }
    save_checkpoint(args.output_dir / "best_model.pt", model, best_threshold, metrics, history,
                    summary, data_provenance=data_provenance)
    splits["test"][["Path", "patient_id", "target"]].assign(
        probability=test_probabilities,
        prediction=(test_probabilities >= best_threshold).astype(int),
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Test metrics: {metrics}")
    print(f"Saved checkpoint: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
