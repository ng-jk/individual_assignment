"""Build a compact 224px CheXpert training directory from Stanford batch files.

The current Stanford full-resolution release is split into a metadata/validation
batch and three training-image batches.  This utility reads images directly
from each ZIP (or an already extracted batch directory), resizes them, and
writes the layout consumed by ``train.py`` without first expanding the full
release on disk.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PATIENT_PART = re.compile(r"patient\d+", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory containing the Stanford batch ZIPs/directories")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Compact CheXpert directory to create")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 3, 4],
                        help="Training batches to process (default: 2 3 4)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Write a filtered CSV for a partial smoke-test dataset")
    return parser.parse_args(argv)


def _is_batch(path: Path, number: int) -> bool:
    return re.search(rf"batch\s+{number}(?:\D|$)", path.name, re.IGNORECASE) is not None


def discover_batch_source(source_dir: Path, number: int) -> Path:
    candidates = [path for path in source_dir.iterdir() if _is_batch(path, number)]
    directories = sorted(path for path in candidates if path.is_dir())
    archives = sorted(path for path in candidates if path.is_file() and path.suffix.lower() == ".zip")
    if directories:
        return directories[0]
    if archives:
        return archives[0]
    raise FileNotFoundError(
        f"CheXpert batch {number} was not found in {source_dir}. "
        f"Download every required batch before final preparation."
    )


def _relative_train_path(raw_path: str) -> Path:
    parts = PurePosixPath(str(raw_path).replace("\\", "/")).parts
    patient_index = next(
        (index for index, part in enumerate(parts) if PATIENT_PART.fullmatch(part)), None
    )
    if patient_index is None:
        raise ValueError(f"No CheXpert patient component found in path: {raw_path}")
    return Path("train", *parts[patient_index:])


def _safe_target(output_dir: Path, relative_path: Path) -> Path:
    target = (output_dir / relative_path).resolve()
    output_root = output_dir.resolve()
    if output_root not in target.parents:
        raise ValueError(f"Unsafe output path: {relative_path}")
    return target


def _save_resized(image: Image.Image, target: Path, size: int, quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    temporary = target.with_suffix(target.suffix + ".tmp")
    resized.save(temporary, format="JPEG", quality=quality)
    os.replace(temporary, target)


def _process_zip_chunk(archive: str, members: list[str], output_dir: str,
                       size: int, quality: int, overwrite: bool) -> dict[str, int]:
    written = skipped = 0
    output = Path(output_dir)
    with zipfile.ZipFile(archive) as bundle:
        for member in members:
            target = _safe_target(output, _relative_train_path(member))
            if target.exists() and not overwrite:
                skipped += 1
                continue
            with bundle.open(member) as image_stream, Image.open(image_stream) as image:
                _save_resized(image, target, size, quality)
            written += 1
    return {"written": written, "skipped": skipped}


def _process_directory_chunk(files: list[str], source_dir: str, output_dir: str,
                             size: int, quality: int, overwrite: bool) -> dict[str, int]:
    written = skipped = 0
    source = Path(source_dir)
    output = Path(output_dir)
    for filename in files:
        image_path = Path(filename)
        target = _safe_target(output, _relative_train_path(image_path.relative_to(source).as_posix()))
        if target.exists() and not overwrite:
            skipped += 1
            continue
        with Image.open(image_path) as image:
            _save_resized(image, target, size, quality)
        written += 1
    return {"written": written, "skipped": skipped}


def _chunks(items: list[str], count: int) -> list[list[str]]:
    count = max(1, min(count, len(items)))
    chunk_size = math.ceil(len(items) / count)
    return [items[index:index + chunk_size]
            for index in range(0, len(items), chunk_size)]


def process_batch(source: Path, output_dir: Path, size: int, quality: int,
                  workers: int, overwrite: bool,
                  required_paths: set[str] | None = None) -> dict[str, object]:
    if source.is_file():
        with zipfile.ZipFile(source) as bundle:
            items = [name for name in bundle.namelist()
                     if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES]
        worker = _process_zip_chunk
        worker_args = lambda chunk: (str(source), chunk, str(output_dir), size, quality, overwrite)
    else:
        items = [str(path) for path in source.rglob("*")
                 if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        worker = _process_directory_chunk
        worker_args = lambda chunk: (chunk, str(source), str(output_dir), size, quality, overwrite)

    source_images = len(items)
    if required_paths is not None:
        items = [item for item in items
                 if _relative_train_path(
                     Path(item).relative_to(source).as_posix() if source.is_dir() else item
                 ).as_posix() in required_paths]
    if not items:
        return {"source": str(source.resolve()), "source_images": source_images,
                "images": 0, "written": 0, "skipped": 0}
    totals = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(worker, *worker_args(chunk))
                   for chunk in _chunks(items, workers)]
        for future in as_completed(futures):
            result = future.result()
            totals["written"] += result["written"]
            totals["skipped"] += result["skipped"]
    return {"source": str(source.resolve()), "source_images": source_images,
            "images": len(items), **totals}


def read_train_csv(batch_one: Path) -> pd.DataFrame:
    if batch_one.is_file():
        with zipfile.ZipFile(batch_one) as bundle:
            members = [name for name in bundle.namelist()
                       if PurePosixPath(name).name.lower() == "train.csv"]
            if len(members) != 1:
                raise ValueError(f"Expected one train.csv in {batch_one}, found {len(members)}")
            with bundle.open(members[0]) as stream:
                frame = pd.read_csv(io.BytesIO(stream.read()))
    else:
        matches = list(batch_one.rglob("train.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one train.csv in {batch_one}, found {len(matches)}")
        frame = pd.read_csv(matches[0])
    if "Path" not in frame.columns:
        raise ValueError("CheXpert train.csv does not contain the required Path column")
    frame = frame.copy()
    frame["Path"] = frame["Path"].map(lambda value: _relative_train_path(str(value)).as_posix())
    return frame


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if args.size < 1 or args.workers < 1 or not 1 <= args.quality <= 100:
        raise ValueError("size/workers must be positive and quality must be between 1 and 100")
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_one = discover_batch_source(source_dir, 1)
    batch_sources = [discover_batch_source(source_dir, number) for number in args.batches]
    source_frame = read_train_csv(batch_one)
    required_columns = {"Cardiomegaly", "Frontal/Lateral"}
    missing_columns = required_columns.difference(source_frame.columns)
    if missing_columns:
        raise ValueError(f"CheXpert train.csv is missing columns: {sorted(missing_columns)}")
    frame = source_frame[
        source_frame["Frontal/Lateral"].eq("Frontal")
        & source_frame["Cardiomegaly"].isin([0.0, 1.0])
    ].copy()
    required_paths = set(frame["Path"])
    print(
        f"Preparing {len(frame)} eligible frontal images with certain cardiomegaly labels "
        f"from {len(source_frame)} metadata rows",
        flush=True,
    )

    batch_results = []
    for number, source in zip(args.batches, batch_sources):
        print(f"Processing CheXpert training batch {number}: {source}", flush=True)
        result = process_batch(source, output_dir, args.size, args.quality,
                               args.workers, args.overwrite, required_paths)
        result["batch"] = number
        batch_results.append(result)
        print(f"Completed batch {number}: {result}", flush=True)

    exists = frame["Path"].map(lambda value: (output_dir / value).is_file())
    missing = int((~exists).sum())
    if missing and not args.allow_incomplete:
        examples = frame.loc[~exists, "Path"].head(5).tolist()
        raise FileNotFoundError(
            f"Prepared dataset is missing {missing} images. Examples: {examples}. "
            "Ensure batches 2, 3, and 4 are all available, then rerun; existing images are skipped."
        )
    output_frame = frame.loc[exists].reset_index(drop=True) if missing else frame
    output_frame.to_csv(output_dir / "train.csv", index=False)
    manifest = {
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "image_size": args.size,
        "jpeg_quality": args.quality,
        "batches": batch_results,
        "metadata_rows": int(len(source_frame)),
        "eligible_rows": int(len(frame)),
        "prepared_rows": int(len(output_frame)),
        "missing_rows": missing,
        "incomplete": bool(missing),
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
