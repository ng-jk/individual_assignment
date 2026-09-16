"""Create a tiny, non-medical CheXpert-shaped dataset for pipeline smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cardioexplain.demo_data import create_synthetic_chexpert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=60)
    args = parser.parse_args()
    print(create_synthetic_chexpert(args.output, patients=args.patients))


if __name__ == "__main__":
    main()
