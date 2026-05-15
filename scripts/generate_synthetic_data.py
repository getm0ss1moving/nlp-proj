from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coe_lift.data import generate_coe_lift_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CoE-LIFT JSONL data.")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--train_groups", type=int, default=1200)
    parser.add_argument("--eval_groups", type=int, default=300)
    parser.add_argument("--examples_per_task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_probe_shifts", action="store_true")
    args = parser.parse_args()

    metadata = generate_coe_lift_dataset(
        output_dir=args.output_dir,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        examples_per_task=args.examples_per_task,
        seed=args.seed,
        include_probe_shifts=not args.no_probe_shifts,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
