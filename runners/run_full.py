#!/usr/bin/env python3
"""Orchestrate continuous full multi-model generation + validation.

Default grid (after prepare_full):
  80 profiles × 6 topics × 4 conditions × 3 models = 5760

Examples:
  python optimized/runners/run_full.py --prepare-only
  python optimized/runners/run_full.py --smoke --mock
  python optimized/runners/run_full.py --workers 6
  python optimized/runners/run_full.py --skip-prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OPTIMIZED_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = OPTIMIZED_DIR.parent
STEPS = OPTIMIZED_DIR / "steps"
RUNNERS = OPTIMIZED_DIR / "runners"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=None,
        help="Override profile count passed to prepare_full (default 80).",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_prepare:
        prep = [
            py,
            str(RUNNERS / "prepare_full.py"),
            "--seed",
            str(args.seed),
        ]
        if args.n_synthetic is not None:
            prep.extend(["--n-synthetic", str(args.n_synthetic)])
        run(prep)
        if args.prepare_only:
            print("Prepare-only complete.")
            return

    gen_cmd = [
        py,
        str(STEPS / "step6_full_generation.py"),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.mock:
        gen_cmd.extend(["--mock-generation", "--mock-liwc"])
    if args.smoke:
        gen_cmd.append("--smoke")
    run(gen_cmd)

    print("\nFull continuous generation complete.")
    print("  generations: optimized/results/full/generations/")
    print("  Next: step6_embedding_probe / recovery viz / step8_bfi_validation")


if __name__ == "__main__":
    main()
