#!/usr/bin/env python3
"""Run the continuous BFI–LIWC optimized pipeline through the pilot.

Steps 1–4 are local/CPU-only. Step 5 calls DeepSeek + LIWC-22 unless mocked.

Examples:
  python optimized/runners/run_pilot.py --mock
  python optimized/runners/run_pilot.py --mock --n-synthetic 8 --limit 24
  python optimized/runners/run_pilot.py                 # real API + LIWC-22
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OPTIMIZED_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = OPTIMIZED_DIR.parent
STEPS = OPTIMIZED_DIR / "steps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock generation and LIWC scoring (no API / no LIWC app).",
    )
    parser.add_argument("--mock-generation", action="store_true")
    parser.add_argument("--mock-liwc", action="store_true")
    parser.add_argument("--n-synthetic", type=int, default=12)
    parser.add_argument("--n-reps", type=int, default=1)
    parser.add_argument(
        "--contents",
        nargs="+",
        default=["weekend", "technology"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional Step 5 plan-row cap (smoke test).",
    )
    parser.add_argument(
        "--skip-through",
        type=int,
        default=0,
        help="Skip steps ≤ N (1–5) if prior outputs already exist.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional Step 3/5 condition subset (e.g. lex_fewshot).",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Step 5: keep prior generations not being regenerated.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    mock_gen = args.mock or args.mock_generation
    mock_liwc = args.mock or args.mock_liwc
    py = sys.executable

    step3 = [
        py,
        str(STEPS / "step3_generation_plan.py"),
        "--n-reps",
        str(args.n_reps),
        "--contents",
        *args.contents,
    ]
    if args.conditions:
        step3.extend(["--conditions", *args.conditions])

    steps = [
        (1, [py, str(STEPS / "step1_author_profiles.py")]),
        (
            2,
            [
                py,
                str(STEPS / "step2_gaussian_copula.py"),
                "--n-synthetic",
                str(args.n_synthetic),
                "--seed",
                str(args.seed),
            ],
        ),
        (3, step3),
        (4, [py, str(STEPS / "step4_prompt_components.py")]),
    ]

    step5 = [
        py,
        str(STEPS / "step5_pilot_generation.py"),
        "--seed",
        str(args.seed),
    ]
    if mock_gen:
        step5.append("--mock-generation")
    if mock_liwc:
        step5.append("--mock-liwc")
    if args.limit is not None:
        step5.extend(["--limit", str(args.limit)])
    if args.conditions:
        step5.extend(["--conditions", *args.conditions])
    if args.merge_existing:
        step5.append("--merge-existing")
    steps.append((5, step5))

    for step_id, cmd in steps:
        if step_id <= args.skip_through:
            print(f"Skipping step {step_id}")
            continue
        run(cmd)

    print("\nPilot pipeline complete. Outputs under optimized/results/pilot/")


if __name__ == "__main__":
    main()
