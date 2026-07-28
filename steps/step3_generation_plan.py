#!/usr/bin/env python3
"""Step 3: Expand synthetic authors into a matched generation plan.

Each synthetic profile is a stable author identity. Essays may vary by topic,
model, repetition, and steering condition; O/N/LIWC targets stay fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
from pathlib import Path

import pandas as pd

from common import (
    CONTENT_CONDITIONS,
    DEFAULT_N_REPS,
    DEFAULT_PILOT_CONTENTS,
    OPTIMIZED_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_DIR,
    FULL_RESULTS_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)

DEFAULT_INPUT = PILOT_RESULTS_DIR / "step2_synthetic_profiles.csv"
DEFAULT_OUTPUT = PILOT_RESULTS_DIR / "step3_generation_plan.csv"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step3_generation_plan.report.json"
DEFAULT_MODEL = "deepseek-chat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--contents",
        nargs="+",
        default=list(DEFAULT_PILOT_CONTENTS),
        choices=sorted(CONTENT_CONDITIONS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(STEERING_CONDITIONS),
        choices=list(STEERING_CONDITIONS),
    )
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def build_generation_plan(
    profiles: pd.DataFrame,
    *,
    contents: list[str],
    conditions: list[str],
    n_reps: int,
    model: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, profile in profiles.iterrows():
        profile_id = str(profile["profile_id"])
        for content in contents:
            for condition in conditions:
                for rep in range(n_reps):
                    rows.append(
                        {
                            "plan_id": (
                                f"{profile_id}__{content}__{condition}__r{rep}"
                            ),
                            "profile_id": profile_id,
                            "content": content,
                            "steering_condition": condition,
                            "repetition": rep,
                            "model": model,
                            "target_O": float(profile["O"]),
                            "target_N": float(profile["N"]),
                        }
                    )
    plan = pd.DataFrame(rows)
    # Completion check: equal exposure across conditions.
    counts = plan.groupby("steering_condition").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"Uneven condition exposure: {counts.to_dict()}")
    return plan


def main() -> None:
    args = parse_args()
    profiles = pd.read_csv(_resolve_out(args.input))
    plan = build_generation_plan(
        profiles,
        contents=args.contents,
        conditions=args.conditions,
        n_reps=args.n_reps,
        model=args.model,
    )
    output = _resolve_out(args.output)
    report = _resolve_out(args.report)
    ensure_results_dir(output.parent)
    plan.to_csv(output, index=False)
    write_json(
        report,
        {
            "output": str(output),
            "n_profiles": int(profiles["profile_id"].nunique()),
            "n_plan_rows": int(len(plan)),
            "contents": args.contents,
            "conditions": args.conditions,
            "n_reps": args.n_reps,
            "model": args.model,
            "rows_per_condition": {
                k: int(v)
                for k, v in plan.groupby("steering_condition").size().items()
            },
        },
    )
    print(f"Wrote generation plan ({len(plan)} rows) -> {output}")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
