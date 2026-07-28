#!/usr/bin/env python3
"""Prepare full continuous multi-model generation inputs.

Samples synthetic authors directly from a Gaussian copula (fixed seed; QC
reject-and-replace only — no tertile cell quotas), builds the matched plan,
and prompt components (persona / LIWC / nearest-neighbor few-shot).

Default grid:
  80 profiles × 6 narrative topics × 4 conditions × 3 models × 1 rep = 5760
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_FULL_CONTENTS,
    DEFAULT_N_SYNTHETIC_FULL,
    DEFAULT_SEED,
    FULL_CONTENT_CONDITIONS,
    FULL_RESULTS_DIR,
    OPTIMIZED_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)

FULL_DIR = FULL_RESULTS_DIR
STEPS = OPTIMIZED_DIR / "steps"
MODELS = ("deepseek-v3", "qwen3-32b", "gpt-4o-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=DEFAULT_N_SYNTHETIC_FULL,
        help="Number of Gaussian-copula synthetic profiles (default 80).",
    )
    parser.add_argument(
        "--contents",
        nargs="+",
        default=list(DEFAULT_FULL_CONTENTS),
        choices=sorted(FULL_CONTENT_CONDITIONS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(STEERING_CONDITIONS),
        choices=list(STEERING_CONDITIONS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
    )
    parser.add_argument("--n-reps", type=int, default=1)
    parser.add_argument("--skip-step1", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(OPTIMIZED_DIR.parent))


def build_multi_model_plan(
    profiles: pd.DataFrame,
    *,
    contents: list[str],
    conditions: list[str],
    models: list[str],
    n_reps: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, profile in profiles.iterrows():
        profile_id = str(profile["profile_id"])
        for model in models:
            for content in contents:
                for condition in conditions:
                    for rep in range(n_reps):
                        rows.append(
                            {
                                "plan_id": (
                                    f"{model}__{profile_id}__{content}__"
                                    f"{condition}__r{rep}"
                                ),
                                "profile_id": profile_id,
                                "content": content,
                                "steering_condition": condition,
                                "repetition": rep,
                                "model_key": model,
                                "target_O": float(profile["O"]),
                                "target_N": float(profile["N"]),
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    py = sys.executable
    ensure_results_dir(FULL_DIR)

    if not args.skip_step1:
        run([py, str(STEPS / "step1_author_profiles.py")])

    profiles_path = FULL_DIR / "step2_synthetic_profiles.csv"
    run(
        [
            py,
            str(STEPS / "step2_gaussian_copula.py"),
            "--n-synthetic",
            str(args.n_synthetic),
            "--seed",
            str(args.seed),
            "-o",
            str(profiles_path),
            "-r",
            str(FULL_DIR / "step2_synthetic_profiles.report.json"),
        ]
    )

    selected = pd.read_csv(profiles_path)
    if "cell" in selected.columns:
        selected = selected.drop(columns=["cell"])
        selected.to_csv(profiles_path, index=False)
    print(f"Wrote Gaussian profiles -> {profiles_path} ({len(selected)})")

    plan = build_multi_model_plan(
        selected,
        contents=args.contents,
        conditions=args.conditions,
        models=args.models,
        n_reps=args.n_reps,
    )
    plan_path = FULL_DIR / "step3_generation_plan.csv"
    plan.to_csv(plan_path, index=False)
    print(f"Wrote full plan ({len(plan)} rows) -> {plan_path}")

    run(
        [
            py,
            str(STEPS / "step4_prompt_components.py"),
            "--synthetic",
            str(profiles_path),
            "-o",
            str(FULL_DIR / "step4_prompt_components.csv"),
            "-r",
            str(FULL_DIR / "step4_prompt_components.report.json"),
            "--exemplar-cache",
            str(FULL_DIR / "step4_fewshot_exemplars.json"),
            "--rebuild-exemplars",
        ]
    )

    write_json(
        FULL_DIR / "prepare_full.report.json",
        {
            "n_profiles": int(len(selected)),
            "n_plan_rows": int(len(plan)),
            "sampling": "gaussian_copula_direct",
            "contents": args.contents,
            "conditions": args.conditions,
            "models": args.models,
            "n_reps": args.n_reps,
            "rows_per_condition": {
                k: int(v)
                for k, v in plan.groupby("steering_condition").size().items()
            },
            "rows_per_model": {
                k: int(v) for k, v in plan.groupby("model_key").size().items()
            },
            "expected_total": int(len(plan)),
        },
    )
    print(
        f"Grid: {len(selected)} profiles × {len(args.contents)} topics × "
        f"{len(args.conditions)} conditions × {len(args.models)} models × "
        f"{args.n_reps} reps = {len(plan)}"
    )


if __name__ == "__main__":
    main()
