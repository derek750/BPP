#!/usr/bin/env python3
"""Step 7: Secondary 9-cell LIWC classification on continuous full generations.

Primary continuous trait recovery is Validation 2 (`step6_embedding_probe.py`:
Ridge O/N from mpnet embeddings trained on humans only).

This script is a secondary discrete diagnostic for comparability with earlier
categorical analyses: each synthetic profile carries a tertile cell label
(from continuous O/N via human cutoffs). A logistic classifier is trained on
observed 8-d LIWC rates to recover that cell, by steering condition and model.

Also reports pooled-over-models accuracies and writes a simple bar figure.
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    FULL_RESULTS_DIR,
    OPTIMIZED_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)

FULL_DIR = FULL_RESULTS_DIR
DEFAULT_GENERATIONS = FULL_DIR / "generations" / "full_generations.csv"
DEFAULT_OUTPUT_DIR = FULL_DIR / "classification"
CHANCE_9 = 1.0 / 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=2)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def categories_from_frame(frame: pd.DataFrame) -> list[str]:
    return [
        c.replace("obs_", "", 1)
        for c in frame.columns
        if c.startswith("obs_")
    ]


def cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int,
) -> float:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    max_splits = int(counts.min())
    if max_splits < 2:
        return float("nan")
    n_splits = min(n_splits, max_splits)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = cross_val_predict(pipe, X, y, cv=cv)
    return float(np.mean(preds == y))


def evaluate_slice(
    frame: pd.DataFrame,
    categories: list[str],
    *,
    seed: int,
    n_splits: int,
    min_class_count: int,
) -> dict:
    usable = frame[
        frame["error"].isna()
        & frame["scored"].astype(bool)
        & frame["cell"].notna()
    ].copy()
    if usable.empty:
        return {"n": 0, "n_classes": 0, "cv_accuracy": float("nan")}

    # Drop rare cells that break stratified CV.
    counts = usable["cell"].value_counts()
    keep_cells = counts[counts >= min_class_count].index
    usable = usable[usable["cell"].isin(keep_cells)].copy()
    if usable.empty or usable["cell"].nunique() < 2:
        return {
            "n": int(len(usable)),
            "n_classes": int(usable["cell"].nunique()) if len(usable) else 0,
            "cv_accuracy": float("nan"),
            "dropped_rare_cells": int((counts < min_class_count).sum()),
        }

    X = usable[[f"obs_{c}" for c in categories]].to_numpy(dtype=float)
    y = usable["cell"].to_numpy()
    acc = cv_accuracy(X, y, seed=seed, n_splits=n_splits)
    return {
        "n": int(len(usable)),
        "n_classes": int(len(np.unique(y))),
        "cv_accuracy": acc,
        "chance": CHANCE_9,
        "cell_counts": usable["cell"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    gens_path = _resolve(args.generations)
    out_dir = _resolve(args.output_dir)
    ensure_results_dir(out_dir)

    frame = pd.read_csv(gens_path)
    if "cell" not in frame.columns:
        raise SystemExit("Generations CSV missing 'cell' column.")
    categories = categories_from_frame(frame)
    if not categories:
        raise SystemExit("No obs_* LIWC columns found — score generations first.")

    rows: list[dict] = []
    detail: dict = {"by_condition_model": {}, "by_condition": {}, "by_model": {}}

    for condition in sorted(frame["steering_condition"].unique()):
        for model in sorted(frame["model_key"].unique()):
            subset = frame[
                (frame["steering_condition"] == condition)
                & (frame["model_key"] == model)
            ]
            metrics = evaluate_slice(
                subset,
                categories,
                seed=args.seed,
                n_splits=args.n_splits,
                min_class_count=args.min_class_count,
            )
            detail["by_condition_model"][f"{condition}__{model}"] = metrics
            rows.append(
                {
                    "steering_condition": condition,
                    "model_key": model,
                    "representation": "LIWC (8-d)",
                    "n": metrics["n"],
                    "n_classes": metrics["n_classes"],
                    "cv_accuracy": metrics["cv_accuracy"],
                    "chance": CHANCE_9,
                }
            )

        subset = frame[frame["steering_condition"] == condition]
        metrics = evaluate_slice(
            subset,
            categories,
            seed=args.seed,
            n_splits=args.n_splits,
            min_class_count=args.min_class_count,
        )
        detail["by_condition"][condition] = metrics
        rows.append(
            {
                "steering_condition": condition,
                "model_key": "ALL",
                "representation": "LIWC (8-d)",
                "n": metrics["n"],
                "n_classes": metrics["n_classes"],
                "cv_accuracy": metrics["cv_accuracy"],
                "chance": CHANCE_9,
            }
        )

    for model in sorted(frame["model_key"].unique()):
        subset = frame[frame["model_key"] == model]
        metrics = evaluate_slice(
            subset,
            categories,
            seed=args.seed,
            n_splits=args.n_splits,
            min_class_count=args.min_class_count,
        )
        detail["by_model"][model] = metrics

    summary = pd.DataFrame(rows)
    cond_order = [c for c in STEERING_CONDITIONS if c in set(summary["steering_condition"])]
    summary["steering_condition"] = pd.Categorical(
        summary["steering_condition"], categories=cond_order, ordered=True
    )
    summary = summary.sort_values(["steering_condition", "model_key"]).reset_index(drop=True)
    summary["steering_condition"] = summary["steering_condition"].astype(str)

    csv_path = out_dir / "classification_cv_summary.csv"
    summary.to_csv(csv_path, index=False)
    write_json(out_dir / "classification.report.json", {"detail": detail, "summary_csv": str(csv_path)})

    # Bar figure: accuracy by condition × model (+ ALL).
    plot_df = summary[summary["model_key"] != "ALL"].copy()
    if not plot_df.empty and plot_df["cv_accuracy"].notna().any():
        models = sorted(plot_df["model_key"].unique())
        conditions = [c for c in STEERING_CONDITIONS if c in set(plot_df["steering_condition"])]
        x = np.arange(len(conditions))
        width = 0.8 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, model in enumerate(models):
            vals = []
            for cond in conditions:
                hit = plot_df[
                    (plot_df["steering_condition"] == cond)
                    & (plot_df["model_key"] == model)
                ]
                vals.append(float(hit["cv_accuracy"].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + i * width, vals, width=width, label=model)
        ax.axhline(CHANCE_9, color="gray", linestyle="--", linewidth=1, label="chance (1/9)")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.set_ylabel("5-fold CV accuracy (LIWC 8-d → cell)")
        ax.set_title("Continuous full-gen: cell classification by strategy × model")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(0.5, float(np.nanmax(plot_df["cv_accuracy"])) + 0.05))
        fig.tight_layout()
        fig_path = out_dir / "classification_cv_by_model.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Wrote figure -> {fig_path}")

    print(f"Wrote classification summary -> {csv_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
