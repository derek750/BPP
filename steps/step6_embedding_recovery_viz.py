#!/usr/bin/env python3
"""Continuous embedding recovery figure (replaces 9-cell LDA visual).

Plots author-level assigned vs mpnet-Ridge predicted O/N by steering condition.
Primary visual for Validation 2 under the continuous Gaussian design.
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import FULL_RESULTS_DIR, OPTIMIZED_DIR, ensure_results_dir

DEFAULT_AUTHOR = FULL_RESULTS_DIR / "embedding" / "embedding_author_level.csv"
DEFAULT_OUT = (
    OPTIMIZED_DIR.parent
    / "papers"
    / "paper"
    / "figures"
    / "fig_embedding_recovery.png"
)

CONDITION_ORDER = ("persona_only", "liwc_only", "persona_liwc", "lex_fewshot")
CONDITION_LABELS = {
    "persona_only": "Persona",
    "liwc_only": "LIWC",
    "persona_liwc": "Persona+LIWC",
    "lex_fewshot": "LIWC+Fewshot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-level", type=Path, default=DEFAULT_AUTHOR)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITION_ORDER),
    )
    return parser.parse_args()


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    r, _ = stats.spearmanr(x, y)
    return float(r) if r == r else float("nan")


def main() -> None:
    args = parse_args()
    path = args.author_level
    if not path.is_absolute():
        path = (OPTIMIZED_DIR / path).resolve()
    if not path.exists():
        raise SystemExit(f"Author-level embedding CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"steering_condition", "target_O", "target_N", "pred_O", "pred_N"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    conditions = [c for c in args.conditions if c in set(df["steering_condition"])]
    if not conditions:
        raise SystemExit("No requested conditions present in author-level CSV.")

    n_cols = len(conditions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = df[df["steering_condition"] == cond]
        o_x = sub["target_O"].to_numpy(dtype=float)
        o_y = sub["pred_O"].to_numpy(dtype=float)
        n_x = sub["target_N"].to_numpy(dtype=float)
        n_y = sub["pred_N"].to_numpy(dtype=float)
        ax.scatter(o_x, o_y, s=18, alpha=0.65, label="Openness", color="#1f77b4")
        ax.scatter(n_x, n_y, s=18, alpha=0.65, label="Neuroticism", color="#d62728")
        lims = [0, 100]
        ax.plot(lims, lims, ls="--", lw=1, color="0.5", zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        rho_o = _rho(o_x, o_y)
        rho_n = _rho(n_x, n_y)
        title = CONDITION_LABELS.get(cond, cond)
        ax.set_title(f"{title}\n$\\rho_O$={rho_o:.2f}, $\\rho_N$={rho_n:.2f}", fontsize=10)
        ax.set_xlabel("Assigned")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Predicted (mpnet Ridge)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Embedding recovery: assigned vs predicted O/N (author-level)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    out = args.output
    if not out.is_absolute():
        out = (OPTIMIZED_DIR / out).resolve()
    ensure_results_dir(out.parent)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    # Also copy into full embedding results for the package.
    alt = FULL_RESULTS_DIR / "embedding" / "fig_embedding_recovery.png"
    ensure_results_dir(alt.parent)
    fig.savefig(alt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {alt}")


if __name__ == "__main__":
    main()
