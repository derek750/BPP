#!/usr/bin/env python3
"""Cell discriminability viz for continuous full generations.

Same layout as ``scripts/stage2_step8_personality_discriminability_viz.py``:
LDA shows structure that predicted-O/N and PCA hide.

  - Top: supervised LDA-2 of mpnet-personality embeddings by strategy
  - Bottom-left: unsupervised PCA contrast
  - Bottom-right: 5-fold CV accuracy by representation × strategy

Example:
  python optimized/step6_embedding_discriminability_viz.py \\
    --output-dir results/full/embedding \\
    --generations results/full/generations/full_generations.csv \\
    --title "Continuous full"
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
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OPTIMIZED_DIR, PILOT_RESULTS_DIR, FULL_RESULTS_DIR, RESULTS_DIR, STEERING_CONDITIONS, ensure_results_dir
from step6_embedding_viz import (
    CELL_COLORS,
    CELL_ORDER,
    cell_legend_handles,
    ordered_conditions,
    point_size,
    scatter_pca,
)

DEFAULT_OUTPUT_DIR = FULL_RESULTS_DIR / "embedding"
DEFAULT_GENERATIONS = FULL_RESULTS_DIR / "generations" / "full_generations.csv"
CONDITION_ORDER_SHORT = list(STEERING_CONDITIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    p.add_argument("--per-sample", type=Path, default=None)
    p.add_argument("--embeddings", type=Path, default=None)
    p.add_argument("--title", type=str, default="Continuous full")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def cv_accuracy(X: np.ndarray, y: np.ndarray, *, seed: int, n_splits: int = 5) -> float:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(pipe, X, y, cv=cv)
    return float(np.mean(preds == y))


def lda_xy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_classes = len(np.unique(y))
    lda = LinearDiscriminantAnalysis(n_components=min(2, n_classes - 1))
    xy = lda.fit_transform(X, y)
    if xy.shape[1] == 1:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    return xy


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def human_trait_axes(
    human_emb: np.ndarray,
    human: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous analog of high−low cell centroid axes from human windows."""
    o = human["O"].to_numpy(dtype=float)
    n = human["N"].to_numpy(dtype=float)
    o_hi = human_emb[o >= np.quantile(o, 2 / 3)]
    o_lo = human_emb[o <= np.quantile(o, 1 / 3)]
    n_hi = human_emb[n >= np.quantile(n, 2 / 3)]
    n_lo = human_emb[n <= np.quantile(n, 1 / 3)]
    o_axis = l2_normalize(
        (l2_normalize(o_hi.mean(axis=0, keepdims=True))
         - l2_normalize(o_lo.mean(axis=0, keepdims=True)))
    )[0]
    n_axis = l2_normalize(
        (l2_normalize(n_hi.mean(axis=0, keepdims=True))
         - l2_normalize(n_lo.mean(axis=0, keepdims=True)))
    )[0]
    return o_axis, n_axis


def main() -> None:
    args = parse_args()
    out = _resolve(args.output_dir)
    ensure_results_dir(out)

    gen_path = _resolve(args.generations)
    per_sample_path = _resolve(args.per_sample or (out / "embedding_per_sample.csv"))
    emb_path = _resolve(args.embeddings or (out / "personality_embeddings.npz"))
    human_path = out / "human_windows.csv"

    gens = pd.read_csv(gen_path)
    scored = pd.read_csv(per_sample_path)
    npz = np.load(emb_path, allow_pickle=True)
    gen_emb = np.asarray(npz["generation_embeddings"], dtype=np.float32)
    human_emb = np.asarray(npz["human_embeddings"], dtype=np.float32)

    # Align generations to embedding/per-sample order via plan_id when possible.
    if "plan_id" in scored.columns and "plan_id" in gens.columns:
        gens = gens.set_index("plan_id").loc[scored["plan_id"]].reset_index()
    if len(gens) != len(scored) or len(scored) != len(gen_emb):
        raise RuntimeError(
            f"Row mismatch: gens={len(gens)} scored={len(scored)} emb={len(gen_emb)}"
        )

    if "cell" not in scored.columns or scored["cell"].isna().all():
        raise SystemExit("per-sample table needs a cell column for discriminability.")

    liwc_cols = [c for c in gens.columns if c.startswith("obs_")]
    if not liwc_cols:
        raise SystemExit("No obs_* LIWC columns found in generations CSV.")

    cells = [c for c in CELL_ORDER if c in set(scored["cell"])]
    cells += sorted(set(scored["cell"]) - set(cells))
    conditions = ordered_conditions(scored)
    y = scored["cell"].astype(str).to_numpy()
    chance = 1.0 / len(cells)

    # O/N axes: prefer human high−low axes; fall back to probe predictions.
    if human_path.exists():
        human = pd.read_csv(human_path)
        if len(human) == len(human_emb):
            o_axis, n_axis = human_trait_axes(human_emb, human)
            X_on = np.column_stack([gen_emb @ o_axis, gen_emb @ n_axis])
            on_label = "O/N axes"
        else:
            X_on = scored[["pred_O", "pred_N"]].to_numpy(dtype=float)
            on_label = "Pred O/N"
    else:
        X_on = scored[["pred_O", "pred_N"]].to_numpy(dtype=float)
        on_label = "Pred O/N"

    X_liwc = gens[liwc_cols].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for cond in conditions:
        mask = scored["steering_condition"].to_numpy() == cond
        y_c = y[mask]
        emb_c = gen_emb[mask]
        pca_c = PCA(n_components=2, random_state=args.seed).fit_transform(emb_c)
        lda_c = lda_xy(emb_c, y_c)
        for name, X in (
            ("LIWC (8-d)", X_liwc[mask]),
            ("Emb 768-d", emb_c),
            (on_label, X_on[mask]),
            ("PCA-2", pca_c),
            ("LDA-2", lda_c),
        ):
            acc = cv_accuracy(X, y_c, seed=args.seed)
            rows.append(
                {
                    "steering_condition": cond,
                    "representation": name,
                    "cv_accuracy": acc,
                    "n": int(mask.sum()),
                }
            )
    acc_df = pd.DataFrame(rows)
    acc_path = out / "discriminability_cv.csv"
    acc_df.to_csv(acc_path, index=False)
    print(f"Wrote {acc_path}")
    print(acc_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    n_cond = len(conditions)
    fig = plt.figure(figsize=(3.8 * n_cond + 0.4, 8.2))
    gs = fig.add_gridspec(2, n_cond, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22)
    s = point_size(len(scored) // max(n_cond, 1))

    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[0, i])
        mask = scored["steering_condition"].to_numpy() == cond
        xy = lda_xy(gen_emb[mask], y[mask])
        scatter_pca(ax, xy, y[mask], cells, s=s, alpha=0.5)
        sub_acc = float(
            acc_df.loc[
                (acc_df["steering_condition"] == cond)
                & (acc_df["representation"] == "Emb 768-d"),
                "cv_accuracy",
            ].iloc[0]
        )
        ax.set_title(f"{cond}\n768-d CV acc={sub_acc:.0%}", fontsize=10)
        ax.set_xlabel("LD1", fontsize=8)
        if i == 0:
            ax.set_ylabel("LD2", fontsize=8)

    # PCA contrast: strongest Emb 768-d condition
    emb_acc = (
        acc_df[acc_df["representation"] == "Emb 768-d"]
        .sort_values("cv_accuracy", ascending=False)
    )
    contrast_cond = str(emb_acc.iloc[0]["steering_condition"])
    ax_pca = fig.add_subplot(gs[1, 0])
    mask = scored["steering_condition"].to_numpy() == contrast_cond
    pca_xy = PCA(n_components=2, random_state=args.seed).fit_transform(gen_emb[mask])
    scatter_pca(ax_pca, pca_xy, y[mask], cells, s=max(6, s - 4), alpha=0.45)
    pca_acc = float(
        acc_df.loc[
            (acc_df["steering_condition"] == contrast_cond)
            & (acc_df["representation"] == "PCA-2"),
            "cv_accuracy",
        ].iloc[0]
    )
    ax_pca.set_title(
        f"PCA contrast ({contrast_cond})\nPCA-2 CV acc={pca_acc:.0%}",
        fontsize=9,
    )
    ax_pca.set_xlabel("PC1", fontsize=8)
    ax_pca.set_ylabel("PC2", fontsize=8)

    ax_bar = fig.add_subplot(gs[1, 1:])
    reps = ["LIWC (8-d)", "Emb 768-d", on_label, "PCA-2", "LDA-2"]
    plot_conds = [c for c in CONDITION_ORDER_SHORT if c in conditions] or conditions
    x = np.arange(len(reps))
    width = 0.8 / max(len(plot_conds), 1)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    for j, cond in enumerate(plot_conds):
        vals = [
            float(
                acc_df.loc[
                    (acc_df["steering_condition"] == cond)
                    & (acc_df["representation"] == r),
                    "cv_accuracy",
                ].iloc[0]
            )
            for r in reps
        ]
        ax_bar.bar(
            x + j * width - 0.4 + width / 2,
            vals,
            width=width * 0.9,
            label=cond,
            color=palette[j % len(palette)],
        )
    ax_bar.axhline(chance, color="#666", ls="--", lw=1, label=f"chance ({chance:.0%})")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(reps, rotation=15, ha="right", fontsize=8)
    ax_bar.set_ylabel("5-fold CV accuracy", fontsize=9)
    ax_bar.set_ylim(0, max(0.75, float(acc_df["cv_accuracy"].max()) + 0.08))
    ax_bar.set_title(
        "Cell classification by representation (same labels, held-out CV)",
        fontsize=9,
    )
    ax_bar.legend(frameon=False, fontsize=7, loc="upper right")

    fig.legend(
        handles=cell_legend_handles(cells),
        loc="lower center",
        ncol=min(len(cells), 5),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"{args.title} · cell discriminability "
        f"(LDA shows structure that O/N & PCA hide)",
        fontsize=11,
        y=0.995,
    )
    path = out / "personality_embedding_viz_discriminability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Vertical LDA-only figure (single-column paper layout)
    # ------------------------------------------------------------------
    plot_conds = [c for c in CONDITION_ORDER_SHORT if c in conditions] or conditions
    n_v = len(plot_conds)
    fig_v, axes_v = plt.subplots(
        n_v,
        1,
        figsize=(3.25, 7.6),
        sharex=False,
        sharey=False,
    )
    if n_v == 1:
        axes_v = [axes_v]
    s_v = max(7, point_size(len(scored) // max(n_v, 1)) - 2)
    for ax, cond in zip(axes_v, plot_conds):
        mask = scored["steering_condition"].to_numpy() == cond
        xy = lda_xy(gen_emb[mask], y[mask])
        scatter_pca(ax, xy, y[mask], cells, s=s_v, alpha=0.55)
        sub_acc = float(
            acc_df.loc[
                (acc_df["steering_condition"] == cond)
                & (acc_df["representation"] == "Emb 768-d"),
                "cv_accuracy",
            ].iloc[0]
        )
        ax.set_title(f"{cond}  ·  768-d CV={sub_acc:.0%}", fontsize=8.5, pad=2)
        ax.set_xlabel("LD1", fontsize=7)
        ax.set_ylabel("LD2", fontsize=7)
        ax.tick_params(labelsize=6.5)
    fig_v.legend(
        handles=cell_legend_handles(cells),
        loc="lower center",
        ncol=3,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig_v.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6)
    path_v = out / "personality_embedding_viz_lda_vertical.png"
    fig_v.savefig(path_v, dpi=220, bbox_inches="tight")
    print(f"Wrote {path_v}")
    plt.close(fig_v)

    # ------------------------------------------------------------------
    # Horizontal LDA-only row (full-width paper page)
    # ------------------------------------------------------------------
    def _short_cell(c: str) -> str:
        parts = c.replace("__", "_").split("_")
        o = {"low": "↓", "med": "–", "high": "↑"}.get(parts[0], "?")
        n = {"low": "↓", "med": "–", "high": "↑"}.get(parts[2], "?")
        return f"O{o}N{n}"

    n_row = len(plot_conds)
    fig_r, axes_r = plt.subplots(
        1,
        n_row,
        figsize=(3.55 * n_row + 0.2, 4.6),
        sharex=False,
        sharey=False,
    )
    if n_row == 1:
        axes_r = [axes_r]
    s_r = max(8, point_size(len(scored) // max(n_row, 1)) - 1)
    for ax, cond in zip(axes_r, plot_conds):
        mask = scored["steering_condition"].to_numpy() == cond
        xy = lda_xy(gen_emb[mask], y[mask])
        scatter_pca(ax, xy, y[mask], cells, s=s_r, alpha=0.55)
        sub_acc = float(
            acc_df.loc[
                (acc_df["steering_condition"] == cond)
                & (acc_df["representation"] == "Emb 768-d"),
                "cv_accuracy",
            ].iloc[0]
        )
        ax.set_title(f"{cond}  ·  CV={sub_acc:.0%}", fontsize=9.5, pad=3)
        ax.set_xlabel("LD1", fontsize=8)
        ax.set_ylabel("LD2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    short_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CELL_COLORS.get(c, "#888888"),
            markersize=7,
            label=_short_cell(c),
        )
        for c in cells
    ]
    fig_r.legend(
        handles=short_handles,
        loc="lower center",
        ncol=min(len(cells), 9),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig_r.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.85)
    path_r = out / "personality_embedding_viz_lda_row.png"
    fig_r.savefig(path_r, dpi=220, bbox_inches="tight")
    print(f"Wrote {path_r}")
    plt.close(fig_r)

    for cond in plot_conds:
        for rep in ("Emb 768-d", on_label, "PCA-2"):
            acc = float(
                acc_df.loc[
                    (acc_df["steering_condition"] == cond)
                    & (acc_df["representation"] == rep),
                    "cv_accuracy",
                ].iloc[0]
            )
            n = int((scored["steering_condition"] == cond).sum())
            k = int(round(acc * n))
            p = stats.binomtest(k, n, chance, alternative="greater").pvalue
            print(f"  {cond:16s} {rep:12s} acc={acc:.3f} p={p:.2e}")


if __name__ == "__main__":
    main()
