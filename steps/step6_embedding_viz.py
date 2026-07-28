#!/usr/bin/env python3
"""Render continuous mpnet-personality visuals (pilot or full).

Analogous to ``scripts/stage2_step8_personality_embedding_viz.py``:

  - personality_embedding_viz.png
      pred O/N scatter faceted by strategy + recovery-ρ ranking
  - personality_embedding_viz_pca_by_strategy.png
      PCA faceted by strategy
  - personality_embedding_viz_by_model.png  (multi-model only)
      rows = strategy, columns = model

Example:
  python optimized/step6_embedding_viz.py \\
    --output-dir results/full/embedding \\
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
from sklearn.decomposition import PCA

from common import OPTIMIZED_DIR, PILOT_RESULTS_DIR, FULL_RESULTS_DIR, RESULTS_DIR, STEERING_CONDITIONS, ensure_results_dir

DEFAULT_OUTPUT_DIR = PILOT_RESULTS_DIR / "step6_embedding"
DEFAULT_PER_SAMPLE = DEFAULT_OUTPUT_DIR / "embedding_per_sample.csv"
DEFAULT_EMBEDDINGS = DEFAULT_OUTPUT_DIR / "personality_embeddings.npz"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "embedding_condition_summary.csv"

CELL_ORDER = [
    "low_O__low_N",
    "low_O__med_N",
    "low_O__high_N",
    "med_O__low_N",
    "med_O__med_N",
    "med_O__high_N",
    "high_O__low_N",
    "high_O__med_N",
    "high_O__high_N",
]
CELL_COLORS = {
    "low_O__low_N": "#4C78A8",
    "low_O__med_N": "#72B7B2",
    "low_O__high_N": "#F58518",
    "med_O__low_N": "#54A24B",
    "med_O__med_N": "#B279A2",
    "med_O__high_N": "#FF9DA6",
    "high_O__low_N": "#9D755D",
    "high_O__med_N": "#BAB0AC",
    "high_O__high_N": "#E45756",
}

CONDITION_ORDER = list(STEERING_CONDITIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--per-sample", type=Path, default=None)
    p.add_argument("--embeddings", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--title", type=str, default="Continuous")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def tertile_labels(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "med", "high"]).astype(str)


def ensure_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use existing cell labels when present; else soft tertiles of targets."""
    out = frame.copy()
    if "cell" in out.columns and out["cell"].notna().any():
        out["cell"] = out["cell"].astype(str)
        return out, "profile cell"
    o_lvl = tertile_labels(out["target_O"])
    n_lvl = tertile_labels(out["target_N"])
    out["cell"] = [f"{o}_O__{n}_N" for o, n in zip(o_lvl, n_lvl)]
    return out, "target tertile cell (viz only)"


def ordered_conditions(frame: pd.DataFrame) -> list[str]:
    present = set(frame["steering_condition"])
    ordered = [c for c in CONDITION_ORDER if c in present]
    ordered += sorted(present - set(ordered))
    return ordered


def facet_grid_shape(n: int) -> tuple[int, int]:
    if n <= 4:
        return 1, n
    if n <= 6:
        return 2, int(np.ceil(n / 2))
    return 3, int(np.ceil(n / 3))


def point_size(n: int) -> float:
    if n >= 1000:
        return 8
    if n >= 400:
        return 12
    if n >= 100:
        return 18
    return 28


def cell_legend_handles(cells: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CELL_COLORS.get(c, "#888888"),
            markersize=7,
            label=c,
        )
        for c in cells
    ]


def scatter_pred(
    ax,
    frame: pd.DataFrame,
    cells: list[str],
    *,
    s: float,
    alpha: float = 0.55,
) -> None:
    for cell in cells:
        sub = frame[frame["cell"] == cell]
        if sub.empty:
            continue
        ax.scatter(
            sub["pred_O"],
            sub["pred_N"],
            c=CELL_COLORS.get(cell, "#888888"),
            s=s,
            alpha=alpha,
            linewidths=0,
        )
    ax.axhline(frame["pred_N"].mean(), color="#999", lw=0.6, ls="--")
    ax.axvline(frame["pred_O"].mean(), color="#999", lw=0.6, ls="--")


def scatter_pca(
    ax,
    xy: np.ndarray,
    cells_arr: np.ndarray,
    cells: list[str],
    *,
    s: float,
    alpha: float = 0.5,
) -> None:
    for cell in cells:
        m = cells_arr == cell
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c=CELL_COLORS.get(cell, "#888888"),
            s=s,
            alpha=alpha,
            linewidths=0,
        )


def main() -> None:
    args = parse_args()
    out = _resolve(args.output_dir)
    ensure_results_dir(out)

    per_sample_path = _resolve(args.per_sample or (out / "embedding_per_sample.csv"))
    emb_path = _resolve(args.embeddings or (out / "personality_embeddings.npz"))
    summary_path = _resolve(args.summary or (out / "embedding_condition_summary.csv"))

    scored = pd.read_csv(per_sample_path)
    scored, cell_legend_title = ensure_cells(scored)
    if "model_key" not in scored.columns:
        scored["model_key"] = scored["model"].astype(str) if "model" in scored.columns else "all"

    npz = np.load(emb_path, allow_pickle=True)
    gen_emb = npz["generation_embeddings"]
    human_emb = npz["human_embeddings"]
    if len(scored) != len(gen_emb):
        raise RuntimeError(
            f"Row mismatch: per-sample={len(scored)} vs gen_emb={len(gen_emb)}"
        )

    summary = pd.read_csv(summary_path)
    rho_lookup = {
        (r.steering_condition, r.trait): float(r.rho)
        for r in summary.itertuples()
        if pd.notna(r.rho)
    }

    cells = [c for c in CELL_ORDER if c in set(scored["cell"])]
    cells += sorted(set(scored["cell"]) - set(cells))
    conditions = ordered_conditions(scored)
    models = sorted(scored["model_key"].astype(str).unique())
    has_multi_model = len(models) > 1 and models != ["all"]
    s = point_size(len(scored))

    pca = PCA(n_components=2, random_state=args.seed)
    xy = pca.fit_transform(np.vstack([human_emb, gen_emb]))
    gen_xy = xy[len(human_emb) :]
    scored = scored.reset_index(drop=True)
    scored["pc1"] = gen_xy[:, 0]
    scored["pc2"] = gen_xy[:, 1]

    # ------------------------------------------------------------------
    # Main: pred O/N faceted by strategy + ρ bars
    # ------------------------------------------------------------------
    n_cond = len(conditions)
    nrows, ncols = facet_grid_shape(n_cond)
    fig = plt.figure(figsize=(3.6 * ncols + 3.6, 3.5 * nrows + 0.5))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.95])

    axes = []
    for i, cond in enumerate(conditions):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)
        sub = scored[scored["steering_condition"] == cond]
        scatter_pred(ax, sub, cells, s=s)
        rho_o = rho_lookup.get((cond, "O"), float("nan"))
        rho_n = rho_lookup.get((cond, "N"), float("nan"))
        ax.set_title(
            f"{cond}\nρ_O={rho_o:.2f}  ρ_N={rho_n:.2f}  n={len(sub)}",
            fontsize=9,
        )
        if r == nrows - 1:
            ax.set_xlabel("Predicted Openness", fontsize=8)
        if c == 0:
            ax.set_ylabel("Predicted Neuroticism", fontsize=8)

    for j in range(n_cond, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.add_subplot(gs[r, c]).axis("off")

    all_o = scored["pred_O"].to_numpy()
    all_n = scored["pred_N"].to_numpy()
    pad_o = 0.05 * (all_o.max() - all_o.min() + 1e-6)
    pad_n = 0.05 * (all_n.max() - all_n.min() + 1e-6)
    xlim = (all_o.min() - pad_o, all_o.max() + pad_o)
    ylim = (all_n.min() - pad_n, all_n.max() + pad_n)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax_bar = fig.add_subplot(gs[:, -1])
    rho_o_vals = [rho_lookup.get((c, "O"), 0.0) for c in conditions]
    rho_n_vals = [rho_lookup.get((c, "N"), 0.0) for c in conditions]
    order = np.argsort([-(a + b) / 2 for a, b in zip(rho_o_vals, rho_n_vals)])
    cond_ranked = [conditions[i] for i in order]
    rho_o_ranked = [rho_o_vals[i] for i in order]
    rho_n_ranked = [rho_n_vals[i] for i in order]
    y = np.arange(len(cond_ranked))
    height = 0.35
    ax_bar.barh(y + height / 2, rho_o_ranked, height=height, color="#4C78A8", label="O ρ")
    ax_bar.barh(y - height / 2, rho_n_ranked, height=height, color="#E45756", label="N ρ")
    ax_bar.axvline(0, color="#999", lw=0.7)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(cond_ranked, fontsize=8)
    ax_bar.set_xlabel("Target–pred Spearman ρ", fontsize=8)
    ax_bar.set_title("Recovery ranking", fontsize=9)
    ax_bar.legend(fontsize=7, frameon=False, loc="lower right")

    fig.legend(
        handles=cell_legend_handles(cells),
        loc="lower center",
        ncol=min(len(cells), 5),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.42, -0.04),
        title=cell_legend_title,
        title_fontsize=7,
    )
    fig.suptitle(
        f"{args.title} · predicted O/N by strategy · dwulff/mpnet-personality",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path1 = out / "personality_embedding_viz.png"
    fig.savefig(path1, dpi=160, bbox_inches="tight")
    print(f"Wrote {path1}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # PCA faceted by strategy
    # ------------------------------------------------------------------
    fig_p, axes_p = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.3 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, cond in enumerate(conditions):
        r, c = divmod(i, ncols)
        ax = axes_p[r][c]
        sub = scored[scored["steering_condition"] == cond]
        scatter_pca(
            ax,
            sub[["pc1", "pc2"]].to_numpy(),
            sub["cell"].to_numpy(),
            cells,
            s=s,
        )
        rho_o = rho_lookup.get((cond, "O"), float("nan"))
        rho_n = rho_lookup.get((cond, "N"), float("nan"))
        ax.set_title(f"{cond}\nρ_O={rho_o:.2f}  ρ_N={rho_n:.2f}", fontsize=9)
        if r == nrows - 1:
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        if c == 0:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    for j in range(n_cond, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_p[r][c].axis("off")

    fig_p.legend(
        handles=cell_legend_handles(cells),
        loc="lower center",
        ncol=min(len(cells), 5),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        title=cell_legend_title,
        title_fontsize=7,
    )
    fig_p.suptitle(
        f"{args.title} · PCA by strategy (color = cell)",
        fontsize=11,
        y=1.02,
    )
    fig_p.tight_layout()
    path_pca = out / "personality_embedding_viz_pca_by_strategy.png"
    fig_p.savefig(path_pca, dpi=160, bbox_inches="tight")
    print(f"Wrote {path_pca}")
    plt.close(fig_p)

    # ------------------------------------------------------------------
    # Multi-model grid
    # ------------------------------------------------------------------
    if has_multi_model:
        by_model_path = out / "embedding_condition_model_summary.csv"
        by_model_rho: dict[tuple[str, str, str], float] = {}
        if by_model_path.exists():
            bm = pd.read_csv(by_model_path)
            for r in bm.itertuples():
                if pd.notna(r.rho):
                    by_model_rho[(r.steering_condition, str(r.model_key), r.trait)] = float(
                        r.rho
                    )

        fig2, axes2 = plt.subplots(
            n_cond,
            len(models),
            figsize=(3.8 * len(models), 3.3 * n_cond),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i, cond in enumerate(conditions):
            for j, model in enumerate(models):
                ax = axes2[i][j]
                sub = scored[
                    (scored["steering_condition"] == cond)
                    & (scored["model_key"].astype(str) == model)
                ]
                scatter_pred(ax, sub, cells, s=max(6, s - 4), alpha=0.45)
                rho_o = by_model_rho.get((cond, model, "O"), float("nan"))
                rho_n = by_model_rho.get((cond, model, "N"), float("nan"))
                if i == 0:
                    ax.set_title(model, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{cond}\nPredicted N", fontsize=8)
                if i == n_cond - 1:
                    ax.set_xlabel("Predicted O", fontsize=8)
                ax.text(
                    0.02,
                    0.98,
                    f"ρO={rho_o:.2f}\nρN={rho_n:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="#333",
                )
        for ax_row in axes2:
            for ax in ax_row:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        fig2.legend(
            handles=cell_legend_handles(cells),
            loc="lower center",
            ncol=min(len(cells), 5),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
            title=cell_legend_title,
            title_fontsize=7,
        )
        fig2.suptitle(
            f"{args.title} · predicted O/N by strategy × model · dwulff/mpnet-personality",
            fontsize=11,
            y=1.01,
        )
        fig2.tight_layout()
        path2 = out / "personality_embedding_viz_by_model.png"
        fig2.savefig(path2, dpi=160, bbox_inches="tight")
        print(f"Wrote {path2}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
