#!/usr/bin/env python3
"""Step 9: profile-clustered bootstrap CIs and planned condition contrasts.

Resamples synthetic profiles with replacement and retains every
condition × model observation attached to each selected profile (RA Feedback §5).

Primary estimates (author / profile level):
  LIWC: MAE_LIWC, ρ_LIWC (Fisher-z mean), β_LIWC
  Embedding: ρ_O, ρ_N
  BFI: ρ_O, ρ_N, standardised MAE_O / MAE_N, mean MAE_BFI

Planned contrasts (within model):
  liwc_only − persona_only
  persona_liwc − persona_only
  persona_liwc − liwc_only
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    DEFAULT_CONTROL_CATEGORIES,
    FULL_RESULTS_DIR,
    OPTIMIZED_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)
import step5_pilot_generation as pilot  # noqa: E402
import step8_bfi_validation as bfi  # noqa: E402

DEFAULT_LIWC = FULL_RESULTS_DIR / "generations" / "full_author_level_liwc.csv"
DEFAULT_EMB = FULL_RESULTS_DIR / "embedding" / "embedding_author_level.csv"
DEFAULT_BFI = FULL_RESULTS_DIR / "bfi" / "bfi_per_sample.csv"
DEFAULT_OUT = FULL_RESULTS_DIR / "inference"
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 7

PLANNED_CONTRASTS = (
    ("liwc_only", "persona_only"),
    ("persona_liwc", "persona_only"),
    ("persona_liwc", "liwc_only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liwc-author", type=Path, default=DEFAULT_LIWC)
    parser.add_argument("--embedding-author", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--bfi-per-sample", type=Path, default=DEFAULT_BFI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CONTROL_CATEGORIES),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) < 1e-8:
        return float("nan")
    rho = float(stats.spearmanr(x, y).statistic)
    return rho if rho == rho else float("nan")


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) < 1e-8:
        return float("nan")
    slope = float(stats.linregress(x, y).slope)
    return slope if slope == slope else float("nan")


def liwc_metrics(subset: pd.DataFrame, categories: list[str]) -> dict[str, float]:
    if subset.empty:
        return {
            "MAE_LIWC": float("nan"),
            "rho_LIWC": float("nan"),
            "beta_LIWC": float("nan"),
            "n_profiles": 0,
        }
    cat_metrics = {}
    for category in categories:
        t_col, o_col = f"target_{category}", f"obs_{category}"
        if t_col not in subset.columns or o_col not in subset.columns:
            continue
        t = subset[t_col].to_numpy(dtype=float)
        o = subset[o_col].to_numpy(dtype=float)
        cat_metrics[category] = pilot.calibration_metrics(t, o)
    mean_cal = pilot.mean_calibration(cat_metrics) if cat_metrics else {
        "mean_rho": float("nan"),
        "mean_slope": float("nan"),
    }
    return {
        "MAE_LIWC": float(np.nanmean(subset["standardized_mae"].to_numpy(dtype=float))),
        "rho_LIWC": float(mean_cal["mean_rho"]),
        "beta_LIWC": float(mean_cal["mean_slope"]),
        "n_profiles": int(len(subset)),
    }


def emb_metrics(subset: pd.DataFrame) -> dict[str, float]:
    if subset.empty:
        return {"rho_O": float("nan"), "rho_N": float("nan"), "n_profiles": 0}
    return {
        "rho_O": _spearman(
            subset["target_O"].to_numpy(dtype=float),
            subset["pred_O"].to_numpy(dtype=float),
        ),
        "rho_N": _spearman(
            subset["target_N"].to_numpy(dtype=float),
            subset["pred_N"].to_numpy(dtype=float),
        ),
        "n_profiles": int(len(subset)),
    }


def build_bfi_author(per_sample: pd.DataFrame) -> pd.DataFrame:
    sub = per_sample[per_sample["bfi_parse_ok"].fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["model_key", "steering_condition", "profile_id"], as_index=False)
        .agg(
            target_O=("target_O", "first"),
            target_N=("target_N", "first"),
            bfi_O=("bfi_O", "mean"),
            bfi_N=("bfi_N", "mean"),
            n_essays=("plan_id", "count"),
        )
    )


def bfi_metrics(
    subset: pd.DataFrame,
    *,
    scales: dict[str, float],
) -> dict[str, float]:
    if subset.empty or len(subset) < 3:
        return {
            "rho_O": float("nan"),
            "rho_N": float("nan"),
            "standardized_mae_O": float("nan"),
            "standardized_mae_N": float("nan"),
            "MAE_BFI": float("nan"),
            "slope_O": float("nan"),
            "slope_N": float("nan"),
            "n_profiles": int(len(subset)),
        }
    t_o = subset["target_O"].to_numpy(dtype=float)
    t_n = subset["target_N"].to_numpy(dtype=float)
    o = subset["bfi_O"].to_numpy(dtype=float)
    n = subset["bfi_N"].to_numpy(dtype=float)
    mae_o = float(np.mean(np.abs(o - t_o))) / scales["O"]
    mae_n = float(np.mean(np.abs(n - t_n))) / scales["N"]
    return {
        "rho_O": _spearman(t_o, o),
        "rho_N": _spearman(t_n, n),
        "standardized_mae_O": mae_o,
        "standardized_mae_N": mae_n,
        "MAE_BFI": float(np.mean([mae_o, mae_n])),
        "slope_O": _slope(t_o, o),
        "slope_N": _slope(t_n, n),
        "n_profiles": int(len(subset)),
    }


def estimate_block(
    *,
    liwc: pd.DataFrame,
    emb: pd.DataFrame,
    bfi_author: pd.DataFrame,
    profiles: list[str],
    model_key: str,
    condition: str,
    categories: list[str],
    scales: dict[str, float],
) -> dict[str, float]:
    prof_set = set(profiles)
    liwc_sub = liwc[
        (liwc["model_key"] == model_key)
        & (liwc["steering_condition"] == condition)
        & (liwc["profile_id"].isin(prof_set))
    ]
    emb_sub = emb[
        (emb["model_key"] == model_key)
        & (emb["steering_condition"] == condition)
        & (emb["profile_id"].isin(prof_set))
    ]
    bfi_sub = bfi_author[
        (bfi_author["model_key"] == model_key)
        & (bfi_author["steering_condition"] == condition)
        & (bfi_author["profile_id"].isin(prof_set))
    ] if not bfi_author.empty else bfi_author

    out: dict = {
        "model_key": model_key,
        "steering_condition": condition,
    }
    out.update({f"liwc_{k}": v for k, v in liwc_metrics(liwc_sub, categories).items()})
    out.update({f"emb_{k}": v for k, v in emb_metrics(emb_sub).items()})
    out.update({f"bfi_{k}": v for k, v in bfi_metrics(bfi_sub, scales=scales).items()})
    return out


def _ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    clean = samples[np.isfinite(samples)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(clean)),
        float(np.quantile(clean, alpha / 2)),
        float(np.quantile(clean, 1 - alpha / 2)),
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_results_dir(_resolve(args.output_dir))
    liwc = pd.read_csv(_resolve(args.liwc_author))
    emb = pd.read_csv(_resolve(args.embedding_author))
    bfi_path = _resolve(args.bfi_per_sample)
    bfi_author = build_bfi_author(pd.read_csv(bfi_path)) if bfi_path.exists() else pd.DataFrame()
    scales = bfi.load_human_trait_scales()
    categories = list(args.categories)

    profiles = sorted(set(liwc["profile_id"].astype(str)))
    models = sorted(set(liwc["model_key"].astype(str)))
    conditions = [c for c in STEERING_CONDITIONS if c in set(liwc["steering_condition"])]

    # Point estimates on the full sample.
    point_rows = [
        estimate_block(
            liwc=liwc,
            emb=emb,
            bfi_author=bfi_author,
            profiles=profiles,
            model_key=model,
            condition=condition,
            categories=categories,
            scales=scales,
        )
        for model in models
        for condition in conditions
    ]
    point = pd.DataFrame(point_rows)
    point_path = out_dir / "profile_point_estimates.csv"
    point.to_csv(point_path, index=False)

    rng = np.random.default_rng(args.seed)
    boot_records: list[dict] = []
    n = len(profiles)
    for b in range(args.n_boot):
        drawn = rng.choice(profiles, size=n, replace=True).tolist()
        for model in models:
            for condition in conditions:
                row = estimate_block(
                    liwc=liwc,
                    emb=emb,
                    bfi_author=bfi_author,
                    profiles=drawn,
                    model_key=model,
                    condition=condition,
                    categories=categories,
                    scales=scales,
                )
                row["boot_id"] = b
                boot_records.append(row)
        if (b + 1) % 200 == 0:
            print(f"bootstrap {b + 1}/{args.n_boot}")

    boot = pd.DataFrame(boot_records)
    boot_path = out_dir / "profile_bootstrap_draws.csv"
    # Keep draws optional/large — write a compact summary instead of full draws by default.
    # Full draws can be huge; write summary + contrasts only, plus a sample of draws.
    boot.sample(n=min(len(boot), 5000), random_state=args.seed).to_csv(
        out_dir / "profile_bootstrap_draws_sample.csv", index=False
    )

    metric_cols = [
        c
        for c in boot.columns
        if c
        not in (
            "model_key",
            "steering_condition",
            "boot_id",
            "liwc_n_profiles",
            "emb_n_profiles",
            "bfi_n_profiles",
        )
        and not c.endswith("_n_profiles")
    ]

    ci_rows: list[dict] = []
    for model in models:
        for condition in conditions:
            sub = boot[
                (boot["model_key"] == model) & (boot["steering_condition"] == condition)
            ]
            point_row = point[
                (point["model_key"] == model) & (point["steering_condition"] == condition)
            ]
            for metric in metric_cols:
                if metric not in sub.columns:
                    continue
                mean, lo, hi = _ci(sub[metric].to_numpy(dtype=float))
                point_val = (
                    float(point_row[metric].iloc[0])
                    if len(point_row) and metric in point_row.columns
                    else float("nan")
                )
                ci_rows.append(
                    {
                        "model_key": model,
                        "steering_condition": condition,
                        "metric": metric,
                        "point": round(point_val, 4) if point_val == point_val else None,
                        "boot_mean": round(mean, 4) if mean == mean else None,
                        "ci95_lo": round(lo, 4) if lo == lo else None,
                        "ci95_hi": round(hi, 4) if hi == hi else None,
                    }
                )
    ci = pd.DataFrame(ci_rows)
    ci_path = out_dir / "profile_bootstrap_ci.csv"
    ci.to_csv(ci_path, index=False)

    contrast_rows: list[dict] = []
    for model in models:
        for a, b_cond in PLANNED_CONTRASTS:
            sub_a = boot[
                (boot["model_key"] == model) & (boot["steering_condition"] == a)
            ].set_index("boot_id")
            sub_b = boot[
                (boot["model_key"] == model) & (boot["steering_condition"] == b_cond)
            ].set_index("boot_id")
            shared = sub_a.index.intersection(sub_b.index)
            point_a = point[
                (point["model_key"] == model) & (point["steering_condition"] == a)
            ]
            point_b = point[
                (point["model_key"] == model) & (point["steering_condition"] == b_cond)
            ]
            for metric in metric_cols:
                if metric not in sub_a.columns:
                    continue
                delta = (
                    sub_a.loc[shared, metric].to_numpy(dtype=float)
                    - sub_b.loc[shared, metric].to_numpy(dtype=float)
                )
                mean, lo, hi = _ci(delta)
                pa = (
                    float(point_a[metric].iloc[0])
                    if len(point_a) and metric in point_a.columns
                    else float("nan")
                )
                pb = (
                    float(point_b[metric].iloc[0])
                    if len(point_b) and metric in point_b.columns
                    else float("nan")
                )
                contrast_rows.append(
                    {
                        "model_key": model,
                        "contrast": f"{a} - {b_cond}",
                        "metric": metric,
                        "point_delta": round(pa - pb, 4)
                        if pa == pa and pb == pb
                        else None,
                        "boot_mean_delta": round(mean, 4) if mean == mean else None,
                        "ci95_lo": round(lo, 4) if lo == lo else None,
                        "ci95_hi": round(hi, 4) if hi == hi else None,
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts_path = out_dir / "profile_bootstrap_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    report = {
        "step": "Optimized continuous / Step 9 — profile-clustered bootstrap",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_profiles": len(profiles),
        "models": models,
        "conditions": conditions,
        "planned_contrasts": [f"{a} - {b}" for a, b in PLANNED_CONTRASTS],
        "mean_rho_aggregation": "fisher_z",
        "outputs": {
            "point_estimates": point_path.name,
            "bootstrap_ci": ci_path.name,
            "contrasts": contrasts_path.name,
            "draws_sample": "profile_bootstrap_draws_sample.csv",
        },
        "notes": [
            "Resampling unit is profile_id; all condition×model rows for a drawn profile are kept.",
            "LIWC ρ uses Fisher-z mean across the eight control categories.",
            "BFI metrics use author-level means over scored topics.",
        ],
    }
    report_path = out_dir / "profile_bootstrap.report.json"
    write_json(report_path, report)

    print(f"Wrote {point_path}")
    print(f"Wrote {ci_path}")
    print(f"Wrote {contrasts_path}")
    print(f"Wrote {report_path}")
    # Show a small headline slice.
    headline = ci[
        ci["metric"].isin(
            [
                "liwc_MAE_LIWC",
                "liwc_rho_LIWC",
                "liwc_beta_LIWC",
                "emb_rho_O",
                "emb_rho_N",
                "bfi_rho_O",
                "bfi_rho_N",
                "bfi_MAE_BFI",
            ]
        )
    ]
    print("\n=== Headline CIs (first 24 rows) ===")
    print(headline.head(24).to_string(index=False))


if __name__ == "__main__":
    main()
