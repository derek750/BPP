#!/usr/bin/env python3
"""Step 2: Fit a Gaussian copula and sample synthetic author profiles.

Preserves empirical marginals (via quantile matching) and the Gaussian-rank
dependency structure among O, N, and the selected LIWC categories.

Sampling follows the RA continuous rule: draw directly from the fitted copula
with a fixed seed. Invalid values and exact duplicates are rejected and
replaced by the next draw from the same seeded sequence.
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
    OPTIMIZED_DIR,
    PILOT_RESULTS_DIR,
    clip_liwc_rate,
    ensure_results_dir,
    write_json,
)

DEFAULT_INPUT = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_OUTPUT = PILOT_RESULTS_DIR / "step2_synthetic_profiles.csv"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step2_synthetic_profiles.report.json"
DEFAULT_N_SYNTHETIC = 12
DEFAULT_SEED = 7
TRAIT_COLS = ("O", "N")
MAX_DRAW_MULTIPLIER = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-synthetic", type=int, default=DEFAULT_N_SYNTHETIC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    skip = {"author_id", "n_essays", "total_word_count", "liwc_wc", "profile_id", "cell"}
    cols = [c for c in frame.columns if c not in skip]
    ordered = [c for c in TRAIT_COLS if c in cols]
    ordered.extend([c for c in cols if c not in ordered])
    return ordered


def _to_gaussian_scores(values: np.ndarray) -> np.ndarray:
    """Empirical rank -> uniform -> standard normal (Gaussian scores)."""
    n = len(values)
    ranks = stats.rankdata(values, method="average")
    uniforms = ranks / (n + 1.0)
    uniforms = np.clip(uniforms, 1e-6, 1.0 - 1e-6)
    return stats.norm.ppf(uniforms)


def fit_gaussian_copula(frame: pd.DataFrame, cols: list[str]) -> dict:
    matrix = frame[cols].to_numpy(dtype=float)
    gauss = np.column_stack([_to_gaussian_scores(matrix[:, j]) for j in range(len(cols))])
    corr = np.corrcoef(gauss, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    return {
        "cols": cols,
        "corr": corr_psd,
        "empirical": {c: frame[c].to_numpy(dtype=float).copy() for c in cols},
    }


def _draw_batch(model: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    cols: list[str] = model["cols"]
    z = rng.multivariate_normal(mean=np.zeros(len(cols)), cov=model["corr"], size=n)
    uniforms = stats.norm.cdf(z)
    rows: dict[str, np.ndarray] = {}
    for j, col in enumerate(cols):
        empirical = np.sort(model["empirical"][col])
        sampled = np.quantile(empirical, uniforms[:, j], method="linear")
        sampled = np.clip(sampled, 0.0, 100.0)
        rows[col] = sampled
    out = pd.DataFrame(rows)
    for col in cols:
        if col not in TRAIT_COLS:
            out[col] = out[col].map(clip_liwc_rate)
    return out


def _row_valid(row: pd.Series, cols: list[str]) -> bool:
    vals = row[cols].to_numpy(dtype=float)
    return bool(np.all(np.isfinite(vals)) and np.all((vals >= 0.0) & (vals <= 100.0)))


def sample_profiles(
    model: dict,
    n: int,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Sample n valid unique profiles; reject invalid/dupes via next seeded draw."""
    rng = np.random.default_rng(seed)
    cols: list[str] = model["cols"]
    accepted: list[pd.Series] = []
    seen: set[tuple[float, ...]] = set()
    n_drawn = 0
    n_rejected_invalid = 0
    n_rejected_dupes = 0
    max_draws = max(n * MAX_DRAW_MULTIPLIER, n + 100)

    while len(accepted) < n and n_drawn < max_draws:
        batch_n = min(max(n - len(accepted), 1) * 2, max_draws - n_drawn)
        batch = _draw_batch(model, batch_n, rng)
        n_drawn += batch_n
        for _, row in batch.iterrows():
            if len(accepted) >= n:
                break
            if not _row_valid(row, cols):
                n_rejected_invalid += 1
                continue
            key = tuple(np.round(row[cols].to_numpy(dtype=float), 6).tolist())
            if key in seen:
                n_rejected_dupes += 1
                continue
            seen.add(key)
            accepted.append(row)

    if len(accepted) < n:
        raise RuntimeError(
            f"Could not collect {n} valid unique profiles after {n_drawn} draws "
            f"(invalid={n_rejected_invalid}, dupes={n_rejected_dupes})."
        )

    out = pd.DataFrame(accepted).reset_index(drop=True)
    out.insert(0, "profile_id", [f"syn_{i:04d}" for i in range(len(out))])
    meta = {
        "n_requested": n,
        "n_drawn": n_drawn,
        "n_rejected_invalid": n_rejected_invalid,
        "n_rejected_dupes": n_rejected_dupes,
        "n_accepted": len(out),
    }
    return out, meta


def validate_synthetic(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
) -> dict:
    report: dict = {
        "n_human": int(len(human)),
        "n_synthetic": int(len(synthetic)),
        "cols": cols,
        "marginal_mean_abs_diff": {},
        "marginal_std_abs_diff": {},
        "corr_frobenius_diff": None,
        "corr_O_liwc_human": {},
        "corr_O_liwc_synth": {},
        "corr_N_liwc_human": {},
        "corr_N_liwc_synth": {},
        "invalid_values": 0,
        "n_duplicate_profiles": 0,
    }

    for col in cols:
        report["marginal_mean_abs_diff"][col] = float(
            abs(human[col].mean() - synthetic[col].mean())
        )
        report["marginal_std_abs_diff"][col] = float(
            abs(human[col].std(ddof=1) - synthetic[col].std(ddof=1))
        )

    human_corr = human[cols].corr().to_numpy()
    synth_corr = synthetic[cols].corr().to_numpy()
    report["corr_frobenius_diff"] = float(np.linalg.norm(human_corr - synth_corr, ord="fro"))

    liwc_cols = [c for c in cols if c not in TRAIT_COLS]
    for col in liwc_cols:
        report["corr_O_liwc_human"][col] = float(human["O"].corr(human[col]))
        report["corr_O_liwc_synth"][col] = float(synthetic["O"].corr(synthetic[col]))
        report["corr_N_liwc_human"][col] = float(human["N"].corr(human[col]))
        report["corr_N_liwc_synth"][col] = float(synthetic["N"].corr(synthetic[col]))

    invalid = 0
    for col in cols:
        vals = synthetic[col]
        invalid += int(((vals < 0) | (vals > 100)).sum())
    report["invalid_values"] = invalid

    feature_rows = synthetic[cols].round(6)
    report["n_duplicate_profiles"] = int(feature_rows.duplicated().sum())
    report["passed_qc"] = invalid == 0 and report["n_duplicate_profiles"] == 0
    return report


def main() -> None:
    args = parse_args()
    human_path = _resolve_out(args.input)
    human = pd.read_csv(human_path)
    cols = _feature_cols(human)
    model = fit_gaussian_copula(human, cols)
    synthetic, sample_meta = sample_profiles(model, args.n_synthetic, seed=args.seed)
    qc = validate_synthetic(human, synthetic, cols)

    output = _resolve_out(args.output)
    report_path = _resolve_out(args.report)
    ensure_results_dir(output.parent)
    synthetic.to_csv(output, index=False)
    write_json(
        report_path,
        {
            "output": str(output),
            "seed": args.seed,
            "sampling": sample_meta,
            "corr_matrix": {
                "cols": cols,
                "matrix": model["corr"].tolist(),
            },
            "qc": qc,
        },
    )
    print(f"Wrote {len(synthetic)} synthetic profiles -> {output}")
    print(
        f"Sampling draws={sample_meta['n_drawn']} | "
        f"rejected_invalid={sample_meta['n_rejected_invalid']} | "
        f"rejected_dupes={sample_meta['n_rejected_dupes']}"
    )
    print(
        f"QC passed={qc['passed_qc']} | invalid={qc['invalid_values']} | "
        f"dupes={qc['n_duplicate_profiles']} | corr F-diff={qc['corr_frobenius_diff']:.3f}"
    )
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
