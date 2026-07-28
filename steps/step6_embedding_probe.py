#!/usr/bin/env python3
"""Step 6: Embedding-based continuous O/N recovery with mpnet-personality.

Validation 2. Adapts personality-embedding analyses to continuous targets:

  1. Sample length-matched human windows labeled with continuous O/N.
  2. Embed with ``dwulff/mpnet-personality``.
  3. Fit Ridge probes for O and N on a held-out *author* split (no generation
     text is used in training).
  4. Apply the fixed probes to all generations; report predicted vs target
     O/N by steering condition (ρ, calibration slope, MAE).

Example:
  python optimized/steps/step6_embedding_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

from common import (
    DEFAULT_AUTHORS,
    DEFAULT_SEED,
    OPTIMIZED_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_DIR,
    FULL_RESULTS_DIR,
    ensure_results_dir,
    resolve_path,
    write_json,
)

DEFAULT_GENERATIONS = PILOT_RESULTS_DIR / "step5_pilot" / "pilot_generations.csv"
DEFAULT_HUMAN_PROFILES = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_OUTPUT_DIR = PILOT_RESULTS_DIR / "step6_embedding"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step6_embedding" / "embedding_probe.report.json"
DEFAULT_MODEL = "dwulff/mpnet-personality"
DEFAULT_WINDOW_WORDS = 200
DEFAULT_N_HUMAN_WINDOWS = 800
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RIDGE_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--human-profiles", type=Path, default=DEFAULT_HUMAN_PROFILES)
    parser.add_argument("--authors", type=Path, default=Path(DEFAULT_AUTHORS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--window-words", type=int, default=DEFAULT_WINDOW_WORDS)
    parser.add_argument("--n-human-windows", type=int, default=DEFAULT_N_HUMAN_WINDOWS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda/mps/cpu auto).",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse cached embeddings.npz if present.",
    )
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def load_embedding_model(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. "
            "Install with: pip install -U sentence-transformers"
        ) from exc
    kwargs = {}
    if device:
        kwargs["device"] = device
    print(f"Loading {model_name} ...")
    return SentenceTransformer(model_name, **kwargs)


def encode_texts(model, texts: list[str], *, batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def sample_human_windows(
    authors_path: Path,
    profiles: pd.DataFrame,
    *,
    window_words: int,
    n_windows: int,
    seed: int,
) -> pd.DataFrame:
    """Length-matched windows labeled with continuous author O/N."""
    o_by_id = {
        int(i): float(o) for i, o in zip(profiles["author_id"], profiles["O"])
    }
    n_by_id = {
        int(i): float(n) for i, n in zip(profiles["author_id"], profiles["N"])
    }
    wanted_ids = set(o_by_id)

    essays: dict[int, list[str]] = {}
    resolved = resolve_path(authors_path)
    row_idx = 0
    print(f"Scanning author texts for windows (>= {max(40, window_words // 4)} tokens) ...")
    for chunk in pd.read_csv(resolved, usecols=["text"], chunksize=200):
        for text in chunk["text"]:
            if row_idx in wanted_ids:
                tokens = str(text).split()
                if len(tokens) >= max(40, window_words // 4):
                    essays[row_idx] = tokens
            row_idx += 1

    usable_ids = sorted(essays)
    if len(usable_ids) < 50:
        raise RuntimeError(f"Too few usable essays for windows: {len(usable_ids)}")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_windows):
        author_id = int(rng.choice(usable_ids))
        tokens = essays[author_id]
        if len(tokens) <= window_words:
            window = tokens
        else:
            start = int(rng.integers(0, len(tokens) - window_words + 1))
            window = tokens[start : start + window_words]
        rows.append(
            {
                "window_id": i,
                "author_id": author_id,
                "O": o_by_id[author_id],
                "N": n_by_id[author_id],
                "text": " ".join(window),
                "n_words": len(window),
            }
        )
    return pd.DataFrame(rows)


def fit_probes(
    X: np.ndarray,
    y_o: np.ndarray,
    y_n: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    alpha: float,
    seed: int,
) -> dict:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(X, y_o, groups))
    probe_o = Ridge(alpha=alpha)
    probe_n = Ridge(alpha=alpha)
    probe_o.fit(X[train_idx], y_o[train_idx])
    probe_n.fit(X[train_idx], y_n[train_idx])

    def _eval(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
        return {
            "rho": float(stats.spearmanr(true, pred).statistic),
            "pearson_r": float(stats.pearsonr(true, pred).statistic),
            "slope": float(stats.linregress(true, pred).slope),
            "intercept": float(stats.linregress(true, pred).intercept),
            "mae": float(np.mean(np.abs(pred - true))),
        }

    pred_o_test = probe_o.predict(X[test_idx])
    pred_n_test = probe_n.predict(X[test_idx])
    return {
        "probe_o": probe_o,
        "probe_n": probe_n,
        "n_train_windows": int(len(train_idx)),
        "n_test_windows": int(len(test_idx)),
        "n_train_authors": int(len(np.unique(groups[train_idx]))),
        "n_test_authors": int(len(np.unique(groups[test_idx]))),
        "human_heldout_O": _eval(pred_o_test, y_o[test_idx]),
        "human_heldout_N": _eval(pred_n_test, y_n[test_idx]),
        "test_idx": test_idx,
        "train_idx": train_idx,
    }


def condition_recovery(
    frame: pd.DataFrame,
    pred_o: np.ndarray,
    pred_n: np.ndarray,
    *,
    group_cols: tuple[str, ...] = ("steering_condition",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["pred_O"] = pred_o
    out["pred_N"] = pred_n
    out["err_O"] = out["pred_O"] - out["target_O"]
    out["err_N"] = out["pred_N"] - out["target_N"]

    rows: list[dict] = []
    for keys, grp in out.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        for trait, target_col, pred_col in (
            ("O", "target_O", "pred_O"),
            ("N", "target_N", "pred_N"),
        ):
            t = grp[target_col].to_numpy(dtype=float)
            p = grp[pred_col].to_numpy(dtype=float)
            if len(t) < 3 or np.std(t) < 1e-8:
                rho = slope = pearson = float("nan")
            else:
                rho = float(stats.spearmanr(t, p).statistic)
                pearson = float(stats.pearsonr(t, p).statistic)
                slope = float(stats.linregress(t, p).slope)
            row = {
                **key_map,
                "trait": trait,
                "n": int(len(grp)),
                "rho": round(rho, 4) if rho == rho else None,
                "pearson_r": round(pearson, 4) if pearson == pearson else None,
                "calibration_slope": round(slope, 4) if slope == slope else None,
                "mae": round(float(np.mean(np.abs(p - t))), 4),
                "mean_pred": round(float(p.mean()), 4),
                "mean_target": round(float(t.mean()), 4),
            }
            rows.append(row)
    return out, pd.DataFrame(rows)


def author_level_recovery(
    per_sample: pd.DataFrame,
    *,
    extra_groups: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average predictions within profile × condition [(× model)], then score."""
    group_cols = ["profile_id", "steering_condition", *extra_groups]
    rows: list[dict] = []
    grouped = per_sample.groupby(group_cols, as_index=False).agg(
        target_O=("target_O", "first"),
        target_N=("target_N", "first"),
        pred_O=("pred_O", "mean"),
        pred_N=("pred_N", "mean"),
        n_essays=("plan_id", "count"),
    )
    summary_groups = ["steering_condition", *extra_groups]
    for keys, grp in grouped.groupby(summary_groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(summary_groups, keys))
        for trait, target_col, pred_col in (
            ("O", "target_O", "pred_O"),
            ("N", "target_N", "pred_N"),
        ):
            t = grp[target_col].to_numpy(dtype=float)
            p = grp[pred_col].to_numpy(dtype=float)
            if len(t) < 3 or np.std(t) < 1e-8:
                rho = slope = float("nan")
            else:
                rho = float(stats.spearmanr(t, p).statistic)
                slope = float(stats.linregress(t, p).slope)
            rows.append(
                {
                    **key_map,
                    "trait": trait,
                    "n_authors": int(len(grp)),
                    "rho": round(rho, 4) if rho == rho else None,
                    "calibration_slope": round(slope, 4) if slope == slope else None,
                    "mae": round(float(np.mean(np.abs(p - t))), 4),
                }
            )
    return grouped, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = _resolve_out(args.output_dir)
    ensure_results_dir(output_dir)
    emb_path = output_dir / "personality_embeddings.npz"
    human_windows_path = output_dir / "human_windows.csv"

    gens = pd.read_csv(_resolve_out(args.generations))
    gens = gens[gens["error"].isna() | (gens["error"] == "")].copy()
    if gens.empty:
        raise RuntimeError("No successful generations to embed.")
    # Normalize model column across pilot / full schemas.
    if "model_key" not in gens.columns:
        if "model" in gens.columns:
            gens["model_key"] = gens["model"].astype(str)
        else:
            gens["model_key"] = "unknown"
    profiles = pd.read_csv(_resolve_out(args.human_profiles))
    has_multi_model = gens["model_key"].nunique() > 1
    print(
        f"Generations: {len(gens)} | conditions={sorted(gens['steering_condition'].unique())} "
        f"| models={sorted(gens['model_key'].unique())}"
    )

    if args.reuse_embeddings and emb_path.exists() and human_windows_path.exists():
        print(f"Reusing cached embeddings -> {emb_path}")
        human = pd.read_csv(human_windows_path)
        npz = np.load(emb_path)
        human_emb = npz["human_embeddings"]
        gen_emb = npz["generation_embeddings"]
        if len(human_emb) != len(human) or len(gen_emb) != len(gens):
            raise RuntimeError(
                "Cached embedding counts do not match current windows/generations; "
                "re-run without --reuse-embeddings."
            )
    else:
        human = sample_human_windows(
            args.authors,
            profiles,
            window_words=args.window_words,
            n_windows=args.n_human_windows,
            seed=args.seed,
        )
        human.to_csv(human_windows_path, index=False)
        print(
            f"Sampled {len(human)} human windows "
            f"(~{args.window_words} words) from {human['author_id'].nunique()} authors"
        )

        model = load_embedding_model(args.model, args.device)
        print("Encoding human windows ...")
        human_emb = encode_texts(
            model, human["text"].tolist(), batch_size=args.batch_size
        )
        print("Encoding generations ...")
        gen_emb = encode_texts(
            model, gens["text"].fillna("").astype(str).tolist(), batch_size=args.batch_size
        )
        np.savez_compressed(
            emb_path,
            human_embeddings=human_emb,
            generation_embeddings=gen_emb,
            human_author_ids=human["author_id"].to_numpy(dtype=int),
            generation_plan_ids=gens["plan_id"].astype(str).to_numpy(),
        )
        print(f"Wrote embeddings -> {emb_path}")

    y_o = human["O"].to_numpy(dtype=float)
    y_n = human["N"].to_numpy(dtype=float)
    groups = human["author_id"].to_numpy(dtype=int)
    fitted = fit_probes(
        human_emb,
        y_o,
        y_n,
        groups,
        test_size=args.test_size,
        alpha=args.ridge_alpha,
        seed=args.seed,
    )
    print(
        "Human held-out probe: "
        f"O ρ={fitted['human_heldout_O']['rho']:.3f} "
        f"slope={fitted['human_heldout_O']['slope']:.3f} | "
        f"N ρ={fitted['human_heldout_N']['rho']:.3f} "
        f"slope={fitted['human_heldout_N']['slope']:.3f}"
    )

    pred_o = fitted["probe_o"].predict(gen_emb)
    pred_n = fitted["probe_n"].predict(gen_emb)
    per_sample, essay_summary = condition_recovery(gens, pred_o, pred_n)
    author_extra = ("model_key",) if has_multi_model else ()
    author_frame, author_summary = author_level_recovery(
        per_sample, extra_groups=author_extra
    )
    by_model_summary = None
    if has_multi_model:
        _, by_model_summary = condition_recovery(
            gens, pred_o, pred_n, group_cols=("steering_condition", "model_key")
        )

    per_sample_path = output_dir / "embedding_per_sample.csv"
    essay_summary_path = output_dir / "embedding_condition_summary.csv"
    author_sample_path = output_dir / "embedding_author_level.csv"
    author_summary_path = output_dir / "embedding_author_summary.csv"
    by_model_path = output_dir / "embedding_condition_model_summary.csv"

    export_cols = [
        "plan_id",
        "profile_id",
        "cell",
        "content",
        "steering_condition",
        "repetition",
        "model_key",
        "model",
        "target_O",
        "target_N",
        "pred_O",
        "pred_N",
        "err_O",
        "err_N",
        "n_words",
        "standardized_mae",
    ]
    export_cols = [c for c in export_cols if c in per_sample.columns]
    per_sample[export_cols].to_csv(per_sample_path, index=False)
    essay_summary.to_csv(essay_summary_path, index=False)
    author_frame.to_csv(author_sample_path, index=False)
    author_summary.to_csv(author_summary_path, index=False)
    if by_model_summary is not None:
        by_model_summary.to_csv(by_model_path, index=False)

    report_path = _resolve_out(args.report)
    write_json(
        report_path,
        {
            "model": args.model,
            "window_words": args.window_words,
            "n_human_windows": int(len(human)),
            "n_generations": int(len(gens)),
            "embedding_dim": int(human_emb.shape[1]),
            "ridge_alpha": args.ridge_alpha,
            "human_heldout_O": fitted["human_heldout_O"],
            "human_heldout_N": fitted["human_heldout_N"],
            "n_train_windows": fitted["n_train_windows"],
            "n_test_windows": fitted["n_test_windows"],
            "n_train_authors": fitted["n_train_authors"],
            "n_test_authors": fitted["n_test_authors"],
            "models": sorted(gens["model_key"].unique().tolist()),
            "outputs": {
                "per_sample": str(per_sample_path),
                "essay_summary": str(essay_summary_path),
                "author_level": str(author_sample_path),
                "author_summary": str(author_summary_path),
                "by_model_summary": str(by_model_path) if by_model_summary is not None else None,
                "embeddings": str(emb_path),
                "human_windows": str(human_windows_path),
            },
        },
    )

    print(f"\nWrote essay-level recovery -> {essay_summary_path}")
    print(essay_summary.to_string(index=False))
    if by_model_summary is not None:
        print(f"\nWrote condition×model recovery -> {by_model_path}")
        print(by_model_summary.to_string(index=False))
    print(f"\nWrote author-level recovery -> {author_summary_path}")
    print(author_summary.to_string(index=False))
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
