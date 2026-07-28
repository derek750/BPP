"""Author-level LIWC alignment: concatenate essays → re-score LIWC.

Primary Validation 1 metric. For each synthetic author × steering condition
(× model), concatenate all topic essays and score LIWC once on the combined
text. This mirrors how human author profiles are built and avoids averaging
sparse per-essay category rates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import STEERING_CONDITIONS


ScoreFn = Callable[[list[str]], list[dict[str, float]]]


def categories_from_frame(frame: pd.DataFrame) -> list[str]:
    return [
        c.replace("target_", "", 1)
        for c in frame.columns
        if c.startswith("target_") and c not in ("target_O", "target_N")
    ]


def _group_cols(frame: pd.DataFrame) -> list[str]:
    cols = ["profile_id", "steering_condition"]
    if "model_key" in frame.columns:
        cols.append("model_key")
    return cols


def build_author_concat_groups(
    generations: pd.DataFrame,
    *,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """One row per profile × condition [× model] with concatenated essay text."""
    frame = generations.copy()
    frame = frame[
        frame["error"].isna()
        | (frame["error"].astype(str).str.strip() == "")
    ]
    frame = frame[frame["text"].notna() & (frame["text"].astype(str).str.strip() != "")]
    if frame.empty:
        return pd.DataFrame()

    cats = categories or categories_from_frame(frame)
    group_cols = _group_cols(frame)
    rows: list[dict] = []
    for keys, grp in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        ordered = grp
        if "content" in grp.columns:
            ordered = grp.sort_values(["content", "repetition"] if "repetition" in grp.columns else "content")
        texts = [str(t).strip() for t in ordered["text"].tolist() if str(t).strip()]
        if not texts:
            continue
        concat = " ".join(texts)
        row: dict = {
            **key_map,
            "n_essays": len(texts),
            "concat_text": concat,
            "n_words_concat": len(concat.split()),
        }
        if "cell" in ordered.columns:
            row["cell"] = str(ordered["cell"].iloc[0])
        if "target_O" in ordered.columns:
            row["target_O"] = float(ordered["target_O"].iloc[0])
        if "target_N" in ordered.columns:
            row["target_N"] = float(ordered["target_N"].iloc[0])
        for category in cats:
            tcol = f"target_{category}"
            if tcol in ordered.columns:
                row[tcol] = float(ordered[tcol].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def score_author_concatenations(
    author_groups: pd.DataFrame,
    *,
    categories: list[str],
    scales: dict[str, float],
    score_fn: ScoreFn,
    mock_score_fn: Callable[..., dict[str, float]] | None = None,
    mock: bool = False,
    batch_size: int = 40,
    corpus_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score concatenated texts and attach obs_* + MAE columns."""
    if author_groups.empty:
        return author_groups.copy()

    out = author_groups.copy()
    texts = out["concat_text"].tolist()
    measured_list: list[dict[str, float]] = []

    if mock:
        if mock_score_fn is None:
            raise ValueError("mock=True requires mock_score_fn")
        means = corpus_means or {c: 0.0 for c in categories}
        for _, row in out.iterrows():
            targets = {c: float(row[f"target_{c}"]) for c in categories}
            digest = hashlib.sha256(
                f"author:{row.get('profile_id')}:{row.get('steering_condition')}:"
                f"{row.get('model_key', '')}".encode()
            ).hexdigest()
            measured_list.append(
                mock_score_fn(
                    condition=str(row["steering_condition"]),
                    profile_id=str(row["profile_id"]),
                    content="author_concat",
                    repetition=0,
                    targets=targets,
                    scales=scales,
                    corpus_means=means,
                    categories=categories,
                    seed=int(digest[:16], 16),
                )
            )
    else:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            measured_list.extend(score_fn(batch))

    raw_maes: list[float] = []
    std_maes: list[float] = []
    for i, measured in enumerate(measured_list):
        for category in categories:
            out.at[out.index[i], f"obs_{category}"] = measured[category]
        targets = {c: float(out.iloc[i][f"target_{c}"]) for c in categories}
        raw = []
        std = []
        for category in categories:
            diff = measured[category] - targets[category]
            raw.append(abs(diff))
            std.append(abs(diff) / scales[category])
        raw_maes.append(float(np.mean(raw)))
        std_maes.append(float(np.mean(std)))

    out["raw_mae"] = raw_maes
    out["standardized_mae"] = std_maes
    out["scored"] = True
    return out.drop(columns=["concat_text"])


def summarize_author_level(
    author_scored: pd.DataFrame,
    categories: list[str],
    *,
    calibration_metrics_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    mean_calibration_fn: Callable[[dict], dict[str, float]],
) -> pd.DataFrame:
    """MAE + mean per-category ρ/slope by condition [× model]."""
    if author_scored.empty:
        return pd.DataFrame()

    has_model = "model_key" in author_scored.columns
    rows: list[dict] = []

    if has_model:
        keys = sorted(
            {
                (str(r.steering_condition), str(r.model_key))
                for r in author_scored.itertuples(index=False)
            }
        )
        for condition, model in keys:
            subset = author_scored[
                (author_scored["steering_condition"] == condition)
                & (author_scored["model_key"] == model)
            ]
            rows.append(
                _summary_row(
                    subset,
                    categories,
                    condition=condition,
                    model_key=model,
                    calibration_metrics_fn=calibration_metrics_fn,
                    mean_calibration_fn=mean_calibration_fn,
                )
            )
        for condition in STEERING_CONDITIONS:
            subset = author_scored[author_scored["steering_condition"] == condition]
            if subset.empty:
                continue
            rows.append(
                _summary_row(
                    subset,
                    categories,
                    condition=condition,
                    model_key="ALL",
                    calibration_metrics_fn=calibration_metrics_fn,
                    mean_calibration_fn=mean_calibration_fn,
                )
            )
    else:
        for condition in sorted(author_scored["steering_condition"].unique()):
            subset = author_scored[author_scored["steering_condition"] == condition]
            rows.append(
                _summary_row(
                    subset,
                    categories,
                    condition=str(condition),
                    model_key=None,
                    calibration_metrics_fn=calibration_metrics_fn,
                    mean_calibration_fn=mean_calibration_fn,
                )
            )
    return pd.DataFrame(rows)


def _summary_row(
    subset: pd.DataFrame,
    categories: list[str],
    *,
    condition: str,
    model_key: str | None,
    calibration_metrics_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    mean_calibration_fn: Callable[[dict], dict[str, float]],
) -> dict:
    cat_metrics = {}
    for category in categories:
        cat_metrics[category] = calibration_metrics_fn(
            subset[f"target_{category}"].to_numpy(dtype=float),
            subset[f"obs_{category}"].to_numpy(dtype=float),
        )
    mean_cal = mean_calibration_fn(cat_metrics)
    row: dict = {
        "steering_condition": condition,
        "n_authors": int(len(subset)),
        "mean_standardized_mae": round(float(np.nanmean(subset["standardized_mae"])), 4),
        "mean_category_rho": round(mean_cal["mean_rho"], 4)
        if mean_cal["mean_rho"] == mean_cal["mean_rho"]
        else None,
        "mean_category_slope": round(mean_cal["mean_slope"], 4)
        if mean_cal["mean_slope"] == mean_cal["mean_slope"]
        else None,
    }
    if model_key is not None:
        row["model_key"] = model_key
    return row


def run_author_level_from_generations(
    generations: pd.DataFrame,
    *,
    categories: list[str],
    scales: dict[str, float],
    score_fn: ScoreFn,
    calibration_metrics_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    mean_calibration_fn: Callable[[dict], dict[str, float]],
    mock: bool = False,
    mock_score_fn: Callable[..., dict[str, float]] | None = None,
    corpus_means: dict[str, float] | None = None,
    batch_size: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: concat → score → per-author CSV + condition summary."""
    groups = build_author_concat_groups(generations, categories=categories)
    scored = score_author_concatenations(
        groups,
        categories=categories,
        scales=scales,
        score_fn=score_fn,
        mock_score_fn=mock_score_fn,
        mock=mock,
        batch_size=batch_size,
        corpus_means=corpus_means,
    )
    summary = summarize_author_level(
        scored,
        categories,
        calibration_metrics_fn=calibration_metrics_fn,
        mean_calibration_fn=mean_calibration_fn,
    )
    return scored, summary
