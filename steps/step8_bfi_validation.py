#!/usr/bin/env python3
"""Step 8: essay-only BFI-44 validation (Validation 3).

An independent BFI-based evaluator rates the personality expressed in each
generated essay. The evaluator sees only the essay text and the BFI-44
questionnaire — not target O/N, steering condition, LIWC targets, or the
generation prompt — avoiding role-self-report / prompt-echo circularity.

Trait means are mapped to PANDORA's 0–100 scale via (mean − 1) / 4 × 100.

Default: all full narrative topics × all profiles × 4 conditions × 3 models.
Primary metrics (author-level mean BFI over topics):
  MAE_BFI (standardised absolute error), ρ_BFI (Spearman), β_BFI (calibration slope).
Essay-level rows are retained as secondary.
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    DEFAULT_AUTHORS,
    DEFAULT_FULL_CONTENTS,
    FULL_RESULTS_DIR,
    OLD_PIPELINE_DIR,
    OPTIMIZED_DIR,
    SCRIPTS_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import stage2_step8_full_generation as full  # noqa: E402

import step6_full_generation as gen  # noqa: E402

DEFAULT_GENERATIONS = FULL_RESULTS_DIR / "generations" / "full_generations.csv"
DEFAULT_OUTPUT_DIR = FULL_RESULTS_DIR / "bfi"
DEFAULT_PER_SAMPLE = DEFAULT_OUTPUT_DIR / "bfi_per_sample.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "bfi_summary.csv"
DEFAULT_AUTHOR_SUMMARY = DEFAULT_OUTPUT_DIR / "bfi_author_summary.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "bfi.report.json"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2500
DEFAULT_CHECKPOINT_EVERY = 10
# Protocol id written into report / used to invalidate old role-self-report rows.
BFI_PROTOCOL = "essay_only_rater_v1"

# John & Srivastava (1999) BFI-44 — reverse-keyed item indices (1-based).
BFI_SCALES: dict[str, list[int]] = {
    "E": [1, -6, 11, 16, -21, 26, -31, 36],
    "A": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
    "C": [3, -8, 13, -18, -23, 28, 33, 38, -43],
    "N": [4, -9, 14, 19, -24, 29, -34, 39],
    "O": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
}

BFI_ITEMS: list[str] = [
    "Is talkative",
    "Tends to find fault with others",
    "Does a thorough job",
    "Is depressed, blue",
    "Is original, comes up with new ideas",
    "Is reserved",
    "Is helpful and unselfish with others",
    "Can be somewhat careless",
    "Is relaxed, handles stress well",
    "Is curious about many different things",
    "Is full of energy",
    "Starts quarrels with others",
    "Is a reliable worker",
    "Can be tense",
    "Is ingenious, a deep thinker",
    "Generates a lot of enthusiasm",
    "Has a forgiving nature",
    "Tends to be disorganized",
    "Worries a lot",
    "Has an active imagination",
    "Tends to be quiet",
    "Is generally trusting",
    "Tends to be lazy",
    "Is emotionally stable, not easily upset",
    "Is inventive",
    "Has an assertive personality",
    "Can be cold and aloof",
    "Perseveres until the task is finished",
    "Can be moody",
    "Values artistic, aesthetic experiences",
    "Is sometimes shy, inhibited",
    "Is considerate and kind to almost everyone",
    "Does things efficiently",
    "Remains calm in tense situations",
    "Prefers work that is routine",
    "Is outgoing, sociable",
    "Is sometimes rude to others",
    "Makes plans and follows through with them",
    "Gets nervous easily",
    "Likes to reflect, play with ideas",
    "Has few artistic interests",
    "Likes to cooperate with others",
    "Is easily distracted",
    "Is sophisticated in art, music, or literature",
]

BFI_COLS = [
    "bfi_protocol",
    "bfi_scored",
    "bfi_parse_ok",
    "bfi_prompt",
    "bfi_raw",
    "bfi_error",
    "bfi_O",
    "bfi_C",
    "bfi_E",
    "bfi_A",
    "bfi_N",
    "bfi_n_items_parsed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-sample", type=Path, default=DEFAULT_PER_SAMPLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--author-summary",
        type=Path,
        default=DEFAULT_AUTHOR_SUMMARY,
        help="Profile-level aggregated BFI summary (primary).",
    )
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--contents",
        nargs="+",
        default=list(DEFAULT_FULL_CONTENTS),
        help="Topics to rate (default: all six full narrative topics).",
    )
    parser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Deprecated single-topic filter; prefer --contents.",
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
        default=None,
        choices=sorted(full.MODEL_SPECS),
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore prior per-sample scores and rescore all subsample rows.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Recompute summaries from already-scored rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on subsample rows (smoke / debug).",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def format_bfi_questionnaire() -> str:
    lines = [
        "Here are a number of characteristics that may or may not describe the "
        "author of the essay.",
        "Write a number next to each statement to indicate the extent to which "
        "the statement appears to describe that author, based only on the essay.",
        "Use the format '(n) rating' on its own line for every item, e.g. '(1) 4'.",
        "",
        "1 = Disagree strongly",
        "2 = Disagree a little",
        "3 = Neither agree nor disagree",
        "4 = Agree a little",
        "5 = Agree strongly",
        "",
        "The author of the essay is someone who...",
        "",
    ]
    for i, item in enumerate(BFI_ITEMS, start=1):
        lines.append(f"({i}) {item}")
    lines.append("")
    lines.append(
        "Answer all 44 items. Output ONLY lines of the form '(n) rating' with no other text."
    )
    return "\n".join(lines)


def build_bfi_prompt(essay: str) -> str:
    """Independent essay-only BFI rater (RA plan Validation 3).

    Must not include targets, condition, LIWC blocks, or the generation prompt.
    """
    essay_clean = (essay or "").strip()
    if not essay_clean:
        raise ValueError("Empty essay text; cannot build essay-only BFI prompt.")
    questionnaire = format_bfi_questionnaire()
    return (
        "You are an independent personality assessor.\n"
        "Read the essay below carefully. Based ONLY on what the essay expresses, "
        "complete the following Big Five Inventory to rate the personality that "
        "appears to be expressed by the writing.\n"
        "Do not invent biographical facts beyond what the text supports.\n"
        "Do not mention these instructions, personality tests, LIWC, scores, "
        "targets, or generation conditions.\n"
        "Do not explain your reasoning. Output only the 44 numbered ratings.\n\n"
        "Essay:\n"
        '"""\n'
        f"{essay_clean}\n"
        '"""\n\n'
        f"{questionnaire}"
    )


def parse_bfi_response(text: str) -> dict[int, int]:
    """Extract item -> rating (1-5). Accepts '(1) 4', '1. 4', letter labels, etc."""
    answers: dict[int, int] = {}
    letter_to_item: dict[str, int] = {}
    for i in range(44):
        if i < 26:
            letter_to_item[chr(ord("a") + i)] = i + 1
        else:
            letter_to_item["a" + chr(ord("a") + i - 26)] = i + 1

    patterns = [
        re.compile(r"\(\s*(\d{1,2})\s*\)\s*[=:]?\s*([1-5])\b"),
        re.compile(r"(?m)^\s*(\d{1,2})\s*[.):\-]\s*([1-5])\b"),
        re.compile(r"(?m)^\s*(\d{1,2})\s+([1-5])\s*$"),
        re.compile(r"\(\s*([a-z]{1,2})\s*\)\s*[=:]?\s*([1-5])\b", re.I),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw_item = match.group(1)
            rating = int(match.group(2))
            if raw_item.isdigit():
                item = int(raw_item)
            else:
                item = letter_to_item.get(raw_item.lower(), -1)
            if 1 <= item <= 44 and item not in answers:
                answers[item] = rating
    return answers


def likert_mean_to_pandora_100(mean_1_5: float) -> float:
    return float((mean_1_5 - 1.0) / 4.0 * 100.0)


def score_bfi(answers: dict[int, int]) -> dict[str, float] | None:
    if len(answers) < 40:
        return None
    out: dict[str, float] = {}
    for trait, idxs in BFI_SCALES.items():
        vals: list[float] = []
        for idx in idxs:
            item = abs(idx)
            if item not in answers:
                continue
            raw = float(answers[item])
            vals.append((6.0 - raw) if idx < 0 else raw)
        if len(vals) < max(1, len(idxs) - 1):
            return None
        out[trait] = likert_mean_to_pandora_100(float(np.mean(vals)))
    return out


def mock_bfi_answers(essay: str, seed: int) -> str:
    """Deterministic mock ratings from essay length hash (no target leakage)."""
    rng = np.random.default_rng((seed + len(essay)) % (2**32))
    centers = {
        "O": 3.0 + float(rng.normal(0, 0.4)),
        "N": 3.0 + float(rng.normal(0, 0.4)),
        "E": 3.0,
        "A": 3.0,
        "C": 3.0,
    }

    item_trait: dict[int, tuple[str, bool]] = {}
    for trait, idxs in BFI_SCALES.items():
        for idx in idxs:
            item_trait[abs(idx)] = (trait, idx < 0)

    lines = []
    for item in range(1, 45):
        trait, is_rev = item_trait[item]
        desired = centers[trait] + float(rng.normal(0, 0.55))
        raw = (6.0 - desired) if is_rev else desired
        rating = int(np.clip(round(raw), 1, 5))
        lines.append(f"({item}) {rating}")
    return "\n".join(lines)


def select_subsample(
    generations: pd.DataFrame,
    *,
    contents: list[str],
    conditions: list[str],
    models: list[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    mask = (
        (generations["content"].isin(contents))
        & (generations["steering_condition"].isin(conditions))
        & (generations["repetition"] == 0)
    )
    if models:
        mask &= generations["model_key"].isin(models)
    sub = generations.loc[mask].copy()
    if sub.empty:
        raise SystemExit(
            f"No subsample rows for contents={contents!r}, conditions={conditions}."
        )
    empty_text = sub["text"].isna() | (sub["text"].astype(str).str.strip() == "")
    if empty_text.any():
        print(f"Skipping {int(empty_text.sum())} rows with empty essay text")
        sub = sub.loc[~empty_text].copy()
    if sub.empty:
        raise SystemExit("All selected rows have empty essay text.")
    sub = sub.sort_values(
        ["model_key", "steering_condition", "profile_id", "content", "plan_id"]
    ).reset_index(drop=True)
    if limit is not None:
        sub = sub.head(limit).copy()
    print(
        f"Subsample: {len(sub)} rows "
        f"({sub['model_key'].nunique()} models × "
        f"{sub['steering_condition'].nunique()} conditions × "
        f"{sub['profile_id'].nunique()} profiles × "
        f"{sub['content'].nunique()} topics)"
    )
    return sub


def ensure_bfi_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = ("bfi_scored", "bfi_parse_ok")
    str_cols = ("bfi_protocol", "bfi_prompt", "bfi_raw", "bfi_error")
    float_cols = ("bfi_O", "bfi_C", "bfi_E", "bfi_A", "bfi_N", "bfi_n_items_parsed")
    for col in bool_cols:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    for col in str_cols:
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="object")
        else:
            df[col] = df[col].astype("object")
    for col in float_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    if len(x) < 3:
        return None, None
    rho, p = stats.spearmanr(x, y)
    rho_out = float(rho) if rho == rho else None
    p_out = float(p) if p == p else None
    return rho_out, p_out


def _calibration_slope(target: pd.Series, observed: pd.Series) -> float | None:
    t = target.to_numpy(dtype=float)
    o = observed.to_numpy(dtype=float)
    if len(t) < 3 or float(np.std(t)) < 1e-8:
        return None
    slope = float(stats.linregress(t, o).slope)
    return slope if slope == slope else None


def load_human_trait_scales(authors_path: Path | None = None) -> dict[str, float]:
    """Human O/N SDs for standardised MAE_BFI (matches LIWC MAE standardisation)."""
    path = authors_path or (OLD_PIPELINE_DIR / Path(DEFAULT_AUTHORS).name)
    if not path.exists():
        # Fallback: repo-relative default used by Stage 1.
        path = OLD_PIPELINE_DIR / "pandora-authors-train.csv"
    human = pd.read_csv(path)
    return {
        "O": float(max(human["O"].std(ddof=1), 1e-6)),
        "N": float(max(human["N"].std(ddof=1), 1e-6)),
    }


def _trait_recovery_metrics(
    target: pd.Series,
    observed: pd.Series,
    *,
    scale: float,
) -> dict[str, float | None]:
    rho, p = _spearman(target, observed)
    err = (observed - target).abs().to_numpy(dtype=float)
    mae = float(np.mean(err)) if len(err) else float("nan")
    std_mae = mae / scale if scale > 0 else float("nan")
    slope = _calibration_slope(target, observed)
    return {
        "rho": None if rho is None else round(rho, 4),
        "p": p,
        "mae": round(mae, 4) if mae == mae else None,
        "standardized_mae": round(std_mae, 4) if std_mae == std_mae else None,
        "slope": None if slope is None else round(slope, 4),
    }


def summarize_essay_level(
    df: pd.DataFrame,
    *,
    trait_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Secondary: essay-level MAE_BFI / ρ_BFI / β_BFI (topics counted separately)."""
    scales = trait_scales or load_human_trait_scales()
    sub = df[df["bfi_parse_ok"].fillna(False)].copy()
    rows = []
    for (model_key, condition), g in sub.groupby(["model_key", "steering_condition"]):
        met_o = _trait_recovery_metrics(g["target_O"], g["bfi_O"], scale=scales["O"])
        met_n = _trait_recovery_metrics(g["target_N"], g["bfi_N"], scale=scales["N"])
        std_vals = [
            v
            for v in (met_o["standardized_mae"], met_n["standardized_mae"])
            if v is not None
        ]
        slope_vals = [v for v in (met_o["slope"], met_n["slope"]) if v is not None]
        rows.append(
            {
                "level": "essay",
                "model_key": model_key,
                "steering_condition": condition,
                "n": len(g),
                "mean_bfi_O": round(float(g["bfi_O"].mean()), 4),
                "mean_bfi_N": round(float(g["bfi_N"].mean()), 4),
                "rho_O_vs_target": met_o["rho"],
                "p_O": met_o["p"],
                "rho_N_vs_target": met_n["rho"],
                "p_N": met_n["p"],
                "mae_O": met_o["mae"],
                "mae_N": met_n["mae"],
                "standardized_mae_O": met_o["standardized_mae"],
                "standardized_mae_N": met_n["standardized_mae"],
                "mean_standardized_mae": (
                    round(float(np.mean(std_vals)), 4) if std_vals else None
                ),
                "slope_O": met_o["slope"],
                "slope_N": met_n["slope"],
                "mean_slope": (
                    round(float(np.mean(slope_vals)), 4) if slope_vals else None
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_author_level(
    df: pd.DataFrame,
    *,
    trait_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Primary: mean BFI over topics, then MAE_BFI / ρ_BFI / β_BFI."""
    scales = trait_scales or load_human_trait_scales()
    sub = df[df["bfi_parse_ok"].fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame()
    author = (
        sub.groupby(["model_key", "steering_condition", "profile_id"], as_index=False)
        .agg(
            target_O=("target_O", "first"),
            target_N=("target_N", "first"),
            bfi_O=("bfi_O", "mean"),
            bfi_N=("bfi_N", "mean"),
            n_essays=("plan_id", "count"),
        )
    )
    rows = []
    for (model_key, condition), g in author.groupby(["model_key", "steering_condition"]):
        met_o = _trait_recovery_metrics(g["target_O"], g["bfi_O"], scale=scales["O"])
        met_n = _trait_recovery_metrics(g["target_N"], g["bfi_N"], scale=scales["N"])
        std_vals = [
            v
            for v in (met_o["standardized_mae"], met_n["standardized_mae"])
            if v is not None
        ]
        slope_vals = [v for v in (met_o["slope"], met_n["slope"]) if v is not None]
        rows.append(
            {
                "level": "author",
                "model_key": model_key,
                "steering_condition": condition,
                "n_profiles": int(len(g)),
                "mean_n_essays": round(float(g["n_essays"].mean()), 2),
                "mean_bfi_O": round(float(g["bfi_O"].mean()), 4),
                "mean_bfi_N": round(float(g["bfi_N"].mean()), 4),
                "rho_O_vs_target": met_o["rho"],
                "p_O": met_o["p"],
                "rho_N_vs_target": met_n["rho"],
                "p_N": met_n["p"],
                "mae_O": met_o["mae"],
                "mae_N": met_n["mae"],
                "standardized_mae_O": met_o["standardized_mae"],
                "standardized_mae_N": met_n["standardized_mae"],
                "mean_standardized_mae": (
                    round(float(np.mean(std_vals)), 4) if std_vals else None
                ),
                "slope_O": met_o["slope"],
                "slope_N": met_n["slope"],
                "mean_slope": (
                    round(float(np.mean(slope_vals)), 4) if slope_vals else None
                ),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    df: pd.DataFrame,
    *,
    per_sample_path: Path,
    summary_path: Path,
    author_summary_path: Path,
    report_path: Path,
    contents: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_results_dir(per_sample_path.parent)
    keep = [
        "plan_id",
        "profile_id",
        "content",
        "steering_condition",
        "repetition",
        "model_key",
        "api_model",
        "target_O",
        "target_N",
        *BFI_COLS,
    ]
    keep = [c for c in keep if c in df.columns]
    current = df[keep].copy()

    if per_sample_path.exists():
        prior = pd.read_csv(per_sample_path)
        for col in keep:
            if col not in prior.columns:
                prior[col] = (
                    False if col in ("bfi_scored", "bfi_parse_ok") else pd.NA
                )
        prior = prior[[c for c in keep if c in prior.columns]]
        if "bfi_protocol" in prior.columns and "bfi_protocol" in current.columns:
            prior = prior[prior["bfi_protocol"].fillna("") == BFI_PROTOCOL]
        prior = prior[~prior["plan_id"].isin(current["plan_id"])]
        combined = pd.concat([prior, current], ignore_index=True)
    else:
        combined = current

    combined = combined.sort_values(
        ["model_key", "steering_condition", "profile_id", "content", "plan_id"]
    ).reset_index(drop=True)
    combined.to_csv(per_sample_path, index=False)

    trait_scales = load_human_trait_scales()
    essay_summary = summarize_essay_level(combined, trait_scales=trait_scales)
    essay_summary.to_csv(summary_path, index=False)
    author_summary = summarize_author_level(combined, trait_scales=trait_scales)
    author_summary.to_csv(author_summary_path, index=False)

    resolved_models: dict[str, str] = {}
    if "api_model" in combined.columns:
        for model_key, g in combined.groupby("model_key"):
            vals = g["api_model"].dropna().astype(str)
            vals = vals[vals.str.len() > 0]
            if not vals.empty:
                resolved_models[str(model_key)] = str(vals.iloc[0])
    report = {
        "step": "Optimized continuous / Step 8 — essay-only BFI-44 validation",
        "protocol": BFI_PROTOCOL,
        "protocol_note": (
            "Independent BFI rater sees only the generated essay + questionnaire. "
            "No target O/N, steering condition, LIWC targets, or generation prompt."
        ),
        "n_subsample": int(len(combined)),
        "n_scored": int(combined["bfi_scored"].fillna(False).sum()),
        "n_parse_ok": int(combined["bfi_parse_ok"].fillna(False).sum()),
        "contents": contents,
        "n_topics": int(combined["content"].nunique()) if "content" in combined.columns else 0,
        "score_scale": "0-100",
        "score_scale_note": (
            "Trait composites aligned with PANDORA Big Five (0–100). "
            "Items answered 1–5; mapped via (mean − 1) / 4 × 100."
        ),
        "primary_metrics": {
            "MAE_BFI": "standardised |bfi − target| using human O/N SD",
            "rho_BFI": "Spearman ρ(target_O/N, mean_bfi_O/N over topics)",
            "beta_BFI": "OLS slope of mean_bfi on target (per trait)",
        },
        "human_trait_scales": {
            "sd_O": round(trait_scales["O"], 4),
            "sd_N": round(trait_scales["N"], 4),
        },
        "api_models_used": resolved_models,
        "outputs": {
            "per_sample": per_sample_path.name,
            "essay_summary": summary_path.name,
            "author_summary": author_summary_path.name,
        },
        "author_summary": author_summary.to_dict(orient="records"),
        "essay_summary": essay_summary.to_dict(orient="records"),
    }
    write_json(report_path, report)
    return author_summary, essay_summary


def main() -> None:
    args = parse_args()
    dotenv_path = args.dotenv if args.dotenv.is_absolute() else (OPTIMIZED_DIR.parent / args.dotenv)
    gen.load_dotenv(dotenv_path)

    generations_path = _resolve(args.generations)
    per_sample_path = _resolve(args.per_sample)
    summary_path = _resolve(args.summary)
    author_summary_path = _resolve(args.author_summary)
    report_path = _resolve(args.report)
    ensure_results_dir(_resolve(args.output_dir))

    contents = list(args.contents)
    if args.content:
        contents = [args.content]

    if not generations_path.exists():
        raise SystemExit(f"Generations CSV not found: {generations_path}")

    generations = pd.read_csv(generations_path)
    if "text" not in generations.columns:
        raise SystemExit("Generations CSV missing required 'text' column.")

    subsample = select_subsample(
        generations,
        contents=contents,
        conditions=list(args.conditions),
        models=args.models,
        limit=args.limit,
    )
    empty_text = subsample["text"].isna() | (subsample["text"].astype(str).str.strip() == "")
    if empty_text.any():
        bad = subsample.loc[empty_text, "plan_id"].head(10).tolist()
        raise SystemExit(f"Empty essay text for plan_ids: {bad}")

    if per_sample_path.exists() and not args.force:
        prior = pd.read_csv(per_sample_path)
        prior = ensure_bfi_columns(prior)
        if "bfi_protocol" in prior.columns:
            prior = prior[prior["bfi_protocol"].fillna("") == BFI_PROTOCOL]
        else:
            prior = prior.iloc[0:0]
        keep_prior = ["plan_id", *BFI_COLS]
        keep_prior = [c for c in keep_prior if c in prior.columns]
        if len(prior) and keep_prior:
            subsample = subsample.drop(
                columns=[c for c in BFI_COLS if c in subsample.columns],
                errors="ignore",
            )
            subsample = subsample.merge(prior[keep_prior], on="plan_id", how="left")
    subsample = ensure_bfi_columns(subsample)
    if args.force:
        for col in BFI_COLS:
            if col in ("bfi_scored", "bfi_parse_ok"):
                subsample[col] = False
            elif col in ("bfi_O", "bfi_C", "bfi_E", "bfi_A", "bfi_N", "bfi_n_items_parsed"):
                subsample[col] = np.nan
            else:
                subsample[col] = pd.NA
        subsample["bfi_protocol"] = BFI_PROTOCOL

    if args.score_only:
        author_summary, essay_summary = write_outputs(
            subsample,
            per_sample_path=per_sample_path,
            summary_path=summary_path,
            author_summary_path=author_summary_path,
            report_path=report_path,
            contents=contents,
        )
        print("=== Author-level BFI (primary) ===")
        print(author_summary.to_string(index=False))
        print("\n=== Essay-level BFI (secondary) ===")
        print(essay_summary.to_string(index=False))
        print(f"\nWrote {per_sample_path}")
        print(f"Wrote {author_summary_path}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {report_path}")
        return

    models = sorted(subsample["model_key"].unique())
    model_resolved = {key: full.resolve_model(key) for key in models}
    if not args.mock and not args.dry_run:
        missing = [k for k, cfg in model_resolved.items() if not cfg["api_key"]]
        if missing:
            raise SystemExit(f"Missing API keys for models: {missing}")

    id_to_idx = {str(pid): i for i, pid in enumerate(subsample["plan_id"].astype(str))}
    pending_ids = [
        pid
        for pid in subsample["plan_id"].astype(str)
        if (not bool(subsample.at[id_to_idx[pid], "bfi_scored"]))
        or (not bool(subsample.at[id_to_idx[pid], "bfi_parse_ok"]))
        or str(subsample.at[id_to_idx[pid], "bfi_protocol"]) != BFI_PROTOCOL
    ]
    for pid in pending_ids:
        idx = id_to_idx[pid]
        if bool(subsample.at[idx, "bfi_scored"]) and (
            not bool(subsample.at[idx, "bfi_parse_ok"])
            or str(subsample.at[idx, "bfi_protocol"]) != BFI_PROTOCOL
        ):
            subsample.at[idx, "bfi_scored"] = False

    model_priority = {"gpt-4o-mini": 0, "deepseek-v3": 1, "qwen3-32b": 2}

    def _pending_key(pid: str) -> tuple[int, str]:
        idx = id_to_idx[pid]
        return (model_priority.get(str(subsample.at[idx, "model_key"]), 9), pid)

    pending_ids = sorted(pending_ids, key=_pending_key)
    print(f"Protocol: {BFI_PROTOCOL}")
    print(f"Pending BFI calls: {len(pending_ids)} / {len(subsample)}")
    if pending_ids:
        print(f"  first pending: {pending_ids[0]}")

    if args.dry_run:
        row = subsample.iloc[0]
        prompt = build_bfi_prompt(str(row["text"]))
        print(f"Dry run OK. Example prompt ({len(prompt)} chars):\n")
        print(prompt[:1100], "...\n")
        if "Essay:" not in prompt:
            raise SystemExit("Dry-run failed: prompt missing Essay: block.")
        if "Pretend you are" in prompt:
            raise SystemExit("Dry-run failed: prompt still contains role-play steering.")
        return

    lock = threading.Lock()
    completed = 0

    def run_one(pid: str) -> None:
        nonlocal completed
        idx = id_to_idx[pid]
        row = subsample.iloc[idx]
        model_key = str(row["model_key"])
        prompt = build_bfi_prompt(str(row["text"]))
        raw = ""
        error = ""
        answers: dict[int, int] = {}
        scores = None
        parse_ok = False
        api_model = ""
        try:
            if args.mock:
                api_model = "mock"
                raw = mock_bfi_answers(
                    str(row["text"]),
                    seed=abs(hash(pid)) % (2**32),
                )
            else:
                cfg = model_resolved[model_key]
                api_model = str(cfg["model"])
                raw = full.chat_complete(
                    prompt,
                    api_key=str(cfg["api_key"]),
                    base_url=str(cfg["base_url"]),
                    model=str(cfg["model"]),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    extra=dict(cfg["extra"]),  # type: ignore[arg-type]
                )
            answers = parse_bfi_response(raw)
            scores = score_bfi(answers)
            parse_ok = scores is not None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        with lock:
            subsample.at[idx, "bfi_protocol"] = BFI_PROTOCOL
            subsample.at[idx, "bfi_prompt"] = prompt
            subsample.at[idx, "bfi_raw"] = raw
            subsample.at[idx, "bfi_error"] = error
            subsample.at[idx, "api_model"] = api_model
            if error:
                subsample.at[idx, "bfi_scored"] = False
                subsample.at[idx, "bfi_parse_ok"] = False
                print(
                    f"[{completed + 1}/{len(pending_ids)}] {pid} ERROR: {error}",
                    flush=True,
                )
            else:
                subsample.at[idx, "bfi_parse_ok"] = parse_ok
                subsample.at[idx, "bfi_n_items_parsed"] = len(answers)
                subsample.at[idx, "bfi_scored"] = True
                if parse_ok and scores is not None:
                    subsample.at[idx, "bfi_O"] = scores["O"]
                    subsample.at[idx, "bfi_C"] = scores["C"]
                    subsample.at[idx, "bfi_E"] = scores["E"]
                    subsample.at[idx, "bfi_A"] = scores["A"]
                    subsample.at[idx, "bfi_N"] = scores["N"]
                print(
                    f"[{completed + 1}/{len(pending_ids)}] {pid} "
                    f"parse_ok={parse_ok} items={len(answers)} ({api_model})",
                    flush=True,
                )
            completed += 1
            do_checkpoint = (
                completed % args.checkpoint_every == 0
                or completed == len(pending_ids)
            )
            snapshot = subsample.copy() if do_checkpoint else None

        if snapshot is not None:
            write_outputs(
                snapshot,
                per_sample_path=per_sample_path,
                summary_path=summary_path,
                author_summary_path=author_summary_path,
                report_path=report_path,
                contents=contents,
            )
            print(f"  checkpoint wrote ({completed} new calls)", flush=True)

    workers = 1 if args.mock else max(1, args.workers)
    print(f"Running with {workers} worker(s)")
    if workers == 1:
        for pid in pending_ids:
            run_one(pid)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, pid) for pid in pending_ids]
            for fut in as_completed(futures):
                fut.result()

    author_summary, essay_summary = write_outputs(
        subsample,
        per_sample_path=per_sample_path,
        summary_path=summary_path,
        author_summary_path=author_summary_path,
        report_path=report_path,
        contents=contents,
    )

    print("\n=== Author-level BFI (primary) ===")
    print(author_summary.to_string(index=False))
    print("\n=== Essay-level BFI (secondary) ===")
    print(essay_summary.to_string(index=False))
    print(f"\nWrote {per_sample_path}")
    print(f"Wrote {author_summary_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
