#!/usr/bin/env python3
"""Step 6: Full multi-model continuous generation (4 strategies × 3 models).

Reads the prepare_full plan + prompt components, generates with checkpointing /
resume, then LIWC-scores and summarizes MAE / ρ / slope by condition × model.

Essay-level summaries are secondary. Author-level (concat essays → re-LIWC) is
the Validation 1 primary metric.

Example:
  python optimized/steps/step6_full_generation.py --workers 6
  python optimized/steps/step6_full_generation.py --mock-generation --mock-liwc --smoke
  python optimized/steps/step6_full_generation.py --score-only
  python optimized/steps/step6_full_generation.py --score-author-level-only
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CONTENT_CONDITIONS,
    DEFAULT_LIWC_CLI,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TARGET_WORDS,
    DEFAULT_TEMPERATURE,
    FULL_CONTENT_CONDITIONS,
    OPTIMIZED_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_DIR,
    FULL_RESULTS_DIR,
    SCRIPTS_DIR,
    STEERING_CONDITIONS,
    ensure_results_dir,
    write_json,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import stage2_step8_full_generation as full  # noqa: E402

import step5_pilot_generation as pilot  # noqa: E402

import liwc_author_level as author_liwc  # noqa: E402

FULL_DIR = FULL_RESULTS_DIR
DEFAULT_PLAN = FULL_DIR / "step3_generation_plan.csv"
DEFAULT_COMPONENTS = FULL_DIR / "step4_prompt_components.csv"
DEFAULT_HUMAN = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_OUTPUT_DIR = FULL_DIR / "generations"
DEFAULT_GENERATIONS = DEFAULT_OUTPUT_DIR / "full_generations.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "full_condition_model_summary.csv"
DEFAULT_AUTHOR_LEVEL = DEFAULT_OUTPUT_DIR / "full_author_level_liwc.csv"
DEFAULT_AUTHOR_SUMMARY = DEFAULT_OUTPUT_DIR / "full_author_condition_model_summary.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "full_generation.report.json"

# Make narrative topics visible to prompt builder.
CONTENT_CONDITIONS.update(FULL_CONTENT_CONDITIONS)
pilot.CONTENT_CONDITIONS.update(FULL_CONTENT_CONDITIONS)


@dataclass
class FullResult:
    plan_id: str
    profile_id: str
    cell: str
    content: str
    steering_condition: str
    repetition: int
    model_key: str
    api_model: str
    prompt: str
    text: str
    backend: str
    target_O: float
    target_N: float
    targets: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    measured: dict[str, float] = field(default_factory=dict)
    raw_mae: float = float("nan")
    standardized_mae: float = float("nan")
    type_token_ratio: float = float("nan")
    self_identification: bool = False
    n_words: int = 0
    scored: bool = False


_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--author-level",
        type=Path,
        default=DEFAULT_AUTHOR_LEVEL,
        help="Per author×condition×model concat→re-LIWC rows (primary).",
    )
    parser.add_argument(
        "--author-summary",
        type=Path,
        default=DEFAULT_AUTHOR_SUMMARY,
        help="Author-level condition×model summary (Validation 1 primary).",
    )
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--liwc-cli", type=Path, default=Path(DEFAULT_LIWC_CLI))
    parser.add_argument("--liwc-batch-size", type=int, default=40)
    parser.add_argument("--mock-generation", action="store_true")
    parser.add_argument("--mock-liwc", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--score-author-level-only",
        action="store_true",
        help="Skip generation; concat existing essays and re-score LIWC at author level.",
    )
    parser.add_argument(
        "--resummarize-author-only",
        action="store_true",
        help=(
            "Recompute author-level LIWC summaries from an existing "
            "full_author_level_liwc.csv (no LIWC CLI / no generation)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 profile × 1 topic × all conditions × all models.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=list(STEERING_CONDITIONS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=sorted(full.MODEL_SPECS),
    )
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def load_dotenv(path: Path) -> None:
    resolved = path if path.is_absolute() else (OPTIMIZED_DIR.parent / path)
    if not resolved.exists():
        return
    for line in resolved.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def results_to_frame(results: list[FullResult], categories: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        row = {
            "plan_id": r.plan_id,
            "profile_id": r.profile_id,
            "cell": r.cell,
            "content": r.content,
            "steering_condition": r.steering_condition,
            "repetition": r.repetition,
            "model_key": r.model_key,
            "api_model": r.api_model,
            "backend": r.backend,
            "target_O": r.target_O,
            "target_N": r.target_N,
            "n_words": r.n_words,
            "type_token_ratio": r.type_token_ratio,
            "self_identification": r.self_identification,
            "raw_mae": r.raw_mae,
            "standardized_mae": r.standardized_mae,
            "scored": r.scored,
            "error": r.error,
            "text": r.text,
            "prompt": r.prompt,
        }
        for c in categories:
            row[f"target_{c}"] = r.targets.get(c)
            row[f"obs_{c}"] = r.measured.get(c)
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_results(frame: pd.DataFrame, categories: list[str]) -> list[FullResult]:
    out: list[FullResult] = []
    for _, row in frame.iterrows():
        targets = {
            c: float(row[f"target_{c}"])
            for c in categories
            if f"target_{c}" in frame.columns and pd.notna(row.get(f"target_{c}"))
        }
        measured = {
            c: float(row[f"obs_{c}"])
            for c in categories
            if f"obs_{c}" in frame.columns and pd.notna(row.get(f"obs_{c}"))
        }
        err = row.get("error")
        out.append(
            FullResult(
                plan_id=str(row["plan_id"]),
                profile_id=str(row["profile_id"]),
                cell=str(row.get("cell", "")),
                content=str(row["content"]),
                steering_condition=str(row["steering_condition"]),
                repetition=int(row["repetition"]),
                model_key=str(row["model_key"]),
                api_model=str(row.get("api_model", "")),
                prompt=str(row.get("prompt", "")),
                text=str(row.get("text", "") or ""),
                backend=str(row.get("backend", "")),
                target_O=float(row["target_O"]),
                target_N=float(row["target_N"]),
                targets=targets,
                error=None if pd.isna(err) or err == "" else str(err),
                measured=measured,
                raw_mae=float(row["raw_mae"]) if pd.notna(row.get("raw_mae")) else float("nan"),
                standardized_mae=(
                    float(row["standardized_mae"])
                    if pd.notna(row.get("standardized_mae"))
                    else float("nan")
                ),
                type_token_ratio=(
                    float(row["type_token_ratio"])
                    if pd.notna(row.get("type_token_ratio"))
                    else float("nan")
                ),
                self_identification=bool(row.get("self_identification", False)),
                n_words=int(row["n_words"]) if pd.notna(row.get("n_words")) else 0,
                scored=bool(row.get("scored", False)),
            )
        )
    return out


def save_checkpoint(results: list[FullResult], path: Path, categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(results, categories).to_csv(path, index=False)


def score_results(
    results: list[FullResult],
    *,
    categories: list[str],
    scales: dict[str, float],
    corpus_means: dict[str, float],
    liwc_cli: Path,
    mock_liwc: bool,
    batch_size: int,
    force: bool = False,
) -> tuple[int, str | None]:
    to_score = [
        r
        for r in results
        if r.error is None and r.text and (force or not r.scored)
    ]
    if not to_score:
        return 0, None

    if mock_liwc:
        for r in to_score:
            measured = pilot.mock_score_row(
                condition=r.steering_condition,
                profile_id=r.profile_id,
                content=r.content,
                repetition=r.repetition,
                targets=r.targets,
                scales=scales,
                corpus_means=corpus_means,
                categories=categories,
                seed=hash(r.plan_id) % (2**31),
            )
            raw, std = pilot.profile_mae(measured, r.targets, scales, categories)
            r.measured = measured
            r.raw_mae = raw
            r.standardized_mae = std
            r.scored = True
        return len(to_score), None

    try:
        for start in range(0, len(to_score), batch_size):
            batch = to_score[start : start + batch_size]
            measured_list = pilot.score_with_liwc_cli(
                [r.text for r in batch],
                liwc_cli=liwc_cli,
                categories=categories,
            )
            for r, measured in zip(batch, measured_list):
                raw, std = pilot.profile_mae(measured, r.targets, scales, categories)
                r.measured = measured
                r.raw_mae = raw
                r.standardized_mae = std
                r.scored = True
            print(f"  scored {min(start + batch_size, len(to_score))}/{len(to_score)}")
        return len(to_score), None
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def summarize_by_condition_model(
    results: list[FullResult],
    categories: list[str],
) -> pd.DataFrame:
    """Essay-level MAE + mean per-category ρ/slope by steering_condition × model_key.

    Secondary to author-level concat→re-LIWC summaries.
    """
    valid = [r for r in results if r.error is None and r.scored and r.measured]
    rows: list[dict] = []
    keys = sorted({(r.steering_condition, r.model_key) for r in results})
    for condition, model in keys:
        subset = [
            r for r in valid if r.steering_condition == condition and r.model_key == model
        ]
        if not subset:
            rows.append(
                {
                    "steering_condition": condition,
                    "model_key": model,
                    "n": 0,
                }
            )
            continue
        cat_metrics = {}
        for category in categories:
            t = np.array([r.targets[category] for r in subset], dtype=float)
            o = np.array([r.measured[category] for r in subset], dtype=float)
            cat_metrics[category] = pilot.calibration_metrics(t, o)
        mean_cal = pilot.mean_calibration(cat_metrics)
        mae = np.array([r.standardized_mae for r in subset], dtype=float)
        rows.append(
            {
                "steering_condition": condition,
                "model_key": model,
                "n": len(subset),
                "mean_standardized_mae": round(float(np.nanmean(mae)), 4),
                "mean_category_rho": round(mean_cal["mean_rho"], 4)
                if mean_cal["mean_rho"] == mean_cal["mean_rho"]
                else None,
                "mean_category_slope": round(mean_cal["mean_slope"], 4)
                if mean_cal["mean_slope"] == mean_cal["mean_slope"]
                else None,
                "self_identification_rate": round(
                    float(np.mean([r.self_identification for r in subset])), 4
                ),
                "mean_ttr": round(
                    float(np.nanmean([r.type_token_ratio for r in subset])), 4
                ),
            }
        )

    # Pooled over models per condition.
    for condition in STEERING_CONDITIONS:
        subset = [r for r in valid if r.steering_condition == condition]
        if not subset:
            continue
        cat_metrics = {}
        for category in categories:
            t = np.array([r.targets[category] for r in subset], dtype=float)
            o = np.array([r.measured[category] for r in subset], dtype=float)
            cat_metrics[category] = pilot.calibration_metrics(t, o)
        mean_cal = pilot.mean_calibration(cat_metrics)
        mae = np.array([r.standardized_mae for r in subset], dtype=float)
        rows.append(
            {
                "steering_condition": condition,
                "model_key": "ALL",
                "n": len(subset),
                "mean_standardized_mae": round(float(np.nanmean(mae)), 4),
                "mean_category_rho": round(mean_cal["mean_rho"], 4)
                if mean_cal["mean_rho"] == mean_cal["mean_rho"]
                else None,
                "mean_category_slope": round(mean_cal["mean_slope"], 4)
                if mean_cal["mean_slope"] == mean_cal["mean_slope"]
                else None,
                "self_identification_rate": round(
                    float(np.mean([r.self_identification for r in subset])), 4
                ),
                "mean_ttr": round(
                    float(np.nanmean([r.type_token_ratio for r in subset])), 4
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        cond_order = [c for c in STEERING_CONDITIONS if c in set(summary["steering_condition"])]
        summary["steering_condition"] = pd.Categorical(
            summary["steering_condition"], categories=cond_order, ordered=True
        )
        summary = summary.sort_values(
            ["steering_condition", "model_key"]
        ).reset_index(drop=True)
        summary["steering_condition"] = summary["steering_condition"].astype(str)
    return summary


def run_and_write_author_level(
    generations: pd.DataFrame,
    *,
    categories: list[str],
    scales: dict[str, float],
    corpus_means: dict[str, float],
    liwc_cli: Path,
    mock_liwc: bool,
    batch_size: int,
    author_level_path: Path,
    author_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concat essays per profile×condition×model → re-LIWC → write primary CSVs."""
    print(
        "Author-level LIWC (concat essays → re-score) ..."
        + (" [mock]" if mock_liwc else "")
    )

    def _score_fn(texts: list[str]) -> list[dict[str, float]]:
        return pilot.score_with_liwc_cli(
            texts, liwc_cli=liwc_cli, categories=categories
        )

    author_rows, author_summary = author_liwc.run_author_level_from_generations(
        generations,
        categories=categories,
        scales=scales,
        score_fn=_score_fn,
        calibration_metrics_fn=pilot.calibration_metrics,
        mean_calibration_fn=pilot.mean_calibration,
        mock=mock_liwc,
        mock_score_fn=pilot.mock_score_row,
        corpus_means=corpus_means,
        batch_size=batch_size,
    )
    author_level_path.parent.mkdir(parents=True, exist_ok=True)
    author_rows.to_csv(author_level_path, index=False)
    author_summary.to_csv(author_summary_path, index=False)
    print(f"Wrote author-level LIWC -> {author_level_path} ({len(author_rows)} rows)")
    print(f"Wrote author-level summary -> {author_summary_path}")
    if not author_summary.empty:
        print("\n=== Author-level summary (Validation 1 primary) ===")
        print(author_summary.to_string(index=False))
    return author_rows, author_summary


def generate_one(
    row: pd.Series,
    *,
    comp_by_id: pd.DataFrame,
    categories: list[str],
    model_cfg: dict,
    args: argparse.Namespace,
) -> FullResult:
    profile_id = str(row["profile_id"])
    condition = str(row["steering_condition"])
    content = str(row["content"])
    model_key = str(row["model_key"])
    comp = comp_by_id.loc[profile_id]
    targets = {c: float(comp[f"target_{c}"]) for c in categories}
    fewshot = str(comp.get("fewshot_component", "") or "")
    prompt = pilot.build_prompt(
        condition,
        persona=str(comp["persona_component"]),
        liwc_block=str(comp["liwc_component"]),
        fewshot_block=fewshot,
        content_key=content,
        target_words=args.target_words,
    )
    result = FullResult(
        plan_id=str(row["plan_id"]),
        profile_id=profile_id,
        cell=str(row.get("cell", comp.get("cell", ""))),
        content=content,
        steering_condition=condition,
        repetition=int(row["repetition"]),
        model_key=model_key,
        api_model=str(model_cfg["model"]),
        prompt=prompt,
        text="",
        backend="mock" if args.mock_generation else model_key,
        target_O=float(row["target_O"]),
        target_N=float(row["target_N"]),
        targets=targets,
    )
    try:
        if args.mock_generation:
            result.text = full.mock_generate(
                prompt,
                target_words=args.target_words,
                seed=args.seed + abs(hash(result.plan_id)) % 10_000,
            )
        else:
            text = ""
            last_err = ""
            attempts = max(1, int(args.retries) + 1)
            for attempt in range(attempts):
                try:
                    text = full.chat_complete(
                        prompt,
                        api_key=str(model_cfg["api_key"]),
                        base_url=str(model_cfg["base_url"]),
                        model=str(model_cfg["model"]),
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.request_timeout,
                        retries=1,
                        extra=model_cfg.get("extra") or {},
                    )
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    text = ""
                if isinstance(text, str) and text.strip():
                    break
            if not (isinstance(text, str) and text.strip()):
                result.error = last_err or "empty_generation"
                result.text = ""
            else:
                result.text = text
        result.n_words = len(result.text.split()) if result.text else 0
        result.type_token_ratio = pilot.type_token_ratio(result.text) if result.text else 0.0
        result.self_identification = (
            pilot.has_self_identification(result.text) if result.text else False
        )
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
    return result


def main() -> None:
    args = parse_args()
    load_dotenv(args.dotenv)

    plan = pd.read_csv(_resolve(args.plan))
    components = pd.read_csv(_resolve(args.components))
    human = pd.read_csv(_resolve(args.human))
    categories = pilot._categories_from_components(components)
    scales = {c: float(max(human[c].std(ddof=1), 1e-6)) for c in categories}
    corpus_means = {c: float(human[c].mean()) for c in categories}

    if "fewshot_component" not in components.columns:
        raise RuntimeError("Missing fewshot_component — re-run prepare_full.py")

    if args.conditions:
        plan = plan[plan["steering_condition"].isin(args.conditions)].copy()
    if args.models:
        plan = plan[plan["model_key"].isin(args.models)].copy()
    if args.smoke:
        first_profile = plan["profile_id"].iloc[0]
        first_content = plan["content"].iloc[0]
        plan = plan[
            (plan["profile_id"] == first_profile) & (plan["content"] == first_content)
        ].copy()
        print(f"SMOKE: profile={first_profile} content={first_content} rows={len(plan)}")

    output_dir = _resolve(args.output_dir)
    ensure_results_dir(output_dir)
    generations_path = _resolve(args.generations)
    summary_path = _resolve(args.summary)
    author_level_path = _resolve(args.author_level)
    author_summary_path = _resolve(args.author_summary)
    report_path = _resolve(args.report)

    model_keys = sorted(plan["model_key"].unique().tolist())
    model_resolved = {key: full.resolve_model(key) for key in model_keys}
    if (
        not args.mock_generation
        and not args.score_only
        and not args.score_author_level_only
    ):
        missing = [k for k, cfg in model_resolved.items() if not cfg["api_key"]]
        if missing:
            raise SystemExit(f"Missing API keys for models: {missing}")

    results: list[FullResult] = []
    done_ids: set[str] = set()
    if generations_path.exists():
        existing = pd.read_csv(generations_path)
        results = frame_to_results(existing, categories)
        for r in results:
            text_ok = isinstance(r.text, str) and bool(r.text.strip())
            if text_ok or r.error:
                done_ids.add(r.plan_id)
        print(f"Resuming: {len(done_ids)} existing rows in {generations_path.name}")

    by_id = {r.plan_id: r for r in results}

    if args.resummarize_author_only:
        if not author_level_path.exists():
            raise SystemExit(f"Missing author-level LIWC CSV: {author_level_path}")
        scored = pd.read_csv(author_level_path)
        cats = categories or author_liwc.categories_from_frame(scored)
        author_summary = author_liwc.summarize_author_level(
            scored,
            cats,
            calibration_metrics_fn=pilot.calibration_metrics,
            mean_calibration_fn=pilot.mean_calibration,
        )
        author_summary.to_csv(author_summary_path, index=False)
        write_json(
            report_path.with_name("full_author_level.report.json"),
            {
                "n_author_rows": int(len(scored)),
                "primary_metric": "author_level_concat_reliwc",
                "mean_rho_aggregation": "fisher_z",
                "author_level": str(author_level_path),
                "author_summary": str(author_summary_path),
                "summary": author_summary.to_dict(orient="records"),
            },
        )
        print(f"Wrote author summary -> {author_summary_path}")
        print(author_summary.to_string(index=False))
        return

    if args.score_author_level_only:
        if not generations_path.exists():
            raise SystemExit(f"Missing generations CSV: {generations_path}")
        gens = pd.read_csv(generations_path)
        if args.conditions:
            gens = gens[gens["steering_condition"].isin(args.conditions)].copy()
        if args.models:
            gens = gens[gens["model_key"].isin(args.models)].copy()
        author_rows, author_summary = run_and_write_author_level(
            gens,
            categories=categories,
            scales=scales,
            corpus_means=corpus_means,
            liwc_cli=args.liwc_cli,
            mock_liwc=args.mock_liwc,
            batch_size=args.liwc_batch_size,
            author_level_path=author_level_path,
            author_summary_path=author_summary_path,
        )
        write_json(
            report_path.with_name("full_author_level.report.json"),
            {
                "n_author_rows": int(len(author_rows)),
                "primary_metric": "author_level_concat_reliwc",
                "essay_level_summary": str(summary_path),
                "author_level": str(author_level_path),
                "author_summary": str(author_summary_path),
                "summary": author_summary.to_dict(orient="records"),
            },
        )
        return

    if args.score_only:
        n_scored, err = score_results(
            list(by_id.values()),
            categories=categories,
            scales=scales,
            corpus_means=corpus_means,
            liwc_cli=args.liwc_cli,
            mock_liwc=args.mock_liwc,
            batch_size=args.liwc_batch_size,
            force=True,
        )
        save_checkpoint(list(by_id.values()), generations_path, categories)
        summary = summarize_by_condition_model(list(by_id.values()), categories)
        summary.to_csv(summary_path, index=False)
        author_rows, author_summary = run_and_write_author_level(
            results_to_frame(list(by_id.values()), categories),
            categories=categories,
            scales=scales,
            corpus_means=corpus_means,
            liwc_cli=args.liwc_cli,
            mock_liwc=args.mock_liwc,
            batch_size=args.liwc_batch_size,
            author_level_path=author_level_path,
            author_summary_path=author_summary_path,
        )
        write_json(
            report_path.with_name("full_score_only.report.json"),
            {
                "n_scored": n_scored,
                "error": err,
                "essay_level_summary": summary.to_dict(orient="records"),
                "author_level_summary": author_summary.to_dict(orient="records"),
            },
        )
        if err:
            raise SystemExit(f"LIWC scoring failed: {err}")
        print("\nEssay-level (secondary):")
        print(summary.to_string(index=False))
        return

    pending = plan[~plan["plan_id"].isin(done_ids)].copy()
    print(
        f"Full generation: {len(pending)} pending / {len(plan)} planned | "
        f"models={model_keys} | workers={args.workers}"
    )
    if pending.empty:
        print("Nothing to generate.")
    else:
        comp_by_id = components.set_index("profile_id")
        completed = 0
        lock = threading.Lock()

        def _task(row: pd.Series) -> FullResult:
            return generate_one(
                row,
                comp_by_id=comp_by_id,
                categories=categories,
                model_cfg=model_resolved[str(row["model_key"])],
                args=args,
            )

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_task, row): str(row["plan_id"])
                for _, row in pending.iterrows()
            }
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    by_id[result.plan_id] = result
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        with _print_lock:
                            print(f"  generated {completed}/{len(pending)}")
                    if completed % args.checkpoint_every == 0:
                        save_checkpoint(list(by_id.values()), generations_path, categories)

        save_checkpoint(list(by_id.values()), generations_path, categories)

    # Score newly generated (and any unscored) rows.
    print("Scoring LIWC (essay-level) ...")
    n_scored, err = score_results(
        list(by_id.values()),
        categories=categories,
        scales=scales,
        corpus_means=corpus_means,
        liwc_cli=args.liwc_cli,
        mock_liwc=args.mock_liwc,
        batch_size=args.liwc_batch_size,
        force=False,
    )
    if err:
        print(f"WARNING: LIWC scoring error: {err}")
    save_checkpoint(list(by_id.values()), generations_path, categories)

    summary = summarize_by_condition_model(list(by_id.values()), categories)
    summary.to_csv(summary_path, index=False)
    author_rows, author_summary = run_and_write_author_level(
        results_to_frame(list(by_id.values()), categories),
        categories=categories,
        scales=scales,
        corpus_means=corpus_means,
        liwc_cli=args.liwc_cli,
        mock_liwc=args.mock_liwc,
        batch_size=args.liwc_batch_size,
        author_level_path=author_level_path,
        author_summary_path=author_summary_path,
    )
    write_json(
        report_path,
        {
            "n_planned": int(len(plan)),
            "n_rows": int(len(by_id)),
            "n_scored_this_run": n_scored,
            "score_error": err,
            "models": model_keys,
            "conditions": sorted(plan["steering_condition"].unique().tolist()),
            "categories": categories,
            "primary_liwc_metric": "author_level_concat_reliwc",
            "essay_level_summary": summary.to_dict(orient="records"),
            "author_level_summary": author_summary.to_dict(orient="records"),
            "outputs": {
                "generations": str(generations_path),
                "essay_level_summary": str(summary_path),
                "author_level": str(author_level_path),
                "author_summary": str(author_summary_path),
            },
        },
    )
    print(f"\nWrote generations -> {generations_path}")
    print(f"Wrote essay-level summary (secondary) -> {summary_path}")
    print(f"Wrote author-level summary (primary) -> {author_summary_path}")
    print(f"Report -> {report_path}")
    if not summary.empty:
        print("\nEssay-level (secondary):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
