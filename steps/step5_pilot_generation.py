#!/usr/bin/env python3
"""Step 5: Pilot generation under persona_only / liwc_only / persona_liwc / lex_fewshot.

Matched essays: same profile, topic, model, and repetition; only the steering
block differs. Validates with MAE, target–output correlation, and calibration
slope (essay-level secondary; author-level = concat essays → re-LIWC primary).

``lex_fewshot`` = author-specific LIWC targets + nearest-neighbor human excerpts.

Example:
  python optimized/steps/step5_pilot_generation.py --mock-generation --mock-liwc
  python optimized/steps/step5_pilot_generation.py --conditions lex_fewshot --merge-existing
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    CONTENT_CONDITIONS,
    DEFAULT_LIWC_CLI,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TARGET_WORDS,
    DEFAULT_TEMPERATURE,
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

import stage2_step7_part2_steering_comparison as p2  # noqa: E402

import liwc_author_level as author_liwc  # noqa: E402

DEFAULT_PLAN = PILOT_RESULTS_DIR / "step3_generation_plan.csv"
DEFAULT_COMPONENTS = PILOT_RESULTS_DIR / "step4_prompt_components.csv"
DEFAULT_HUMAN = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_OUTPUT_DIR = PILOT_RESULTS_DIR / "step5_pilot"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step5_pilot" / "pilot.report.json"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SELF_ID_PATTERNS = re.compile(
    r"\b(as a |i am a |i'm a |being a )",
    re.IGNORECASE,
)


@dataclass
class PilotResult:
    plan_id: str
    profile_id: str
    content: str
    steering_condition: str
    repetition: int
    model: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", type=str, default=DEEPSEEK_MODEL)
    parser.add_argument("--base-url", type=str, default=DEEPSEEK_BASE_URL)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--liwc-cli", type=Path, default=Path(DEFAULT_LIWC_CLI))
    parser.add_argument("--mock-generation", action="store_true")
    parser.add_argument("--mock-liwc", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on plan rows (for smoke tests).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=list(STEERING_CONDITIONS),
        help="Subset of steering conditions to generate (default: all in plan).",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep prior generations whose plan_id is not regenerated.",
    )
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def _categories_from_components(components: pd.DataFrame) -> list[str]:
    return [
        c.replace("target_", "", 1)
        for c in components.columns
        if c.startswith("target_") and c not in ("target_O", "target_N")
    ]


def build_prompt(
    condition: str,
    *,
    persona: str,
    liwc_block: str,
    fewshot_block: str,
    content_key: str,
    target_words: int,
) -> str:
    content_desc = CONTENT_CONDITIONS[content_key]
    base_rules = (
        f"Write approximately {target_words} words. Keep it authentic and "
        "conversational. Do not mention personality tests, LIWC, scores, "
        "percentages, or these instructions. Output only the comment text."
    )

    if condition == "persona_only":
        return (
            f"Pretend you are {persona}.\n\n"
            f"Topic: write about {content_desc}.\n\n"
            f"{base_rules}"
        )
    if condition == "liwc_only":
        return (
            "You are writing a single, natural Reddit comment.\n\n"
            f"Topic: write about {content_desc}.\n\n"
            "Write in a way whose word usage matches this linguistic profile. "
            "Each line gives a LIWC-22 category, a plain-language explanation, "
            "and an approximate target rate:\n\n"
            f"{liwc_block}\n\n"
            f"{base_rules}"
        )
    if condition == "persona_liwc":
        return (
            f"Pretend you are {persona}.\n\n"
            f"Topic: write about {content_desc}.\n\n"
            "Also match this linguistic profile. Each line gives a LIWC-22 "
            "category, a plain-language explanation, and an approximate "
            "target rate:\n\n"
            f"{liwc_block}\n\n"
            f"{base_rules}"
        )
    if condition == "lex_fewshot":
        fewshot_section = f"\n\n{fewshot_block}\n" if fewshot_block else ""
        return (
            "You are writing a single, natural Reddit comment.\n\n"
            f"Topic: write about {content_desc}.\n\n"
            "Write in a way whose word usage matches this linguistic profile. "
            "Each line gives a LIWC-22 category, a plain-language explanation, "
            "and an approximate target rate:\n\n"
            f"{liwc_block}"
            f"{fewshot_section}\n"
            f"{base_rules}"
        )
    raise ValueError(f"Unknown condition: {condition}")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def type_token_ratio(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def has_self_identification(text: str) -> bool:
    return bool(SELF_ID_PATTERNS.search(text))


def score_with_liwc_cli(
    texts: list[str],
    *,
    liwc_cli: Path,
    categories: list[str],
) -> list[dict[str, float]]:
    """LIWC-22-cli scoring that drops phantom low-WC rows."""
    if not liwc_cli.exists():
        raise FileNotFoundError(
            f"LIWC-22-cli not found at {liwc_cli}. Install LIWC-22 or use --mock-liwc."
        )
    cleaned = [t.replace("\r", " ").replace("\n", " ").strip() for t in texts]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_csv = tmp_path / "texts.csv"
        output_csv = tmp_path / "liwc.csv"
        pd.DataFrame({"text": cleaned}).to_csv(input_csv, index=False)
        cmd = [
            str(liwc_cli),
            "-m", "wc",
            "-i", str(input_csv),
            "-o", str(output_csv),
            "-sh", "no",
            "-skip", "1",
            "-ci", "1",
            "-d", "LIWC22",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"LIWC-22-cli failed:\n{result.stderr or result.stdout}"
            )
        liwc = pd.read_csv(output_csv)

    missing = [c for c in categories if c not in liwc.columns]
    if missing:
        raise ValueError(f"LIWC output missing categories: {missing}")
    if "WC" not in liwc.columns:
        raise ValueError("LIWC output missing WC column needed for phantom-row filter.")

    measured: list[dict[str, float]] = []
    row_idx = 0
    n_rows = len(liwc)
    for _ in cleaned:
        if row_idx >= n_rows:
            raise RuntimeError("LIWC returned fewer usable rows than input texts.")
        while (
            row_idx + 1 < n_rows
            and float(liwc.iloc[row_idx]["WC"]) <= 2
            and float(liwc.iloc[row_idx + 1]["WC"]) > 2
        ):
            row_idx += 1
        row = liwc.iloc[row_idx]
        measured.append({category: float(row[category]) for category in categories})
        row_idx += 1
    return measured


def mock_score_row(
    *,
    condition: str,
    profile_id: str,
    content: str,
    repetition: int,
    targets: dict[str, float],
    scales: dict[str, float],
    corpus_means: dict[str, float],
    categories: list[str],
    seed: int,
) -> dict[str, float]:
    digest = hashlib.sha256(
        f"{seed}:{condition}:{profile_id}:{content}:{repetition}".encode()
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    # LIWC-aware conditions track targets; persona hugs corpus mean (MAE paradox).
    if condition == "persona_only":
        bias = 0.35
        center = corpus_means
    elif condition == "liwc_only":
        bias = 0.55
        center = targets
    elif condition == "lex_fewshot":
        # Slightly tighter tracking than plain LIWC (exemplars help in mock).
        bias = 0.40
        center = targets
    else:  # persona_liwc
        bias = 0.45
        center = {
            c: 0.5 * targets[c] + 0.5 * corpus_means[c] for c in categories
        }

    measured: dict[str, float] = {}
    for category in categories:
        noise = rng.normal(0.0, bias) * scales[category]
        measured[category] = max(0.0, center[category] + noise)
    return measured


def profile_mae(
    measured: dict[str, float],
    targets: dict[str, float],
    scales: dict[str, float],
    categories: list[str],
) -> tuple[float, float]:
    raw = []
    std = []
    for category in categories:
        diff = measured[category] - targets[category]
        raw.append(abs(diff))
        std.append(abs(diff) / scales[category])
    return float(np.mean(raw)), float(np.mean(std))


def calibration_metrics(
    targets: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    mask = np.isfinite(targets) & np.isfinite(observed)
    if mask.sum() < 3:
        return {"rho": float("nan"), "slope": float("nan"), "intercept": float("nan")}
    x = targets[mask]
    y = observed[mask]
    rho = float(stats.spearmanr(x, y).statistic)
    slope, intercept, *_ = stats.linregress(x, y)
    return {"rho": rho, "slope": float(slope), "intercept": float(intercept)}


def mean_calibration(per_category: dict) -> dict[str, float]:
    """Average category-level ρ via Fisher z (RA plan); slopes via arithmetic mean."""
    rhos = [v["rho"] for v in per_category.values() if v["rho"] == v["rho"]]
    slopes = [v["slope"] for v in per_category.values() if v["slope"] == v["slope"]]
    if rhos:
        clipped = np.clip(np.asarray(rhos, dtype=float), -0.999999, 0.999999)
        mean_rho = float(np.tanh(np.mean(np.arctanh(clipped))))
    else:
        mean_rho = float("nan")
    return {
        "mean_rho": mean_rho,
        "mean_slope": float(np.mean(slopes)) if slopes else float("nan"),
        "mean_rho_arithmetic": float(np.mean(rhos)) if rhos else float("nan"),
    }


def aggregate_validation(
    results: list[PilotResult],
    categories: list[str],
) -> dict:
    """Essay-level metrics only. Author-level uses concat→re-LIWC separately."""
    valid = [r for r in results if r.error is None and r.measured]
    conditions = sorted({r.steering_condition for r in results})
    by_condition: dict = {}
    for condition in conditions:
        subset = [r for r in valid if r.steering_condition == condition]
        if not subset:
            by_condition[condition] = {"n": 0}
            continue
        mae = np.array([r.standardized_mae for r in subset], dtype=float)
        cat_metrics: dict = {}
        for category in categories:
            t = np.array([r.targets[category] for r in subset], dtype=float)
            o = np.array([r.measured[category] for r in subset], dtype=float)
            cat_metrics[category] = calibration_metrics(t, o)
        # Pooled after within-category z-scoring (avoids between-category scale artifact).
        all_tz = []
        all_oz = []
        for category in categories:
            t = np.array([r.targets[category] for r in subset], dtype=float)
            o = np.array([r.measured[category] for r in subset], dtype=float)
            t_sd = float(np.std(t, ddof=1)) if len(t) > 1 else 0.0
            o_sd = float(np.std(o, ddof=1)) if len(o) > 1 else 0.0
            if t_sd < 1e-8 or o_sd < 1e-8:
                continue
            all_tz.extend(((t - t.mean()) / t_sd).tolist())
            all_oz.extend(((o - o.mean()) / o_sd).tolist())
        pooled = calibration_metrics(np.array(all_tz), np.array(all_oz))
        mean_cal = mean_calibration(cat_metrics)

        by_condition[condition] = {
            "n": len(subset),
            "mean_standardized_mae": float(np.nanmean(mae)),
            "self_identification_rate": float(
                np.mean([r.self_identification for r in subset])
            ),
            "mean_ttr": float(np.nanmean([r.type_token_ratio for r in subset])),
            "mean_per_category": mean_cal,
            "pooled_z_target_output": pooled,
            "per_category": cat_metrics,
        }

    return {
        "essay_level": by_condition,
        "n_valid": len(valid),
        "n_errors": sum(1 for r in results if r.error is not None),
    }


def write_summary(
    results: list[PilotResult],
    categories: list[str],
    *,
    output_dir: Path,
    report_path: Path,
    mock_generation: bool,
    mock_liwc: bool,
    scales: dict[str, float],
    corpus_means: dict[str, float],
    liwc_cli: Path,
) -> pd.DataFrame:
    frame = results_to_frame(results, categories)
    generations_path = output_dir / "pilot_generations.csv"
    frame.to_csv(generations_path, index=False)

    validation = aggregate_validation(results, categories)
    summary_path = output_dir / "pilot_condition_summary.csv"
    summary_rows = []
    for condition, payload in validation["essay_level"].items():
        if payload.get("n", 0) == 0:
            continue
        mean_cal = payload["mean_per_category"]
        pooled = payload["pooled_z_target_output"]
        summary_rows.append(
            {
                "steering_condition": condition,
                "n": payload["n"],
                "mean_standardized_mae": round(payload["mean_standardized_mae"], 4),
                "mean_category_rho": round(mean_cal["mean_rho"], 4)
                if mean_cal["mean_rho"] == mean_cal["mean_rho"]
                else None,
                "mean_category_slope": round(mean_cal["mean_slope"], 4)
                if mean_cal["mean_slope"] == mean_cal["mean_slope"]
                else None,
                "pooled_z_rho": round(pooled["rho"], 4)
                if pooled["rho"] == pooled["rho"]
                else None,
                "self_identification_rate": round(
                    payload["self_identification_rate"], 4
                ),
                "mean_ttr": round(payload["mean_ttr"], 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        order = [c for c in STEERING_CONDITIONS if c in set(summary["steering_condition"])]
        extra = [c for c in summary["steering_condition"] if c not in order]
        summary["steering_condition"] = pd.Categorical(
            summary["steering_condition"], categories=order + extra, ordered=True
        )
        summary = summary.sort_values("steering_condition").reset_index(drop=True)
        summary["steering_condition"] = summary["steering_condition"].astype(str)
    summary.to_csv(summary_path, index=False)

    author_path = output_dir / "pilot_author_level_liwc.csv"
    author_summary_path = output_dir / "pilot_author_condition_summary.csv"
    author_summary_records: list[dict] = []
    try:

        def _score_fn(texts: list[str]) -> list[dict[str, float]]:
            return score_with_liwc_cli(
                texts, liwc_cli=liwc_cli, categories=categories
            )

        author_rows, author_summary = author_liwc.run_author_level_from_generations(
            frame,
            categories=categories,
            scales=scales,
            score_fn=_score_fn,
            calibration_metrics_fn=calibration_metrics,
            mean_calibration_fn=mean_calibration,
            mock=mock_liwc,
            mock_score_fn=mock_score_row,
            corpus_means=corpus_means,
        )
        author_rows.to_csv(author_path, index=False)
        author_summary.to_csv(author_summary_path, index=False)
        author_summary_records = author_summary.to_dict(orient="records")
        validation["author_level"] = {
            row["steering_condition"]: {
                "n_authors": row["n_authors"],
                "mean_standardized_mae": row["mean_standardized_mae"],
                "mean_per_category": {
                    "mean_rho": row["mean_category_rho"],
                    "mean_slope": row["mean_category_slope"],
                },
            }
            for row in author_summary_records
        }
        validation["author_level_method"] = "concat_reliwc"
    except Exception as exc:  # noqa: BLE001
        validation["author_level_error"] = str(exc)
        print(f"WARNING: author-level LIWC failed: {exc}")

    write_json(
        report_path,
        {
            "generations": str(generations_path),
            "summary": str(summary_path),
            "author_level": str(author_path),
            "author_summary": str(author_summary_path),
            "categories": categories,
            "mock_generation": mock_generation,
            "mock_liwc": mock_liwc,
            "primary_liwc_metric": "author_level_concat_reliwc",
            "validation": validation,
            "author_level_summary": author_summary_records,
        },
    )
    print(f"\nWrote generations -> {generations_path}")
    print(f"Wrote essay-level summary (secondary) -> {summary_path}")
    print(f"Wrote author-level summary (primary) -> {author_summary_path}")
    print(f"Report -> {report_path}")
    if not summary.empty:
        print("\nEssay-level condition summary (secondary):")
        print(summary.to_string(index=False))
    if author_summary_records:
        print("\nAuthor-level condition summary (primary):")
        print(pd.DataFrame(author_summary_records).to_string(index=False))
    return summary


def load_dotenv_key() -> None:
    env_path = OPTIMIZED_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    args = parse_args()
    load_dotenv_key()

    plan = pd.read_csv(_resolve_out(args.plan))
    components = pd.read_csv(_resolve_out(args.components))
    human = pd.read_csv(_resolve_out(args.human))
    categories = _categories_from_components(components)
    scales = {c: float(max(human[c].std(ddof=1), 1e-6)) for c in categories}
    corpus_means = {c: float(human[c].mean()) for c in categories}

    if "fewshot_component" not in components.columns:
        raise RuntimeError(
            "step4 components missing fewshot_component — re-run "
            "optimized/steps/step4_prompt_components.py"
        )

    if args.conditions:
        plan = plan[plan["steering_condition"].isin(args.conditions)].copy()
    if args.limit is not None:
        plan = plan.head(args.limit).copy()

    comp_by_id = components.set_index("profile_id")
    results: list[PilotResult] = []

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.mock_generation and not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set (or pass --mock-generation).")

    print(
        f"Pilot generation: {len(plan)} rows | categories={categories} | "
        f"conditions={sorted(plan['steering_condition'].unique().tolist())} | "
        f"mock_gen={args.mock_generation} mock_liwc={args.mock_liwc}"
    )

    for i, row in plan.iterrows():
        profile_id = str(row["profile_id"])
        condition = str(row["steering_condition"])
        content = str(row["content"])
        comp = comp_by_id.loc[profile_id]
        targets = {c: float(comp[f"target_{c}"]) for c in categories}
        fewshot = str(comp.get("fewshot_component", "") or "")
        prompt = build_prompt(
            condition,
            persona=str(comp["persona_component"]),
            liwc_block=str(comp["liwc_component"]),
            fewshot_block=fewshot,
            content_key=content,
            target_words=args.target_words,
        )

        result = PilotResult(
            plan_id=str(row["plan_id"]),
            profile_id=profile_id,
            content=content,
            steering_condition=condition,
            repetition=int(row["repetition"]),
            model=str(row.get("model", args.model)),
            prompt=prompt,
            text="",
            backend="mock" if args.mock_generation else "deepseek",
            target_O=float(row["target_O"]),
            target_N=float(row["target_N"]),
            targets=targets,
        )

        try:
            if args.mock_generation:
                result.text = p2.mock_generate(
                    prompt, target_words=args.target_words, seed=args.seed + int(i)
                )
            else:
                result.text = p2.deepseek_generate(
                    prompt,
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.request_timeout,
                    retries=args.retries,
                )
            result.n_words = len(result.text.split())
            result.type_token_ratio = type_token_ratio(result.text)
            result.self_identification = has_self_identification(result.text)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
        results.append(result)
        if (len(results) % 10) == 0:
            print(f"  generated {len(results)}/{len(plan)}")

    # Score LIWC.
    ok_idx = [i for i, r in enumerate(results) if r.error is None]
    if args.mock_liwc:
        for i in ok_idx:
            r = results[i]
            measured = mock_score_row(
                condition=r.steering_condition,
                profile_id=r.profile_id,
                content=r.content,
                repetition=r.repetition,
                targets=r.targets,
                scales=scales,
                corpus_means=corpus_means,
                categories=categories,
                seed=args.seed,
            )
            raw, std = profile_mae(measured, r.targets, scales, categories)
            r.measured = measured
            r.raw_mae = raw
            r.standardized_mae = std
    else:
        texts = [results[i].text for i in ok_idx]
        if texts:
            scored = score_with_liwc_cli(
                texts, liwc_cli=args.liwc_cli, categories=categories
            )
            for i, measured in zip(ok_idx, scored):
                r = results[i]
                raw, std = profile_mae(measured, r.targets, scales, categories)
                r.measured = measured
                r.raw_mae = raw
                r.standardized_mae = std

    output_dir = _resolve_out(args.output_dir)
    ensure_results_dir(output_dir)
    generations_path = output_dir / "pilot_generations.csv"
    new_frame = results_to_frame(results, categories)

    if args.merge_existing and generations_path.exists():
        old = pd.read_csv(generations_path)
        regenerated = set(new_frame["plan_id"].astype(str))
        kept = old[~old["plan_id"].astype(str).isin(regenerated)].copy()
        merged = pd.concat([kept, new_frame], ignore_index=True)
        print(
            f"Merged: kept {len(kept)} prior rows + {len(new_frame)} new "
            f"= {len(merged)} total"
        )
        all_results = frame_to_results(merged, categories)
    else:
        all_results = results

    write_summary(
        all_results,
        categories,
        output_dir=output_dir,
        report_path=_resolve_out(args.report),
        mock_generation=args.mock_generation,
        mock_liwc=args.mock_liwc,
        scales=scales,
        corpus_means=corpus_means,
        liwc_cli=args.liwc_cli,
    )


if __name__ == "__main__":
    main()
