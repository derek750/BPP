#!/usr/bin/env python3
"""Step 4: Convert synthetic profiles into persona / LIWC / few-shot blocks.

Persona and LIWC components stay strictly separate:
  - persona: continuous O/N described relative to the human sample
  - LIWC: category explanations + approximate target rates
  - fewshot: excerpts from nearest human authors by standardized LIWC distance
    (continuous stand-in for same-cell exemplars)
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForBootstrap

_OPTIMIZED_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DEFAULT_AUTHORS,
    DEFAULT_EXEMPLAR_CANDIDATES,
    DEFAULT_EXEMPLAR_WORDS,
    DEFAULT_N_EXEMPLARS,
    LIWC_PLAIN_LANGUAGE,
    OPTIMIZED_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_DIR,
    FULL_RESULTS_DIR,
    SCRIPTS_DIR,
    ensure_results_dir,
    percentile_rank,
    resolve_path,
    strength_phrase,
    write_json,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import stage2_step7_part3_lexical_signatures as p3  # noqa: E402

DEFAULT_HUMAN = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_SYNTH = PILOT_RESULTS_DIR / "step2_synthetic_profiles.csv"
DEFAULT_OUTPUT = PILOT_RESULTS_DIR / "step4_prompt_components.csv"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step4_prompt_components.report.json"
DEFAULT_EXEMPLAR_CACHE = PILOT_RESULTS_DIR / "step4_fewshot_exemplars.json"

MOCK_EXEMPLAR = (
    "honestly i think most weekends end up feeling the same, i just wander "
    "around the house and worry about the week ahead even though nothing bad "
    "ever actually happens, which is kind of funny when you think about it"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTH)
    parser.add_argument("--authors", type=Path, default=Path(DEFAULT_AUTHORS))
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--exemplar-cache", type=Path, default=DEFAULT_EXEMPLAR_CACHE)
    parser.add_argument("--n-exemplars", type=int, default=DEFAULT_N_EXEMPLARS)
    parser.add_argument("--exemplar-words", type=int, default=DEFAULT_EXEMPLAR_WORDS)
    parser.add_argument(
        "--exemplar-candidates",
        type=int,
        default=DEFAULT_EXEMPLAR_CANDIDATES,
    )
    parser.add_argument("--rebuild-exemplars", action="store_true")
    parser.add_argument(
        "--mock-exemplars",
        action="store_true",
        help="Use canned excerpts (no author-text load).",
    )
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def _liwc_cols(synth: pd.DataFrame) -> list[str]:
    skip = {"profile_id", "O", "N", "cell", "author_id", "n_essays", "total_word_count", "liwc_wc"}
    return [c for c in synth.columns if c not in skip]


def build_persona_component(o: float, n: float, human: pd.DataFrame) -> str:
    o_pct = percentile_rank(human["O"], o)
    n_pct = percentile_rank(human["N"], n)
    o_phrase = strength_phrase(o_pct)
    n_phrase = strength_phrase(n_pct)
    return (
        "a Reddit commenter who tends to seek novelty, ideas, and new experiences "
        f"{o_phrase}, and who tends to feel anxiety, worry, or emotional ups-and-downs "
        f"{n_phrase}"
    )


def build_liwc_component(
    row: pd.Series,
    categories: list[str],
    human: pd.DataFrame,
) -> str:
    lines: list[str] = []
    for category in categories:
        target = float(row[category])
        corpus_mean = float(human[category].mean())
        explanation = LIWC_PLAIN_LANGUAGE.get(
            category, f"words in the {category} category"
        )
        if target >= corpus_mean * 1.15:
            relative = "higher than typical"
        elif target <= corpus_mean * 0.85:
            relative = "lower than typical"
        else:
            relative = "about typical"
        lines.append(
            f"- {category}: {explanation}; aim for {relative} "
            f"(target ≈ {target:.2f}% of words)"
        )
    return "\n".join(lines)


def format_fewshot_block(excerpts: list[str]) -> str:
    lines = [
        "Here are excerpts from writers with a closely matching linguistic "
        "profile. Match their voice, tone, and word choice — not their topics:"
    ]
    for i, excerpt in enumerate(excerpts, start=1):
        lines.append(f'Example {i}: "{excerpt}"')
    return "\n\n".join(lines)


def select_nearest_exemplars(
    synthetic: pd.DataFrame,
    human: pd.DataFrame,
    categories: list[str],
    authors_path: Path,
    *,
    n_exemplars: int,
    excerpt_words: int,
    n_candidates: int,
    mock: bool,
) -> dict[str, dict]:
    """For each synthetic profile, pick nearest human authors by z-scored LIWC."""
    if mock:
        return {
            str(pid): {
                "author_ids": [],
                "distances": [],
                "excerpts": [MOCK_EXEMPLAR] * n_exemplars,
            }
            for pid in synthetic["profile_id"]
        }

    human_x = human[categories].to_numpy(dtype=float)
    mu = human_x.mean(axis=0)
    sd = human_x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    human_z = (human_x - mu) / sd
    human_ids = human["author_id"].to_numpy(dtype=int)

    # Collect candidate author ids across all synthetic profiles first.
    candidates_by_profile: dict[str, list[tuple[int, float]]] = {}
    all_ids: set[int] = set()
    for _, row in synthetic.iterrows():
        profile_id = str(row["profile_id"])
        target = np.array([float(row[c]) for c in categories], dtype=float)
        target_z = (target - mu) / sd
        dists = np.linalg.norm(human_z - target_z, axis=1)
        order = np.argsort(dists)[:n_candidates]
        pairs = [(int(human_ids[i]), float(dists[i])) for i in order]
        candidates_by_profile[profile_id] = pairs
        all_ids.update(pid for pid, _ in pairs)

    print(f"  loading {len(all_ids)} nearest-neighbor author texts ...")
    texts = p3.load_texts_by_essay_id(authors_path, all_ids)

    exemplars: dict[str, dict] = {}
    for profile_id, pairs in candidates_by_profile.items():
        usable: list[tuple[int, float, str]] = []
        for author_id, dist in pairs:
            raw = texts.get(author_id)
            if not raw:
                continue
            cleaned = p3.clean_source_text(str(raw))
            tokens = cleaned.split()
            if len(tokens) < max(40, excerpt_words // 2):
                continue
            excerpt = p3.excerpt_from_text(cleaned, excerpt_words)
            usable.append((author_id, dist, excerpt))
            if len(usable) >= n_exemplars:
                break
        if len(usable) < n_exemplars:
            # Fall back to shorter texts if needed.
            for author_id, dist in pairs:
                if any(author_id == u[0] for u in usable):
                    continue
                raw = texts.get(author_id)
                if not raw:
                    continue
                cleaned = p3.clean_source_text(str(raw))
                if len(cleaned.split()) < 20:
                    continue
                usable.append(
                    (author_id, dist, p3.excerpt_from_text(cleaned, excerpt_words))
                )
                if len(usable) >= n_exemplars:
                    break
        if not usable:
            usable = [(-1, float("nan"), MOCK_EXEMPLAR) for _ in range(n_exemplars)]
        exemplars[profile_id] = {
            "author_ids": [u[0] for u in usable[:n_exemplars]],
            "distances": [round(u[1], 4) for u in usable[:n_exemplars]],
            "excerpts": [u[2] for u in usable[:n_exemplars]],
        }
    return exemplars


def build_prompt_components(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    exemplars: dict[str, dict],
) -> pd.DataFrame:
    categories = _liwc_cols(synthetic)
    rows: list[dict] = []
    for _, row in synthetic.iterrows():
        profile_id = str(row["profile_id"])
        persona = build_persona_component(float(row["O"]), float(row["N"]), human)
        liwc = build_liwc_component(row, categories, human)
        ex = exemplars.get(profile_id, {"excerpts": [MOCK_EXEMPLAR]})
        fewshot = format_fewshot_block(list(ex["excerpts"]))

        banned_in_persona = ("LIWC", "%", "emo_", "Affect", "adverb", "swear")
        if any(tok in persona for tok in banned_in_persona):
            raise RuntimeError(f"Persona leaked LIWC content: {persona}")
        banned_in_liwc = ("Openness", "Neuroticism", "BFI", "personality")
        if any(tok.lower() in liwc.lower() for tok in banned_in_liwc):
            raise RuntimeError(f"LIWC block leaked personality labels: {liwc}")

        out: dict = {
            "profile_id": profile_id,
            "target_O": float(row["O"]),
            "target_N": float(row["N"]),
            "persona_component": persona,
            "liwc_component": liwc,
            "fewshot_component": fewshot,
            "fewshot_author_ids": json.dumps(ex.get("author_ids", [])),
            "fewshot_distances": json.dumps(ex.get("distances", [])),
            "o_percentile": percentile_rank(human["O"], float(row["O"])),
            "n_percentile": percentile_rank(human["N"], float(row["N"])),
        }
        for category in categories:
            out[f"target_{category}"] = float(row[category])
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    human = pd.read_csv(_resolve_out(args.human))
    synthetic = pd.read_csv(_resolve_out(args.synthetic))
    categories = _liwc_cols(synthetic)

    cache_path = _resolve_out(args.exemplar_cache)
    if cache_path.exists() and not args.rebuild_exemplars and not args.mock_exemplars:
        exemplars = json.loads(cache_path.read_text())
        if all(str(pid) in exemplars for pid in synthetic["profile_id"]):
            print(f"Using cached few-shot exemplars -> {cache_path}")
        else:
            exemplars = None
    else:
        exemplars = None

    if exemplars is None:
        print("Selecting nearest-neighbor few-shot exemplars ...")
        exemplars = select_nearest_exemplars(
            synthetic,
            human,
            categories,
            resolve_path(args.authors),
            n_exemplars=args.n_exemplars,
            excerpt_words=args.exemplar_words,
            n_candidates=args.exemplar_candidates,
            mock=args.mock_exemplars,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(exemplars, indent=2) + "\n")
        print(f"Cached exemplars -> {cache_path}")

    components = build_prompt_components(human, synthetic, exemplars)
    output = _resolve_out(args.output)
    report = _resolve_out(args.report)
    ensure_results_dir(output.parent)
    components.to_csv(output, index=False)
    write_json(
        report,
        {
            "output": str(output),
            "n_profiles": int(len(components)),
            "categories": categories,
            "n_exemplars": args.n_exemplars,
            "example_persona": components.iloc[0]["persona_component"],
            "example_liwc_preview": components.iloc[0]["liwc_component"][:400],
            "example_fewshot_preview": components.iloc[0]["fewshot_component"][:400],
        },
    )
    print(f"Wrote prompt components for {len(components)} profiles -> {output}")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
