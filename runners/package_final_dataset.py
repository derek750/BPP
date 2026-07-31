#!/usr/bin/env python3
"""Package the continuous BFI–LIWC pipeline as a FAIR reproducibility release.

Builds a self-contained archive under ``optimized/results/final_dataset/``
comprising:

  - human and synthetic author profiles (continuous O/N + LIWC targets)
  - paired generation–measurement records (full multi-model grid)
  - author-level LIWC alignment, continuous embedding recovery, essay-only BFI-44
  - reproduction code, schema documentation, and worked examples

FAIR mapping (Wilkinson et al., 2016):

  - Findability: ``datapackage.json`` + ``CITATION.cff``
  - Accessibility: CC BY 4.0 (data) + MIT (code)
  - Interoperability: Parquet, JSONL, CSV, JSON
  - Reusability: ``SCHEMA.md``, README, examples

Does not upload to Zenodo; produces a local folder/zip ready for deposit.
The LIWC-22 dictionary is commercial and is never redistributed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_OPTIMIZED_DIR = Path(__file__).resolve().parents[1]
if str(_OPTIMIZED_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZED_DIR))

from common import (  # noqa: E402
    DEFAULT_CONTROL_CATEGORIES,
    FULL_RESULTS_DIR,
    OPTIMIZED_DIR,
    PILOT_RESULTS_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    STEERING_CONDITIONS,
    write_json,
)

DEFAULT_OUTPUT = RESULTS_DIR / "final_dataset"
VERSION = "1.1.0"

CONTENT_TO_CP = {
    "memorable_journey": "CP1",
    "difficult_decision": "CP2",
    "daily_life_change": "CP3",
    "meaningful_relationship": "CP4",
    "childhood_memory": "CP5",
    "turning_point": "CP6",
}

CODE_FILES = [
    "common.py",
    "liwc_author_level.py",
    "README.md",
    "runners/prepare_full.py",
    "runners/run_pilot.py",
    "runners/run_full.py",
    "runners/build_paper_table.py",
    "runners/package_final_dataset.py",
    "steps/step1_author_profiles.py",
    "steps/step2_gaussian_copula.py",
    "steps/step3_generation_plan.py",
    "steps/step4_prompt_components.py",
    "steps/step5_pilot_generation.py",
    "steps/step6_full_generation.py",
    "steps/step6_embedding_probe.py",
    "steps/step6_embedding_viz.py",
    "steps/step6_embedding_discriminability_viz.py",
    "steps/step6_embedding_recovery_viz.py",
    "steps/step8_bfi_validation.py",
    "steps/step9_profile_bootstrap.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--human-profiles",
        type=Path,
        default=PILOT_RESULTS_DIR / "step1_author_profiles.csv",
    )
    parser.add_argument(
        "--synthetic-profiles",
        type=Path,
        default=FULL_RESULTS_DIR / "step2_synthetic_profiles.csv",
    )
    parser.add_argument(
        "--synthetic-raw",
        type=Path,
        default=FULL_RESULTS_DIR / "step2_synthetic_profiles_raw.csv",
    )
    parser.add_argument(
        "--prompt-components",
        type=Path,
        default=FULL_RESULTS_DIR / "step4_prompt_components.csv",
    )
    parser.add_argument(
        "--fewshot-exemplars",
        type=Path,
        default=FULL_RESULTS_DIR / "step4_fewshot_exemplars.json",
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        default=FULL_RESULTS_DIR / "step3_generation_plan.csv",
    )
    parser.add_argument(
        "--generations",
        type=Path,
        default=FULL_RESULTS_DIR / "generations" / "full_generations.csv",
    )
    parser.add_argument(
        "--author-level",
        type=Path,
        default=FULL_RESULTS_DIR / "generations" / "full_author_level_liwc.csv",
    )
    parser.add_argument(
        "--author-summary",
        type=Path,
        default=FULL_RESULTS_DIR
        / "generations"
        / "full_author_condition_model_summary.csv",
    )
    parser.add_argument(
        "--essay-summary",
        type=Path,
        default=FULL_RESULTS_DIR / "generations" / "full_condition_model_summary.csv",
    )
    parser.add_argument(
        "--bfi-summary",
        type=Path,
        default=FULL_RESULTS_DIR / "bfi" / "bfi_summary.csv",
    )
    parser.add_argument(
        "--bfi-per-sample",
        type=Path,
        default=FULL_RESULTS_DIR / "bfi" / "bfi_per_sample.csv",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=FULL_RESULTS_DIR / "embedding",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=FULL_RESULTS_DIR / "classification",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=FULL_RESULTS_DIR / "tables",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also write final_dataset.zip next to the output folder.",
    )
    parser.add_argument(
        "--skip-embeddings-npz",
        action="store_true",
        help="Omit personality_embeddings.npz (~9 MB) from the release.",
    )
    return parser.parse_args()


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required {label}: {path}")
    return path


def categories_from_frame(frame: pd.DataFrame) -> list[str]:
    found = [
        c.replace("target_", "", 1)
        for c in frame.columns
        if c.startswith("target_") and c not in ("target_O", "target_N")
    ]
    ordered = [c for c in DEFAULT_CONTROL_CATEGORIES if c in found]
    for cat in found:
        if cat not in ordered:
            ordered.append(cat)
    return ordered


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(sanitize(record), ensure_ascii=False) + "\n")


def build_paired_records(
    generations: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in generations.iterrows():
        content = str(row["content"])
        target = {cat: float(row[f"target_{cat}"]) for cat in categories}
        measured = {
            cat: float(row[f"obs_{cat}"])
            if pd.notna(row.get(f"obs_{cat}"))
            else None
            for cat in categories
        }
        control_error = {
            cat: (
                abs(float(row[f"obs_{cat}"]) - float(row[f"target_{cat}"]))
                if pd.notna(row.get(f"obs_{cat}"))
                else None
            )
            for cat in categories
        }
        rows.append(
            {
                "generation_id": str(row["plan_id"]),
                "profile_id": str(row["profile_id"]),
                "cell": str(row["cell"]) if pd.notna(row.get("cell")) else None,
                "content_prompt_id": CONTENT_TO_CP.get(content),
                "content": content,
                "steering_condition": str(row["steering_condition"]),
                "repetition": int(row["repetition"]) if pd.notna(row.get("repetition")) else 0,
                "model": str(row["model_key"]),
                "api_model": str(row["api_model"]) if pd.notna(row.get("api_model")) else None,
                "target_O": float(row["target_O"]),
                "target_N": float(row["target_N"]),
                "generated_text": row.get("text"),
                "prompt": row.get("prompt"),
                "target_profile": target,
                "measured_profile": measured,
                "control_error": control_error,
                "raw_mae": float(row["raw_mae"]) if pd.notna(row.get("raw_mae")) else None,
                "standardized_mae": (
                    float(row["standardized_mae"])
                    if pd.notna(row.get("standardized_mae"))
                    else None
                ),
                "word_count": int(row["n_words"]) if pd.notna(row.get("n_words")) else None,
                "type_token_ratio": (
                    float(row["type_token_ratio"])
                    if pd.notna(row.get("type_token_ratio"))
                    else None
                ),
                "self_identification": (
                    bool(row["self_identification"])
                    if pd.notna(row.get("self_identification"))
                    else None
                ),
                "scored": bool(row["scored"]) if pd.notna(row.get("scored")) else None,
                "error": row.get("error") if pd.notna(row.get("error")) else None,
            }
        )
    return pd.DataFrame(rows)


def _optional_int(row: pd.Series, *names: str) -> int | None:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return int(row[name])
    return None


def build_human_profile_records(
    human: pd.DataFrame,
    categories: list[str],
) -> list[dict]:
    records: list[dict] = []
    for _, row in human.iterrows():
        liwc = {cat: float(row[cat]) for cat in categories if cat in human.columns}
        records.append(
            {
                "author_id": str(row["author_id"]) if "author_id" in human.columns else None,
                "openness": float(row["O"]),
                "neuroticism": float(row["N"]),
                "cell": (
                    str(row["cell"])
                    if "cell" in human.columns and pd.notna(row.get("cell"))
                    else None
                ),
                "n_essays": _optional_int(row, "n_essays"),
                "n_words": _optional_int(row, "n_words", "total_word_count", "liwc_wc"),
                "liwc_profile": liwc,
            }
        )
    return records


def build_synthetic_profile_records(
    synthetic: pd.DataFrame,
    categories: list[str],
) -> list[dict]:
    records: list[dict] = []
    for _, row in synthetic.iterrows():
        liwc = {cat: float(row[cat]) for cat in categories if cat in synthetic.columns}
        target_o = float(row["O"]) if "O" in synthetic.columns else float(row["target_O"])
        target_n = float(row["N"]) if "N" in synthetic.columns else float(row["target_N"])
        records.append(
            {
                "profile_id": str(row["profile_id"]),
                "target_O": target_o,
                "target_N": target_n,
                "cell": (
                    str(row["cell"])
                    if "cell" in synthetic.columns and pd.notna(row.get("cell"))
                    else None
                ),
                "liwc_target_profile": liwc,
            }
        )
    return records


def _relativize_path_value(value: str, release_subdir: str) -> str:
    """Rewrite absolute/local paths to release-relative basenames."""
    name = Path(value).name
    if not name:
        return value
    return f"{release_subdir}/{name}"


def _copy_json_relativize_paths(
    src: Path, dest: Path, *, release_subdir: str
) -> None:
    """Copy a JSON report, replacing absolute filesystem paths with release paths."""
    payload = json.loads(src.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        payload["outputs"] = {
            key: _relativize_path_value(str(val), release_subdir)
            if isinstance(val, str) and ("/" in val or "\\" in val)
            else val
            for key, val in outputs.items()
        }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_code(out: Path) -> list[str]:
    code_root = out / "code" / "optimized"
    copied: list[str] = []
    for relative in CODE_FILES:
        src = OPTIMIZED_DIR / relative
        if not src.exists():
            continue
        dest = code_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(f"optimized/{relative}")
    requirements = REPO_ROOT / "requirements.txt"
    if requirements.exists():
        shutil.copy2(requirements, out / "code" / "requirements.txt")
        copied.append("requirements.txt")
    return copied


def make_zip(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file():
                continue
            if any(part == ".git" for part in path.parts):
                continue
            archive.write(path, arcname=str(Path(out.name) / path.relative_to(out)))
    return zip_path


def write_licenses(out: Path) -> None:
    (out / "LICENSE-DATA.txt").write_text(
        "Data in this release are licensed under Creative Commons Attribution 4.0 "
        "International (CC BY 4.0).\n"
        "https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )
    (out / "LICENSE-CODE.txt").write_text(
        "MIT License\n\n"
        "Copyright (c) 2026 Controled Generation authors\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n",
        encoding="utf-8",
    )


def write_docs(out: Path, meta: dict) -> None:
    write_licenses(out)

    readme = f"""# Continuous BFI–LIWC Controlled Personality Text Dataset

Anonymous code-and-data supplement for continuous Openness / Neuroticism
personality-conditioned generation. This archive contains the profiles, full
generation grid, validation tables, figures, and reproduction scripts used in
the paper.

Synthetic authors are sampled from a Gaussian copula over continuous trait and
LIWC targets (no tertile / 9-cell quotas). Paper metrics are continuous only.

## What is included

- **Generated data**: all {meta['n_paired']} essays (`generated_text` + prompts) with
  targets, measured LIWC, and fidelity metrics
- **Human profiles**: {meta['n_human']} author-level records derived from PANDORA
  (continuous O/N + eight LIWC rates; no raw Reddit dump)
- **Synthetic profiles**: {meta['n_synthetic']} Gaussian-copula authors used in the grid
- **Validation**: author-level LIWC alignment, embedding probe recovery, essay-only
  BFI-44 scores and summaries
- **Code**: pipeline scripts under `code/optimized/` (MIT)

## What is not included

- Raw PANDORA comment / essay corpus (only derived profiles + short few-shot excerpts)
- LIWC-22 dictionary (commercial; category rates in the tables were scored with it)
- API keys / `.env` files (regeneration against live model APIs requires your own keys)

## Design (as executed)

- **{meta['n_synthetic']} synthetic authors** × **{meta['n_topics']} topics** ×
  **{meta['n_conditions']} conditions** × **{meta['n_models']} models** =
  **{meta['n_paired']}** generations
- Topics: {", ".join(meta['contents'])}
- Conditions: {", ".join(meta['conditions'])}
- Models: {", ".join(meta['models'])}
- Control-target LIWC categories: `{meta['categories']}`

### Validation layers

1. **LIWC profile alignment** — concat topics per author–condition–model, re-score LIWC (primary); essay-level secondary
2. **Embedding signal recovery** — Ridge probe on `mpnet-personality`, trained on human windows only
3. **BFI score consistency** — independent essay-only BFI-44 rater; primary metrics are author-level means then Spearman ρ

## Contents

| Path | Description |
|------|-------------|
| `data/profiles/human_author_profiles.*` | {meta['n_human']} human author profiles (continuous O/N + 8 LIWC rates) |
| `data/profiles/synthetic_author_profiles.*` | {meta['n_synthetic']} Gaussian-copula synthetic authors |
| `data/profiles/prompt_components.csv` | Separable persona / LIWC / few-shot prompt blocks |
| `data/profiles/fewshot_exemplars.json` | Nearest-neighbor human excerpts for `lex_fewshot` |
| `data/generations/paired_generations.*` | {meta['n_paired']} essay-level generation–measurement records |
| `data/generations/author_level_liwc.*` | Primary Validation 1: concat topics → re-scored LIWC |
| `data/generations/*_summary.csv` | Author-level (primary) and essay-level (secondary) aggregates |
| `data/validation/bfi_*` | Essay-only BFI-44 (Validation 3; {meta['n_bfi']} scored essays) |
| `data/validation/embedding_*` | Continuous embedding probe recovery (Validation 2) |
| `data/validation/discriminability_cv.csv` | Nine-way cell CV accuracies for the LDA figure |
| `data/validation/paper_results_table.csv` | Main results table used in the paper |
| `data/figures/` | Paper embedding LDA / recovery figures and diagnostics |
| `code/optimized/` | Reproduction scripts |
| `code/requirements.txt` | Python dependencies |
| `examples/load_example.py` | Worked loading example |
| `SCHEMA.md` | Field-level schema |
| `datapackage.json` | Frictionless Data Package descriptor |
| `CITATION.cff` | Citation metadata (anonymous for review) |
| `metadata/release_manifest.json` | Build provenance |

Field definitions: see `SCHEMA.md`.

## Quick start

```bash
pip install -r code/requirements.txt
python examples/load_example.py
```

Inspect paper aggregates without regenerating:

```bash
# author × condition × model LIWC summary (primary Validation 1)
# data/generations/author_condition_model_summary.csv

# main paper table
# data/validation/paper_results_table.csv
```

Full regeneration (optional) needs LIWC-22, model API access, and the steps under
`code/optimized/`; see `code/optimized/README.md`.

## Licenses

- **Data**: CC BY 4.0 (`LICENSE-DATA.txt`)
- **Code**: MIT (`LICENSE-CODE.txt`)

## Citation

See `CITATION.cff` (anonymous authors for double-blind review). A public archive
DOI may be added after acceptance.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    schema = """# Schema

## Profiles — `human_author_profiles.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `author_id` | string | PANDORA author identifier |
| `openness` / `neuroticism` | float | Continuous BFI scores (0–100 scale) |
| `cell` | string | Optional legacy tertile label (diagnostic only; not used for sampling) |
| `n_essays` / `n_words` | int | Author essay metadata |
| `liwc_profile` | object | Category → LIWC-22 rate (%) |

## Profiles — `synthetic_author_profiles.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `profile_id` | string | Synthetic author id |
| `target_O` / `target_N` | float | Continuous personality targets |
| `liwc_target_profile` | object | Category → target LIWC-22 rate (%) |

## Generations — `paired_generations.jsonl` (one object per essay)

| Field | Type | Description |
|-------|------|-------------|
| `generation_id` | string | Stable plan / generation id |
| `profile_id` | string | Synthetic author |
| `content_prompt_id` | string | CP1–CP6 |
| `content` | string | Topic slug |
| `steering_condition` | string | `persona_only` / `liwc_only` / `persona_liwc` / `lex_fewshot` |
| `model` | string | Generator key (`deepseek-v3`, `gpt-4o-mini`, `qwen3-32b`) |
| `target_O` / `target_N` | float | Author personality targets |
| `generated_text` | string | Model output |
| `prompt` | string | Full generation prompt |
| `target_profile` | object | Control-target category → target % |
| `measured_profile` | object | Control-target category → observed % |
| `control_error` | object | Absolute raw error per category |
| `raw_mae` / `standardized_mae` | float | Essay-level LIWC MAE |
| `word_count` / `type_token_ratio` | number | Surface quality metrics |
| `self_identification` | bool | Regex screen for prompt leakage |

## Generations — `author_level_liwc.*`

One row per `profile_id` × `steering_condition` × `model`. Observed LIWC rates come from
**concatenating all topic essays** for that author–condition–model and re-scoring LIWC
once (primary Validation 1 metric).

## Validation — BFI (`bfi_per_sample.parquet`, `bfi_author_summary.csv`)

Essay-only BFI-44 protocol (`essay_only_rater_v1`): the rater sees only the essay text
for all six topics. Trait composites are on the PANDORA 0–100 scale: items answered
1–5, mapped via `(mean − 1) / 4 × 100`. Primary metrics average BFI over topics per
author then compute Spearman ρ(`target_O`, mean `bfi_O`) / ρ(`target_N`, mean `bfi_N`).
Raw rater prompts/responses are omitted from the release.

## Validation — Embedding

`embedding_per_sample.csv` stores probe predictions for each generation.
`embedding_author_summary.csv` stores author-level ρ_O / ρ_N (primary).
`personality_embeddings.npz` stores raw `mpnet-personality` vectors when included.
The Ridge probes were trained only on human windows (no generation leakage).

## Content prompt map

| Id | Slug |
|----|------|
| CP1 | memorable_journey |
| CP2 | difficult_decision |
| CP3 | daily_life_change |
| CP4 | meaningful_relationship |
| CP5 | childhood_memory |
| CP6 | turning_point |
"""
    (out / "SCHEMA.md").write_text(schema, encoding="utf-8")

    citation = f"""cff-version: 1.2.0
message: If you use this dataset, please cite it.
title: "Continuous BFI–LIWC Controlled Personality Text Dataset"
version: "{meta['version']}"
date-released: "{meta['date']}"
license: CC-BY-4.0
type: dataset
authors:
  - name: "Anonymous Authors"
abstract: >
  A FAIR synthetic-text dataset for personality-conditioned generation with continuous
  Openness and Neuroticism targets, author-specific LIWC-22 control profiles, matched
  persona / LIWC / joint / few-shot steering, and multi-level validation
  (author-level LIWC alignment, embedding recovery, essay-only BFI-44). Contains
  {meta['n_paired']} paired LLM generations across {meta['n_models']} models.
"""
    (out / "CITATION.cff").write_text(citation, encoding="utf-8")

    datapackage = {
        "name": "continuous-bfi-liwc-controlled-personality-text",
        "title": "Continuous BFI–LIWC Controlled Personality Text Dataset",
        "version": meta["version"],
        "created": meta["date"],
        "licenses": [
            {
                "name": "CC-BY-4.0",
                "path": "LICENSE-DATA.txt",
                "title": "Creative Commons Attribution 4.0 International",
            }
        ],
        "keywords": [
            "personality",
            "controlled generation",
            "LIWC",
            "Big Five",
            "BFI-44",
            "Gaussian copula",
        ],
        "resources": [
            {
                "name": "human_author_profiles",
                "path": "data/profiles/human_author_profiles.parquet",
                "format": "parquet",
            },
            {
                "name": "synthetic_author_profiles",
                "path": "data/profiles/synthetic_author_profiles.parquet",
                "format": "parquet",
            },
            {
                "name": "paired_generations",
                "path": "data/generations/paired_generations.parquet",
                "format": "parquet",
                "mediatype": "application/vnd.apache.parquet",
            },
            {
                "name": "author_level_liwc",
                "path": "data/generations/author_level_liwc.parquet",
                "format": "parquet",
            },
            {
                "name": "bfi_per_sample",
                "path": "data/validation/bfi_per_sample.parquet",
                "format": "parquet",
            },
            {
                "name": "paper_results_table",
                "path": "data/validation/paper_results_table.csv",
                "format": "csv",
            },
        ],
    }
    (out / "datapackage.json").write_text(
        json.dumps(datapackage, indent=2) + "\n", encoding="utf-8"
    )

    example = '''#!/usr/bin/env python3
"""Worked example: load profiles, generations, and author-level LIWC summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    humans = pd.read_parquet(ROOT / "data/profiles/human_author_profiles.parquet")
    synth = pd.read_parquet(ROOT / "data/profiles/synthetic_author_profiles.parquet")
    print(f"Human authors: {len(humans)}  |  Synthetic profiles: {len(synth)}")

    gens = pd.read_parquet(ROOT / "data/generations/paired_generations.parquet")
    print(f"Paired generations: {len(gens)}")
    print(
        gens.groupby(["model", "steering_condition"])["standardized_mae"]
        .mean()
        .round(3)
        .unstack()
    )

    author = pd.read_parquet(ROOT / "data/generations/author_level_liwc.parquet")
    print(f"\\nAuthor-level LIWC rows: {len(author)}")
    summary = pd.read_csv(
        ROOT / "data/generations/author_condition_model_summary.csv"
    )
    print(summary.head(8).to_string(index=False))

    hit = gens[
        (gens["steering_condition"] == "persona_liwc")
        & (gens["model"] == "gpt-4o-mini")
    ].iloc[0]
    print("\\nExample generation (truncated):")
    print(str(hit["generated_text"])[:280], "...")
    print("Target LIWC:", json.loads(hit["target_profile"]) if isinstance(hit["target_profile"], str) else hit["target_profile"])


if __name__ == "__main__":
    main()
'''
    (out / "examples" / "load_example.py").write_text(example, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()

    human_path = require(args.human_profiles, "human profiles")
    synth_path = require(args.synthetic_profiles, "synthetic profiles")
    components_path = require(args.prompt_components, "prompt components")
    gens_path = require(args.generations, "generations")
    author_path = require(args.author_level, "author-level LIWC")
    author_summary_path = require(args.author_summary, "author-level summary")
    essay_summary_path = require(args.essay_summary, "essay-level summary")
    bfi_summary_path = require(args.bfi_summary, "BFI summary")
    bfi_per_path = require(args.bfi_per_sample, "BFI per-sample")

    if out.exists():
        shutil.rmtree(out)
    for sub in (
        "data/profiles",
        "data/generations",
        "data/validation",
        "data/figures",
        "examples",
        "metadata",
        "code",
    ):
        (out / sub).mkdir(parents=True)

    human = pd.read_csv(human_path)
    synthetic = pd.read_csv(synth_path)
    generations = pd.read_csv(gens_path)
    author_level = pd.read_csv(author_path)
    categories = categories_from_frame(generations)

    # --- Profiles ---
    human_records = build_human_profile_records(human, categories)
    write_jsonl(out / "data/profiles/human_author_profiles.jsonl", human_records)
    human.to_parquet(out / "data/profiles/human_author_profiles.parquet", index=False)
    human.to_csv(out / "data/profiles/human_author_profiles.csv", index=False)

    o_col = "O" if "O" in synthetic.columns else "target_O"
    n_col = "N" if "N" in synthetic.columns else "target_N"
    synth_export = synthetic.copy()
    if o_col != "target_O":
        synth_export = synth_export.rename(columns={o_col: "target_O", n_col: "target_N"})
    synth_records = build_synthetic_profile_records(synth_export, categories)
    write_jsonl(out / "data/profiles/synthetic_author_profiles.jsonl", synth_records)
    synth_export.to_parquet(
        out / "data/profiles/synthetic_author_profiles.parquet", index=False
    )
    synth_export.to_csv(out / "data/profiles/synthetic_author_profiles.csv", index=False)

    if args.synthetic_raw.exists():
        shutil.copy2(
            args.synthetic_raw,
            out / "data/profiles/synthetic_author_profiles_raw_pool.csv",
        )

    shutil.copy2(components_path, out / "data/profiles/prompt_components.csv")
    if args.fewshot_exemplars.exists():
        shutil.copy2(
            args.fewshot_exemplars, out / "data/profiles/fewshot_exemplars.json"
        )
    if args.generation_plan.exists():
        shutil.copy2(args.generation_plan, out / "data/profiles/generation_plan.csv")

    # --- Generations ---
    paired = build_paired_records(generations, categories)
    write_jsonl(
        out / "data/generations/paired_generations.jsonl",
        paired.to_dict(orient="records"),
    )
    paired_pq = paired.copy()
    for col in ("target_profile", "measured_profile", "control_error"):
        paired_pq[col] = paired_pq[col].map(lambda value: json.dumps(sanitize(value)))
    paired_pq.to_parquet(out / "data/generations/paired_generations.parquet", index=False)

    author_level.to_parquet(
        out / "data/generations/author_level_liwc.parquet", index=False
    )
    author_level.to_csv(out / "data/generations/author_level_liwc.csv", index=False)
    shutil.copy2(
        author_summary_path,
        out / "data/generations/author_condition_model_summary.csv",
    )
    shutil.copy2(
        essay_summary_path,
        out / "data/generations/essay_condition_model_summary.csv",
    )

    # --- Validation ---
    shutil.copy2(bfi_summary_path, out / "data/validation/bfi_summary.csv")
    bfi_author = FULL_RESULTS_DIR / "bfi" / "bfi_author_summary.csv"
    if bfi_author.exists():
        shutil.copy2(bfi_author, out / "data/validation/bfi_author_summary.csv")
    bfi_report = FULL_RESULTS_DIR / "bfi" / "bfi.report.json"
    if bfi_report.exists():
        shutil.copy2(bfi_report, out / "data/validation/bfi.report.json")
    bfi = pd.read_csv(bfi_per_path)
    drop_cols = [c for c in ("bfi_prompt", "bfi_raw") if c in bfi.columns]
    bfi_release = bfi.drop(columns=drop_cols)
    bfi_release.to_parquet(out / "data/validation/bfi_per_sample.parquet", index=False)
    bfi_release.to_csv(out / "data/validation/bfi_per_sample.csv", index=False)

    emb_dir = args.embedding_dir
    for name in (
        "embedding_per_sample.csv",
        "embedding_condition_summary.csv",
        "embedding_condition_model_summary.csv",
        "embedding_author_level.csv",
        "embedding_author_summary.csv",
        "embedding_probe.report.json",
        "human_windows.csv",
    ):
        src = emb_dir / name
        if not src.exists():
            continue
        dest = out / "data/validation" / name
        if name.endswith(".report.json"):
            _copy_json_relativize_paths(src, dest, release_subdir="data/validation")
        else:
            shutil.copy2(src, dest)
    npz = emb_dir / "personality_embeddings.npz"
    if npz.exists() and not args.skip_embeddings_npz:
        shutil.copy2(npz, out / "data/validation/personality_embeddings.npz")

    tables_dir = args.tables_dir
    for name in (
        "full_results_paper_table.csv",
        "full_results_paper_table_fragment.tex",
        "full_results_paper_table.tex",
    ):
        src = tables_dir / name
        if src.exists():
            dest_name = (
                "paper_results_table.csv"
                if name == "full_results_paper_table.csv"
                else name
            )
            shutil.copy2(src, out / "data/validation" / dest_name)

    # --- Figures / embedding diagnostics ---
    for name in (
        "fig_embedding_recovery.png",
        "personality_embedding_viz_lda_row.png",
        "personality_embedding_viz_lda_vertical.png",
        "personality_embedding_viz_discriminability.png",
        "discriminability_cv.csv",
    ):
        src = emb_dir / name
        if not src.exists():
            continue
        dest_dir = out / "data/figures" if name.endswith(".png") else out / "data/validation"
        shutil.copy2(src, dest_dir / name)

    paper_fig_dir = REPO_ROOT / "papers" / "main" / "figures"
    paper_copies = {
        "fig_embedding_recovery.png": "fig_embedding_recovery.png",
        "fig_embedding_lda_row.png": "fig_embedding_lda_row.png",
        "full_results_paper_table.png": "paper_results_table.png",
    }
    for src_name, dest_name in paper_copies.items():
        src = paper_fig_dir / src_name
        if src.exists():
            shutil.copy2(src, out / "data/figures" / dest_name)

    table_png = tables_dir / "full_results_paper_table.png"
    if table_png.exists() and not (out / "data/figures/paper_results_table.png").exists():
        shutil.copy2(table_png, out / "data/figures/paper_results_table.png")

    copied_code = copy_code(out)

    meta = {
        "step": "Optimized continuous pipeline — final dataset packaging",
        "version": VERSION,
        "date": date.today().isoformat(),
        "n_human": int(len(human)),
        "n_synthetic": int(len(synthetic)),
        "n_paired": int(len(paired)),
        "n_author_level": int(len(author_level)),
        "n_bfi": int(len(bfi_release)),
        "n_models": int(paired["model"].nunique()),
        "n_conditions": int(paired["steering_condition"].nunique()),
        "n_topics": int(paired["content"].nunique()),
        "models": sorted(paired["model"].astype(str).unique().tolist()),
        "conditions": sorted(paired["steering_condition"].astype(str).unique().tolist()),
        "contents": sorted(paired["content"].astype(str).unique().tolist()),
        "categories": categories,
        "content_prompt_map": CONTENT_TO_CP,
        "steering_conditions_canonical": list(STEERING_CONDITIONS),
        "bfi_protocol": "essay_only_rater_v1",
        "primary_liwc_metric": "author_level_concat_reliwc",
        "code_scripts": copied_code,
        "fair": {
            "findability": ["datapackage.json", "CITATION.cff"],
            "accessibility": [
                "LICENSE-DATA.txt (CC BY 4.0)",
                "LICENSE-CODE.txt (MIT)",
            ],
            "interoperability": ["parquet", "jsonl", "csv", "json"],
            "reusability": ["SCHEMA.md", "examples/load_example.py", "README.md"],
        },
        "sources": {
            "human_profiles": str(human_path.relative_to(REPO_ROOT)),
            "synthetic_profiles": str(synth_path.relative_to(REPO_ROOT)),
            "generations": str(gens_path.relative_to(REPO_ROOT)),
            "author_level_liwc": str(author_path.relative_to(REPO_ROOT)),
            "bfi_per_sample": str(bfi_per_path.relative_to(REPO_ROOT)),
        },
        "note": (
            "Zenodo/DOI registration is manual after upload; update CITATION.cff "
            "when a DOI is minted. LIWC-22 dictionary is not redistributed."
        ),
    }
    write_json(out / "metadata" / "release_manifest.json", meta)
    write_docs(out, meta)

    zip_path = make_zip(out) if args.zip else None
    nbytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    report = {
        "status": "ok",
        "output_dir": str(out),
        "n_human": meta["n_human"],
        "n_synthetic": meta["n_synthetic"],
        "n_paired": meta["n_paired"],
        "n_author_level": meta["n_author_level"],
        "n_bfi": meta["n_bfi"],
        "zip": str(zip_path) if zip_path else None,
        "bytes": nbytes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
