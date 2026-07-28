"""Shared constants and helpers for the continuous BFI–LIWC pipeline.

Defines control-target categories, steering conditions, topic prompts, path
anchors, and small utilities used across pilot and full-run steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPTIMIZED_DIR = Path(__file__).resolve().parent
REPO_ROOT = OPTIMIZED_DIR.parent
OLD_PIPELINE_DIR = REPO_ROOT / "old_pipeline"
SCRIPTS_DIR = OLD_PIPELINE_DIR / "scripts"
RESULTS_DIR = OPTIMIZED_DIR / "results"
PILOT_RESULTS_DIR = RESULTS_DIR / "pilot"
FULL_RESULTS_DIR = RESULTS_DIR / "full"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from repo_paths import anchor_output, resolve_path  # noqa: E402

DEFAULT_CELL_ASSIGNMENTS = "results/stage1_step2/pandora-authors-cell-assignments.csv"
DEFAULT_LIWC = "liwc22/LIWC-22 Results - pandora-authors-train - LIWC Analysis.csv"
DEFAULT_AUTHORS = "pandora-authors-train.csv"
DEFAULT_VALIDATED_TARGETS = "results/stage1_step6/pandora-validated-control-targets.csv"
DEFAULT_LIWC_CLI = "/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-cli"

DEFAULT_N_CONTROL_TARGETS = 8

# Top-8 Stage-2 control targets (validated; matches paper Table 1 ordering).
DEFAULT_CONTROL_CATEGORIES = (
    "emo_sad",
    "focusfuture",
    "mental",
    "certitude",
    "fatigue",
    "Affect",
    "adverb",
    "swear",
)

LIWC_PLAIN_LANGUAGE: dict[str, str] = {
    "emo_sad": "words expressing sadness or low mood",
    "focusfuture": "words oriented toward the future (plans, will, gonna)",
    "mental": "words referring to mental states or cognition about the mind",
    "certitude": "words expressing certainty or conviction",
    "fatigue": "words about tiredness or fatigue",
    "Affect": "overall affective / emotional language",
    "adverb": "adverbs that modify how actions or states are described",
    "swear": "swear / curse words",
}

CONTENT_CONDITIONS: dict[str, str] = {
    "weekend": "what you did last weekend",
    "technology": "your opinion on a recent technology trend",
    "food": "a meal or kind of food you really enjoy",
    "travel": "a place you would like to visit and why",
    "movie": "your thoughts on a movie or show you watched recently",
    "work": "a typical day in your work or studies",
}

# Full-scale narrative topics (match Stage 2 Step 8).
FULL_CONTENT_CONDITIONS: dict[str, str] = {
    "memorable_journey": "a memorable journey",
    "difficult_decision": "a difficult decision",
    "daily_life_change": "a recent change in daily life",
    "meaningful_relationship": "a meaningful relationship",
    "childhood_memory": "an early childhood memory",
    "turning_point": "a personal turning point",
}

STEERING_CONDITIONS = ("persona_only", "liwc_only", "persona_liwc", "lex_fewshot")

# Continuous analog of cell few-shot: nearest human authors by LIWC distance.
DEFAULT_N_EXEMPLARS = 2
DEFAULT_EXEMPLAR_WORDS = 100
DEFAULT_EXEMPLAR_CANDIDATES = 40

DEFAULT_PILOT_CONTENTS = ("weekend", "technology")
DEFAULT_FULL_CONTENTS = tuple(FULL_CONTENT_CONDITIONS.keys())
DEFAULT_N_SYNTHETIC = 12
DEFAULT_N_SYNTHETIC_FULL = 80
DEFAULT_N_REPS = 1
DEFAULT_TARGET_WORDS = 200
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 800
DEFAULT_SEED = 7

# Human sample tertile edges from Stage 1 Step 2 (for classification labels).
HUMAN_O_TERTILE_EDGES = (19.0, 58.0)
HUMAN_N_TERTILE_EDGES = (28.3333, 71.0)

def ensure_results_dir(path: Path | None = None) -> Path:
    out = path or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def select_control_categories(
    validated_path: Path,
    n: int = DEFAULT_N_CONTROL_TARGETS,
) -> list[str]:
    resolved = resolve_path(validated_path)
    if not resolved.exists():
        return list(DEFAULT_CONTROL_CATEGORIES[:n])
    validated = pd.read_csv(resolved)
    targets = validated[validated["validated_control_target"] == True].copy()  # noqa: E712
    if targets.empty:
        return list(DEFAULT_CONTROL_CATEGORIES[:n])
    targets = targets.sort_values("max_abs_cohens_d", ascending=False)
    cats = targets["category"].head(n).tolist()
    return cats if cats else list(DEFAULT_CONTROL_CATEGORIES[:n])


def percentile_rank(values: pd.Series, x: float) -> float:
    """Empirical percentile rank of x within values, in [0, 1]."""
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    return float(np.mean(arr <= x))


def strength_phrase(pct: float) -> str:
    """Map a percentile in [0, 1] to a natural-language strength phrase."""
    if pct < 0.15:
        return "much less than most people"
    if pct < 0.35:
        return "somewhat less than most people"
    if pct < 0.65:
        return "about as much as most people"
    if pct < 0.85:
        return "somewhat more than most people"
    return "much more than most people"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def fisher_mean_rho(rhos: list[float] | np.ndarray) -> float:
    """Average correlations on the Fisher-z scale, then invert (RA plan)."""
    arr = np.asarray(list(rhos), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    clipped = np.clip(arr, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def clip_liwc_rate(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def tertile_level(value: float, edges: tuple[float, float]) -> str:
    if value <= edges[0]:
        return "low"
    if value <= edges[1]:
        return "med"
    return "high"


def assign_cell_label(
    o: float,
    n: float,
    *,
    o_edges: tuple[float, float] = HUMAN_O_TERTILE_EDGES,
    n_edges: tuple[float, float] = HUMAN_N_TERTILE_EDGES,
) -> str:
    """Map continuous O/N onto the Stage-1 9-cell grid (for classification viz)."""
    o_level = tertile_level(o, o_edges)
    n_level = tertile_level(n, n_edges)
    return f"{o_level}_O__{n_level}_N"
