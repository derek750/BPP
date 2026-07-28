#!/usr/bin/env python3
"""Step 1: Build author-level continuous BFI–LIWC profiles.

One author = continuous O + continuous N + aggregated LIWC profile (already
author-aggregated in the PANDORA LIWC CSV) + essay/word metadata.
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

from common import (
    DEFAULT_AUTHORS,
    DEFAULT_CELL_ASSIGNMENTS,
    DEFAULT_LIWC,
    DEFAULT_N_CONTROL_TARGETS,
    DEFAULT_VALIDATED_TARGETS,
    OPTIMIZED_DIR,
    RESULTS_DIR,
    PILOT_RESULTS_DIR,
    FULL_RESULTS_DIR,
    ensure_results_dir,
    resolve_path,
    select_control_categories,
    write_json,
)

DEFAULT_OUTPUT = PILOT_RESULTS_DIR / "step1_author_profiles.csv"
DEFAULT_REPORT = PILOT_RESULTS_DIR / "step1_author_profiles.report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liwc", type=Path, default=Path(DEFAULT_LIWC))
    parser.add_argument("--cells", type=Path, default=Path(DEFAULT_CELL_ASSIGNMENTS))
    parser.add_argument("--authors", type=Path, default=Path(DEFAULT_AUTHORS))
    parser.add_argument(
        "--validated-targets",
        type=Path,
        default=Path(DEFAULT_VALIDATED_TARGETS),
    )
    parser.add_argument("--n-control-targets", type=int, default=DEFAULT_N_CONTROL_TARGETS)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-r", "--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (OPTIMIZED_DIR / path).resolve()


def build_author_profiles(
    liwc_path: Path,
    cells_path: Path,
    authors_path: Path,
    categories: list[str],
) -> tuple[pd.DataFrame, dict]:
    liwc = pd.read_csv(resolve_path(liwc_path))
    cells = pd.read_csv(resolve_path(cells_path))
    authors = pd.read_csv(
        resolve_path(authors_path),
        usecols=["comment_count", "word_count"],
    )

    if len(liwc) != len(cells) or len(liwc) != len(authors):
        raise ValueError(
            f"Row-count mismatch: LIWC={len(liwc)}, cells={len(cells)}, "
            f"authors={len(authors)}"
        )
    if not np.allclose(liwc["O"].to_numpy(), cells["O"].to_numpy()):
        raise ValueError("O scores do not align between LIWC and cell tables.")
    if not np.allclose(liwc["N"].to_numpy(), cells["N"].to_numpy()):
        raise ValueError("N scores do not align between LIWC and cell tables.")

    missing = [c for c in categories if c not in liwc.columns]
    if missing:
        raise ValueError(f"LIWC table missing categories: {missing}")

    frame = pd.DataFrame(
        {
            "author_id": cells["essay_id"].astype(int),
            "O": cells["O"].astype(float),
            "N": cells["N"].astype(float),
            "n_essays": authors["comment_count"].astype(int),
            "total_word_count": authors["word_count"].astype(float),
            "liwc_wc": liwc["WC"].astype(float),
        }
    )
    for category in categories:
        frame[category] = liwc[category].astype(float)

    stats: dict = {
        "n_authors": int(len(frame)),
        "categories": categories,
        "O": _summarize(frame["O"]),
        "N": _summarize(frame["N"]),
        "n_essays": _summarize(frame["n_essays"]),
        "total_word_count": _summarize(frame["total_word_count"]),
        "corr_O_N": float(frame["O"].corr(frame["N"])),
        "liwc": {},
        "corr_O_liwc": {},
        "corr_N_liwc": {},
        "zero_rate": {},
    }
    for category in categories:
        stats["liwc"][category] = _summarize(frame[category])
        stats["corr_O_liwc"][category] = float(frame["O"].corr(frame[category]))
        stats["corr_N_liwc"][category] = float(frame["N"].corr(frame[category]))
        stats["zero_rate"][category] = float((frame[category] == 0).mean())

    return frame, stats


def _summarize(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def main() -> None:
    args = parse_args()
    categories = select_control_categories(args.validated_targets, args.n_control_targets)
    frame, stats = build_author_profiles(
        args.liwc, args.cells, args.authors, categories
    )
    output = _resolve_out(args.output)
    report = _resolve_out(args.report)
    ensure_results_dir(output.parent)
    frame.to_csv(output, index=False)
    write_json(report, {"output": str(output), **stats})
    print(f"Wrote {len(frame)} author profiles -> {output}")
    print(f"Report -> {report}")
    print(
        f"O mean={stats['O']['mean']:.1f}, N mean={stats['N']['mean']:.1f}, "
        f"r(O,N)={stats['corr_O_N']:.3f}"
    )


if __name__ == "__main__":
    main()
