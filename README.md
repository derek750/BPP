# Continuous BFI–LIWC Personality Steering

Pipeline for **continuous** Openness / Neuroticism personality-conditioned LLM generation: author-level LIWC targets, Gaussian-copula synthetic authors, matched steering conditions, and multi-level validation (LIWC alignment, embedding recovery, essay-only BFI-44).

Synthetic authors are sampled **directly from a Gaussian copula** (fixed seed; QC reject-and-replace only). Primary paper metrics are continuous: LIWC MAE / ρ / β, embedding ρ_O / ρ_N, and essay-only BFI recovery.

## Layout

| Path | Contents |
|------|----------|
| `common.py` | Shared constants / helpers |
| `liwc_author_level.py` | Concat essays → re-LIWC (Validation 1 primary) |
| `steps/` | Step scripts (`step1` … `step9`, embedding viz) |
| `runners/` | Orchestrators (`run_pilot`, `run_full`, `prepare_full`, `package_final_dataset`) |
| `results/pilot/` | Pilot outputs (steps 1–5 + embedding probe) |
| `results/full/` | Full multi-model outputs (generations, tables, embedding, BFI) |
| `results/final_dataset/` | FAIR release package (profiles, generations, validation, code) |

## Conditions

1. `persona_only` — continuous O/N description only
2. `liwc_only` — author-specific LIWC targets only (hybrid relative + numeric %)
3. `persona_liwc` — both
4. `lex_fewshot` — LIWC targets + nearest-neighbor human excerpts (paper label: **LIWC+Fewshot**)

## Default pilot grid

12 synthetic authors × 2 topics (`weekend`, `technology`) × 4 conditions × 1 rep
= **96** generations.

## Run

From repo root (venv active):

```bash
# End-to-end dry run (no API / no LIWC app)
python runners/run_pilot.py --mock

# Smaller smoke test
python runners/run_pilot.py --mock --n-synthetic 8 --limit 24

# Real DeepSeek + LIWC-22-cli
python runners/run_pilot.py

# Add / re-run only lex_fewshot, keep prior conditions
python steps/step3_generation_plan.py
python steps/step4_prompt_components.py --rebuild-exemplars
python steps/step5_pilot_generation.py --conditions lex_fewshot --merge-existing

# Embedding O/N recovery (mpnet-personality) — Validation 2 primary
python steps/step6_embedding_probe.py
python steps/step6_embedding_recovery_viz.py
```

Requires `DEEPSEEK_API_KEY` in `.env` for non-mock generation, and the LIWC-22
desktop CLI for non-mock scoring.

## Full multi-model run

Default grid after `prepare_full.py`:
**80 profiles × 6 narrative topics × 4 conditions × 3 models = 5,760**

```bash
python runners/prepare_full.py
python runners/run_full.py --skip-prepare --workers 6
# or one-shot:
python runners/run_full.py --workers 6
```

Outputs:
- `results/full/generations/full_generations.csv`
- `results/full/generations/full_author_level_liwc.csv` (concat → re-LIWC; **Validation 1 primary**)
- `results/full/generations/full_author_condition_model_summary.csv` (primary MAE / ρ / slope)
- `results/full/generations/full_condition_model_summary.csv` (essay-level; secondary)

### Validation roles

| Validation | Primary artifact | Notes |
|---|---|---|
| 1 LIWC alignment | `full_author_condition_model_summary.csv` | Concatenate topics → one LIWC pass; **Fisher-z mean ρ** |
| 2 Embedding recovery | `step6_embedding_probe.py` + `step6_embedding_recovery_viz.py` | Humans-only Ridge; continuous O/N; author-level ρ; recovery scatter |
| 3 BFI-44 | `step8_bfi_validation.py` | Essay-only rater on **all topics**; author-level MAE / ρ / β |
| 4 Inference | `step9_profile_bootstrap.py` | Profile-clustered bootstrap CIs + planned contrasts |

Paper table columns: $\mathrm{MAE}_{\mathrm{LIWC}}$, $\rho_{\mathrm{LIWC}}$,
$\beta_{\mathrm{LIWC}}$, emb.\ $\rho_O$/$\rho_N$.
Rebuild with:

```bash
python runners/build_paper_table.py
# refresh LIWC ρ after Fisher-z change (no LIWC CLI):
python steps/step6_full_generation.py --resummarize-author-only
# profile-clustered bootstrap (default 2000 draws):
python steps/step9_profile_bootstrap.py
```

### BFI-44 validation (essay-only rater; Validation 3)

Independent BFI-based evaluator rates each generated essay. The evaluator
sees **only the essay text** + BFI-44 questionnaire — not target O/N,
steering condition, LIWC targets, or the generation prompt. Default grid:
**80 profiles × 6 topics × 4 conditions × 3 models = 5,760**.
Primary summary: mean BFI over topics within profile×condition×model, then Spearman ρ.

```bash
python steps/step8_bfi_validation.py --workers 6
# smoke:
python steps/step8_bfi_validation.py --mock --limit 12
# force full rescore:
python steps/step8_bfi_validation.py --force --workers 6
```

Outputs under `results/full/bfi/` (`bfi_per_sample.csv`, `bfi_summary.csv`,
`bfi_author_summary.csv`, `bfi.report.json`).
Primary metrics: $\mathrm{MAE}_{\mathrm{BFI}}$, $\rho_O$/$\rho_N$, $\beta_{\mathrm{BFI}}$.

### Author-level tradeoff

Author-level tradeoff holds on all three models: persona wins MAE with ~null ρ;
LIWC methods track (higher $\rho_{\mathrm{LIWC}}$) but overshoot ($\beta{>}1$) and raise MAE.
Persona+LIWC often leads embedding $\rho_O$/$\rho_N$.

Relative to `persona_only` (use **mean per-category** ρ / slope, not raw
stacked pooling — category base-rate differences inflate pooled ρ):

- `liwc_only` / `persona_liwc` should show **higher mean category ρ** and a
  **mean calibration slope closer to 1**
- MAE alone is insufficient (persona can look good by hugging the corpus mean)

## FAIR final dataset package

Build a self-contained reproducibility archive (profiles, paired generations,
author-level LIWC, embedding/BFI tables, code, schema):

```bash
python runners/package_final_dataset.py --zip
```

Writes `results/final_dataset/` and `results/final_dataset.zip`
(CC BY 4.0 data, MIT code). Ready for Zenodo deposit; update `CITATION.cff`
after a DOI is minted. LIWC-22 is not redistributed.

## Notes

- Copula uses Gaussian-rank transforms + empirical quantile matching (no extra
  packages beyond numpy/scipy/pandas).
- Continuous-run LIWC prompts are **hybrid** (relative + numeric %).
- Rare LIWC categories remain sparse; QC clips rates to [0, 100].
