# Archived scripts (2026-08-04, job scripts added 2026-08-06)

Superseded by benchmark v1. Kept for provenance — several produced figures
still cited in `docs/centaur-eval-handoff.md`.

The formal Track P entry point is `scripts/submit_roster.sh` driving
`scripts/score_model.slurm`. Nothing here is a submission path any more; the
two job scripts below were moved out of `scripts/` precisely so that
`sbatch scripts/<name>.slurm` can no longer start a retired-model run.

| script | why archived |
|---|---|
| `e0_e3_minitaur.slurm` | v0 production job for the merged Minitaur checkpoint, writing the old five-column CSV. The merged checkpoint left the formal roster (design §6.2: merging to BF16 and requantizing costs 0.182 nat), and the schema is now the three tables of §2.5. |
| `smoke_e0_e3.slurm` | v0 smoke and FP16 arm for the same retired model. Its numerical cross-check role is now served offline — design §8.4 re-confirms r = 1.00000 against the official 36 families with zero GPU. |
| `build_report_figures.py` | fig1–4, built on v0 runs and the old aggregation. Its shared helpers moved and are still live: styling to `scripts/experiments/_plotting.py`, loading and aggregation to `mt.evaluation.aggregate`. |
| `build_per_task_figures.py` | fig5–8; `dot_rows` / `short_name` / `VIOLET` moved to `_plotting.py` |
| `build_task_descriptive_figures.py` | fig11–13 |
| `build_canonical_choice_figures.py` | fig14; needs the canonical-space tables |
| `build_adaptation_gap_figures.py` | fig15 |
| `measure_numerical_floor.py` | the numerical-floor section was dropped from the design |
| `measure_choice_baselines.py` | duplicated `mt.models.baselines.sequence.score_sequence_online` instead of calling it, and scored the rules by accuracy — a metric the design later dropped. **The baselines themselves are still required** (registry R4–R7); `scripts/experiments/run_sequence_baselines.py` is the runner for them. |

These import each other and `build_report_figures`; running one from here may
need `PYTHONPATH` to include both this directory and `scripts/experiments`.
