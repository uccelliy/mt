# v0 — Centaur evaluation (archived 2026-08-04)

Everything the Centaur-evaluation phase produced, superseded by benchmark v1
(`docs/benchmark design.md`). Kept because the scientific findings in
`docs/centaur-eval-handoff.md` cite these files, not because anything here
feeds the new pipeline.

| | |
|---|---|
| `scoring/` | E0 / E3 / P0 / E2 score CSVs, 593 MB |
| `analysis/` | derived per-task tables |
| `figures/` | fig1–fig16 |

**Do not mix with v1 results.** Three reasons:

1. **Schema.** These CSVs have five columns — `experiment, participant,
   choice_index, nll, num_tokens`. Benchmark v1 stores three tables per run
   (design §2) and none of it is here: not the model's own choice, not the
   per-position top-k, not the legal-option logprobs.
2. **Configuration.** Run at `--batch-tokens 16384` on a consumer card, not
   the pinned `8192` + `volta16` (design §8 item 3).
3. **Known dirt.** `collsiöö2023MCPL` appears as `collsi枚枚2023MCPL` in the
   Minitaur CSV (CP936 residue — the scores are valid, only the name is
   wrong), and ten `zorowitz2023data` sessions need the UTF-8 re-scored
   version in `minitaur8b_e0_zorowitz_utf8_4bit.csv`.

The `nll` column is the **sum** of the token NLLs inside one choice, which is
exactly the first level of the v1 aggregation, so the three v1 metrics can be
computed from these files as a zero-GPU preview. Store such previews apart
from real v1 results.
