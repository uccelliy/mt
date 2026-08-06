# PROJECT.md

## What This Project Is

`mt` builds a benchmark for auditing what language models actually learn about
human behavior. The design of record is **`docs/benchmark design.md`** — it owns
the metrics, the storage schema, the aggregation rule, the protocol clauses, and
the analysis registry. This file owns the surrounding context that does not
belong in a construction document: compute budget, conclusion boundaries, and
scope.

The problem it targets: claims about "foundation models of cognition" rest on
aggregate held-out likelihood numbers, and a raw NLL is not readable. It is
neither the probability the model assigned to the human's choice nor an accuracy,
and a difference of 0.01 nat means nothing without a reference. The benchmark
keeps NLL (it is the only quantity that checks against published numbers) and
adds two readable transforms on top of it: a probability, and a ratio against a
counting baseline.

**This project is not a replication of Centaur.** Reproducing the paper's
numbers is not the goal — runs are 4-bit and are never mixed into a column with
published BF16 results. Centaur is the *subject* of the audit and a reference
line, never a denominator.

---

## End Goal

**1. A reusable evaluation protocol.** One pinned configuration under which many
models can be scored and compared, with counting baselines as the non-neural
floor and a fixed rule for what may be asserted.

**2. A foundation model**, held to that same protocol.

The protocol makes the comparison honest; the foundation model is what the setup
exists to test.

Classical cognitive-model baselines were part of an earlier plan and were
removed from the evaluation path. They return in one specific place: Track S
fits the official `openloop/models.py` cognitive models to human and to
model-generated trajectories and compares the fitted parameters. Recover the
earlier implementations from git history if a broader use appears.

---

## Current Stage

Benchmark v1: Psych-101-test only, full test set, no sampling tier.

Two tracks, defined in `benchmark design.md` §7:

- **Track P** — teacher-forced scoring. Reuses `src/mt/evaluation/`.
- **Track S** — open-loop simulation on the four official tasks that fall in the
  test set, reusing `~/wkspace/Llama-3.1-Centaur-70B/openloop/` with the model
  loader swapped for ours.

See `ARCHITECTURE.md` for per-module stability status.

---

## Compute Budget

Cost atom, extrapolated from measurement: the five-anchor E3 full grid took
about 24h on a single RTX 5060 Ti at NF4. On 4×V100 data-parallel that becomes
roughly **E0 ≈ 2–3h, E3 ≈ 8–12h** of 4-GPU wall.

At 8B a single job sits well inside the `gpu` partition's 47h cap. **The
bottleneck is queueing and fairshare, not walltime.**

| Item | Cost |
|---|---|
| Unified Track P `full + E3`, per model | full ≈3h plus finite-window grid ≈12h: **≈15h of 4-GPU wall** |
| Unified Track P, current roster | 6 models ≈90h ≈360 GPU-hour |
| Track S | Sequential per trial, cannot batch within a session, so GPU
utilization is poor. Small task count keeps it affordable. |
| 70B | Single-process model sharding, throughput ≈1× against 8B's 4× data
parallel — roughly 35× slower. A full pass does not fit in 47h, so it **requires
verified shard-and-resume** (registry A10) as a hard prerequisite. |

**Cut order if the queue makes the total infeasible** — decided in advance, not
improvised: drop the context-window curve first, then the 70B reference, and cut
the model count last. Model count is the primary research question; the window
curve already has complete two-model results to lean on.

---

## Conclusion Boundaries

- **The three metrics are not three pieces of evidence.** $L_f$, $p_f$ and $R_f$
  are transforms of the same NLL. They improve readability, not statistical
  independence.
- **Teacher-forced and open-loop numbers cannot be subtracted.** One is one-step
  prediction, the other accumulates its own error, and their probabilities are
  conditioned on different histories (design §2.6).
- **E3 is a marked-text window, not a trial-memory assay.** The test JSONL has no
  trial/phase offsets; its `marked-choice-segment` unit can contain several
  choices or inter-trial feedback. Interpret it as an input intervention.
- **Track S covers four tasks**, all reinforcement-learning or exploration
  paradigms, with one fitted parameter each. It does not stand in for the
  75-task grid and it is not a full phenotype vector.
- **Track S environments are yoked**: the reward schedule comes from the human
  CSV, so the model is not running in a freshly sampled environment.
- **The legal-option set is a lower bound** — the union of `<<...>>` responses
  within a session. Coverage is reported per task; tasks with very large option
  spaces do not report the option-level distribution at all.
- **`enkavi2019gonogo` is degenerate** and excluded: humans produce no `<<>>`
  marker when withholding a press, so every scored position has a one-element
  legal set and any model "hits" it.
- **All results are runtime NF4**, and this cluster physically cannot hold a 70B
  BF16 anchor (140 GB against 4×32 GB).

---

## What Is Out of Scope

Benchmark v1:

- P500 — designed and then parked; the design text is retained but not executed
- Noise-ceiling estimation
- Centaur as a denominator; it is a reference line only
- Merged-weight 70B — deployment damage is measured and avoidable; always
  4-bit base plus the official adapter
- Reasoning / thinking models — teacher-forced scoring has no single definition
  when the answer is conditioned on a self-sampled chain of thought
- RT as a prediction target
- A cross-task total score as a claim
- Refactoring — get results first; restructure once the code has mass

Permanently:

- General-purpose machine learning infrastructure
- A generic data-contract or dataset-abstraction layer — removed deliberately;
  handle each dataset at the point of use
- Reproducing the published Centaur numbers
- Real-time or production inference

---

## Work Boundaries

Full-grid scoring runs are expensive. Check the registry in
`benchmark design.md` §10 for what is already complete and reuse cached results
rather than re-running.

Quantized results are never mixed into a column with published BF16 numbers.

Scientific logic that a second script would need goes in `src/mt/`, not in a
script. `scripts/` owns run configuration, file I/O, and output shaping only.

---

## Key Terminology

**Marked text** — the Psych-101 convention where the tokens a model is scored on
are wrapped in `<<...>>`. Owned by `mt.models.llm.supervision`.

**Choice** — one `<<...>>` span; one behavioral decision. It may tokenize into
several tokens, in which case its NLL is the **sum** over those tokens, so that
$e^{-c}$ is literally the probability of the string the human produced.

**Baseline** — a counting rule giving the non-neural floor: `uniform`,
`base_rate`, `sticky`, `bigram`, all prequential within a session.

**Track P / Track S** — teacher-forced prediction versus open-loop simulation.

**Foundation model** — a single model trained across cognitive tasks that
generalizes to unseen tasks and populations.

---

## Planned Order of Work

1. Benchmark v1 on Psych-101-test  ← current focus
2. Foundation model
