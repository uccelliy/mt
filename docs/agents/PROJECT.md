# PROJECT.md

## What This Project Is

`mt` is a research platform for auditing what language models actually learn
about human behavior. It targets one problem:

- It is unclear how much of an LLM's fit to human behavior comes from behavioral
  finetuning versus from general pretraining plus in-context learning

Claims about "foundation models of cognition" rest on aggregate held-out
likelihood numbers. Those numbers do not separate the two sources, so the
headline claim has never actually been tested.

**This project is not a replication of Centaur.** Reproducing the paper's
numbers is explicitly not the goal — the runs here are 4-bit and use a
research-grade full-context protocol, and they are never mixed into a column
with published BF16 results. Centaur is the *subject* of the audit, not a
target to match.

---

## End Goal

**1. Controlled LLM evaluation protocol**
A scoring setup that isolates where an LLM's behavioral fit actually comes
from — context length, behavioral finetuning, task-identity information — with
a matched base-model control at every step, and count-based baselines as the
non-neural floor.

**2. Foundation model**
A model trained across multiple cognitive tasks that generalizes to new tasks
and populations — held to the same controlled protocol.

The protocol makes the comparison honest; the foundation model is what the
setup exists to test.

Classical cognitive-model baselines (Rescorla-Wagner, Prospect Theory,
Generalized Context Model, etc.) were part of an earlier plan and have been
removed — they never got wired into the evaluation path, and the count-based
baselines in `mt.models.baselines` serve the "non-neural floor" role that the
comparison actually needs. Recover them from git history if that changes.

---

## Current Stage

Centaur evaluation, phase one: LLM inference-side scoring only — no finetuning
of our own.

**What is active:** `src/mt/models/llm/` and `src/mt/evaluation/` — marked-text
supervision, transcript scoring, context windows
**What is supporting:** `src/mt/models/baselines/` — count-based sequence and
population baselines

See `ARCHITECTURE.md` for detailed stability status of each module.

---

## Immediate Focus

Finishing the Centaur evaluation experiment grid.

The central question: on Psych-101 transcripts, how much of Centaur's advantage
over a matched Llama-3.1-8B base model survives once context length is
controlled for? Current answer at 8B/NF4 is "very little, and it is
concentrated at trial 0" — the remaining work is to establish why.

Experiment definitions and status live in `docs/centaur-eval-design.md` §12 and
`docs/centaur-eval-handoff.md` §1. Read the handoff before starting a run.

## Current Work Boundaries

Scoring runs over the full 75-experiment grid are expensive. Check the handoff
for what is already complete and reuse cached results rather than re-running.

Quantized (4-bit) results are never mixed into a column with published
BF16 Centaur numbers. They are a compatible protocol, not a replication.

Scientific logic that a second script would need goes in `src/mt/`, not in a
script. `scripts/` owns run configuration, file I/O, and output shaping only.

---

## What Is Out of Scope

- General-purpose machine learning infrastructure
- A generic data-contract or dataset-abstraction layer — removed deliberately;
  handle each dataset at the point of use
- Classical cognitive-model implementations — removed; see End Goal
- Reproducing the published Centaur numbers
- Real-time or production inference

---

## Key Terminology

**Marked text** — The Psych-101 transcript convention where the tokens a model
is scored on are wrapped in `<<...>>`. Owned by `mt.models.llm.supervision`.

**Baseline** — A count-based model giving the non-neural floor for a task.
Transcript-space (`uniform`, `base_rate`, `sticky`, `bigram`) and
canonical-space population variants live in `mt.models.baselines`.

**Foundation model** — A single model trained across multiple cognitive tasks
that can generalize to unseen tasks and populations.

**Paradigm** — A class of cognitive experiment (e.g. reinforcement learning,
categorization, decision under risk).

## Planned Order of Work

1. Centaur evaluation experiment grid  ← current focus
2. Foundation model
