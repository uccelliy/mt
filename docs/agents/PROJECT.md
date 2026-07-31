# PROJECT.md

## What This Project Is

`mt` is a research platform for systematic comparison of human cognition models.
It targets three connected problems that currently prevent the cognitive science
and AI communities from making cumulative progress:

- It is unclear how much of an LLM's fit to human behavior comes from behavioral
  finetuning versus from general pretraining plus in-context learning
- Existing cognitive model implementations are difficult to reproduce and benchmark
- LLMs have never been systematically evaluated against classical cognitive models

These problems are treated as a single system: without a controlled evaluation
protocol, claims about "foundation models of cognition" cannot be checked;
without reproducible baselines, comparisons are meaningless; without a
foundation model, there is nothing to compare against.

---

## End Goal

A complete system with three components:

**1. Controlled LLM evaluation protocol**
A scoring setup that isolates where an LLM's behavioral fit actually comes
from — context length, behavioral finetuning, task-identity information — with
a matched base-model control at every step.

**2. Reproducible baseline suite**
Clean, citable implementations of classical cognitive models
(Rescorla-Wagner, Prospect Theory, Generalized Context Model, etc.)
that the community can use as benchmarks.

**3. Foundation model**
A model trained across multiple cognitive tasks that generalizes to new tasks
and populations — evaluated systematically against the baseline suite.

These three components are designed to be used together. The evaluation
protocol makes the comparison honest. The baselines give the LLM numbers a
scale to be read against. The foundation model is what the whole setup exists
to test.

---

## Current Stage

Centaur evaluation, phase one: LLM inference-side scoring only — no finetuning
of our own, no cognitive-model replication. Core cognitive model
implementations are working but are not on the current path.

**What is stable:** `src/mt/models/` — cognitive models and baselines
**What is active:** `src/mt/models/llm/` and `src/mt/evaluation/` — transcript
scoring, context windows, sequence baselines
**What is legacy:** `src/mt/training/` and the dataframe model contracts

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

`src/mt/training/` and the dataframe-based model contracts are not on the
current path. Do not extend them, and do not treat them as constraints on
evaluation code.

---

## What Is Out of Scope

- General-purpose machine learning infrastructure
- A generic data-contract or dataset-abstraction layer — removed deliberately;
  handle each dataset at the point of use
- Real-time or production inference

---

## Key Terminology

**Cognitive model** — A mathematical model of human decision-making or
learning behavior (e.g. Rescorla-Wagner, Prospect Theory).

**Baseline** — A community-standard cognitive model implementation used as
a reproducible reference for comparison.

**Marked text** — The Psych-101 transcript convention where the tokens a model
is scored on are wrapped in `<<...>>`. Owned by `mt.models.llm.supervision`.

**Foundation model** — A single model trained across multiple cognitive tasks
that can generalize to unseen tasks and populations.

**Paradigm** — A class of cognitive experiment (e.g. reinforcement learning,
categorization, decision under risk).

## Planned Order of Work

1. Centaur evaluation experiment grid  ← current focus
2. Cognitive-model baselines on the same tasks, for a common scale
3. Training pipeline rework
4. Foundation model
