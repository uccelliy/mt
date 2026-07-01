# PROJECT.md

## What This Project Is

`mt` is a research platform for systematic comparison of human cognition models.
It targets three connected problems that currently prevent the cognitive science
and AI communities from making cumulative progress:

- No unified way to compare cognitive models across tasks and datasets
- Existing cognitive model implementations are difficult to reproduce and benchmark
- LLMs have never been systematically evaluated against classical cognitive models

These three problems are treated as a single system: without a shared data
standard, benchmarking is impossible; without reproducible baselines, comparisons
are meaningless; without a foundation model, there is nothing to compare against.

---

## End Goal

A complete system with three components:

**1. Data contract standard**
A shared format for cognitive datasets that enables any model to be evaluated
on any dataset without custom preprocessing per experiment.

**2. Reproducible baseline suite**
Clean, citable implementations of classical cognitive models
(Rescorla-Wagner, Prospect Theory, Generalized Context Model, etc.)
that the community can use as benchmarks.

**3. Foundation model**
A model trained across multiple cognitive tasks that generalizes to new tasks
and populations — evaluated systematically against the baseline suite.

These three components are designed to be used together. The data standard
makes the baselines reproducible. The baselines make the foundation model
comparable. The foundation model gives the data standard a reason to exist.

---

## Current Stage

Early exploration. Core cognitive model implementations are working.
The immediate focus is designing the data contract standard — this is the
foundation everything else depends on.

**What is stable:** `src/mt/models/` — cognitive models and baselines
**What is in progress:** `src/mt/data/` — data contract system
**What is pending:** evaluation pipeline, training pipeline, CLI

See `ARCHITECTURE.md` for detailed stability status of each module.

---

## Immediate Focus

Designing and implementing the data contract standard in `src/mt/data/`.

The contract must solve: how do datasets from different cognitive tasks
and paradigms get represented in a unified format that any model can consume
without task-specific preprocessing code?

No other component should be extended until the data contract design is settled.

## Current Work Boundaries

The CLI is unstable and is not a compatibility target during the data-contract
redesign. Do not preserve or extend CLI behavior while implementing replacement
data modules unless a task explicitly asks for CLI migration.

`src/mt/evaluation/baseline_comparison/` is also unstable. Its scripts are not
acceptance criteria for replacement data modules and must not constrain the new
data API. Migrate or repair them later as part of the approved evaluation
refactor, not during the current module-by-module data work.

---

## What Is Out of Scope

- General-purpose machine learning infrastructure
- Support for non-cognitive datasets
- Real-time or production inference
- Any dataset that does not follow the data contract standard

---

## Key Terminology

**Cognitive model** — A mathematical model of human decision-making or
learning behavior (e.g. Rescorla-Wagner, Prospect Theory).

**Baseline** — A community-standard cognitive model implementation used as
a reproducible reference for comparison.

**Data contract** — A schema that specifies exactly which columns a dataset
must provide and how they map to model inputs.

**Foundation model** — A single model trained across multiple cognitive tasks
that can generalize to unseen tasks and populations.

**Paradigm** — A class of cognitive experiment (e.g. reinforcement learning,
categorization, decision under risk).

## Planned Order of Work

1. Data contract standard  ← current focus
2. Evaluation pipeline refactor
3. Training pipeline refactor
4. Foundation model
