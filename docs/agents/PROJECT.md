# PROJECT.md

## What This Project Is

`mt` develops trial-native behavioral models and measurements on the project's
own P500 tasks. The program separates input representation, response dynamics,
and cross-model executive-control structure so each has an independent scientific
question and evidence standard.

The current program contains three papers:

1. **Trial-native state vs. language serialization:** does compressing structured
   trials into natural language change choice learning, sample/compute efficiency,
   and transfer to unseen instruments or task families?
2. **Shared response dynamics:** can one model jointly explain choice, RT, and
   nonresponse, and does shared timing structure transfer across participants and
   tasks?
3. **Unity and diversity of executive control across language models:** do
   differences among relatively independent checkpoints follow human-inspired EF
   measurement structure, and do those factors generalize across lineages and
   predict external interactive abilities?

The design of record is **`docs/benchmark design.md`**. Psych-101/Centaur work is
frozen under `docs/archive/` and is not part of the new experiment matrix.
Paper-level designs are `docs/p500-structured-input-paper.md`,
`docs/p500-choice-rt-paper.md`, and
`docs/p500-llm-executive-function-paper.md`.

## Current Stage

P500 conversion currently covers 100 participants, 21 task families, 62
instruments, 777,403 trials, and 934,410 choice targets. The raw data contain
structured stimulus/option/response fields and RT, but the existing standard
table and JSONL were built for natural-language transcript training. Formal model
runs have not started.

The immediate stage is **P500 Trial-State v1**, not model scaling:

- preserve typed stimulus, candidate-action, condition, and adaptive-state fields;
- define the minimal sufficient state for every family;
- retain choice-level `trial_uid`, run, trial, and step mappings;
- implement participant-group splits with no leakage;
- validate RT observation/censoring semantics and per-family coverage;
- provide equal-information text and structured views from one canonical record.

In parallel, Paper B performs its per-family RT/event measurement audit, and
Paper C enters **C0 EF measurement audit**: map tasks to distinct inhibition,
updating, and shifting paradigms; verify reconstructable no-leak stimuli; and
build a small agent-facing task-execution smoke suite. This does not authorize a
full model roster yet.

## History Condition

Reducing history is an optional Paper A diagnostic, not the main contribution. It may
remove some prior cross-trial human choices, RTs, and outcomes, but never removes
task state required for the current response: n-back stimulus memory, within-trial
sequence prefixes, adaptive difficulty, current phase, stimulus, or the legal
action set. Any representation×history claim crosses the same full and
short-history conditions with both text and structured inputs; otherwise it is
only a model robustness check. Paper B fixes one history budget.

The Paper A/B model is a population-level predictor on held-out participants.
Participant ID is an index for splitting, not an input. Leave-one-task-family-out
is the later foundation-model test; until it succeeds, describe the result as a
multitask P500 model rather than a general foundation model of cognition.

## Paper Experiments

Paper A is choice-only. It compares two frozen equal-information text renderers
with typed structured state under the same candidate-action head, split, targets,
updates, and seeds. It includes data-efficiency, held-out instrument/family, a
structured ablation, and a targeted second-architecture replication. It owns and
then freezes Trial-State v1 and the structured interface.

Paper B reuses that frozen interface. Its core conditions are structured
choice-only, RT/event-only, joint choice+RT/event, and shuffled RT/event. It must
model RT/event distributions, missing/censoring and no-response semantics, and
test shared versus independent timing models in low-data or held-out tasks.
Text+RT is not a required condition.

Paper C does not train the behavioral model from Papers A/B. It administers a
separate no-leak, compact P500 executive-function battery to frozen external
checkpoints. It preregisters unitary, correlated three-factor, revised
Unity–Diversity, and method-factor measurement models; the primary estimator is
trial-level confirmatory MIRT/GLLVM. C0 uses 6–10 checkpoints for interface smoke,
C1 uses roughly 30–50 checkpoints for measurement development, and C2 begins only
after construct, reliability, ceiling/floor, lineage, and Monte Carlo power gates.
Its primary roster uses immutable, text-capable checkpoints and losslessly
symbolizable tasks; true-visual VLM tasks are a separate multi-group sensitivity.

## Conclusion Boundaries

- Text-versus-structured claims require identical underlying information, action
  encoding, targets, splits, and matched training budgets.
- Paper A claims require directionally consistent results across two renderers
  and a task-transfer layer; a single-template gain is a renderer effect.
- A joint RT choice gain must beat within-task/condition shuffled-RT and matched
  auxiliary controls before being attributed to RT's behavioral content.
- RT coverage and timing semantics differ by family; no global claim may hide that.
- Paper B can succeed without a choice gain only if RT/event distributions are
  calibrated and shared timing dynamics transfer to low-data or unseen tasks.
- Paper C requires at least three no-leak, independent paradigms per EF construct;
  variants or difficulty levels of one instrument do not count as independent
  indicators.
- In Paper C, checkpoint is the measured unit and base lineage is the dependency
  and external-validation cluster. Prompt/seed/quantization repeats do not
  increase sample size.
- Similar human/LLM factors describe performance covariance, not shared internal
  mechanisms. Claim shared measurement only to the level supported by
  common indicators and linked scales; different modalities support congruence,
  not automatic metric invariance.
- Paper C leaderboard analysis is criterion validity: it must propagate factor
  uncertainty, control common ability/scale/training/lineage, and use
  lineage-held-out prediction. Ordinary correlations are descriptive.
- A short-history effect is a context-robustness diagnostic, not the main model
  contribution or evidence for a human cognitive mechanism.
- Held-out participants do not substitute for held-out task families.
- P500-only results do not establish transfer across all cognition.
- Local NF4 Centaur results are historical motivation and are never mixed with
  published BF16 numbers.

## What Is Out of Scope Now

- New Psych-101 or Centaur runs and analyses;
- the old LLM roster, 70B, Track S, E6, A8–A19, and context-window sweeps;
- full-session prompt engineering as the primary modeling path;
- a free-generation output interface in the main comparison;
- a single scalar combining choice and RT;
- a full classical cognitive-model suite;
- a full Papers A/B model roster or simultaneous ICL/LoRA/full-finetuning matrix;
- Papers A/B human-behavior open-loop simulation, internal-mechanism claims, or
  a generic ML framework;
- a new nonlinear CFA/VAE/generative psychometrics method; it remains a distant
  follow-up unless the confirmatory generalized model shows stable nonlinear
  misfit and a separate methods project is explicitly opened.

Paper C's full roster remains excluded before its C0/C1 measurement gates. Its
frozen C2 roster is in scope once those gates and the power plan pass.
Paper C's controlled task administration, including model-driven adaptive state,
is not the excluded Papers A/B open-loop simulation.

## Work Order

1. Track and verify the P500 converter/schema/QA while keeping raw private data
   ignored.
2. Build Trial-State v1 and pass sufficient-state, identity, split, and RT gates.
3. In parallel, complete Paper B's RT audit and Paper C's task→construct,
   stimulus-availability, leakage, and modality audit.
4. Validate Paper A baselines and overfit a tiny sample; build Paper C's no-leak
   agent battery and pass a 6–10-checkpoint C0 smoke.
5. Run Paper A on held-out participants, data fractions, and task-transfer folds;
   run Paper C's 30–50-checkpoint C1 measurement pilot without making a final CFA
   claim.
6. Freeze the structured representation, action head, history budget, and main
   training configuration using train/validation only.
7. Run Paper B choice-only, RT-only, joint, and shuffled controls, followed by
   low-data/task-transfer and preregistered behavioral-effect analyses.
8. If C1 passes, use Monte Carlo power to freeze Paper C's items, lineage-aware C2
   roster, candidate measurement models, and external outcomes; then run the
   confirmatory measurement model, justified human–LLM comparison, and
   lineage-held-out external-validity analyses in that order.
9. Add the Paper A history diagnostic only after its representation result is stable.

Do not expand any paper beyond its registered gates: Papers A/B require
reproducible structured-input and RT evidence, while Paper C first requires a
reliable no-leak measurement battery.

## Engineering Boundaries

Dataset-specific extraction remains explicit by P500 family. The shared layer is
the model-facing trial-state schema, not a universal data framework. Scientific
logic reused by multiple runners belongs in `src/mt/`; scripts own run
configuration, file I/O, and output shaping. Raw private data, large derived
tables, and checkpoints remain gitignored; converter code, schema docs, QA tests,
and compact summaries must be tracked.

Paper C's task-execution environment is distinct from the human-response renderer:
some current text views expose derived answer information or omit actual visual
state. Its tracked compact artifacts will include the construct/leakage audit,
indicator specification, lineage registry, item/prompt manifests, and QA
summaries. Raw multi-model responses remain gitignored.
