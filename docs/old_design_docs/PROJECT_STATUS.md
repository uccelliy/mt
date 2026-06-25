# Project Status — 2026-06-10

Snapshot of where `mt` is, what we did this session, and what's open.

## Context

- **Researcher:** PhD year 1, AI × computational cognitive science
- **Project:** `mt` — foundation behavioral cognitive model, structured trial data, building on Centaur (Binz & Schulz 2025)
- **Supervisor:** endorsed comparing against Centaur as the baseline

## What `mt` already has

- `src/mt/data/` — data contract, loading, transforms, splitting, views, tensors
- `src/mt/models/cognitive/` — 13 classical cognitive baselines (Centaur supplement, formula-first)
- `src/mt/evaluation/` — generalization, neural alignment, baseline comparison
- `src/mt/models/llm/` + `experiments/llm/` — LLM training scaffold
- RT modeling: TODO in `README.md`
- **Current state:** framework-only, design phase, no end-to-end training yet

## Data landscape (updated this session)

| Domain | Source | Status |
|---|---|---|
| **Decision-making** (bandit, prospect theory, two-step, RL) | Centaur's Psych-101 (Apache 2.0; HuggingFace `marcelbinz/Psych-101`) | Public, ready to use |
| **Basic cognitive abilities** (perception, attention, memory, cognitive control, etc.) | **User's own experimental data** | Complementary to centaur |

**Framing (revised this session):**
> Centaur is the "decision-making foundation model". `mt` targets the **complete cognitive foundation model** — decision + basic cognition, with cross-domain transfer as the unique contribution.

## What we did this session

### 1. `.harness/` agent team + `AGENTS.md`

Created a 4-rein team aligned to the project's 3 core modules. Lives in the repo, awaiting user review and commit.

| Path | Role |
|---|---|
| `AGENTS.md` | Root agent config (open-agents.md standard) |
| `.harness/agent.md` | Harness (orchestrator) |
| `.harness/reins/developer/agent.md` | General dev, utils, glue |
| `.harness/reins/tester/agent.md` | pytest/ruff/smoke/contract gates |
| `.harness/reins/data-expert/agent.md` | `src/mt/data/` — **dual-schema contract** (decision-trial + basic-cognition-trial) |
| `.harness/reins/llm-expert/agent.md` | LLM training, cognitive baselines, cross-domain transfer, RT |

These files were updated mid-session when the user clarified they have their own basic-cognition data — the framing changed from "beyond centaur" to "complete cognitive foundation model".

### 2. Deep-research report

- **Path:** `/tmp/mavis-deep-research/20260610-122843-structured-trial-data-vs-centaur/turn_001/final.md`
- **Format:** 10-section research report, ~10 pages worth
- **Key findings:**
  - Centaur's stated limitation: no RT prediction (a clean `mt` contribution)
  - Centaur model + dataset are public (Apache 2.0)
  - **NSO 2026 critique (Zhejiang Univ)** only applies to NL encoding — strongest methodological argument for `mt`
  - **Orr et al. 2025 "Not Even Wrong"** — framing decision (cognitive model vs behavior model)
  - Tokenization of cognitive trial data is an open methodological problem
- **3 research directions proposed** (originally for centaur-only comparison; now extendable with cross-domain transfer):
  - **A (primary):** Structured vs NL head-to-head on Psych-101 + cross-domain transfer
  - **B (clean increment):** RT as 2nd output head
  - **C (defensive):** "Not Even Wrong" probe suite

## Open questions (blocking full research design)

- [ ] **Which specific basic-cognition tasks does the user have?** Memory? Attention? Perception? Mix? (Decides tokenization + which classical cognitive baselines to add + framing of cross-domain transfer)
- [ ] **User data scale:** #participants, #trials per participant
- [ ] **User data schema:** which fields per trial (stimulus, response, accuracy, RT, condition parameters)

## Suggested next steps (after user clarifies open questions)

1. User clarifies basic-cognition task types + scale
2. Re-design with full info: pick primary framing (A extended, B, C, or combination)
3. Integrate Psych-101 into `mt` data contract
4. Design structured tokenization that handles both schemas
5. Run pilot: small-scale structured vs NL on a Psych-101 subset
6. Paper outline draft for discussion with supervisor

## Environment

- macOS, zsh
- Node v26.3.0 via nvm (`~/.zshrc` already loads nvm)
- `mavis` CLI works; daemon on port 15321 (MiniMax Code.app, electron)
- Working directory: `/Users/ruochen.yin/wkspace/mt/`

## References

- Binz et al. 2025, *Nature* — Centaur. [arXiv 2410.20268]
- Orr et al. 2025, "Not Even Wrong". [arXiv 2510.03311]
- Schulze Buschoff et al. 2025, visual cognition follow-up. [arXiv 2502.15678]
- Psych-101: <https://huggingface.co/datasets/marcelbinz/Psych-101>
- NSO 2026 critique (Zhejiang Univ, Jan 26 2026)
- See report §9 for full bibliography
