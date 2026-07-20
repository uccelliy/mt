Great practical question. Here's a structured recommendation for coding project docs that agents can reliably work from.

## The Core Principle

Docs need to be **unambiguous, not exhaustive**. The goal is zero guessing — the agent should never have to infer what you mean.

---

## Recommended Doc Structure for a Coding Project

### 1. `PROJECT.md` — The North Star

The single most important file. Covers:

- **What** the project does (1–2 paragraphs, plain language)
- **Why** it exists (the problem being solved)
- **Who** uses it
- **Key constraints** (performance, scale, security, budget)
- **What's out of scope** — this is often skipped but critical

### 2. `ARCHITECTURE.md` — How it's built

- Tech stack with **version numbers** (e.g. `Node 20, React 18.3, PostgreSQL 15`)
- Folder structure with purpose of each directory
- Data flow diagram or description
- Key design decisions and **why** you made them
- What patterns you follow (e.g. "we use repository pattern, no direct DB calls in controllers")

### 3. `CONVENTIONS.md` — How to write code here

This is the **biggest gap** most projects have. Cover:

```
Naming:
- Files:        kebab-case       (user-profile.ts)
- Components:   PascalCase       (UserProfile.tsx)
- Functions:    camelCase        (getUserById)
- Constants:    UPPER_SNAKE      (MAX_RETRY_COUNT)
- DB tables:    snake_case       (user_profiles)
- DB columns:   snake_case       (created_at)
- CSS classes:  BEM or utility   (depends on your stack)

Structure:
- One component per file
- Tests co-located: user-profile.test.ts next to user-profile.ts
- Shared types in /types, never inline in components
```

### 4. `TASKS.md` or per-feature specs — What to build

For each feature or task an agent will work on:

- **Input / Output** clearly defined
- **Acceptance criteria** (numbered, checkable)
- **Edge cases** explicitly listed
- **What NOT to do** (anti-patterns, things to avoid)
- Links to relevant existing files to modify

### 5. `ERRORS.md` — Known issues and gotchas

- Known bugs and their workarounds
- Third-party quirks (e.g. "Library X has a bug with Y, use Z instead")
- Things that look wrong but are intentional

---

## Naming Convention for the Docs Themselves

```
/docs
  PROJECT.md          ← vision, scope, constraints
  ARCHITECTURE.md     ← stack, structure, patterns
  CONVENTIONS.md      ← naming, formatting, style rules
  GLOSSARY.md         ← domain terms defined precisely
  ERRORS.md           ← gotchas, known issues
  /features
    auth.md           ← spec for authentication feature
    payments.md       ← spec for payments feature
  /agents
    agent-rules.md    ← rules specific to AI agents
    context.md        ← what the agent should always load first
```

---

## The `agent-rules.md` File — Often Overlooked

This is a file **written specifically for the AI**, not humans. Include:

- "Always check existing utils before writing new ones"
- "Never modify migrations, create new ones only"
- "Run tests after every change"
- "If unsure about scope, stop and ask — don't guess"
- "Preferred libraries for X" (e.g. "use `date-fns`, not `moment`")
- What files are **off-limits**

---

## The Quality Bar by Task Type

|Task Complexity|Minimum Docs Needed|
|---|---|
|Bug fix|File location + reproduction steps + expected behavior|
|New function|Input/output types + edge cases + where it lives|
|New feature|Full feature spec + architecture context|
|Refactor|Current behavior + target behavior + what must not change|
|New project|All 5 doc types above|

---

## The Single Best Test

> _"Could a skilled developer who has never seen this project complete this task correctly using only these docs?"_

If yes — your docs are agent-ready. If no — find the gap and fill it. Agents are essentially that developer, except they won't ask follow-up questions unless you tell them to.


OK, this part is very important for me. current codebase already used a lot AI agent to generate codes but its code style is hugely differently from me. It is very hard for me to read. From when to break a line to how to order functions. A brief knowledge of my style. when I define params for a function I won't break each line for each params, otherwise I will write until the line is full for a window. And I like put util functions at last of a file. It is because I used to use C++ and C# more frequently. The main princple is I want to mimic a mature code style like a very old school developer and I want to make my code is easy for everyone(first is me of course). You can specific some styles questions letting me choose from them.

Great place to start — and the right instinct. AI agents default to overly "modern" Python style (black-formatted, one-param-per-line, heavy docstrings) which is very different from a seasoned systems developer's style.

Before I draft anything, let me ask the most impactful questions first — these decisions affect every single file:

Q: Docstring style — how much documentation do you want on functions? A: One-line only — if the name is clear, no docstring needed

Q: Type hints — how strict? A: Only where the type is non-obvious

Q: Within a file, what is your preferred order of sections? A: Ranked: 1. Imports, 2. Constants / module-level variables, 3. Main class(es), 4. Standalone functions, 5. Utility / helper functions

Good clear answers. Now the next batch — these cover the decisions that affect line-by-line readability most directly:

Q: Line length limit — what is your window width? A: 80 characters (classic terminal width)

Q: String quotes — which do you prefer? A: Double for strings, single for chars/keys

Q: Comments — where do they go? A: Above for blocks, inline for quick notes

Good, one final round — these two affect the visual rhythm of code most directly:

Q: Blank lines — how much vertical space between functions and methods? A: One blank line everywhere — keep it compact

Q: Private / internal functions — how do you signal them? A: Single underscore for private methods, no prefix for private module functions