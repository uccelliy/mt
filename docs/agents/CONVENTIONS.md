# CONVENTIONS.md

## 1. File Structure

Every file follows this order, top to bottom:

1. `from __future__ import annotations` (always first if present)
2. Imports (see Import Rules below)
3. Constants and module-level variables
4. Main class(es)
5. Standalone functions
6. Utility / helper functions

Exception: a constant whose value must instantiate a class defined in the same
file may appear immediately after that class. All constants without that
dependency remain above the main classes.

---

## 2. Imports

Group imports in this order, separated by one blank line:

1. Standard library
2. Third-party packages
3. Local modules (`mt.*`)

Within each group, sort alphabetically.

```python
# correct
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mt.models.llm.supervision import find_target_spans
```

Never use wildcard imports (`from module import *`).
Never import inside a function unless necessary to avoid circular imports.

---

## 3. Naming

| Target | Convention | Example |
|---|---|---|
| Variables | snake_case | `loss_history` |
| Functions | snake_case | `find_target_spans` |
| Classes | PascalCase | `BaselineTables` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT` |
| Private methods | `_single_underscore` | `_preprocess_batch` |
| Private module functions | no prefix | `load_saved_model` |
| Files | snake_case | `rescorla_wagner.py` |

---

## 4. Line Length and Line Breaking

- Hard limit: **80 characters**
- Fill the line to the limit before breaking — never break early
- When a function signature must wrap, continue parameters on the
  same line until 80 characters, then break at a logical boundary

```python
# correct — fill the line, break only when needed
def fit(self, train_data: dict[str, Any], *, save_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None) -> TrainingResult:

# wrong — one param per line (do not do this)
def fit(
    self,
    train_data: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrainingResult:
```

---

## 5. Type Hints

Use type hints only where the type is non-obvious from context.

```python
# correct — non-obvious, hint is useful
def load_parameters(self, path, *, map_location=None,
                    strict: bool = True):

# wrong — obvious from name and usage, hint adds noise
def get_name(self) -> str:
    return self.name
```

Use `|` union syntax, not `Optional` or `Union`:
```python
# correct
map_location: str | None = None

# wrong
map_location: Optional[str] = None
```

---

## 6. Strings

- Double quotes for strings: `"hello"`
- Single quotes for single characters or dict keys: `payload['config']`

```python
# correct
raise NotImplementedError(f"{self.__class__.__name__} must implement score_batch().")
config = payload['config']

# wrong
raise NotImplementedError(f'{self.__class__.__name__} must implement score_batch().')
```

---

## 7. Comments

- Block comments go **above** the code they describe
- Inline comments go on the **same line** for quick notes
- Never state what the code does — state **why**

```python
# correct — explains why, placed above
# schedulefree optimizer also requires mode switching
if hasattr(self.optimizer, "train"):
    self.optimizer.train()

loss_history.append(float(loss.detach().item()))  # detach before converting

# wrong — states the obvious
# append loss to history
loss_history.append(float(loss.detach().item()))
```

---

## 8. Blank Lines

- **One blank line** everywhere — between methods, between functions,
  between logical blocks inside a function
- Never use two blank lines
- Never use blank lines just to pad a file

```python
# correct
def fit(self, train_data):
    self.model.train()
    self.optimizer.train()

    for _ in range(self.num_iter):
        loss = self._step(train_data)

    return loss

def evaluate(self, eval_data):
    ...

# wrong — two blank lines between methods
def fit(self, train_data):
    ...


def evaluate(self, eval_data):
    ...
```

---

## 9. Docstrings

- One-line docstring only, if the function name is not self-explanatory
- End with a period
- No parameter or return documentation in docstrings
- No docstring at all if the name is clear enough

```python
# correct — name is clear, no docstring needed
def zero_grad(self):
    self.optimizer.zero_grad()

# correct — one-liner adds useful context
def parameter_payload(self, ...):
    """Build a serializable parameter payload."""

# wrong — over-documented
def find_target_spans(text):
    """
    Return character spans between supervision markers.

    Args:
        text: the transcript
    Returns:
        list of (start, end) tuples
    """
```

---

## 10. Error Handling

- Raise `NotImplementedError` for unimplemented abstract methods
- Include the class name and method name in the message
- Use `ValueError` for bad inputs, `RuntimeError` for unexpected
  internal states

```python
# correct
raise NotImplementedError(f"{self.__class__.__name__} must implement score_batch().")
raise ValueError(f"Unknown reduction mode: {mode!r}")
```

## 11. Design Principles

Each unit has one responsibility, and that responsibility is obvious from its
name. If describing what a component does requires "and", split it.

The project uses three structural patterns — follow them, do not deviate:

- **Pipeline** (`preprocess_marked_text()` runs find spans → tokenize → mask
  labels → truncate) — the public function owns the fixed order; each stage
  stays independently callable. One step, one transform.
- **Fit/score split** (`fit_tables()` then `score_sequence()`;
  `fit_counts()` then `score_codes()`) — fitting builds state from training
  data and returns it; scoring is a pure function of that state plus one
  sequence. Never fit inside a scoring loop, and never let held-out data reach
  a fit.
- **Strategy** (`truncation="head"|"tail"`, `source_kind="hf"|"file"`) —
  variation points are named parameters on a stable interface. New strategies
  never change the interface.

Transform functions must be pure: same input, same output, no side effects,
no mutation of the input. `fit()` is the only place state is built — document
what state it produces. `transform()` must always operate on a copy.

Every component must be testable in isolation without other components being
correct first. If that is not possible, the boundary is wrong.

Validate inputs at entry points. Never pass invalid state downstream. Errors
surface at their source with a message naming what is wrong and where.
A silent fallback is never acceptable.
