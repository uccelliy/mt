# Conventions

## 1. Purpose

Core principle: code in this repository should be easy to read first.

- Prefer a mature, old-school style over generic AI-generated style
- Keep code compact and readable in a terminal window
- Make code easy for the project owner to read first, then for others
- Do not add style rules that are not listed in this file

---

## 2. Line Length

Core principle: prefer 80 columns; 100 columns is the current hard limit.

- Aim to keep normal code at or below 80 characters
- Never exceed the current Ruff limit of 100 characters
- Keep lines compact when the result is still readable

---

## 3. Function Parameters

Core principle: do not split parameters one per line by default.

- Keep function parameters on the same line until the line is full
- Break parameters only when the line would become too long
- Prefer compact signatures over vertically expanded signatures

Preferred:

```python
def build_trial_view(df, contract, split_name="train"):
    ...
```

Avoid:

```python
def build_trial_view(
    df,
    contract,
    split_name="train",
):
    ...
```

---

## 4. File Order

Core principle: files should go from main structure to helper details.

Use this order inside Python files:

1. Imports
2. Constants and module-level variables
3. Main class or classes
4. Standalone functions
5. Utility and helper functions

Rules:

- Put utility and helper functions at the bottom of the file
- Keep the main class or main public behavior near the top

---

## 5. Docstrings

Core principle: use docstrings only when they help readability.

- If the name is clear, no docstring is needed
- Prefer one-line docstrings
- Do not add long docstrings by default

---

## 6. Type Hints

Core principle: type hints should clarify non-obvious code.

- Add type hints only where the type is non-obvious
- Do not annotate everything by default

---

## 7. Comments

Core principle: comments should explain code without cluttering it.

- Put comments above a block for block-level explanations
- Use inline comments only for quick notes

---

## 8. Blank Lines

Core principle: keep code compact.

- Use one blank line everywhere
- Do not create large vertical gaps between functions or methods

---

## 9. Private Code

Core principle: mark private class methods, but keep module helpers plain.

- Use a single underscore for private methods
- Do not use a prefix for private module functions

---

## 10. Strings

Core principle: use quote style consistently.

- Use double quotes for strings
- Use single quotes for chars and keys
