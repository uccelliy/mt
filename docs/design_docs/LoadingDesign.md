# Code Design: Raw Data Loading

**Status:** Implemented and verified
**Target:** `src/mt/data/_loading.py`
**Date:** 2026-07-01

---

## Purpose

`_loading.py` converts one supported raw data source into an independent
`pandas.DataFrame` and normalizes its column labels. The DataFrame is an
internal staging representation for the later mapping and adaptation stages.

Loading is intentionally unaware of the canonical field registry. It does not
map raw columns, add canonical defaults, filter rows, validate trial identity,
assemble trials, or construct a `TrialCollection`.

The stage boundary is:

```text
CSV, parquet, HuggingFace Dataset, or pandas DataFrame
  -> load(source)
  -> independent raw pandas DataFrame
```

---

## Supported Sources

The first implementation supports exactly four source forms:

| Source | Loading behavior |
|---|---|
| `str` or `Path` ending in `.csv` | `pandas.read_csv()` |
| `str` or `Path` ending in `.parquet` | `pandas.read_parquet()` |
| `pandas.DataFrame` | Return a deep DataFrame copy |
| `datasets.Dataset` | Convert with `Dataset.to_pandas()` |

File suffix matching is case-insensitive. The final suffix must be exactly
`.csv` or `.parquet`; compressed CSV names such as `.csv.gz` are not supported
in the first implementation.

Strings and paths identify files or file-like URIs accepted by pandas. A
string is not interpreted as a HuggingFace dataset identifier. Users who need
a HuggingFace Hub dataset first load and select its split, then pass the
materialized `Dataset`:

```python
dataset = load_dataset("owner/name", split="train")
df = load(dataset)
```

`datasets.DatasetDict` is not accepted because it contains multiple splits
and therefore does not identify one tabular source. `datasets.IterableDataset`
is not accepted because streaming and lazy loading are out of scope. A caller
must select one split and materialize it as a `Dataset` or DataFrame.

JSON, JSONL, TSV, directories, file objects, iterators, and arbitrary mappings
are outside the new loading API. A caller that needs format-specific parsing
may perform it separately and pass the resulting DataFrame to `load()`.

---

## Public API

Define the accepted source type in `_loading.py`:

```python
DataSource = str | Path | pd.DataFrame | Dataset
```

The loading function is:

```python
def load(source: DataSource) -> pd.DataFrame:
    """Load a supported raw source into an independent DataFrame."""
```

Export `DataSource` and `load` from `mt.data`. The future `DataAdapter` uses
this function for its loading stage.

The first implementation does not accept `columns`, reader keyword arguments,
a split name, or a format override. All source columns must be available to
the mapping stage and to the adaptation report. Format-specific options remain
outside the uniform loading interface:

```python
raw = pd.read_csv(path, sep=";")
df = load(raw)
```

---

## Output Invariants

Every successful call returns a `pandas.DataFrame` with these guarantees:

1. The returned DataFrame is independent of a caller-owned DataFrame.
2. Raw row order and column order are preserved.
3. Every column label is normalized to a string.
4. A missing column label becomes the literal string `"None"`; every other
   label uses `str(label)`.
5. Column labels must be unique after normalization.
6. Loading does not add, remove, select, or reorder columns.
7. Loading does not reset or otherwise normalize a DataFrame index.
8. Loading does not validate canonical paths or required fields.
9. Loading does not reject empty data, null values, or duplicate trial
   identities.

Uniqueness is checked after normalization. Raw labels `1` and `"1"`, or
`None` and `"None"`, therefore collide and raise `ValueError`. This gives the
mapping stage one unambiguous string namespace for raw column lookup.

For a DataFrame source, independence is provided by
`source.copy(deep=True)`. Pandas deep copying does not recursively clone every
mutable Python object stored in an object-dtype cell; `_loading.py` does not
promise that stronger behavior. Later `TrialCollection` construction is
responsible for copying numpy-array slot values.

CSV and parquet readers already construct a new DataFrame. A HuggingFace
`Dataset` is Arrow-backed and `to_pandas()` constructs the staging DataFrame.
The implementation may make one final DataFrame copy if needed to keep the
independence invariant uniform, but it must not alter values or ordering.

---

## Dispatch Rules

Dispatch is based only on the source's explicit runtime type or file suffix:

1. A `pandas.DataFrame` is copied.
2. A `datasets.Dataset` is converted to pandas.
3. A `str` or `Path` is dispatched by its case-insensitive final suffix.
4. A `DatasetDict` or `IterableDataset` receives a targeted unsupported-source
   error.
5. Every other type is rejected.

There is no content sniffing, heuristic format inference, schema inspection,
or fallback from one reader to another. An extensionless HuggingFace dataset
identifier must not trigger a network request implicitly.

---

## Failure Semantics

`_loading.py` owns errors only when it can identify the problem at the loading
boundary.

### Unsupported source type

An unsupported Python object raises `TypeError` and names the received type
and supported source forms.

Example:

```text
Unsupported data source type: list. Expected str, Path, pandas.DataFrame, or
datasets.Dataset.
```

### Ambiguous or streaming HuggingFace source

A `DatasetDict` raises `TypeError` explaining that the caller must select one
split. An `IterableDataset` raises `TypeError` explaining that streaming input
must first be materialized.

### Unsupported file format

A string or path whose final suffix is not `.csv` or `.parquet` raises
`ValueError`. The message names the source and the two supported extensions.

Example:

```text
Unsupported data source format for 'trials.json'. Supported file extensions
are '.csv' and '.parquet'.
```

### Duplicate normalized column names

If two raw labels become the same string, loading raises `ValueError` naming
the duplicates. The caller must rename the source columns before adaptation.

### Reader failures

File-not-found, permission, parser, network-filesystem, and parquet-engine
errors are allowed to propagate from pandas or its backend. They must not be
wrapped in a generic loading error, because the original exception contains
the actionable source-level detail.

Missing canonical columns are not loading failures. `_mapping.py` and
`_adapter.py` report them later at the stage that owns those requirements.

`Dataset.to_pandas()` is expected to return a DataFrame because loading never
requests batched output. An unexpected iterator is an internal-state failure
and raises `RuntimeError`; this runtime check also narrows the third-party
union return type for static analysis.

---

## Dependency Rules

`pandas` and `datasets` are existing direct project dependencies. Parquet is
read through `pandas.read_parquet()`; `_loading.py` does not import or call
`pyarrow` directly. No new dependency is required for this module.

Imports of supported source classes belong at module scope. The `datasets`
package is not an optional dependency, so there is no need for a conditional
import or a silent fallback when it is unavailable.

---

## Legacy API Removal

The legacy functions `load_dataframe()` and `load_hf_dataset()` and their
public exports are removed with the replacement implementation. Current CSV
and parquet callers use `load()` directly.

The temporary `_preparation.py` module was later replaced by `_adapter.py` and
removed with the other superseded request/result/report modules. Loading no
longer supports that bridge.

LLM supervision still needs JSON and JSONL for its own training artifacts. It
owns that format-specific reading locally and delegates CSV and parquet to
`load()`. JSON support does not re-enter the canonical loading API.

The removed legacy behaviors are:

- Automatic inclusion of `participant`, `task`, and `trial`
- Column projection during file reads
- JSON and JSONL support in the shared loading module
- Loading a HuggingFace identifier and split into a Dataset

---

## Module Structure

The replacement API should remain small:

```text
DataSource             accepted source type alias
load()                 public source-to-DataFrame entry point
load_path()            private CSV/parquet suffix dispatch
normalize_columns()     private label normalization and uniqueness check
normalize_column_name() private scalar label normalization
```

Targeted checks for unsupported HuggingFace container types may remain inside
`load()` because they clarify entry-point validation. Reader-specific mapping,
canonical validation, and report construction must not enter this module.

---

## Tests

Create `tests/data/test_loading.py` for the new API. It covers these behaviors:

1. A DataFrame source returns equal data in a distinct DataFrame.
2. Mutating the result does not mutate the source DataFrame.
3. DataFrame row order, column order, dtypes, and index are preserved.
4. CSV loads from both a string and a `Path`.
5. Parquet loads from both a string and a `Path`.
6. File suffix matching is case-insensitive.
7. A HuggingFace `Dataset` converts to the expected DataFrame.
8. Empty and non-canonical data load without canonical validation.
9. JSON, JSONL, compressed CSV, and extensionless strings are rejected with a
   message naming the source and supported formats.
10. An unsupported Python type is rejected with a message naming its type.
11. `DatasetDict` tells the caller to select one split.
12. `IterableDataset` tells the caller to materialize the source.
13. A missing supported file path preserves `FileNotFoundError`.
14. Loading does not add legacy default columns or select a subset of columns.
15. Legacy loading functions are not exported from `mt.data`.
16. Numeric and other non-missing labels normalize with `str()`.
17. Missing labels normalize to `"None"`.
18. Duplicate labels after normalization fail clearly.

Tests use local temporary files and in-memory datasets. They do not access the
network or require a HuggingFace Hub account.

---

## Out of Scope

- Mapping raw names to canonical paths
- Canonical defaults or required-field validation
- Column projection or predicate pushdown
- Reader-specific keyword arguments
- Format inference from file content
- HuggingFace dataset identifier resolution
- Automatic split selection from `DatasetDict`
- Streaming or lazy loading
- Chunked reading
- Multi-file concatenation, joins, or directory loading
- JSON, JSONL, TSV, Excel, or compressed CSV support
- Index normalization
- DataFrame value coercion or dtype normalization

---

## Implementation Result

The replacement API is implemented in `src/mt/data/_loading.py`, exported
through `mt.data`, and covered by `tests/data/test_loading.py`. The legacy
loading functions and exports have been removed, and their callers now use
the replacement boundary.

Focused loading tests, all data-layer tests, and the full test suite pass.
Ruff passes for the changed Python files. Repository-wide ruff still reports
pre-existing failures in unrelated evaluation, visualization, model, and
utility modules; those files were not changed as part of this module.

The next replacement module, `_mapping.py`, is implemented separately against
`docs/design_docs/MappingDesign.md`.

---

## Acceptance Criteria

The module is ready for review when:

- `DataSource` and `load()` implement the API above.
- Every successful source produces an independent raw DataFrame.
- Every output column label is a unique string, with missing labels normalized
  to `"None"`.
- Loading performs no canonical mapping or validation.
- Unsupported formats and HuggingFace containers fail with clear errors.
- Native reader failures retain their original exception types.
- Legacy loading functions are absent from the module and public API.
- Existing CSV and parquet callers use `load()`.
- Focused loading tests pass.
- The full test suite and scoped `ruff check` pass.
