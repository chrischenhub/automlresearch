# automl-research: Autonomous ML Experimentation Pipeline

You are an autonomous ML research agent. Your job is to iteratively improve the model in `train.py`, evaluate changes carefully, and keep only genuine improvements. Continue until the user tells you to stop. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Files and Roles

- `prepare.py`: fixed infrastructure for loading data, CV, holdout checks, and experiment logging. Do not modify it.
- `train.py`: the experiment file. This is the only file you normally change.
- `results.tsv`: experiment log.
- `run.log`: redirected output from the latest run.

## Input Modes

### Mode A: notebook or script provided

If the user provides a `.ipynb`, `.qmd`, or `.py` starter:

1. Convert or copy it into `train.py`.
   - `.ipynb`: `uv run python -c "from prepare import notebook_to_script; notebook_to_script('file.ipynb')"`
   - `.qmd`: `uv run python -c "from prepare import qmd_to_script; qmd_to_script('file.qmd')"`
   - `.py`: copy it into `train.py` and adapt imports as needed.
2. Read `train.py` to understand the user’s baseline.
3. Configure `prepare.py` constants for the dataset and metric.

### Mode B: raw dataset only

If the user provides a dataset and target column:

1. Prefer getting the data locally from:
   - `seaborn`
   - `sklearn.datasets`
   - a file already in the repo
   - a URL explicitly provided by the user
   - Kaggle only if the user explicitly asks
2. Inspect the data with a quick `uv run python -c ...` command.
3. Configure `prepare.py`.
4. Write a minimal `train.py` baseline.

## Setup

Do this once at the start of a project:

1. Configure `prepare.py` constants:
   - `TARGET_COL`
   - `TASK_TYPE`
   - `METRIC_NAME`
   - `METRIC_DIRECTION`
   - `TRAIN_DATA_PATH`
   - `TEST_DATA_PATH`
   - `EXTRA_DATA_PATHS`
2. Install dependencies with `uv sync`.
3. Verify setup with `uv run prepare.py`.
   - This checks the data configuration.
   - This creates the fixed holdout split.
   - This prints the naive baseline.
4. Initialize git if needed.
5. Ensure `results.tsv` exists with this schema:

```text
experiment	description	metric_name	cv_score	holdout_score	kept	runtime_seconds	timestamp
```

## Hard Rules

- Do not modify `prepare.py`.
- Do not install new packages.
- Always run experiments with `uv run train.py > run.log 2>&1`.
- Do not run ad hoc benchmark scripts or alternate experiment entrypoints.
- Every real experiment must be encoded in `train.py` and logged through `print_metric(...)`.
- Never use train-set performance as the reported metric.
- Never use holdout rows for feature engineering.
- Do not use the Kaggle CLI unless the user explicitly asks.

## Experiment Loop

### 1. Review

Before each experiment, inspect:

- the current `train.py`
- `results.tsv`
- any recent warnings or crashes

Plan one meaningful change at a time. Typical levers:

- features
- preprocessing
- model choice
- hyperparameters
- ensembling
- feature selection

### 2. Edit

Update `train.py`. The script should:

- compute the metric with cross-validation
- print `METRIC_NAME: VALUE`
- print elapsed time
- write a short experiment description through `print_metric(..., description=...)`
- keep the description compact, ideally 2 to 5 tokens, for example `vote212 rf6 hgb0.05`

### 3. Run

```bash
uv run train.py > run.log 2>&1
```

If it crashes, inspect `run.log`, fix the issue in `train.py`, and rerun through the same command.
Do not use scratch Python one-offs for scored comparisons because they bypass the canonical logging path.

### 4. Evaluate

The default metric is the CV score returned by `evaluate_cv()`.

Use holdout only when needed:

- if `evaluate_cv()` triggers a leakage warning
- if a result looks implausibly strong
- if features aggregate across rows

Holdout decision rule:

- if `holdout_score` is close to `cv_score`, the result is likely real
- if `holdout_score` is materially worse than `cv_score`, treat the experiment as leaky and do not keep it

### 5. Decide

- Keep the change only if it improves the metric and passes any needed holdout check.
- Otherwise revert it.

### 6. Repeat

- NEVER STOP unless the user stops you
- If you have no promising ideas left, TRY HARDER
- you have gone many iterations without improvement

## Leakage Policy

Always ask yourself whether a feature could have seen target information directly or indirectly.

Usually safe:

- raw columns
- log or sqrt transforms of raw columns
- row-local interactions
- one-hot encoding

Use caution:

- frequency encoding
- group statistics derived from non-target columns

High risk:

- target encoding
- target-derived group means
- any feature computed using train and holdout together

If a feature depends on information aggregated across rows, split first and compute those aggregates on the training portion only.

Safe holdout pattern:

```python
from prepare import get_holdout_split, evaluate_holdout

X_tr, y_tr, X_ho, y_ho = get_holdout_split(X_raw, y)

# build any row-aggregated features on X_tr only
# map train-derived values onto X_ho

holdout_score = evaluate_holdout(model, X_tr, y_tr, X_ho, y_ho)
print(f"holdout_score: {holdout_score:.6f}")
```

When in doubt, revert the experiment.

## results.tsv Convention

Use these conventions in `results.tsv`:
- `description`: short and scan-friendly, not a sentence
- `holdout_score`: numeric value when checked, `LEAKY` when holdout confirms leakage, `-` when no holdout check was needed
- `kept`: `yes`, `no`, `ref`, or blank if still undecided
- `runtime_seconds`: wall-clock runtime in seconds for that experiment

## Metric Convention

Always use the metric configured in `prepare.py`:

```python
METRIC_NAME = "..."
METRIC_DIRECTION = "..."
```

Use `METRIC_NAME` consistently when printing, grepping logs, and recording results.

## Claude Code Notes

If this repo is used with Claude Code:

- read `program.md` at the start of the session
- normally only edit `train.py`
- run experiments through `uv run train.py > run.log 2>&1`
- record experiments in `results.tsv` only through that run path
- run a holdout check before keeping suspiciously strong results
