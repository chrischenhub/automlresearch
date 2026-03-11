# automlresearch
[demo.png]
**Autonomous ML experimentation, powered by Claude Code.**
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), adapted for classical ML.

Point your Codex/Claude Code at `program.md` and let it run experiments in a loop.

## How It Works

```
program.md       — agent instructions (read this first)
train.py         — feature engineering, model, hyperparameters (agent modifies this)
prepare.py       — data loading, evaluation, leakage guard (do not modify)
pyproject.toml   — dependencies (pre-installed, agent cannot add new ones)
results.tsv      — experiment log (auto-generated, git-tracked)
```

The agent reads `program.md`, then enters a loop:
1. **Think** — review experiment history, plan one change
2. **Modify** — edit `train.py`
3. **Run** — `uv run train.py > run.log 2>&1`
4. **Evaluate** — check CV score; automatic leakage guard fires if result is suspicious
5. **Keep or revert** — git commit if improved, git checkout if not
6. **Repeat**

Important: there is only one experiment entrypoint. Do not run scored scratch scripts outside `train.py`, because `results.tsv` is meant to be written by the canonical `train.py` path only.

### Two-Layer Validation (Leakage Guard)

The key differentiator: CV scores can be misleading when features leak target information.
This pipeline has a built-in safety net:

- **Layer 1 — CV** (fast, every experiment): standard cross-validation for quick iteration.
- **Layer 2 — Fixed holdout** (triggered automatically on suspicious results): a 20% stratified split saved to `holdout_indices.npy` at setup, never used for feature engineering. If the holdout score diverges from CV, leakage is confirmed.

`evaluate_cv()` automatically calls `leakage_check()` after every run. No manual intervention needed.

## Quick Start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone / copy this repo
git clone <this-repo> my-experiment
cd my-experiment

# 2. Put your data files here (or use seaborn/sklearn built-in datasets)
cp /path/to/train.csv .
cp /path/to/test.csv .    # optional

# 3. Edit prepare.py constants to match your project
#    TARGET_COL, TASK_TYPE, METRIC_NAME, METRIC_DIRECTION, data paths

# 4. Install dependencies + verify setup
uv sync
uv run prepare.py    # creates holdout split, prints naive baseline

# 5. Write or convert your baseline train.py
#    (or let the agent do it from a notebook/CSV)

# 6. Init git
git init && git add prepare.py program.md pyproject.toml train.py
git commit -m "init: baseline setup"
```

## Input Modes

### Mode A — User provides a notebook or script
The agent reads it, learns the data/features/model, converts to `train.py`, and starts experimenting.
Supports `.ipynb`, `.qmd`, and `.py` files.

### Mode B — User provides a dataset
Drop a CSV in the directory (or name a famous dataset like "titanic"), tell the agent the target column.
The agent writes a baseline `train.py` from scratch and starts experimenting.

## Running the Agent

### Fully autonomous mode (recommended)

The agent needs to edit files and run commands without prompting you each time. Launch with:

```bash
claude --dangerously-skip-permissions
```

Then prompt:

```
Read program.md and start a new experiment session.
```

You can walk away — the agent will run experiments, record results, and commit improvements on its own.

### Interactive mode

If you prefer to approve each action, just run `claude` normally. You'll be prompted before every file edit and shell command.

## Available Libraries

All pre-installed via `pyproject.toml`. The agent cannot add new ones.

- **Core**: numpy, pandas, scipy
- **ML**: scikit-learn, xgboost, lightgbm, catboost
- **Imbalanced data**: imbalanced-learn (SMOTE, etc.)
- **Tuning**: optuna
- **Embeddings**: sentence-transformers
- **Visualization**: matplotlib, seaborn
- **Data formats**: pyarrow (parquet), openpyxl (excel)
- **Kaggle**: kaggle (for submissions only, when user requests it)

## License

MIT
