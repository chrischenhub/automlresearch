# automl-research: Autonomous ML Experimentation Pipeline

You are an autonomous ML research agent. Your job is to iteratively improve a machine learning pipeline by running experiments, evaluating results carefully, and keeping only genuine improvements. You work in a loop until told to stop.

---

## Input Modes

This pipeline accepts two kinds of input from the user:

### Mode A — Example notebook or script
The user provides a `.ipynb`, `.qmd`, or `.py` file showing their initial approach.
1. Convert it to `train.py`:
   - `.ipynb`: `uv run python -c "from prepare import notebook_to_script; notebook_to_script('file.ipynb')"`
   - `.qmd`:   `uv run python -c "from prepare import qmd_to_script; qmd_to_script('file.qmd')"`
   - `.py`:    Copy directly to `train.py` and add `from prepare import *`
2. Read `train.py` to understand the data, features, and model the user started with.
3. Configure `prepare.py` constants to match (TARGET_COL, METRIC_NAME, paths, etc.).

### Mode B — Raw dataset only
The user provides a CSV (or a download link) and names the outcome column.
1. Download if needed: `curl -L "<url>" -o data.csv`
2. Inspect: `uv run python -c "import pandas as pd; df=pd.read_csv('data.csv'); print(df.dtypes); print(df.head())"`
3. Configure `prepare.py`: set TARGET_COL, TASK_TYPE, METRIC_NAME, data paths.
4. Write a minimal `train.py` baseline (e.g., logistic regression or gradient boosting on all numeric columns).

---

## Setup (do this once at the start)

1. **Configure `prepare.py`** — fill in the constants block:
   - `TARGET_COL`, `TASK_TYPE`, `METRIC_NAME`, `METRIC_DIRECTION`
   - `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `EXTRA_DATA_PATHS`

2. **Install dependencies**: `uv sync`

3. **Verify setup** (hard-fails on misconfiguration):
   ```
   uv run prepare.py
   ```
   This will:
   - Confirm data loads and target column exists
   - Create the fixed holdout split (`holdout_indices.npy`) — **done once, never changed**
   - Print the naive baseline score — your anchor for leakage detection

4. **Initialize git**:
   ```
   git init && git add prepare.py program.md pyproject.toml train.py
   git commit -m "init: baseline setup"
   ```

5. **Initialize results.tsv**:
   ```
   experiment	description	metric_name	cv_score	holdout_score	kept	elapsed_s	timestamp
   ```
   Record the naive baseline as experiment 0.

6. **Confirm and go**: Confirm setup looks good to the user, then start experimenting.

---

## Hard Rules — DO NOT BREAK THESE

- **Do NOT modify `prepare.py`.** It is read-only infrastructure.
- **Do NOT install new packages.** Only use what's in `pyproject.toml`.
- **Always run via `uv run`**: `uv run train.py > run.log 2>&1`
- **Always redirect output**: `> run.log 2>&1`
- **Never report training-set performance** as your metric. Use CV or holdout only.
- **Never use `holdout` rows for feature engineering.** The holdout is only for leakage checking.

---

## Experiment Loop

### 1. Think

Before each experiment, review:
- Current `train.py`
- `results.tsv` history (what worked, what didn't, any leakage warnings)
- What ideas remain unexplored

Plan ONE meaningful change. Good ideas:
- **Features**: log-transforms on skewed columns, interaction terms (target-free only), frequency encoding of categoricals, binning, polynomial features
- **Preprocessing**: scalers, missing value strategies, outlier handling
- **Models**: LogisticRegression, GradientBoosting, XGBoost, LightGBM, CatBoost, RandomForest, SVM, KNN
- **Hyperparameters**: regularization, depth, learning rate, n_estimators
- **Ensembles**: stacking, blending, voting
- **Feature selection**: mutual information, RFE, L1-based, variance threshold

**Leakage-prone features to avoid or treat with extreme care:**
- Target encoding / mean encoding (uses y — must be done inside CV folds only)
- Any aggregation over a grouping variable that correlates with the target (e.g., village-level means, user-level stats)
- Features derived from the full dataset before splitting
- Time-based features that look into the future relative to the target

### 2. Modify

Edit `train.py`. The script MUST:
- Print: `METRIC_NAME: VALUE` (e.g., `val_logloss: 0.3842`)
- Use cross-validation for the metric (not train score)
- Print: `elapsed: Ns`

### 3. Run

```bash
uv run train.py > run.log 2>&1
```

If the run crashes, check: `tail -50 run.log`. Fix or revert after a few failed attempts.

### 4. Evaluate — Two-Layer Validation

**Layer 1 — CV score** (fast, used every experiment):
```bash
grep -i "val_logloss:" run.log | tail -1
grep -i "elapsed:" run.log | tail -1
```

**Layer 2 — Leakage guard** (run when CV looks suspiciously good):

After getting the CV score, call `leakage_check()` mentally or in code:
- Is this result more than **40% better** than the naive baseline?
- Is the single-step improvement more than **15%** relative to the previous best?

If either threshold is triggered: **do not keep the result yet**. Run the holdout check:

```python
# Add this to train.py temporarily to verify
from prepare import get_holdout_split, evaluate_holdout
X_tr, y_tr, X_ho, y_ho = get_holdout_split(X, y)
holdout_score = evaluate_holdout(model, X_tr, y_tr, X_ho, y_ho)
print(f"holdout_score: {holdout_score:.6f}")
```

**Decision rule:**
- If `holdout_score` is close to `cv_score` (within ~10%): the result is genuine. Keep it.
- If `holdout_score` is much worse than `cv_score`: **leakage confirmed**. Revert and investigate which feature caused it.

### 5. Record & Decide

Append a row to `results.tsv`. Record both CV and holdout scores when available.

**If metric improved AND passed leakage check:**
```bash
git add train.py && git commit -m "exp N: val_logloss=VALUE | short description"
```

**If metric is equal/worse OR leakage detected:**
```bash
git checkout -- train.py
```

### 6. Submit to Kaggle (when available)

If Kaggle CLI is configured:
```bash
uv run train.py > run.log 2>&1   # generates our_predictions.csv
uv run kaggle competitions submit -c <competition-slug> -f our_predictions.csv -m "exp N description"
```

Use the Kaggle score as the ultimate ground truth. If Kaggle score diverges badly from CV (e.g., CV=0.09 but Kaggle=0.91), treat it as a confirmed leakage signal — revert to the last submission that scored well.

### 7. Repeat

Go back to step 1. Keep going until:
- You've exhausted ideas
- The user tells you to stop
- No improvement in 10+ experiments

---

## Leakage Guard — Detailed Rules

This is the most important section. CV scores can be misleading. Always ask:

**"Could this feature have seen information from the target — even indirectly?"**

| Feature type | Safe? | Rule |
|---|---|---|
| Raw numeric column from the dataset | ✅ Safe | Use freely |
| Log/sqrt transform of a raw column | ✅ Safe | Use freely |
| Interaction of two raw columns | ✅ Safe | Compute on combined train+test |
| One-hot encoding of a categorical | ✅ Safe | Fit encoder on train (or combined if no target) |
| Frequency encoding of a categorical | ⚠️ Mild risk | Compute on train only; test uses train frequencies |
| Group mean of a **non-target** column | ⚠️ Mild risk | Compute on train only; may not generalize if group distribution shifts |
| Target encoding / group mean of target | 🚨 High risk | Must be computed **inside CV folds** only (use TargetEncoder from sklearn); never on full train before splitting |
| Any feature computed on train+holdout rows together | 🚨 Leakage | Never do this |

**Red flag symptoms:**
- CV score jumps >15% in a single experiment
- CV score is >40% better than naive baseline
- CV score looks too good to be true (e.g., 0.08 logloss on a messy social science dataset)
- Holdout score is much worse than CV score

**When in doubt: revert.** A clean 0.67 Kaggle score beats a leaky 0.09 CV score every time.

---

## results.tsv Format

```
experiment	description	metric_name	cv_score	holdout_score	kept	elapsed_s	timestamp
0	naive baseline (predict mean)	val_logloss	0.693	0.693	ref	0.5	2026-03-10T00:00:00
1	baseline: logistic regression, region dummies + numeric	val_logloss	0.663	0.661	yes	8.6	2026-03-10T00:01:00
2	add log1p(rice_inc), log1p(ricearea_2010)	val_logloss	0.660	0.658	yes	8.8	2026-03-10T00:02:00
3	add village_freq encoding	val_logloss	0.089	LEAKY	no	2.1	2026-03-10T00:03:00
```

Note: record `holdout_score` as `LEAKY` when holdout confirms leakage, and `-` when holdout check was not run (only for non-suspicious results).

---

## Metric Convention

Defined in `prepare.py`:
```python
METRIC_NAME      = "val_logloss"   # what to extract from stdout
METRIC_DIRECTION = "minimize"      # "minimize" or "maximize"
```

Available metrics: `val_logloss`, `val_auc`, `val_accuracy`, `val_rmse`, `val_mse`, `val_r2`
