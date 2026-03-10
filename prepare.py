"""
prepare.py — Fixed infrastructure for automl-research.
DO NOT MODIFY. The agent only modifies train.py.

This file provides:
  - Data loading utilities
  - Two-layer evaluation: CV (fast) + fixed holdout (leakage guard)
  - Naive baseline for sanity checking
  - Leakage detection helper
  - Constants (metric name, direction, random seed, timeout)
  - Notebook / .qmd conversion utility
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier, DummyRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONSTANTS — Edit these for your project, then never touch prepare.py again.
#
# Checklist before starting:
#   [ ] RANDOM_SEED      — any integer
#   [ ] METRIC_NAME      — val_logloss | val_auc | val_accuracy | val_rmse | val_r2
#   [ ] METRIC_DIRECTION — minimize | maximize
#   [ ] TASK_TYPE        — classification | regression
#   [ ] TARGET_COL       — exact column name in your training CSV
#   [ ] TRAIN_DATA_PATH  — path to training file
#   [ ] TEST_DATA_PATH   — path to test/prediction file (or "" if none)
#   [ ] EXTRA_DATA_PATHS — additional files to auto-join (or {})
# ============================================================================

RANDOM_SEED      = 42
METRIC_NAME      = "val_accuracy"      # printed as "val_accuracy: 0.1234"
METRIC_DIRECTION = "maximize"          # "minimize" or "maximize"
TASK_TYPE        = "classification"    # "classification" or "regression"
TARGET_COL       = "Survived"          # target column name in training data
CV_FOLDS         = 5                   # cross-validation folds
HOLDOUT_FRAC     = 0.2                 # fraction of train held out for leakage guard
EXPERIMENT_TIMEOUT = 300               # seconds — soft guideline for the agent

TRAIN_DATA_PATH  = "train.csv"
TEST_DATA_PATH   = "test.csv"
EXTRA_DATA_PATHS = {}                  # e.g. {"products": "products.csv"}

np.random.seed(RANDOM_SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def _load_file(path: str) -> pd.DataFrame:
    if path.endswith((".csv.gz", ".csv")):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def load_train_data() -> pd.DataFrame:
    """Load training dataset and merge any extra files."""
    df = _load_file(TRAIN_DATA_PATH)
    for _, extra_path in EXTRA_DATA_PATHS.items():
        if os.path.exists(extra_path):
            extra = _load_file(extra_path)
            common = list(set(df.columns) & set(extra.columns))
            if common:
                df = df.merge(extra, on=common, how="left")
    return df


def load_test_data() -> pd.DataFrame:
    """Load test/prediction dataset and merge any extra files."""
    if not TEST_DATA_PATH or not os.path.exists(TEST_DATA_PATH):
        return pd.DataFrame()
    df = _load_file(TEST_DATA_PATH)
    for _, extra_path in EXTRA_DATA_PATHS.items():
        if os.path.exists(extra_path):
            extra = _load_file(extra_path)
            common = list(set(df.columns) & set(extra.columns))
            if common:
                df = df.merge(extra, on=common, how="left")
    return df


# ============================================================================
# HOLDOUT SPLIT — created once, never used for feature engineering
#
# Call create_holdout_split() during setup (once). It saves holdout row
# indices to holdout_indices.npy. All subsequent runs load those same
# indices so the holdout is always identical.
#
# CRITICAL RULE for train.py:
#   Load full df → create features on df → then call get_holdout_split()
#   to separate. Never compute target-derived statistics on holdout rows.
# ============================================================================

HOLDOUT_INDEX_PATH = "holdout_indices.npy"


def create_holdout_split(df: pd.DataFrame = None) -> np.ndarray:
    """
    Create and save a fixed holdout index array (run once during setup).
    Returns the holdout indices.
    """
    if df is None:
        df = load_train_data()
    y = df[TARGET_COL].values

    if TASK_TYPE == "classification":
        _, holdout_idx = train_test_split(
            np.arange(len(df)), test_size=HOLDOUT_FRAC,
            stratify=y, random_state=RANDOM_SEED
        )
    else:
        _, holdout_idx = train_test_split(
            np.arange(len(df)), test_size=HOLDOUT_FRAC,
            random_state=RANDOM_SEED
        )

    np.save(HOLDOUT_INDEX_PATH, holdout_idx)
    print(f"Holdout split saved: {len(holdout_idx):,} rows ({HOLDOUT_FRAC:.0%}) → {HOLDOUT_INDEX_PATH}")
    return holdout_idx


def get_holdout_split(X, y):
    """
    Load the fixed holdout indices and return (X_tr, y_tr, X_ho, y_ho).
    Use X_tr/y_tr for feature engineering and CV. Use X_ho/y_ho only
    for leakage checking — never for fitting or feature engineering.
    """
    if not os.path.exists(HOLDOUT_INDEX_PATH):
        raise FileNotFoundError(
            f"{HOLDOUT_INDEX_PATH} not found. Run `uv run prepare.py` to create it."
        )
    holdout_idx = np.load(HOLDOUT_INDEX_PATH)
    all_idx = np.arange(len(y))
    train_idx = np.setdiff1d(all_idx, holdout_idx)

    if hasattr(X, "iloc"):
        X_tr, X_ho = X.iloc[train_idx], X.iloc[holdout_idx]
    else:
        X_tr, X_ho = X[train_idx], X[holdout_idx]

    if hasattr(y, "iloc"):
        y_tr, y_ho = y.iloc[train_idx], y.iloc[holdout_idx]
    else:
        y_tr, y_ho = y[train_idx], y[holdout_idx]

    return X_tr, y_tr, X_ho, y_ho


# ============================================================================
# EVALUATION
# ============================================================================

def get_cv_splitter():
    if TASK_TYPE == "classification":
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    else:
        return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def get_scoring():
    mapping = {
        "val_logloss":  "neg_log_loss",
        "val_auc":      "roc_auc",
        "val_accuracy": "accuracy",
        "val_rmse":     "neg_root_mean_squared_error",
        "val_mse":      "neg_mean_squared_error",
        "val_r2":       "r2",
    }
    return mapping.get(METRIC_NAME, "neg_log_loss")


def _score_from_raw(scoring: str, raw_scores: np.ndarray) -> float:
    return -raw_scores.mean() if scoring.startswith("neg_") else raw_scores.mean()


def evaluate_cv(model, X, y) -> float:
    """Run CV on (X, y) and return the metric. Fast but may be optimistic if features leak."""
    scoring = get_scoring()
    scores = cross_val_score(model, X, y, cv=get_cv_splitter(), scoring=scoring, n_jobs=-1)
    return _score_from_raw(scoring, scores)


def evaluate_holdout(model, X_tr, y_tr, X_ho, y_ho) -> float:
    """
    Fit on training split, score on holdout. Leakage-free by construction
    (holdout rows were excluded from all feature engineering).
    """
    model.fit(X_tr, y_tr)
    if METRIC_NAME == "val_logloss":
        return log_loss(y_ho, model.predict_proba(X_ho))
    elif METRIC_NAME == "val_auc":
        return roc_auc_score(y_ho, model.predict_proba(X_ho)[:, 1])
    elif METRIC_NAME == "val_accuracy":
        return accuracy_score(y_ho, model.predict(X_ho))
    elif METRIC_NAME in ("val_rmse", "val_mse"):
        preds = model.predict(X_ho)
        mse = mean_squared_error(y_ho, preds)
        return np.sqrt(mse) if METRIC_NAME == "val_rmse" else mse
    elif METRIC_NAME == "val_r2":
        from sklearn.metrics import r2_score
        return r2_score(y_ho, model.predict(X_ho))
    else:
        return log_loss(y_ho, model.predict_proba(X_ho))


def compute_naive_baseline(y) -> float:
    """
    Score of the dumbest possible model (predict mean/majority).
    This is the floor — a real model must beat this comfortably.
    """
    y = np.array(y)
    if TASK_TYPE == "classification":
        dummy = DummyClassifier(strategy="prior")   # predicts class proportions — correct for logloss
    else:
        dummy = DummyRegressor(strategy="mean")

    scoring = get_scoring()
    scores = cross_val_score(dummy, np.zeros((len(y), 1)), y,
                             cv=get_cv_splitter(), scoring=scoring, n_jobs=-1)
    return _score_from_raw(scoring, scores)


def print_metric(value: float, label: str = None):
    """Print the metric in the expected format."""
    tag = label or METRIC_NAME
    print(f"{tag}: {value:.6f}")


# ============================================================================
# LEAKAGE GUARD
#
# Call this after every CV evaluation. It will:
#   1. Warn if CV score is implausibly better than the naive baseline.
#   2. Warn if a single experiment improved the metric by more than
#      LEAKAGE_STEP_THRESHOLD relative to the previous best.
#   3. If either warning triggers, print a strong recommendation to
#      verify with evaluate_holdout() before keeping the change.
#
# These are warnings, not hard stops — the agent decides what to do.
# But the agent MUST run evaluate_holdout() before keeping any flagged result.
# ============================================================================

LEAKAGE_NAIVE_THRESHOLD = 0.40   # CV beats naive by more than 40% → suspicious
LEAKAGE_STEP_THRESHOLD  = 0.15   # single step improves by more than 15% → suspicious


def leakage_check(cv_score: float, naive_baseline: float, prev_best: float = None) -> bool:
    """
    Returns True if the result looks suspicious (possible leakage).
    Prints a clear warning with the reason.
    """
    is_minimize = METRIC_DIRECTION == "minimize"
    suspicious = False
    reasons = []

    # Check 1: implausibly better than naive baseline
    if naive_baseline and naive_baseline != 0:
        if is_minimize:
            improvement_vs_naive = (naive_baseline - cv_score) / naive_baseline
        else:
            improvement_vs_naive = (cv_score - naive_baseline) / abs(naive_baseline)

        if improvement_vs_naive > LEAKAGE_NAIVE_THRESHOLD:
            suspicious = True
            reasons.append(
                f"CV score ({cv_score:.4f}) is {improvement_vs_naive:.0%} better than "
                f"naive baseline ({naive_baseline:.4f}) — threshold is {LEAKAGE_NAIVE_THRESHOLD:.0%}"
            )

    # Check 2: single-step improvement too large
    if prev_best is not None:
        if is_minimize:
            step_improvement = (prev_best - cv_score) / prev_best if prev_best != 0 else 0
        else:
            step_improvement = (cv_score - prev_best) / abs(prev_best) if prev_best != 0 else 0

        if step_improvement > LEAKAGE_STEP_THRESHOLD:
            suspicious = True
            reasons.append(
                f"Single-step improvement of {step_improvement:.0%} "
                f"({prev_best:.4f} → {cv_score:.4f}) exceeds threshold of {LEAKAGE_STEP_THRESHOLD:.0%}"
            )

    if suspicious:
        print("\n" + "!" * 60)
        print("LEAKAGE WARNING — do NOT keep this result yet.")
        for r in reasons:
            print(f"  - {r}")
        print("ACTION: run evaluate_holdout() to confirm before keeping.")
        print("!" * 60 + "\n")

    return suspicious


# ============================================================================
# TIMING
# ============================================================================

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"elapsed: {self.elapsed:.1f}s")


# ============================================================================
# NOTEBOOK / QMD CONVERSION
# ============================================================================

def notebook_to_script(ipynb_path: str, output_path: str = "train.py"):
    """Convert a Jupyter notebook (.ipynb) to a Python script."""
    with open(ipynb_path) as f:
        nb = json.load(f)

    lines = [
        '"""',
        f"Auto-converted from {ipynb_path}",
        "Edit this file for experiments. Do not modify prepare.py.",
        '"""',
        "",
        "from prepare import *",
        "",
    ]
    for i, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).strip()
            if source:
                lines += [f"# {'=' * 60}", f"# Cell {i}", f"# {'=' * 60}", source, ""]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Converted {ipynb_path} → {output_path}")


def qmd_to_script(qmd_path: str, output_path: str = "train.py"):
    """Convert a Quarto (.qmd) file to a Python script by extracting python code blocks."""
    with open(qmd_path) as f:
        content = f.read()

    lines = [
        '"""',
        f"Auto-converted from {qmd_path}",
        "Edit this file for experiments. Do not modify prepare.py.",
        '"""',
        "",
        "from prepare import *",
        "",
    ]

    in_block = False
    block_lines = []
    block_num = 0
    for line in content.splitlines():
        if line.strip().startswith("```{python") and not in_block:
            in_block = True
            block_lines = []
        elif line.strip() == "```" and in_block:
            in_block = False
            source = "\n".join(block_lines).strip()
            if source:
                lines += [f"# {'=' * 60}", f"# Block {block_num}", f"# {'=' * 60}", source, ""]
                block_num += 1
        elif in_block:
            block_lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Converted {qmd_path} → {output_path} ({block_num} code blocks extracted)")


# ============================================================================
# VERIFICATION — hard-fails on misconfiguration
# ============================================================================

def verify_setup() -> bool:
    """
    Check that data files exist, target column is present, and holdout is created.
    Hard-fails (raises) on critical errors so the agent cannot proceed silently.
    """
    print("Verifying setup...")
    errors = []

    # --- Training data ---
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")

    try:
        df = load_train_data()
        print(f"  Train: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {e}")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL='{TARGET_COL}' not found in training data.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  Fix TARGET_COL in prepare.py before proceeding."
        )
    print(f"  Target: '{TARGET_COL}' (mean={df[TARGET_COL].mean():.4f}, "
          f"dtype={df[TARGET_COL].dtype})")

    # --- Test data ---
    if TEST_DATA_PATH and os.path.exists(TEST_DATA_PATH):
        test = load_test_data()
        print(f"  Test:  {test.shape[0]:,} rows x {test.shape[1]:,} cols")
    else:
        print(f"  Test:  not found ({TEST_DATA_PATH}) — predictions won't be generated")

    # --- Extra data ---
    for name, path in EXTRA_DATA_PATHS.items():
        status = "found" if os.path.exists(path) else "NOT FOUND"
        print(f"  Extra '{name}': {status} ({path})")

    # --- Holdout split ---
    if not os.path.exists(HOLDOUT_INDEX_PATH):
        print(f"  Holdout: creating {HOLDOUT_INDEX_PATH}...")
        create_holdout_split(df)
    else:
        ho_idx = np.load(HOLDOUT_INDEX_PATH)
        print(f"  Holdout: {len(ho_idx):,} rows ({len(ho_idx)/len(df):.0%}) — {HOLDOUT_INDEX_PATH}")

    # --- Naive baseline ---
    y = df[TARGET_COL].values
    naive = compute_naive_baseline(y)
    print(f"\n  Naive baseline ({METRIC_NAME}): {naive:.6f}")
    print(f"  Any model must beat this. Flag if CV beats it by >{LEAKAGE_NAIVE_THRESHOLD:.0%}.")

    # --- Config summary ---
    print(f"\n  Metric:    {METRIC_NAME} ({METRIC_DIRECTION})")
    print(f"  Task:      {TASK_TYPE}")
    print(f"  CV folds:  {CV_FOLDS}")
    print(f"  Holdout:   {HOLDOUT_FRAC:.0%} of train (leakage guard)")
    print(f"  Seed:      {RANDOM_SEED}")
    print(f"\nSetup OK!")
    return True


if __name__ == "__main__":
    verify_setup()
