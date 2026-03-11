"""
train.py — Titanic experiment: weighted soft-voting ensemble with title features.
Target: Survived (0/1). Metric: val_accuracy (maximize).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import (
    load_train_data, load_test_data,
    RANDOM_SEED, TARGET_COL,
    evaluate_cv, print_metric, Timer,
)

np.random.seed(RANDOM_SEED)

# ============================================================================
# Load Data
# ============================================================================

with Timer():
    df = load_train_data()
    test_df = load_test_data()
    print(f"  train: {df.shape}, test: {test_df.shape}")

# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features(df):
    df = df.copy()

    df["is_female"] = (df["Sex"] == "female").astype(int)

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["title_Mr"] = (df["Title"] == "Mr").astype(int)
    df["title_Mrs"] = (df["Title"] == "Mrs").astype(int)
    df["title_Miss"] = (df["Title"] == "Miss").astype(int)
    df["title_Master"] = (df["Title"] == "Master").astype(int)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["log_fare"] = np.log1p(df["Fare"])

    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    df["Embarked"] = df["Embarked"].fillna("S")
    df["embarked_C"] = (df["Embarked"] == "C").astype(int)
    df["embarked_Q"] = (df["Embarked"] == "Q").astype(int)

    return df


df = engineer_features(df)
test_df = engineer_features(test_df)

feature_cols = [
    "is_female", "Pclass", "Age", "log_fare",
    "family_size", "is_alone", "SibSp", "Parch",
    "embarked_C", "embarked_Q",
    "title_Mr", "title_Mrs", "title_Miss", "title_Master",
]

X = df[feature_cols].values
y = df[TARGET_COL].astype(int).values
X_test = test_df[feature_cols].values

print(f"X shape: {X.shape}, y mean: {y.mean():.3f}")

# ============================================================================
# Model
# ============================================================================

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegressionCV(Cs=np.logspace(-3, 3, 30), cv=5, scoring="accuracy", max_iter=2000, random_state=RANDOM_SEED)),
])
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=4,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=RANDOM_SEED,
)

model = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("hgb", hgb)],
    voting="soft",
    weights=[2, 1, 2],
)

# ============================================================================
# Evaluate
# ============================================================================

with Timer():
    print("Cross-validation...")
    metric_value = evaluate_cv(model, X, y)

print_metric(metric_value, description="vote212 rf6 hgb0.05")

# ============================================================================
# Generate Predictions
# ============================================================================

model.fit(X, y)
preds = model.predict(X_test)

out = test_df[["PassengerId"]].copy()
out["Survived"] = preds
out.to_csv("submission.csv", index=False)
print(f"Saved submission.csv ({len(out):,} rows)")
