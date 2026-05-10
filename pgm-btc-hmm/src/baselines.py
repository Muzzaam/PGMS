"""
baselines.py
------------
Baseline model comparison for the Bitcoin HMM regime project.

Compares the best HMM against four simpler alternatives on the test set:
  - Majority class (always predict the more common training label)
  - Last sign (predict same direction as current week's return)
  - Logistic regression (direct discriminative model)
  - Gaussian mixture model (same features, no temporal structure)

The HMM result is loaded from the saved feature_family_comparison.csv rather
than hardcoded, so this script stays reproducible if experiments are rerun.

Output: results/tables/baseline_comparison.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from preprocess import chronological_split, scale_features
from evaluate import directional_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
HMM_RESULTS  = PROJECT_ROOT / "results" / "tables" / "feature_family_comparison.csv"
OUT_PATH     = PROJECT_ROOT / "results" / "tables" / "baseline_comparison.csv"

RANDOM_STATE = 42

PSYCHOLOGY_FEATURES = [
    "btc_log_return",
    "btc_4w_vol",
    "btc_volume_log_change",
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the weekly feature CSV, keep only columns needed for the baseline
    comparison, and drop the last row (no observed next-week label).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required = ["Date", "next_week_up"] + PSYCHOLOGY_FEATURES
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def majority_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Predict the most common class from training set for every test week.

    This baseline tests whether any model simply exploits class imbalance
    in the training period.
    """
    majority_class = int(train_df["next_week_up"].mode().iloc[0])
    y_pred = np.full(len(test_df), majority_class, dtype=int)
    return directional_metrics("majority_class", test_df["next_week_up"].values, y_pred)


def last_sign_baseline(test_df: pd.DataFrame) -> dict:
    """
    Predict that next week will have the same return sign as the current week.

    This tests for simple momentum persistence without any model fitting.
    """
    y_pred = (test_df["btc_log_return"] > 0).astype(int).values
    return directional_metrics("last_sign", test_df["next_week_up"].values, y_pred)


def logistic_regression_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Fit a logistic regression on the psychology feature set and evaluate
    directional accuracy on the test set.

    Uses the same feature set as the best HMM to provide a fair comparison
    of discriminative vs. latent-state approaches.
    """
    X_train, X_test, _ = scale_features(train_df, test_df, PSYCHOLOGY_FEATURES)
    y_train = train_df["next_week_up"].values
    y_test  = test_df["next_week_up"].values

    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return directional_metrics("logistic_regression", y_test, y_pred)


def gmm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 3,
) -> dict:
    """
    Fit a Gaussian mixture model on the psychology feature set and predict
    next-week direction using cluster-conditional up-probabilities.

    The GMM uses the same features and cluster count as the best HMM but
    has no transition structure, providing a direct test of whether the
    Markov chain adds value over static clustering.

    Parameters
    ----------
    n_components : int
        Number of GMM clusters. Should match the best HMM state count.
    """
    X_train, X_test, _ = scale_features(train_df, test_df, PSYCHOLOGY_FEATURES)
    y_test = test_df["next_week_up"].values

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    gmm.fit(X_train)

    train_clusters = gmm.predict(X_train)
    test_clusters  = gmm.predict(X_test)

    # Estimate P(next_week_up=1 | cluster) from training labels
    tmp = train_df.copy()
    tmp["cluster"] = train_clusters
    cluster_up_probs = tmp.groupby("cluster")["next_week_up"].mean().to_dict()
    global_prob      = tmp["next_week_up"].mean()

    y_pred_prob = np.array([cluster_up_probs.get(c, global_prob) for c in test_clusters])
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    return directional_metrics("gmm_3_clusters", y_test, y_pred)


def load_hmm_result() -> dict:
    """
    Load the best HMM result from the saved feature family comparison CSV.

    Uses the B_price_psychology family with 3 states, which is the
    best-performing configuration identified in hmm_compare.py.

    Returns
    -------
    dict with keys: 'model', 'accuracy', 'balanced_accuracy'
    """
    hmm_df = pd.read_csv(HMM_RESULTS)
    row = hmm_df[
        (hmm_df["feature_set"] == "B_price_psychology") &
        (hmm_df["n_states"] == 3)
    ].iloc[0]

    return {
        "model":             "hmm_psychology_3_states",
        "accuracy":          row["test_accuracy"],
        "balanced_accuracy": row["test_balanced_accuracy"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all baselines and save the comparison table."""
    df = load_data()
    train_df, test_df = chronological_split(df)

    results = [
        majority_baseline(train_df, test_df),
        last_sign_baseline(test_df),
        logistic_regression_baseline(train_df, test_df),
        gmm_baseline(train_df, test_df, n_components=3),
        load_hmm_result(),
    ]

    results_df = pd.DataFrame(results)[["model", "accuracy", "balanced_accuracy"]]
    results_df.to_csv(OUT_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()