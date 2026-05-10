"""
coverup_test.py
---------------
Masked-feature recovery experiment for the Bitcoin HMM regime project.

Tests the HMM as a generative probabilistic model by hiding one
psychology-linked feature at a time, inferring the latent state sequence
from the remaining features, and reconstructing the masked feature using
state-conditioned means estimated from training data.

Reconstruction quality (MAE and RMSE) is compared against a global-mean
baseline. An improvement over the global mean indicates that the latent
states capture meaningful joint structure among the psychology-linked
variables.

Output: results/tables/coverup_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hmmlearn.hmm import GaussianHMM

from preprocess import chronological_split, scale_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
OUT_PATH     = PROJECT_ROOT / "results" / "tables" / "coverup_results.csv"

N_RESTARTS        = 10
BASE_RANDOM_STATE = 42

FEATURES = [
    "btc_log_return",
    "btc_4w_vol",
    "btc_volume_log_change",
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]

# Features to mask one at a time
MASK_TARGETS = [
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the weekly feature CSV and keep only the psychology feature columns.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required = ["Date"] + FEATURES
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[required].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# HMM fitting (mirrors hmm_compare.py; kept local to keep scripts runnable
# independently without circular imports)
# ---------------------------------------------------------------------------

def fit_best_hmm(
    X_train: np.ndarray,
    n_states: int = 3,
) -> GaussianHMM:
    """
    Fit a Gaussian HMM with diagonal covariance using multiple random
    restarts and return the model with the highest training log-likelihood.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix (visible features only).
    n_states : int
        Number of hidden states. Default 3.

    Returns
    -------
    GaussianHMM
        Best-fitting model across all restarts.
    """
    best_model = None
    best_score = -np.inf

    for i in range(N_RESTARTS):
        seed  = BASE_RANDOM_STATE + i
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=seed,
        )
        try:
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"HMM fitting failed for all {N_RESTARTS} restarts.")

    return best_model


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def reconstruct_masked_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    masked_feature: str,
    n_states: int = 3,
) -> dict:
    """
    Hide one feature from the observation set, infer latent states from
    the remaining features, and reconstruct the masked feature using
    state-conditioned means from the training set.

    The HMM is fitted on visible features only, so the masked feature
    has no influence on the inferred state sequence. Reconstruction uses
    the training-set mean of the masked feature within each state.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training rows with all features available.
    test_df : pd.DataFrame
        Test rows; masked_feature is treated as unknown.
    feature_cols : list of str
        Full feature set (masked_feature will be excluded before fitting).
    masked_feature : str
        The feature to hide and reconstruct.
    n_states : int
        Number of hidden states. Should match the best HMM configuration.

    Returns
    -------
    dict with keys: masked_feature, n_states, mae, rmse
    """
    visible_cols = [c for c in feature_cols if c != masked_feature]

    # Scale and fit on visible features only
    X_train_vis, X_test_vis, _ = scale_features(train_df, test_df, visible_cols)

    hmm = fit_best_hmm(X_train_vis, n_states=n_states)

    train_states = hmm.predict(X_train_vis)
    test_states  = hmm.predict(X_test_vis)

    # Estimate state-conditioned means of the masked feature from training data
    tmp = train_df.copy()
    tmp["state"] = train_states
    state_means  = tmp.groupby("state")[masked_feature].mean().to_dict()
    global_mean  = tmp[masked_feature].mean()

    y_true = test_df[masked_feature].values
    y_pred = np.array([state_means.get(s, global_mean) for s in test_states])

    return {
        "masked_feature": masked_feature,
        "n_states":       n_states,
        "mae":            mean_absolute_error(y_true, y_pred),
        "rmse":           rmse(y_true, y_pred),
    }


def global_mean_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_feature: str,
) -> dict:
    """
    Reconstruct a masked feature using its global training mean for every
    test week (i.e., ignoring any state structure).

    This is the baseline against which the HMM reconstruction is compared.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    masked_feature : str

    Returns
    -------
    dict with keys: masked_feature, baseline, mae, rmse
    """
    y_true      = test_df[masked_feature].values
    mean_value  = train_df[masked_feature].mean()
    y_pred      = np.full(len(test_df), mean_value)

    return {
        "masked_feature": masked_feature,
        "baseline":       "global_mean",
        "mae":            mean_absolute_error(y_true, y_pred),
        "rmse":           rmse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run masked-feature recovery for all target features and save results."""
    df = load_data()
    train_df, test_df = chronological_split(df)

    results = []
    for target in MASK_TARGETS:
        print(f"Masking: {target}")

        hmm_result = reconstruct_masked_feature(
            train_df=train_df,
            test_df=test_df,
            feature_cols=FEATURES,
            masked_feature=target,
            n_states=3,
        )
        hmm_result["baseline"] = "hmm_state_mean"
        results.append(hmm_result)

        base_result = global_mean_baseline(train_df, test_df, target)
        results.append(base_result)

    results_df = pd.DataFrame(results)[
        ["masked_feature", "baseline", "n_states", "mae", "rmse"]
    ]
    results_df.to_csv(OUT_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()