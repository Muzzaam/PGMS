"""
hmm_compare.py
--------------
Main experiment script for the Bitcoin HMM regime project.

Fits Gaussian HMMs with multiple feature families and state counts,
evaluates each configuration on a held-out test set, and saves:
  - Per-family state summary CSVs
  - Decoded state sequence CSVs
  - Transition matrix CSVs
  - Regime visualisation PNGs
  - A combined comparison CSV (results/tables/feature_family_comparison.csv)

Run this script to reproduce all HMM results reported in the paper.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from hmmlearn.hmm import GaussianHMM

from preprocess import chronological_split, scale_features
from evaluate import state_up_probabilities, predict_from_states


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
FIG_DIR      = PROJECT_ROOT / "results" / "figures"
TABLE_DIR    = PROJECT_ROOT / "results" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature family definitions
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "A_price_only": [
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
    ],
    "B_price_psychology": [
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
        "fear_greed_value",
        "btc_momentum_4w",
        "btc_drawdown_12w",
    ],
    "C_price_macro": [
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
        "dxy_log_return",
        "dgs10_change",
    ],
    "D_price_trader": [
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
        "btc_ma_spread_4_12",
        "btc_price_vs_sma12",
    ],
    "E_compact_combined": [
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
        "fear_greed_value",
        "btc_drawdown_12w",
        "dxy_log_return",
        "btc_ma_spread_4_12",
    ],
}

STATE_OPTIONS    = [3, 4]
N_RESTARTS       = 10
BASE_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the pre-built weekly feature CSV, validate required columns,
    and drop the last row (whose next_week_up label has no following week).

    Returns
    -------
    pd.DataFrame
        Clean weekly DataFrame ready for modelling.
    """
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required_cols = [
        "Date", "btc_close", "next_week_up",
        "btc_log_return", "btc_4w_vol", "btc_volume_log_change",
        "fear_greed_value", "dxy_log_return", "dgs10_change",
        "btc_momentum_4w", "btc_drawdown_12w",
        "btc_ma_spread_4_12", "btc_price_vs_sma12",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy().dropna().reset_index(drop=True)
    # Last row has no observed following week for next_week_up
    df = df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------

def fit_best_hmm(
    X_train: np.ndarray,
    n_states: int,
) -> tuple[GaussianHMM, int, float]:
    """
    Fit a Gaussian HMM with diagonal covariance using multiple random
    restarts and return the best model by training log-likelihood.

    Multiple restarts are necessary because the Baum-Welch EM algorithm
    is sensitive to initialisation and can converge to poor local optima.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix, shape (T, n_features).
    n_states : int
        Number of hidden states.

    Returns
    -------
    best_model : GaussianHMM
        Model with the highest training log-likelihood across restarts.
    best_seed : int
        Random seed that produced the best model.
    best_train_loglik : float
        Training log-likelihood of the best model.
    """
    best_model        = None
    best_seed         = None
    best_train_loglik = -np.inf

    for i in range(N_RESTARTS):
        seed = BASE_RANDOM_STATE + i
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=seed,
        )
        try:
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_train_loglik:
                best_train_loglik = score
                best_model        = model
                best_seed         = seed
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"All {N_RESTARTS} HMM fits failed for n_states={n_states}.")

    return best_model, best_seed, best_train_loglik


def try_aic_bic(
    model: GaussianHMM,
    X: np.ndarray,
) -> tuple[float, float]:
    """
    Attempt to compute AIC and BIC for a fitted HMM.

    Returns NaN for either quantity if the model does not expose the
    method (older hmmlearn versions).

    Parameters
    ----------
    model : GaussianHMM
    X : np.ndarray
        Data on which to evaluate AIC/BIC (typically training set).

    Returns
    -------
    aic, bic : float
    """
    aic = np.nan
    bic = np.nan
    try:
        aic = model.aic(X)
    except Exception:
        pass
    try:
        bic = model.bic(X)
    except Exception:
        pass
    return aic, bic


# ---------------------------------------------------------------------------
# Post-fit analysis helpers
# ---------------------------------------------------------------------------

def summarize_states(
    df: pd.DataFrame,
    states: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute per-state summary statistics for interpretation.

    Always includes count, mean return, mean volatility, mean volume change,
    and P(next_week_up). Optional features are included if they appear in
    feature_cols.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with feature and target columns.
    states : np.ndarray
        Decoded state sequence aligned with df rows.
    feature_cols : list of str
        Feature columns used in this model run.

    Returns
    -------
    pd.DataFrame
        One row per state with summary statistics.
    """
    tmp = df.copy()
    tmp["state"] = states

    agg_dict = {
        "count":             ("state", "size"),
        "mean_return":       ("btc_log_return", "mean"),
        "mean_volatility":   ("btc_4w_vol", "mean"),
        "mean_volume_change":("btc_volume_log_change", "mean"),
        "prob_next_week_up": ("next_week_up", "mean"),
    }

    optional_map = {
        "fear_greed_value":  "mean_fear_greed",
        "btc_momentum_4w":   "mean_momentum_4w",
        "btc_drawdown_12w":  "mean_drawdown_12w",
        "dxy_log_return":    "mean_dxy_log_return",
        "dgs10_change":      "mean_dgs10_change",
        "btc_ma_spread_4_12":"mean_ma_spread_4_12",
        "btc_price_vs_sma12":"mean_price_vs_sma12",
    }
    for col, out_name in optional_map.items():
        if col in feature_cols:
            agg_dict[out_name] = (col, "mean")

    summary = (
        tmp.groupby("state")
           .agg(**agg_dict)
           .reset_index()
           .sort_values("state")
    )
    return summary


def save_transition_matrix(
    model: GaussianHMM,
    feature_set_name: str,
    n_states: int,
) -> None:
    """Save the learned transition matrix as a CSV."""
    trans_df = pd.DataFrame(
        model.transmat_,
        columns=[f"to_state_{i}" for i in range(n_states)],
        index=[f"from_state_{i}" for i in range(n_states)],
    )
    out_path = TABLE_DIR / f"{feature_set_name}_transition_matrix_{n_states}_states.csv"
    trans_df.to_csv(out_path)


def plot_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    feature_set_name: str,
    n_states: int,
) -> None:
    """
    Plot BTC closing price with decoded regime states overlaid as
    coloured scatter points and save to the figures directory.
    """
    plot_df = df.copy()
    plot_df["state"] = states

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["Date"], plot_df["btc_close"], alpha=0.35, label="BTC Close")

    for state in sorted(plot_df["state"].unique()):
        sub = plot_df[plot_df["state"] == state]
        plt.scatter(sub["Date"], sub["btc_close"], s=18, label=f"State {state}")

    pretty_name = feature_set_name.replace("_", " ")
    plt.title(f"BTC Regimes: {pretty_name} ({n_states}-State Gaussian HMM)")
    plt.xlabel("Date")
    plt.ylabel("BTC Close")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / f"{feature_set_name}_btc_regimes_{n_states}_states.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    n_states: int,
) -> dict:
    """
    Fit and evaluate a single HMM configuration.

    Splits data chronologically, scales features (fit on train only),
    fits the best HMM across multiple restarts, decodes states, and
    evaluates directional accuracy on the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full weekly dataset.
    feature_set_name : str
        Identifier for the feature family (used in output filenames).
    feature_cols : list of str
        Feature columns for this configuration.
    n_states : int
        Number of hidden states.

    Returns
    -------
    dict
        Result row for the comparison CSV.
    """
    train_df, test_df = chronological_split(df)
    X_train, X_test, scaler = scale_features(train_df, test_df, feature_cols)
    X_all = scaler.transform(df[feature_cols])

    model, best_seed, train_loglik = fit_best_hmm(X_train, n_states)

    test_loglik = model.score(X_test)
    aic, bic    = try_aic_bic(model, X_train)

    train_states = model.predict(X_train)
    test_states  = model.predict(X_test)
    all_states   = model.predict(X_all)

    state_probs, global_prob = state_up_probabilities(train_df, train_states)
    test_pred = predict_from_states(test_states, state_probs, global_prob)

    acc     = accuracy_score(test_df["next_week_up"], test_pred)
    bal_acc = balanced_accuracy_score(test_df["next_week_up"], test_pred)

    # Save per-run artefacts
    summarize_states(df, all_states, feature_cols).to_csv(
        TABLE_DIR / f"{feature_set_name}_state_summary_{n_states}_states.csv",
        index=False,
    )
    decoded = df.copy()
    decoded["state"] = all_states
    decoded.to_csv(
        TABLE_DIR / f"{feature_set_name}_decoded_states_{n_states}_states.csv",
        index=False,
    )
    save_transition_matrix(model, feature_set_name, n_states)
    plot_regimes(df, all_states, feature_set_name, n_states)

    return {
        "feature_set":           feature_set_name,
        "n_features":            len(feature_cols),
        "n_states":              n_states,
        "best_seed":             best_seed,
        "train_loglik":          train_loglik,
        "test_loglik":           test_loglik,
        "aic_train":             aic,
        "bic_train":             bic,
        "test_accuracy":         acc,
        "test_balanced_accuracy":bal_acc,
    }


def main() -> None:
    """Run all feature family / state count combinations and save results."""
    df = load_data()
    print(f"Loaded {len(df)} weekly rows.")

    results = []
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        print(f"\n=== {feature_set_name} ===")
        print("Features:", feature_cols)
        for n_states in STATE_OPTIONS:
            print(f"  Fitting {n_states}-state model...")
            result = evaluate_model(df, feature_set_name, feature_cols, n_states)
            results.append(result)

    results_df = (
        pd.DataFrame(results)
          .sort_values(["feature_set", "n_states"])
    )
    out_path = TABLE_DIR / "feature_family_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved comparison table: {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()