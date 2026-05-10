from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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

STATE_OPTIONS = [3, 4]
TRAIN_RATIO = 0.8
N_RESTARTS = 10
BASE_RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required_cols = [
        "Date",
        "btc_close",
        "next_week_up",
        "btc_log_return",
        "btc_4w_vol",
        "btc_volume_log_change",
        "fear_greed_value",
        "dxy_log_return",
        "dgs10_change",
        "btc_momentum_4w",
        "btc_drawdown_12w",
        "btc_ma_spread_4_12",
        "btc_price_vs_sma12",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy().dropna().reset_index(drop=True)

    # drop last row because next-week label is not genuinely observed after it
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def fit_best_hmm(X_train: np.ndarray, n_states: int):
    best_model = None
    best_seed = None
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
                best_model = model
                best_seed = seed
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"All HMM fits failed for n_states={n_states}")

    return best_model, best_seed, best_train_loglik


def try_aic_bic(model: GaussianHMM, X: np.ndarray):
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


def state_up_probabilities(train_df: pd.DataFrame, train_states: np.ndarray):
    tmp = train_df.copy()
    tmp["state"] = train_states
    probs = tmp.groupby("state")["next_week_up"].mean().to_dict()
    global_prob = tmp["next_week_up"].mean()
    return probs, global_prob


def summarize_states(df: pd.DataFrame, states: np.ndarray, feature_cols):
    tmp = df.copy()
    tmp["state"] = states

    agg_dict = {
        "count": ("state", "size"),
        "mean_return": ("btc_log_return", "mean"),
        "mean_volatility": ("btc_4w_vol", "mean"),
        "mean_volume_change": ("btc_volume_log_change", "mean"),
        "prob_next_week_up": ("next_week_up", "mean"),
    }

    optional_map = {
        "fear_greed_value": "mean_fear_greed",
        "btc_momentum_4w": "mean_momentum_4w",
        "btc_drawdown_12w": "mean_drawdown_12w",
        "dxy_log_return": "mean_dxy_log_return",
        "dgs10_change": "mean_dgs10_change",
        "btc_ma_spread_4_12": "mean_ma_spread_4_12",
        "btc_price_vs_sma12": "mean_price_vs_sma12",
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


def save_transition_matrix(model: GaussianHMM, feature_set_name: str, n_states: int):
    trans_df = pd.DataFrame(
        model.transmat_,
        columns=[f"to_state_{i}" for i in range(n_states)],
        index=[f"from_state_{i}" for i in range(n_states)],
    )
    out_path = TABLE_DIR / f"{feature_set_name}_transition_matrix_{n_states}_states.csv"
    trans_df.to_csv(out_path)


def plot_regimes(df: pd.DataFrame, states: np.ndarray, feature_set_name: str, n_states: int):
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


def evaluate_model(df: pd.DataFrame, feature_set_name: str, feature_cols, n_states: int) -> dict:
    train_df, test_df = chronological_split(df, TRAIN_RATIO)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    X_all = scaler.transform(df[feature_cols])

    model, best_seed, train_loglik = fit_best_hmm(X_train, n_states)
    test_loglik = model.score(X_test)

    aic, bic = try_aic_bic(model, X_train)

    train_states = model.predict(X_train)
    test_states = model.predict(X_test)
    all_states = model.predict(X_all)

    state_probs, global_prob = state_up_probabilities(train_df, train_states)
    test_pred_prob = np.array([state_probs.get(s, global_prob) for s in test_states])
    test_pred = (test_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(test_df["next_week_up"], test_pred)
    bal_acc = balanced_accuracy_score(test_df["next_week_up"], test_pred)

    state_summary = summarize_states(df, all_states, feature_cols)
    state_summary.to_csv(
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
        "feature_set": feature_set_name,
        "n_features": len(feature_cols),
        "n_states": n_states,
        "best_seed": best_seed,
        "train_loglik": train_loglik,
        "test_loglik": test_loglik,
        "aic_train": aic,
        "bic_train": bic,
        "test_accuracy": acc,
        "test_balanced_accuracy": bal_acc,
    }


def main():
    df = load_data()
    results = []

    print(f"Loaded {len(df)} weekly rows")

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        print(f"\n=== {feature_set_name} ===")
        print("Features:", feature_cols)

        for n_states in STATE_OPTIONS:
            print(f"  Fitting {n_states}-state model...")
            result = evaluate_model(df, feature_set_name, feature_cols, n_states)
            results.append(result)

    results_df = pd.DataFrame(results).sort_values(["feature_set", "n_states"])
    results_df.to_csv(TABLE_DIR / "feature_family_comparison.csv", index=False)

    print("\nSaved comparison table:")
    print(TABLE_DIR / "feature_family_comparison.csv")
    print("\nResults:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()