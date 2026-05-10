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


FEATURES = [
    "btc_log_return",
    "btc_4w_vol",
    "btc_volume_log_change",
]

STATE_OPTIONS = [2, 3, 4]
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required = ["Date", "btc_close", "next_week_up"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df = df.dropna().reset_index(drop=True)

    # Drop last row because its next_week_up label is not truly known from a following week
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


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


def summarize_states(df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["state"] = states

    summary = (
        tmp.groupby("state")
        .agg(
            count=("state", "size"),
            mean_return=("btc_log_return", "mean"),
            mean_volatility=("btc_4w_vol", "mean"),
            mean_volume_change=("btc_volume_log_change", "mean"),
            prob_next_week_up=("next_week_up", "mean"),
        )
        .reset_index()
        .sort_values("state")
    )
    return summary


def save_transition_matrix(model: GaussianHMM, n_states: int):
    trans_df = pd.DataFrame(
        model.transmat_,
        columns=[f"to_state_{i}" for i in range(n_states)],
        index=[f"from_state_{i}" for i in range(n_states)],
    )
    trans_df.to_csv(TABLE_DIR / f"transition_matrix_{n_states}_states.csv")


def plot_regimes(df: pd.DataFrame, states: np.ndarray, n_states: int):
    plot_df = df.copy()
    plot_df["state"] = states

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["Date"], plot_df["btc_close"], alpha=0.35, label="BTC Close")

    for state in sorted(plot_df["state"].unique()):
        sub = plot_df[plot_df["state"] == state]
        plt.scatter(sub["Date"], sub["btc_close"], s=18, label=f"State {state}")

    plt.title(f"BTC Regimes from {n_states}-State Gaussian HMM")
    plt.xlabel("Date")
    plt.ylabel("BTC Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"btc_regimes_{n_states}_states.png", dpi=200)
    plt.close()


def evaluate_model(df: pd.DataFrame, n_states: int) -> dict:
    train_df, test_df = chronological_split(df, TRAIN_RATIO)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES])
    X_test = scaler.transform(test_df[FEATURES])
    X_all = scaler.transform(df[FEATURES])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)

    train_loglik = model.score(X_train)
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

    state_summary = summarize_states(df, all_states)
    state_summary.to_csv(TABLE_DIR / f"state_summary_{n_states}_states.csv", index=False)

    full_with_states = df.copy()
    full_with_states["state"] = all_states
    full_with_states.to_csv(TABLE_DIR / f"decoded_states_{n_states}_states.csv", index=False)

    save_transition_matrix(model, n_states)
    plot_regimes(df, all_states, n_states)

    return {
        "n_states": n_states,
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

    for n_states in STATE_OPTIONS:
        print(f"Fitting {n_states}-state HMM...")
        result = evaluate_model(df, n_states)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("n_states")
    results_df.to_csv(TABLE_DIR / "model_comparison_price_only.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {TABLE_DIR / 'model_comparison_price_only.csv'}")
    for n_states in STATE_OPTIONS:
        print(f"- {TABLE_DIR / f'state_summary_{n_states}_states.csv'}")
        print(f"- {TABLE_DIR / f'transition_matrix_{n_states}_states.csv'}")
        print(f"- {TABLE_DIR / f'decoded_states_{n_states}_states.csv'}")
        print(f"- {FIG_DIR / f'btc_regimes_{n_states}_states.png'}")


if __name__ == "__main__":
    main()