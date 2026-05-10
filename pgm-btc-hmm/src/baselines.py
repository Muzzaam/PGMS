from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "baseline_comparison.csv"

TRAIN_RATIO = 0.8
RANDOM_STATE = 42

PSYCHOLOGY_FEATURES = [
    "btc_log_return",
    "btc_4w_vol",
    "btc_volume_log_change",
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    required = ["Date", "next_week_up"] + PSYCHOLOGY_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def majority_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    majority_class = int(train_df["next_week_up"].mode().iloc[0])
    y_pred = np.full(len(test_df), majority_class, dtype=int)
    return evaluate_predictions("majority_class", test_df["next_week_up"], y_pred)


def last_sign_baseline(test_df: pd.DataFrame) -> dict:
    # predict next week will have same sign as current week's return
    y_pred = (test_df["btc_log_return"] > 0).astype(int).values
    return evaluate_predictions("last_sign", test_df["next_week_up"], y_pred)


def logistic_regression_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[PSYCHOLOGY_FEATURES])
    X_test = scaler.transform(test_df[PSYCHOLOGY_FEATURES])

    y_train = train_df["next_week_up"].values
    y_test = test_df["next_week_up"].values

    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result = evaluate_predictions("logistic_regression", y_test, y_pred)
    result["n_features"] = len(PSYCHOLOGY_FEATURES)
    return result


def gmm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, n_components: int = 3) -> dict:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[PSYCHOLOGY_FEATURES])
    X_test = scaler.transform(test_df[PSYCHOLOGY_FEATURES])

    y_test = test_df["next_week_up"].values

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    gmm.fit(X_train)

    train_clusters = gmm.predict(X_train)
    test_clusters = gmm.predict(X_test)

    tmp = train_df.copy()
    tmp["cluster"] = train_clusters

    cluster_up_probs = tmp.groupby("cluster")["next_week_up"].mean().to_dict()
    global_prob = tmp["next_week_up"].mean()

    y_pred_prob = np.array([cluster_up_probs.get(c, global_prob) for c in test_clusters])
    y_pred = (y_pred_prob >= 0.5).astype(int)

    result = evaluate_predictions("gmm_3_clusters", y_test, y_pred)
    result["n_features"] = len(PSYCHOLOGY_FEATURES)
    return result


def hmm_reference_row() -> dict:
    # reference from your current best model
    return {
        "model": "hmm_psychology_3_states",
        "accuracy": 0.5833333333333334,
        "balanced_accuracy": 0.5715102974828375,
        "n_features": len(PSYCHOLOGY_FEATURES),
    }


def main():
    df = load_data()
    train_df, test_df = chronological_split(df)

    results = []
    results.append(majority_baseline(train_df, test_df))
    results.append(last_sign_baseline(test_df))
    results.append(logistic_regression_baseline(train_df, test_df))
    results.append(gmm_baseline(train_df, test_df, n_components=3))
    results.append(hmm_reference_row())

    results_df = pd.DataFrame(results)
    results_df = results_df[["model", "accuracy", "balanced_accuracy", "n_features"]]
    results_df.to_csv(OUT_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()