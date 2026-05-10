from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "coverup_results.csv"

TRAIN_RATIO = 0.8
RANDOM_STATE = 42
N_RESTARTS = 10

FEATURES = [
    "btc_log_return",
    "btc_4w_vol",
    "btc_volume_log_change",
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]

MASK_TARGETS = [
    "fear_greed_value",
    "btc_momentum_4w",
    "btc_drawdown_12w",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    required = ["Date"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[required].dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def fit_best_hmm(X_train: np.ndarray, n_states: int = 3):
    best_model = None
    best_score = -np.inf

    for i in range(N_RESTARTS):
        seed = RANDOM_STATE + i
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
        raise RuntimeError("HMM fitting failed for all restarts.")

    return best_model


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def reconstruct_masked_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    masked_feature: str,
    n_states: int = 3,
):
    visible_cols = [c for c in feature_cols if c != masked_feature]

    # Fit scaler/HMM only on visible features
    scaler = StandardScaler()
    X_train_visible = scaler.fit_transform(train_df[visible_cols])
    X_test_visible = scaler.transform(test_df[visible_cols])

    hmm = fit_best_hmm(X_train_visible, n_states=n_states)

    train_states = hmm.predict(X_train_visible)
    test_states = hmm.predict(X_test_visible)

    tmp = train_df.copy()
    tmp["state"] = train_states

    # Estimate masked feature by state-conditional mean from training set
    state_means = tmp.groupby("state")[masked_feature].mean().to_dict()
    global_mean = tmp[masked_feature].mean()

    y_true = test_df[masked_feature].values
    y_pred = np.array([state_means.get(s, global_mean) for s in test_states])

    return {
        "masked_feature": masked_feature,
        "n_states": n_states,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, masked_feature: str):
    y_true = test_df[masked_feature].values
    mean_value = train_df[masked_feature].mean()
    y_pred = np.full(len(test_df), mean_value)

    return {
        "masked_feature": masked_feature,
        "baseline": "global_mean",
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def main():
    df = load_data()
    train_df, test_df = chronological_split(df)

    results = []

    for target in MASK_TARGETS:
        hmm_result = reconstruct_masked_feature(
            train_df=train_df,
            test_df=test_df,
            feature_cols=FEATURES,
            masked_feature=target,
            n_states=3,
        )
        hmm_result["baseline"] = "hmm_state_mean"
        results.append(hmm_result)

        base_result = mean_baseline(train_df, test_df, target)
        results.append(base_result)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["masked_feature", "baseline", "n_states", "mae", "rmse"]
    ]
    results_df.to_csv(OUT_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()