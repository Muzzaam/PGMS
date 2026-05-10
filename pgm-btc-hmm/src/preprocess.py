"""
preprocess.py
-------------
Shared preprocessing utilities for the Bitcoin HMM regime project.

Provides chronological train/test splitting and feature scaling helpers
used consistently across hmm_compare.py, baselines.py, and coverup_test.py.
Centralising these here ensures that the split ratio and scaling convention
(fit on train, apply to test) are never accidentally varied between scripts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TRAIN_RATIO = 0.8


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train and test sets without shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered DataFrame (earliest row first).
    train_ratio : float
        Fraction of rows to use for training. Default 0.8.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return train_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data and transform both splits.

    The scaler is fitted exclusively on training rows to prevent leakage
    from the test period into feature normalisation.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    feature_cols : list of str
        Column names to scale.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices.
    scaler : StandardScaler
        Fitted scaler (can be applied to new data with scaler.transform).
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])
    return X_train, X_test, scaler