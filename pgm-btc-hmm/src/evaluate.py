"""
evaluate.py
-----------
Shared evaluation utilities for the Bitcoin HMM regime project.

Provides metric computation and state-based prediction helpers used by
hmm_compare.py, baselines.py, and coverup_test.py. Centralising these
ensures consistent metric definitions across all experiments.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def directional_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute accuracy and balanced accuracy for a binary direction prediction.

    Parameters
    ----------
    name : str
        Label for the model or method (used as the 'model' key in output).
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict with keys: 'model', 'accuracy', 'balanced_accuracy'
    """
    return {
        "model":             name,
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def state_up_probabilities(
    train_df: pd.DataFrame,
    train_states: np.ndarray,
    target_col: str = "next_week_up",
) -> tuple[dict, float]:
    """
    Estimate the probability that next week is up for each hidden state,
    using the training set only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame containing the target column.
    train_states : np.ndarray
        Decoded hidden state sequence for the training rows.
    target_col : str
        Column name of the binary target. Default 'next_week_up'.

    Returns
    -------
    state_probs : dict
        Mapping from state index to P(next_week_up=1 | state).
    global_prob : float
        Fallback probability (overall training mean) for unseen states.
    """
    tmp = train_df.copy()
    tmp["state"] = train_states
    state_probs  = tmp.groupby("state")[target_col].mean().to_dict()
    global_prob  = tmp[target_col].mean()
    return state_probs, global_prob


def predict_from_states(
    test_states: np.ndarray,
    state_probs: dict,
    global_prob: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert decoded test states into binary direction predictions.

    Each test week is assigned the training-set up-probability for its
    decoded state; weeks whose state was not seen in training fall back
    to the global training mean.

    Parameters
    ----------
    test_states : np.ndarray
        Decoded hidden state sequence for the test rows.
    state_probs : dict
        State-to-probability mapping from state_up_probabilities().
    global_prob : float
        Fallback probability for unseen states.
    threshold : float
        Decision threshold. Default 0.5.

    Returns
    -------
    np.ndarray of int (0 or 1)
    """
    probs = np.array([state_probs.get(s, global_prob) for s in test_states])
    return (probs >= threshold).astype(int)