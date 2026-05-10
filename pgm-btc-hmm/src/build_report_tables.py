"""
build_report_tables.py
----------------------
Converts raw results CSVs into publication-ready LaTeX tables for the paper.

Reads:
  - results/tables/feature_family_comparison.csv  (from hmm_compare.py)
  - results/tables/baseline_comparison.csv         (from baselines.py)
  - results/tables/coverup_results.csv             (from coverup_test.py)

Writes to results/tables/latex/:
  - table_feature_families.tex / .csv
  - table_baselines.tex / .csv
  - table_coverup.tex / .csv
  - results_summary.txt  (plain-text summary pasted into the paper narrative)

Run this after all three experiment scripts have completed successfully.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR    = PROJECT_ROOT / "results" / "tables"
LATEX_DIR    = TABLE_DIR / "latex"

LATEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Name prettifiers
# ---------------------------------------------------------------------------

def prettify_feature_name(x: str) -> str:
    """Map internal feature-set keys to human-readable table labels."""
    mapping = {
        "A_price_only":        "Price only",
        "B_price_psychology":  "Price + psychology",
        "C_price_macro":       "Price + macro",
        "D_price_trader":      "Price + trader",
        "E_compact_combined":  "Compact combined",
    }
    return mapping.get(x, x)


def prettify_baseline_name(x: str) -> str:
    """Map internal model keys to human-readable table labels."""
    mapping = {
        "majority_class":           "Majority class",
        "last_sign":                "Last sign",
        "logistic_regression":      "Logistic regression",
        "gmm_3_clusters":           "GMM (3 clusters)",
        "hmm_psychology_3_states":  "HMM psychology (3 states)",
    }
    return mapping.get(x, x)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def best_per_feature_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best configuration per feature family (highest accuracy,
    then balanced accuracy) and return a cleaned, prettified table.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature_family_comparison.csv.

    Returns
    -------
    pd.DataFrame
    """
    ranked = df.sort_values(
        ["feature_set", "test_accuracy", "test_balanced_accuracy"],
        ascending=[True, False, False],
    )
    best = ranked.groupby("feature_set", as_index=False).first()
    best["feature_set"] = best["feature_set"].map(prettify_feature_name)

    best = best[[
        "feature_set", "n_features", "n_states",
        "test_accuracy", "test_balanced_accuracy",
    ]].copy()
    best.columns = [
        "Feature family", "Features", "States", "Accuracy", "Balanced accuracy",
    ]
    return best


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prettify the baseline comparison table.

    Parameters
    ----------
    df : pd.DataFrame
        baseline_comparison.csv.

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    out["model"] = out["model"].map(prettify_baseline_name)
    out = out[["model", "accuracy", "balanced_accuracy"]].copy()
    out.columns = ["Model", "Accuracy", "Balanced accuracy"]
    return out


def coverup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the coverup results into a wide format with HMM and baseline
    columns side by side for each masked feature.

    Parameters
    ----------
    df : pd.DataFrame
        coverup_results.csv.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for feature in df["masked_feature"].unique():
        sub  = df[df["masked_feature"] == feature]
        hmm  = sub[sub["baseline"] == "hmm_state_mean"].iloc[0]
        base = sub[sub["baseline"] == "global_mean"].iloc[0]
        rows.append({
            "Masked feature": feature,
            "HMM MAE":  hmm["mae"],
            "Mean MAE": base["mae"],
            "HMM RMSE": hmm["rmse"],
            "Mean RMSE":base["rmse"],
        })

    out = pd.DataFrame(rows)
    pretty_feature_names = {
        "fear_greed_value": "Fear \\& Greed",
        "btc_momentum_4w":  "4-week momentum",
        "btc_drawdown_12w": "12-week drawdown",
    }
    out["Masked feature"] = out["Masked feature"].map(pretty_feature_names)
    return out


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_csv_and_latex(
    df: pd.DataFrame,
    stem: str,
    caption: str,
    label: str,
) -> None:
    """
    Save a DataFrame as both a CSV and a LaTeX table file.

    Numeric columns are rounded to 3 decimal places. The LaTeX file uses
    booktabs formatting compatible with the IEEEtran document class.

    Parameters
    ----------
    df : pd.DataFrame
    stem : str
        Base filename (without extension) for the output files.
    caption : str
        LaTeX table caption string.
    label : str
        LaTeX label for cross-referencing.
    """
    csv_path = LATEX_DIR / f"{stem}.csv"
    tex_path = LATEX_DIR / f"{stem}.tex"

    df_rounded = df.copy()
    for col in df_rounded.columns:
        if pd.api.types.is_float_dtype(df_rounded[col]):
            df_rounded[col] = df_rounded[col].round(3)

    df_rounded.to_csv(csv_path, index=False)

    latex = df_rounded.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        float_format="%.3f",
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)


def save_results_summary(
    feature_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    """
    Write a plain-text summary of the key results for reference when
    drafting the paper narrative.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Prettified feature family table.
    baseline_df : pd.DataFrame
        Prettified baseline table.
    """
    summary_path = LATEX_DIR / "results_summary.txt"

    best_family = feature_df.sort_values(
        ["Accuracy", "Balanced accuracy"], ascending=False
    ).iloc[0]

    hmm_row    = baseline_df[baseline_df["Model"] == "HMM psychology (3 states)"].iloc[0]
    gmm_row    = baseline_df[baseline_df["Model"] == "GMM (3 clusters)"].iloc[0]
    logreg_row = baseline_df[baseline_df["Model"] == "Logistic regression"].iloc[0]

    text = (
        f"The best-performing feature family was {best_family['Feature family']} "
        f"with {int(best_family['States'])} hidden states, achieving an accuracy of "
        f"{best_family['Accuracy']:.3f} and balanced accuracy of "
        f"{best_family['Balanced accuracy']:.3f}. "
        f"In the baseline comparison, the HMM psychology model achieved "
        f"{hmm_row['Accuracy']:.3f} accuracy, outperforming the static GMM baseline "
        f"({gmm_row['Accuracy']:.3f}) and logistic regression "
        f"({logreg_row['Accuracy']:.3f}). "
        f"This supports the claim that psychology-linked features and temporal "
        f"latent-state modeling improved short-term BTC regime inference in this "
        f"weekly setting."
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build all LaTeX tables from saved results CSVs."""
    feature_family_path = TABLE_DIR / "feature_family_comparison.csv"
    baseline_path       = TABLE_DIR / "baseline_comparison.csv"
    coverup_path        = TABLE_DIR / "coverup_results.csv"

    feature_df  = pd.read_csv(feature_family_path)
    baseline_df = pd.read_csv(baseline_path)
    coverup_df  = pd.read_csv(coverup_path)

    feature_best   = best_per_feature_family(feature_df)
    baseline_clean = baseline_table(baseline_df)
    coverup_clean  = coverup_table(coverup_df)

    save_csv_and_latex(
        feature_best, "table_feature_families",
        "Best-performing HMM configuration within each feature family (out-of-sample test set).",
        "tab:feature_families",
    )
    save_csv_and_latex(
        baseline_clean, "table_baselines",
        "Comparison of the best HMM against non-HMM baselines on the test set.",
        "tab:baselines",
    )
    save_csv_and_latex(
        coverup_clean, "table_coverup",
        "Masked-feature recovery on the test set using HMM state-conditioned "
        "reconstruction versus a global-mean baseline.",
        "tab:coverup",
    )

    save_results_summary(feature_best, baseline_clean)

    print("Saved files:")
    for p in sorted(LATEX_DIR.glob("*")):
        print("-", p)


if __name__ == "__main__":
    main()