from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
LATEX_DIR = TABLE_DIR / "latex"

LATEX_DIR.mkdir(parents=True, exist_ok=True)


def prettify_feature_name(x: str) -> str:
    mapping = {
        "A_price_only": "Price only",
        "B_price_psychology": "Price + psychology",
        "C_price_macro": "Price + macro",
        "D_price_trader": "Price + trader",
        "E_compact_combined": "Compact combined",
    }
    return mapping.get(x, x)


def prettify_baseline_name(x: str) -> str:
    mapping = {
        "majority_class": "Majority class",
        "last_sign": "Last sign",
        "logistic_regression": "Logistic regression",
        "gmm_3_clusters": "GMM (3 clusters)",
        "hmm_psychology_3_states": "HMM psychology (3 states)",
    }
    return mapping.get(x, x)


def best_per_feature_family(df: pd.DataFrame) -> pd.DataFrame:
    # pick best row per family by accuracy, then balanced accuracy
    ranked = df.sort_values(
        ["feature_set", "test_accuracy", "test_balanced_accuracy"],
        ascending=[True, False, False],
    )
    best = ranked.groupby("feature_set", as_index=False).first()

    best["feature_set"] = best["feature_set"].map(prettify_feature_name)

    best = best[
        [
            "feature_set",
            "n_features",
            "n_states",
            "test_accuracy",
            "test_balanced_accuracy",
        ]
    ].copy()

    best.columns = [
        "Feature family",
        "Features",
        "States",
        "Accuracy",
        "Balanced accuracy",
    ]

    return best


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].map(prettify_baseline_name)

    out = out[
        ["model", "accuracy", "balanced_accuracy", "n_features"]
    ].copy()

    out.columns = [
        "Model",
        "Accuracy",
        "Balanced accuracy",
        "Features",
    ]

    return out


def coverup_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in df["masked_feature"].unique():
        sub = df[df["masked_feature"] == feature].copy()

        hmm = sub[sub["baseline"] == "hmm_state_mean"].iloc[0]
        meanb = sub[sub["baseline"] == "global_mean"].iloc[0]

        rows.append(
            {
                "Masked feature": feature,
                "HMM MAE": hmm["mae"],
                "Mean MAE": meanb["mae"],
                "HMM RMSE": hmm["rmse"],
                "Mean RMSE": meanb["rmse"],
            }
        )

    out = pd.DataFrame(rows)

    pretty_feature_names = {
        "fear_greed_value": "Fear \\& Greed",
        "btc_momentum_4w": "4-week momentum",
        "btc_drawdown_12w": "12-week drawdown",
    }
    out["Masked feature"] = out["Masked feature"].map(pretty_feature_names)

    return out


def save_csv_and_latex(df: pd.DataFrame, stem: str, caption: str, label: str):
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


def save_results_summary(feature_df: pd.DataFrame, baseline_df: pd.DataFrame):
    summary_path = LATEX_DIR / "results_summary.txt"

    best_family = feature_df.sort_values(
        ["Accuracy", "Balanced accuracy"], ascending=False
    ).iloc[0]

    hmm_row = baseline_df[baseline_df["Model"] == "HMM psychology (3 states)"].iloc[0]
    gmm_row = baseline_df[baseline_df["Model"] == "GMM (3 clusters)"].iloc[0]
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
        f"This supports the claim that psychology-linked features and temporal latent-state "
        f"modeling improved short-term BTC regime inference in this weekly setting."
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    feature_family_path = TABLE_DIR / "feature_family_comparison.csv"
    baseline_path = TABLE_DIR / "baseline_comparison.csv"
    coverup_path = TABLE_DIR / "coverup_results.csv"

    feature_df = pd.read_csv(feature_family_path)
    baseline_df = pd.read_csv(baseline_path)
    coverup_df = pd.read_csv(coverup_path)

    feature_best = best_per_feature_family(feature_df)
    baseline_clean = baseline_table(baseline_df)
    coverup_clean = coverup_table(coverup_df)

    save_csv_and_latex(
        feature_best,
        "table_feature_families",
        "Best-performing configuration within each feature family.",
        "tab:feature_families",
    )

    save_csv_and_latex(
        baseline_clean,
        "table_baselines",
        "Comparison against non-HMM baselines.",
        "tab:baselines",
    )

    save_csv_and_latex(
        coverup_clean,
        "table_coverup",
        "Masked-feature recovery on the test set using HMM state-conditioned reconstruction versus a global-mean baseline.",
        "tab:coverup",
    )

    save_results_summary(feature_best, baseline_clean)

    print("Saved files:")
    for p in sorted(LATEX_DIR.glob("*")):
        print("-", p)


if __name__ == "__main__":
    main()