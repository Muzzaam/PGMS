# Inferring Latent Bitcoin Market Regimes with Hidden Markov Models

PGM Project — Muzzammil Soofie  
School of Computer Science and Applied Mathematics, University of the Witwatersrand

---

## Project Overview

This project applies a Gaussian Hidden Markov Model to infer latent Bitcoin market regimes from weekly observations. Five feature families are compared (price-only, psychology-linked, macro-financial, trader-style, and compact combined), and the best model is evaluated against directional prediction baselines and a masked-feature recovery experiment.

The full write-up is in `report/report.pdf`.

---

## Project Structure

```
pgm-btc-hmm/
├── data/
│   ├── raw/                  # downloaded daily CSVs
│   └── processed/
│       ├── daily_merged.csv
│       └── weekly_features.csv
├── results/
│   ├── figures/              # regime visualisation PNGs
│   └── tables/
│       ├── *.csv             # raw result tables
│       └── latex/            # LaTeX tables used in the report
├── report/
│   └── report.pdf
├── src/
│   ├── load_data.py          # download raw data and build weekly features
│   ├── features.py           # weekly feature engineering logic
│   ├── preprocess.py         # shared splitting and scaling utilities
│   ├── evaluate.py           # shared metric and prediction utilities
│   ├── hmm_compare.py        # main HMM experiment across feature families
│   ├── baselines.py          # baseline model comparison
│   ├── coverup_test.py       # masked-feature recovery experiment
│   └── build_report_tables.py # generate LaTeX tables from results CSVs
├── requirements.txt
└── README.md
```

---

## Setup

Extract the zip folder.
Open the extracted folder in File Explorer.
Click the address bar, type powershell, and press Enter.

Python 3.10+ is required. Create and activate a virtual environment first:


```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Then install dependencies:

```bash
you maybe have to update pip before running this next one

pip install -r requirements.txt
```

## How to Run

Run scripts from the project root in this order:


```bash
python src/load_data.py           # download data and build weekly_features.csv
python src/hmm_compare.py         # fit HMMs across all feature families
python src/baselines.py           # run baseline comparisons
python src/coverup_test.py        # run masked-feature recovery experiment
python src/build_report_tables.py # generate LaTeX tables
```

> **Note:** Pre-generated results are already included in `results/` and `data/processed/`. You do not need to rerun the pipeline to verify the report — the numbers in the report match the pre-generated CSVs exactly. Rerunning will reproduce identical results.

---

## Expected Output

After running the full pipeline:

- `results/tables/feature_family_comparison.csv` — HMM accuracy across feature families
- `results/tables/baseline_comparison.csv` — HMM vs baselines
- `results/tables/coverup_results.csv` — masked-feature MAE and RMSE
- `results/tables/latex/` — LaTeX table files used in the report
- `results/figures/` — regime visualisation plots

---

## Notes

- A convergence warning (`Model is not converging`) may appear for some restarts. This is expected and harmless — 10 random restarts are used per configuration and the best solution by training log-likelihood is selected automatically.
- `baselines.py` loads the HMM result directly from `feature_family_comparison.csv` rather than refitting, so `hmm_compare.py` must be run first.
- All feature scaling is fitted on the training period only and applied to the test period to prevent leakage.
