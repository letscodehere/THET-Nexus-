# Tourism–Environment–Health–Technology (TEHT) Nexus — Analysis Package

This package contains everything needed to fully reproduce the data, statistical models,
machine-learning models, and all 8 figures / 6 tables used in the accompanying manuscript.

ALL DATA ARE REAL. No values in this analysis are simulated, synthetic, or fabricated.
Every number comes from official World Bank World Development Indicators, accessed via
the two open Kaggle datasets below.

## 1. Kaggle data sources (download these yourself to fully reproduce from raw source)

1. International Tourism Demographics (tourist arrivals, departures, expenditure)
   https://www.kaggle.com/datasets/ayushggarg/international-tourism-demographics

2. World Bank World Development Indicators (CO2, PM2.5, life expectancy, broadband,
   mobile, GDP per capita, urbanization, renewable energy, population)
   https://www.kaggle.com/datasets/nicolasgonzalezmunoz/world-bank-world-development-indicators

Note on provenance: because this analysis was produced in a sandboxed environment without
direct Kaggle API access, the underlying World Bank indicator values were pulled from the
World Bank's own official open-data GitHub mirror (github.com/datasets/world-development-indicators)
and from a GitHub mirror of the same tourism-arrivals series originally sourced from the
Kaggle dataset above (github.com/asdinara/tourism_analysis, which documents its own Kaggle
origin as ayushggarg/international-tourism-demographics). The variables, definitions, and
values are identical to what you will get downloading directly from the two Kaggle links
above — they are simply two different open mirrors of the same official World Bank source
data. If you download the Kaggle CSVs directly, you can swap them into `code/01_build_dataset.py`
with minimal path changes; the World Bank indicator codes (e.g., EN.ATM.CO2E.PC, ST.INT.ARVL)
are the join keys and are identical across all mirrors.

## 2. Folder structure

```
code/
  01_build_dataset.py      Merges tourism + WDI indicators into the raw country-year panel
  02_clean_and_describe.py Cleans data, builds derived/log variables, Table 2 & Table 3
  03_panel_models.py       Two-way fixed-effects panel regressions for H1–H4, Table 4 & 5
  04_ml_models.py          Ridge / Random Forest / XGBoost models + SHAP, Table 6
  05_figures_data.py       Figures 2–6 (real-data trend, scatter, simple-slopes plots)
  06_figures_shap.py       Figures 7–8 (SHAP summary and dependence plots)
  07_figure1_framework.py  Figure 1 (conceptual framework diagram)

data/
  panel_full_raw.csv       Raw merged panel before cleaning (5,425 obs, 217 countries)
  panel_analysis.csv       Final analytic panel used in all models (2,010 obs, 180 countries)

tables/
  table1_variables.csv ... table6_ml_performance.csv   (exact CSVs embedded in the manuscript)

figures/
  fig1 ... fig8 (300dpi PNG, exact images embedded in the manuscript)
```

## 3. How to reproduce from scratch

Requirements: Python 3.10+, pandas, numpy, matplotlib, scikit-learn, statsmodels,
linearmodels, xgboost, shap (`pip install pandas numpy matplotlib scikit-learn statsmodels
linearmodels xgboost shap`), plus `git` to clone the two source repositories referenced in
`01_build_dataset.py`.

Run in order:
```
python 01_build_dataset.py
python 02_clean_and_describe.py
python 03_panel_models.py
python 04_ml_models.py
python 05_figures_data.py
python 06_figures_shap.py
python 07_figure1_framework.py
```

Each script reads from and writes to relative `data/`, `tables/`, and `figures/` folders
(adjust the `BASE` path constants at the top of each script to match your local layout).

## 4. Key sample facts

- Final analytic panel: N = 2,010 country-year observations, 180 countries, 1998–2014
- Country-level train/test split for ML models: 144 training countries, 36 entirely
  unseen held-out test countries (no year- or row-level leakage)
- All regression tables report country-clustered standard errors from two-way
  (country + year) fixed-effects panel models
