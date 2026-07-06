import pandas as pd
import numpy as np

BASE = "/home/claude/paper"
df = pd.read_csv(f"{BASE}/data/panel_full_raw.csv")

# ---------- Derived variables ----------
df["tourist_arrivals_p1000"] = df["tourist_arrivals"] / df["population"] * 1000
df["log_arrivals_pc"] = np.log(df["tourist_arrivals_p1000"].clip(lower=0.001))
df["log_gdp_pc"] = np.log(df["gdp_pc"].clip(lower=1))
df["log_pop"] = np.log(df["population"].clip(lower=1))

core_vars = ["log_arrivals_pc", "co2_pc", "life_exp", "pm25", "broadband_p100",
             "mobile_p100", "log_gdp_pc", "urban_pct", "renew_energy_pct", "population"]

clean = df.dropna(subset=core_vars).copy()
clean = clean[(clean["tourist_arrivals_p1000"] > 0) & (clean["co2_pc"] > 0) & (clean["pm25"] > 0)]

# winsorize the 1st/99th pct of highly skewed vars to limit extreme-outlier leverage
for v in ["co2_pc", "tourist_arrivals_p1000", "pm25"]:
    lo, hi = clean[v].quantile([0.01, 0.99])
    clean[v + "_w"] = clean[v].clip(lo, hi)

clean.to_csv(f"{BASE}/data/panel_analysis.csv", index=False)
print("Final analytic panel:", clean.shape)
print("Countries:", clean.country_code.nunique(), "| Years:", clean.year.min(), "-", clean.year.max())
print("Country-years per country (avg):", round(clean.shape[0] / clean.country_code.nunique(), 1))

# ---------- Table 2: Descriptive statistics ----------
desc_vars = {
    "tourist_arrivals_p1000": "Tourist arrivals per 1,000 population",
    "co2_pc": "CO2 emissions (metric tons per capita)",
    "pm25": "PM2.5 air pollution (\u00b5g/m\u00b3, mean annual exposure)",
    "life_exp": "Life expectancy at birth (years)",
    "broadband_p100": "Fixed broadband subscriptions (per 100 people)",
    "mobile_p100": "Mobile cellular subscriptions (per 100 people)",
    "gdp_pc": "GDP per capita (current US$)",
    "urban_pct": "Urban population (% of total)",
    "renew_energy_pct": "Renewable energy consumption (% of total final energy)",
}

rows = []
for v, label in desc_vars.items():
    s = clean[v]
    rows.append({
        "Variable": label,
        "N": int(s.count()),
        "Mean": round(s.mean(), 2),
        "SD": round(s.std(), 2),
        "Min": round(s.min(), 2),
        "Max": round(s.max(), 2),
    })
tab2 = pd.DataFrame(rows)
tab2.to_csv(f"{BASE}/tables/table2_descriptives.csv", index=False)
print("\nTable 2 - Descriptive Statistics")
print(tab2.to_string(index=False))

# ---------- Table 3: Correlation matrix ----------
corr_vars = ["log_arrivals_pc", "co2_pc", "pm25", "life_exp", "broadband_p100",
             "mobile_p100", "log_gdp_pc", "urban_pct", "renew_energy_pct"]
corr_labels = ["Tourism (ln)", "CO2 pc", "PM2.5", "Life Exp.", "Broadband",
               "Mobile", "GDP pc (ln)", "Urban %", "Renew. %"]
corr = clean[corr_vars].corr(method="pearson").round(2)
corr.index = corr_labels
corr.columns = corr_labels
corr.to_csv(f"{BASE}/tables/table3_correlations.csv")
print("\nTable 3 - Correlation Matrix")
print(corr.to_string())
