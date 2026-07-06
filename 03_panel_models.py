import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

BASE = "/home/claude/paper"
df = pd.read_csv(f"{BASE}/data/panel_analysis.csv")

df["arr_c"] = df["log_arrivals_pc"] - df["log_arrivals_pc"].mean()
df["bb_c"]  = df["broadband_p100"] - df["broadband_p100"].mean()
df["pm25_c"] = df["pm25"] - df["pm25"].mean()
df["arr_x_bb"]  = df["arr_c"] * df["bb_c"]
df["pm25_x_bb"] = df["pm25_c"] * df["bb_c"]

panel = df.set_index(["country_code", "year"])

def fit(vars_, dep):
    y = panel[dep]
    X = panel[vars_].assign(const=1)
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)

res_h1 = fit(["arr_c", "log_gdp_pc", "urban_pct", "renew_energy_pct"], "co2_pc")
res_h2 = fit(["pm25_c", "log_gdp_pc", "urban_pct", "mobile_p100"], "life_exp")
res_h3 = fit(["arr_c", "bb_c", "arr_x_bb", "log_gdp_pc", "urban_pct", "renew_energy_pct"], "co2_pc")
res_h4 = fit(["pm25_c", "bb_c", "pm25_x_bb", "log_gdp_pc", "urban_pct", "mobile_p100"], "life_exp")

def cell(res, v):
    if v not in res.params.index:
        return ""
    coef, se, p = res.params[v], res.std_errors[v], res.pvalues[v]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{coef:.3f}{stars} ({se:.3f})"

def meta(res):
    return int(res.nobs), round(float(res.rsquared_within), 3), res.entity_info.total

rows4 = ["Tourism intensity, ln (H1)", "PM2.5 (H2)", "GDP per capita, ln", "Urban population (%)",
         "Renewable energy (%)", "Mobile subscriptions (per 100)", "N", "R2 (within)", "Countries"]
col_h1 = [cell(res_h1, "arr_c"), "", cell(res_h1, "log_gdp_pc"), cell(res_h1, "urban_pct"),
          cell(res_h1, "renew_energy_pct"), ""]
col_h2 = ["", cell(res_h2, "pm25_c"), cell(res_h2, "log_gdp_pc"), cell(res_h2, "urban_pct"),
          "", cell(res_h2, "mobile_p100")]
n1, r1, c1 = meta(res_h1); n2, r2, c2 = meta(res_h2)
col_h1 += [n1, r1, c1]
col_h2 += [n2, r2, c2]
t4 = pd.DataFrame({"DV: CO2 emissions per capita (H1)": col_h1,
                    "DV: Life expectancy at birth (H2)": col_h2}, index=rows4)
t4.to_csv(f"{BASE}/tables/table4_main_effects.csv")
print("=== TABLE 4 ===")
print(t4.to_string())

rows5 = ["Tourism intensity, ln, centered (H3)", "PM2.5, centered (H4)",
         "Digitalization (broadband/100, centered)", "Tourism x Digitalization",
         "PM2.5 x Digitalization", "GDP per capita, ln", "Urban population (%)",
         "Renewable energy (%)", "Mobile subscriptions (per 100)", "N", "R2 (within)", "Countries"]
col_h3 = [cell(res_h3, "arr_c"), "", cell(res_h3, "bb_c"), cell(res_h3, "arr_x_bb"), "",
          cell(res_h3, "log_gdp_pc"), cell(res_h3, "urban_pct"), cell(res_h3, "renew_energy_pct"), ""]
col_h4 = ["", cell(res_h4, "pm25_c"), cell(res_h4, "bb_c"), "", cell(res_h4, "pm25_x_bb"),
          cell(res_h4, "log_gdp_pc"), cell(res_h4, "urban_pct"), "", cell(res_h4, "mobile_p100")]
n3, r3, c3 = meta(res_h3); n4, r4, c4 = meta(res_h4)
col_h3 += [n3, r3, c3]
col_h4 += [n4, r4, c4]
t5 = pd.DataFrame({"DV: CO2 emissions per capita (H3)": col_h3,
                    "DV: Life expectancy at birth (H4)": col_h4}, index=rows5)
t5.to_csv(f"{BASE}/tables/table5_moderation.csv")
print("\n=== TABLE 5 ===")
print(t5.to_string())

import pickle
with open(f"{BASE}/data/h3_model.pkl", "wb") as f:
    pickle.dump({
        "b_arr": res_h3.params["arr_c"], "b_bb": res_h3.params["bb_c"],
        "b_int": res_h3.params["arr_x_bb"], "arr_mean": df["log_arrivals_pc"].mean(),
        "bb_mean": df["broadband_p100"].mean(), "bb_sd": df["broadband_p100"].std(),
        "arr_sd": df["log_arrivals_pc"].std(),
    }, f)

print("\n--- Full H3 model summary ---")
print(res_h3.summary)
print("\n--- Full H4 model summary ---")
print(res_h4.summary)
