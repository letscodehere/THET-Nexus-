import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import pickle
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
BASE = "/home/claude/paper"
df = pd.read_csv(f"{BASE}/data/panel_analysis.csv")

# country-level (entity) held-out split so the test set contains ENTIRELY UNSEEN countries
countries = df["country_code"].unique()
rng = np.random.RandomState(42)
rng.shuffle(countries)
n_test = int(len(countries) * 0.2)
test_countries = set(countries[:n_test])
train_mask = ~df["country_code"].isin(test_countries)

feat_co2 = ["log_arrivals_pc", "broadband_p100", "mobile_p100", "log_gdp_pc",
            "urban_pct", "renew_energy_pct", "log_pop"]
feat_life = ["pm25", "broadband_p100", "mobile_p100", "log_gdp_pc",
             "urban_pct", "log_arrivals_pc", "renew_energy_pct"]

def run_models(features, target, label):
    X, y = df[features], df[target]
    Xtr, Xte = X[train_mask], X[~train_mask]
    ytr, yte = y[train_mask], y[~train_mask]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    results = {}
    preds = {}

    lin = Ridge(alpha=1.0).fit(Xtr_s, ytr)
    preds["Ridge"] = lin.predict(Xte_s)

    rf = RandomForestRegressor(n_estimators=400, max_depth=7, min_samples_leaf=3,
                                random_state=42, n_jobs=-1).fit(Xtr, ytr)
    preds["Random Forest"] = rf.predict(Xte)

    xgbm = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgbm.fit(Xtr, ytr)
    preds["XGBoost"] = xgbm.predict(Xte)

    for name, pred in preds.items():
        results[name] = {
            "R2": round(r2_score(yte, pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
            "MAE": round(mean_absolute_error(yte, pred), 3),
        }
    return results, {"rf": rf, "xgb": xgbm, "ridge": lin, "scaler": scaler,
                      "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "features": features}

res_co2, art_co2 = run_models(feat_co2, "co2_pc", "CO2 per capita")
res_life, art_life = run_models(feat_life, "life_exp", "Life expectancy")

rows = []
for model in ["Ridge", "Random Forest", "XGBoost"]:
    rows.append({"Model": model,
                 "R2 (CO2 pc)": res_co2[model]["R2"], "RMSE (CO2 pc)": res_co2[model]["RMSE"],
                 "MAE (CO2 pc)": res_co2[model]["MAE"],
                 "R2 (Life Exp.)": res_life[model]["R2"], "RMSE (Life Exp.)": res_life[model]["RMSE"],
                 "MAE (Life Exp.)": res_life[model]["MAE"]})
t6 = pd.DataFrame(rows)
t6.to_csv(f"{BASE}/tables/table6_ml_performance.csv", index=False)
print("=== TABLE 6: ML model performance (held-out unseen countries, n_test_countries=%d) ===" % n_test)
print(t6.to_string(index=False))

# Save artifacts for figures
with open(f"{BASE}/data/ml_artifacts.pkl", "wb") as f:
    pickle.dump({"co2": art_co2, "life": art_life}, f)

# SHAP values for CO2 XGBoost model (best nonlinear model) -> Figure 7
explainer = shap.TreeExplainer(art_co2["xgb"])
shap_values = explainer.shap_values(art_co2["Xte"])
with open(f"{BASE}/data/shap_co2.pkl", "wb") as f:
    pickle.dump({"shap_values": shap_values, "X": art_co2["Xte"], "features": feat_co2}, f)

print("\nFeature importance (XGBoost, CO2 model), mean |SHAP|:")
mean_abs = np.abs(shap_values).mean(axis=0)
for f, v in sorted(zip(feat_co2, mean_abs), key=lambda x: -x[1]):
    print(f"  {f}: {v:.3f}")

# SHAP for life expectancy XGBoost model -> Figure 8 (dependence plot: PM2.5 x broadband)
explainer_life = shap.TreeExplainer(art_life["xgb"])
shap_values_life = explainer_life.shap_values(art_life["Xte"])
with open(f"{BASE}/data/shap_life.pkl", "wb") as f:
    pickle.dump({"shap_values": shap_values_life, "X": art_life["Xte"], "features": feat_life}, f)

print("\nFeature importance (XGBoost, Life Expectancy model), mean |SHAP|:")
mean_abs_life = np.abs(shap_values_life).mean(axis=0)
for f, v in sorted(zip(feat_life, mean_abs_life), key=lambda x: -x[1]):
    print(f"  {f}: {v:.3f}")
