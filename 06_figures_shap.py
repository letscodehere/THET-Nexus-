import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

BASE = "/home/claude/paper"
FIG = f"{BASE}/figures"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 300})

label_map = {
    "log_arrivals_pc": "Tourism intensity (ln, arrivals/1,000 pop.)",
    "broadband_p100": "Broadband subscriptions (per 100)",
    "mobile_p100": "Mobile subscriptions (per 100)",
    "log_gdp_pc": "GDP per capita (ln)",
    "urban_pct": "Urban population (%)",
    "renew_energy_pct": "Renewable energy (%)",
    "log_pop": "Population (ln)",
    "pm25": "PM2.5 air pollution",
}

# ---------------- Figure 7: SHAP summary (CO2 model) ----------------
with open(f"{BASE}/data/shap_co2.pkl", "rb") as f:
    d = pickle.load(f)
X = d["X"].rename(columns=label_map)
fig = plt.figure(figsize=(7.5, 5))
shap.summary_plot(d["shap_values"], X, show=False, plot_size=None)
fig = plt.gcf()
fig.suptitle("Figure 7. SHAP Summary Plot: Drivers of CO2 Emissions per Capita\n(XGBoost model, held-out unseen countries)", y=1.04, fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig7_shap_co2.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------------- Figure 8: SHAP dependence plot (PM2.5 x broadband -> life expectancy) ----------------
with open(f"{BASE}/data/shap_life.pkl", "rb") as f:
    d2 = pickle.load(f)
X2 = d2["X"]
feats = d2["features"]
pm25_idx = feats.index("pm25")
bb_idx = feats.index("broadband_p100")

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(X2["pm25"], d2["shap_values"][:, pm25_idx], c=X2["broadband_p100"],
                 cmap="RdYlBu_r", s=32, alpha=0.85, edgecolor="white", linewidth=0.3)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Broadband subscriptions (per 100 people)")
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("PM2.5 air pollution (\u00b5g/m\u00b3, mean annual exposure)")
ax.set_ylabel("SHAP value for PM2.5\n(impact on predicted life expectancy, years)")
ax.set_title("Figure 8. SHAP Dependence Plot: PM2.5 Effect on Predicted Life\nExpectancy, Moderated by Digitalization (XGBoost model)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/fig8_shap_dependence_life.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print("Figures 7-8 saved.")
