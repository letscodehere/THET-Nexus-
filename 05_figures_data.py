import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#444444", "figure.dpi": 300,
})
COLORS = {"hi": "#1b4965", "umi": "#5fa8d3", "lmi": "#bee9e8", "li": "#cae9ff",
          "line1": "#1b4965", "line2": "#c9184a", "bar": "#457b9d"}

BASE = "/home/claude/paper"
FIG = f"{BASE}/figures"
df = pd.read_csv(f"{BASE}/data/panel_analysis.csv")
raw = pd.read_csv(f"{BASE}/data/panel_full_raw.csv")

# ============ Figure 2: Global tourism growth & digital adoption (1998-2014) ============
g = df.groupby("year").agg(arrivals=("tourist_arrivals", "sum"),
                            broadband=("broadband_p100", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax1.plot(g.year, g.arrivals/1e6, color=COLORS["line1"], lw=2.5, marker="o", ms=4, label="Total tourist arrivals")
ax1.set_ylabel("Total international tourist arrivals\n(millions, sample of 180 countries)", color=COLORS["line1"])
ax1.set_xlabel("Year")
ax1.tick_params(axis="y", labelcolor=COLORS["line1"])
ax2 = ax1.twinx()
ax2.plot(g.year, g.broadband, color=COLORS["line2"], lw=2.5, ls="--", marker="s", ms=4, label="Mean broadband penetration")
ax2.set_ylabel("Fixed broadband subscriptions\n(mean, per 100 people)", color=COLORS["line2"])
ax2.tick_params(axis="y", labelcolor=COLORS["line2"])
ax1.spines["right"].set_visible(True)
fig.suptitle("Figure 2. Co-evolution of Global Tourism Growth and Digital Adoption, 1998-2014", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_tourism_digital_trend.png", bbox_inches="tight")
plt.close(fig)

# ============ Figure 3: Dual-axis trend of CO2 pc and tourism intensity (median) ============
g3 = df.groupby("year").agg(co2=("co2_pc", "median"),
                             arr=("tourist_arrivals_p1000", "median")).reset_index()
fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax1.plot(g3.year, g3.co2, color=COLORS["line1"], lw=2.5, marker="o", ms=4)
ax1.set_ylabel("Median CO2 emissions\n(metric tons per capita)", color=COLORS["line1"])
ax1.set_xlabel("Year")
ax1.tick_params(axis="y", labelcolor=COLORS["line1"])
ax2 = ax1.twinx()
ax2.plot(g3.year, g3.arr, color=COLORS["line2"], lw=2.5, ls="--", marker="s", ms=4)
ax2.set_ylabel("Median tourist arrivals\n(per 1,000 population)", color=COLORS["line2"])
ax2.tick_params(axis="y", labelcolor=COLORS["line2"])
ax1.spines["right"].set_visible(True)
fig.suptitle("Figure 3. Median CO2 Emissions and Tourism Intensity, 1998-2014", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_co2_tourism_trend.png", bbox_inches="tight")
plt.close(fig)

# ============ Figure 4: Scatter tourist arrivals (ln) vs CO2 pc, by income group ============
country_avg = df.groupby(["country_code", "income_group"]).agg(
    log_arr=("log_arrivals_pc", "mean"), co2=("co2_pc", "mean")).reset_index()
order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
cmap = {"Low income": "#f77f00", "Lower middle income": "#fcbf49",
        "Upper middle income": "#5fa8d3", "High income": "#1b4965"}
fig, ax = plt.subplots(figsize=(7, 5))
for grp in order:
    sub = country_avg[country_avg.income_group == grp]
    ax.scatter(sub.log_arr, sub.co2, s=45, alpha=0.75, color=cmap[grp], label=grp, edgecolor="white", linewidth=0.4)
z = np.polyfit(country_avg.log_arr, country_avg.co2, 1)
xs = np.linspace(country_avg.log_arr.min(), country_avg.log_arr.max(), 100)
ax.plot(xs, np.polyval(z, xs), color="black", lw=1.8, ls=":")
ax.set_xlabel("Tourist arrivals per 1,000 population (ln), country mean 1998-2014")
ax.set_ylabel("CO2 emissions per capita (metric tons), country mean")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Figure 4. Tourism Intensity and Carbon Emissions Across 180 Countries", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_scatter_tourism_co2.png", bbox_inches="tight")
plt.close(fig)

# ============ Figure 5: Digitalization trend by income group ============
g5 = df.groupby(["year", "income_group"])["broadband_p100"].mean().reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for grp in order:
    sub = g5[g5.income_group == grp]
    ax.plot(sub.year, sub.broadband_p100, marker="o", ms=3.5, lw=2.2, color=cmap[grp], label=grp)
ax.set_xlabel("Year")
ax.set_ylabel("Fixed broadband subscriptions\n(mean, per 100 people)")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Figure 5. Digital Transformation Trajectories by Income Group, 1998-2014", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_digital_by_income.png", bbox_inches="tight")
plt.close(fig)

# ============ Figure 6: Simple-slopes plot for H3 moderation ============
with open(f"{BASE}/data/h3_model.pkl", "rb") as f:
    m = pickle.load(f)
arr_range = np.linspace(-2*m["arr_sd"], 2*m["arr_sd"], 100)
low_bb = -m["bb_sd"]   # -1 SD broadband (centered)
high_bb = m["bb_sd"]   # +1 SD broadband (centered)
pred_low = m["b_arr"]*arr_range + m["b_bb"]*low_bb + m["b_int"]*arr_range*low_bb
pred_high = m["b_arr"]*arr_range + m["b_bb"]*high_bb + m["b_int"]*arr_range*high_bb
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(arr_range + m["arr_mean"], pred_low, color="#c9184a", lw=2.5, label="Low digitalization (\u22121 SD broadband)")
ax.plot(arr_range + m["arr_mean"], pred_high, color="#1b4965", lw=2.5, label="High digitalization (+1 SD broadband)")
ax.set_xlabel("Tourism intensity, ln(arrivals per 1,000 population)")
ax.set_ylabel("Predicted CO2 emissions per capita\n(centered, metric tons)")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Figure 6. Simple-Slopes Plot: Digitalization as a Moderator of the\nTourism-Emissions Relationship (H3)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_simple_slopes.png", bbox_inches="tight")
plt.close(fig)

print("Figures 2-6 saved.")
