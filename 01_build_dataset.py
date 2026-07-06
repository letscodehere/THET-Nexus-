import pandas as pd
import numpy as np
import os

BASE = "/home/claude"
TOUR = f"{BASE}/tourism_repo"
WDI  = f"{BASE}/wdi_repo/indicators"
OUT  = f"{BASE}/paper/data"

# ---------- 1. Country whitelist (real countries, not aggregates) ----------
meta = pd.read_csv(f"{TOUR}/data/input/meta/metadata_country_arvl.csv")
real_countries = meta.loc[meta["Region"].notna(), ["Country Code","Region","IncomeGroup","TableName"]].copy()
real_countries.columns = ["country_code","region","income_group","country_name"]
print("Real countries:", real_countries.shape[0])

# ---------- 2. Tourism arrivals (ST.INT.ARVL) ----------
tour_out = pd.read_csv(f"{TOUR}/data/output/tourism_indicators.csv")
tour_out = tour_out[tour_out["indicator_code"] == "ST.INT.ARVL"].copy()
tour_out = tour_out[pd.to_numeric(tour_out["year"], errors="coerce").notna()]
tour_out["year"] = tour_out["year"].astype(int)
tour_out = tour_out.rename(columns={"country_code":"country_code","indicator_value":"tourist_arrivals"})
tour_out = tour_out[["country_code","year","tourist_arrivals"]]
tour_out = tour_out[tour_out["country_code"].isin(real_countries["country_code"])]
print("Tourism arrivals rows:", tour_out.shape[0], "years:", tour_out.year.min(), "-", tour_out.year.max())

# ---------- 3. WDI indicator loader ----------
def load_wdi(code, newname):
    df = pd.read_csv(f"{WDI}/{code}/data.csv")
    df = df.rename(columns={"Country Code":"country_code","Year":"year","Value":newname})
    df = df[["country_code","year",newname]]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["country_code"].isin(real_countries["country_code"])]
    return df

co2   = load_wdi("en.atm.co2e.pc", "co2_pc")           # CO2 emissions, metric tons per capita
life  = load_wdi("sp.dyn.le00.in", "life_exp")          # life expectancy at birth (years)
bband = load_wdi("it.net.bbnd.p2", "broadband_p100")    # fixed broadband subs per 100 people
mobile= load_wdi("it.cel.sets.p2", "mobile_p100")       # mobile cellular subs per 100 people
beds  = load_wdi("sh.med.beds.zs", "hosp_beds_p1000")   # hospital beds per 1,000 people
rcpt  = load_wdi("st.int.rcpt.xp.zs", "tour_receipts_pctexp")  # tourism receipts, % total exports
pm25  = load_wdi("en.atm.pm25.mc.m3", "pm25")            # PM2.5 air pollution, mean annual exposure
renew = load_wdi("eg.fec.rnew.zs", "renew_energy_pct")   # renewable energy consumption % total
rnd   = load_wdi("gb.xpd.rsdv.gd.zs", "rnd_pct_gdp")     # R&D expenditure % GDP
pop   = load_wdi("sp.pop.totl", "population")             # total population
gdppc = load_wdi("ny.gdp.pcap.cd", "gdp_pc")               # GDP per capita, current US$
urban = load_wdi("sp.urb.totl.in.zs", "urban_pct")         # urban population % of total

# ---------- 4. Merge into single panel ----------
panel = tour_out.copy()
for d in [co2, life, bband, mobile, beds, rcpt, pm25, renew, rnd, pop, gdppc, urban]:
    panel = panel.merge(d, on=["country_code","year"], how="left")

panel = panel.merge(real_countries, on="country_code", how="left")
panel = panel[(panel.year >= 1995) & (panel.year <= 2019)]

print(panel.shape)
print(panel.isna().mean().round(3))

panel.to_csv(f"{OUT}/panel_full_raw.csv", index=False)
print("Saved raw merged panel:", panel.shape)
