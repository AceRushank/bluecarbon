"""
src/eda_and_preprocess.py
--------------------------
Task 1 — Blue Carbon MRV Pipeline (Worthington et al. Global Mangrove Dataset)

Data:  Zenodo 10.5281/zenodo.11283186
       AGB_Carbon_Benefits_Data.csv  (3,983 typological units, global)
       SOC_Benefits_Data.csv         (3,983 typological units, global)
       Country_Statistics.csv        (122 countries, incl. India)

What this script does:
  1. Merges AGB + SOC on the `Type` (unit ID) key.
  2. Engineers target variables: total carbon, carbon density.
  3. Identifies India's statistics from Country_Statistics.csv and
     annotates Indian-context values as reference benchmarks.
  4. Saves two plots and the EDA summary report.

Outputs:
  outputs/indian_mangrove_carbon_distribution.png
  outputs/restoration_carbon_potential.png
  reports/eda_summary.txt

Run from project root:
    python src/eda_and_preprocess.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
AGB_PATH  = os.path.join("data", "AGB_Carbon_Benefits_Data.csv")
SOC_PATH  = os.path.join("data", "SOC_Benefits_Data.csv")
CS_PATH   = os.path.join("data", "Country_Statistics.csv")
REPORT    = os.path.join("reports", "eda_summary.txt")
OUT_DIR   = "outputs"
os.makedirs("reports", exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT     = "#c9d1d9"
GRID     = "#21262d"
INDIA_ORANGE  = "#ff7b2c"
GREEN_AGB     = "#3fb950"
BLUE_SOC      = "#58a6ff"
YELLOW_TOTAL  = "#e3b341"

# ── 1. Load & validate ─────────────────────────────────────────────────────────
print("=" * 65)
print("  Blue Carbon MRV — EDA (Worthington et al. Mangrove Dataset)")
print("=" * 65)

for path in [AGB_PATH, SOC_PATH, CS_PATH]:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Required file not found: {path}")

agb = pd.read_csv(AGB_PATH)
soc = pd.read_csv(SOC_PATH)
cs  = pd.read_csv(CS_PATH)

print(f"\n[OK] AGB  : {agb.shape[0]:,} units x {agb.shape[1]} columns")
print(f"[OK] SOC  : {soc.shape[0]:,} units x {soc.shape[1]} columns")
print(f"[OK] Country stats: {cs.shape[0]} countries")

# ── 2. Merge AGB + SOC on 'Type' (the unit ID) ───────────────────────────────
df = agb.merge(soc, on="Type", how="inner")
print(f"[OK] Merged dataset: {df.shape[0]:,} units x {df.shape[1]} columns")
assert df.shape[0] == 3983, "Expected 3,983 units after merge"

# ── 3. Feature engineering ────────────────────────────────────────────────────
# Extract the typology class (prefix before underscore)
df["typology_class"] = df["Type"].str.split("_").str[0]

# Total carbon stocks: secured + restoration potential (MgC = tC)
df["total_AGB_MgC"]     = df["Mean_AGB_Carbon_Secure"] + df["Mean_AGB_Carbon_Restor"]
df["total_SOC_MgC"]     = df["Mean_SOC_Carbon_Secure"] + df["Mean_SOC_Carbon_Restor"]
df["total_carbon_MgC"]  = df["total_AGB_MgC"] + df["total_SOC_MgC"]

# AGB serves as a proxy for area (biomass stock correlates with extent)
# Carbon density: total carbon per AGB biomass unit (relative density index)
df["AGB_ha"]            = df["AGB"]                       # AGB stock in MgC
df["carbon_density"]    = np.where(
    df["AGB_ha"] > 0,
    df["total_carbon_MgC"] / df["AGB_ha"],
    np.nan
)

# Restoration carbon efficiency (additional tC gained per unit restorable biomass)
df["restoration_efficiency"] = np.where(
    df["AGB_ha"] > 0,
    (df["Mean_AGB_Carbon_Restor"] + df["Mean_SOC_Carbon_Restor"]) / df["AGB_ha"],
    np.nan
)

# Uncertainty band width (model confidence proxy)
df["AGB_uncertainty"]   = df["Hi_AGB_Carbon_Secure"] - df["Low_AGB_Carbon_Secure"]
df["SOC_uncertainty"]   = df["Hi_SOC_Carbon_Secure"] - df["Low_SOC_Carbon_Secure"]

print(f"\n[OK] Engineered features: typology_class, total_AGB_MgC, total_SOC_MgC,")
print(f"     total_carbon_MgC, carbon_density, restoration_efficiency")

# ── 4. India reference from Country_Statistics ────────────────────────────────
india_row  = cs[cs["Name"] == "India"].iloc[0]
india_data = {
    "country"              : "India",
    "area_km2"             : india_row["Area_2020_km2"],
    "gross_loss_km2"       : india_row["Gross_Loss"],
    "restorable_km2"       : india_row["Restorable_km2"],
    "restoration_score"    : india_row["OVERALL_SRE"],
    "mean_AGB_secure_MgC"  : india_row["MeanAGB_Secure"],
    "mean_AGB_restor_MgC"  : india_row["MeanAGB_Restor"],
    "mean_SOC_secure_MgC"  : india_row["MeanSOC_Secure"],
    "mean_SOC_restor_MgC"  : india_row["MeanSOC_Restor"],
    "area_rank"            : int(india_row["Area_Rank"]),
    "proportion_restorable": india_row["Proportion_Restorable"],
}
india_data["total_carbon_MgC"] = (
    india_data["mean_AGB_secure_MgC"] + india_data["mean_AGB_restor_MgC"] +
    india_data["mean_SOC_secure_MgC"] + india_data["mean_SOC_restor_MgC"]
)

print(f"\n[India Summary]")
for k, v in india_data.items():
    if isinstance(v, float):
        print(f"  {k:<30}: {v:,.2f}")
    else:
        print(f"  {k:<30}: {v}")

# ── 5. Global typology-class statistics ───────────────────────────────────────
typ_stats = (df.groupby("typology_class")
               .agg(n_units=("Type","count"),
                    mean_AGB_secure=("Mean_AGB_Carbon_Secure","mean"),
                    mean_SOC_secure=("Mean_SOC_Carbon_Secure","mean"),
                    mean_total_carbon=("total_carbon_MgC","mean"),
                    mean_restoration_efficiency=("restoration_efficiency","mean"))
               .reset_index()
               .sort_values("mean_total_carbon", ascending=False))

print(f"\n[Global typology class breakdown]")
print(typ_stats.to_string(index=False))


# ── 6. Plot 1: AGB vs SOC Carbon by Typology Class (India context) ─────────────
# Build top country comparison: India vs top-5 mangrove nations
top_countries = cs.nlargest(6, "Area_2020_km2")
country_names = top_countries["Name"].tolist()
agb_secure    = top_countries["MeanAGB_Secure"].values / 1e6   # MgC -> TgC for display
agb_restor    = top_countries["MeanAGB_Restor"].values / 1e6
soc_secure    = top_countries["MeanSOC_Secure"].values / 1e6
soc_restor    = top_countries["MeanSOC_Restor"].values / 1e6

fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
fig.suptitle(
    "Global Mangrove Carbon Analysis — Worthington et al.\n"
    "Indian Context vs Top Mangrove Nations (AGB + SOC Carbon Stocks)",
    color="#ffffff", fontsize=14, y=0.98
)

x    = np.arange(len(country_names))
bar_w = 0.20

# ─ Panel A: Stacked secured AGB vs SOC for top countries ─
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(PANEL_BG)
b1 = ax1.bar(x - bar_w*1.5, agb_secure, bar_w, label="AGB Secured",     color=GREEN_AGB,    alpha=0.9)
b2 = ax1.bar(x - bar_w*0.5, agb_restor, bar_w, label="AGB Restoration",  color=GREEN_AGB,    alpha=0.45, hatch="//")
b3 = ax1.bar(x + bar_w*0.5, soc_secure, bar_w, label="SOC Secured",      color=BLUE_SOC,     alpha=0.9)
b4 = ax1.bar(x + bar_w*1.5, soc_restor, bar_w, label="SOC Restoration",  color=BLUE_SOC,     alpha=0.45, hatch="//")

# Highlight India bar group
india_idx = country_names.index("India") if "India" in country_names else -1
if india_idx >= 0:
    ax1.axvspan(india_idx - 0.5, india_idx + 0.5, color=INDIA_ORANGE, alpha=0.08, zorder=0)
    ax1.text(india_idx, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
             "India", color=INDIA_ORANGE, fontsize=9, ha="center", va="bottom", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(country_names, color=TEXT, fontsize=10)
ax1.set_ylabel("Carbon Stock (TgC)", color=TEXT, fontsize=11)
ax1.set_title("AGB vs SOC Carbon — Top 6 Mangrove Nations (Secured + Restoration Potential)",
              color=TEXT, fontsize=11)
ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID, fontsize=9, ncol=4)
ax1.tick_params(colors=TEXT)
ax1.grid(axis="y", color=GRID, linestyle="--", linewidth=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID)

# ─ Panel B: India — AGB vs SOC breakdown (Secured vs Restorable) ─
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(PANEL_BG)

categories  = ["AGB Secured", "AGB Restorable", "SOC Secured", "SOC Restorable"]
india_vals  = [
    india_data["mean_AGB_secure_MgC"] / 1e3,
    india_data["mean_AGB_restor_MgC"] / 1e3,
    india_data["mean_SOC_secure_MgC"] / 1e3,
    india_data["mean_SOC_restor_MgC"] / 1e3,
]
colors_bar  = [GREEN_AGB, "#1d8348", BLUE_SOC, "#1a5276"]
bars = ax2.bar(categories, india_vals, color=colors_bar, alpha=0.88,
               edgecolor="#ffffff22", linewidth=0.5)

for bar, val in zip(bars, india_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.0f} GgC", ha="center", va="bottom", color=TEXT, fontsize=8)

ax2.set_ylabel("Carbon Stock (GgC = 1000 MgC)", color=TEXT, fontsize=10)
ax2.set_title(f"India — Carbon Breakdown\n(Area: {india_data['area_km2']:,.0f} km², Rank #{india_data['area_rank']})",
              color=TEXT, fontsize=10)
ax2.tick_params(colors=TEXT)
ax2.set_xticklabels(categories, color=TEXT, fontsize=8.5, rotation=10, ha="right")
ax2.grid(axis="y", color=GRID, linestyle="--", linewidth=0.5)
for sp in ax2.spines.values(): sp.set_edgecolor(GRID)

# ─ Panel C: Global distribution of AGB:SOC ratio by typology class ─
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(PANEL_BG)

# Log-scale scatter: AGB secured vs SOC secured per unit
sample = df.sample(min(800, len(df)), random_state=42)
sc = ax3.scatter(
    np.log1p(sample["Mean_AGB_Carbon_Secure"]),
    np.log1p(sample["Mean_SOC_Carbon_Secure"]),
    c=sample["restoration_efficiency"].clip(0, 20),
    cmap="YlOrRd", s=8, alpha=0.65, edgecolors="none"
)
# India reference lines (log scale of India's mean AGB/SOC secure values)
india_agb_log = np.log1p(india_data["mean_AGB_secure_MgC"])
india_soc_log = np.log1p(india_data["mean_SOC_secure_MgC"])
ax3.axvline(india_agb_log, color=INDIA_ORANGE, linestyle="--", linewidth=1.2, alpha=0.7)
ax3.axhline(india_soc_log, color=INDIA_ORANGE, linestyle="--", linewidth=1.2, alpha=0.7,
            label="India mean")

cbar3 = fig.colorbar(sc, ax=ax3, pad=0.02)
cbar3.set_label("Restoration Efficiency", color=TEXT, fontsize=8)
cbar3.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar3.ax.yaxis.get_ticklabels(), color=TEXT)

ax3.set_xlabel("log(AGB Carbon Secured, MgC)", color=TEXT, fontsize=9)
ax3.set_ylabel("log(SOC Carbon Secured, MgC)", color=TEXT, fontsize=9)
ax3.set_title("AGB vs SOC (3,983 units, log scale)\nColour = Restoration Efficiency",
              color=TEXT, fontsize=10)
ax3.legend(facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID, fontsize=8)
ax3.tick_params(colors=TEXT)
ax3.grid(color=GRID, linestyle="--", linewidth=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor(GRID)

out1 = os.path.join(OUT_DIR, "indian_mangrove_carbon_distribution.png")
plt.savefig(out1, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"\n[OK] Plot 1 saved -> {out1}")


# ── 7. Plot 2: Restoration potential vs total carbon gain ─────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6), facecolor=DARK_BG)
fig2.suptitle(
    "Restoration Carbon Potential — Global Mangrove Units\n"
    "(Worthington et al., 3,983 typological units)",
    color="#ffffff", fontsize=13, y=1.01
)

# ─ Left: AGB restoration potential vs total carbon gain ─
ax_l = axes2[0]
ax_l.set_facecolor(PANEL_BG)

restor = df["Mean_AGB_Carbon_Restor"].clip(0, df["Mean_AGB_Carbon_Restor"].quantile(0.98))
total  = df["total_carbon_MgC"].clip(0, df["total_carbon_MgC"].quantile(0.98))

sc2 = ax_l.scatter(
    np.log1p(restor),
    np.log1p(total),
    c=np.log1p(df["Mean_SOC_Carbon_Restor"].clip(0, df["Mean_SOC_Carbon_Restor"].quantile(0.98))),
    cmap="plasma", s=6, alpha=0.5, edgecolors="none"
)
cbar2 = fig2.colorbar(sc2, ax=ax_l, pad=0.02)
cbar2.set_label("log(SOC Restoration MgC)", color=TEXT, fontsize=8)
cbar2.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT)

ax_l.set_xlabel("log(AGB Restoration Potential, MgC)", color=TEXT, fontsize=10)
ax_l.set_ylabel("log(Total Carbon Stock, MgC)",         color=TEXT, fontsize=10)
ax_l.set_title("Restoration Potential vs Total Carbon Gain\n(each dot = 1 mangrove unit)",
               color=TEXT, fontsize=10)
ax_l.tick_params(colors=TEXT)
ax_l.grid(color=GRID, linestyle="--", linewidth=0.5)
for sp in ax_l.spines.values(): sp.set_edgecolor(GRID)

# ─ Right: Cumulative carbon distribution by typology class ─
ax_r = axes2[1]
ax_r.set_facecolor(PANEL_BG)

typ_agg = (df.groupby("typology_class")
             .agg(total_carbon=("total_carbon_MgC","sum"),
                  total_agb=("total_AGB_MgC","sum"),
                  total_soc=("total_SOC_MgC","sum"),
                  n_units=("Type","count"))
             .sort_values("total_carbon", ascending=True))

y_pos  = np.arange(len(typ_agg))
w_agb  = typ_agg["total_agb"].values  / 1e6
w_soc  = typ_agg["total_soc"].values  / 1e6

ax_r.barh(y_pos, w_agb, label="AGB", color=GREEN_AGB, alpha=0.88, edgecolor="#ffffff11")
ax_r.barh(y_pos, w_soc, left=w_agb, label="SOC",  color=BLUE_SOC,  alpha=0.88, edgecolor="#ffffff11")

# Annotate n_units
for i, (_, row) in enumerate(typ_agg.iterrows()):
    total_val = (row["total_agb"] + row["total_soc"]) / 1e6
    ax_r.text(total_val + 0.5, i, f" n={row['n_units']}", va="center",
              color=TEXT, fontsize=8)

ax_r.set_yticks(y_pos)
ax_r.set_yticklabels(typ_agg.index, color=TEXT, fontsize=9)
ax_r.set_xlabel("Total Carbon Stock (TgC)", color=TEXT, fontsize=10)
ax_r.set_title("Carbon by Mangrove Typology Class\n(AGB + SOC, global)",
               color=TEXT, fontsize=10)
ax_r.legend(facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID, fontsize=9)
ax_r.tick_params(colors=TEXT)
ax_r.grid(axis="x", color=GRID, linestyle="--", linewidth=0.5)
for sp in ax_r.spines.values(): sp.set_edgecolor(GRID)

out2 = os.path.join(OUT_DIR, "restoration_carbon_potential.png")
plt.savefig(out2, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"[OK] Plot 2 saved -> {out2}")


# ── 8. Save EDA summary report ────────────────────────────────────────────────
lines = [
    "=" * 65,
    "  Blue Carbon MRV — EDA Summary (Worthington et al. 2024)",
    "=" * 65,
    "",
    "Source: Zenodo DOI 10.5281/zenodo.11283186",
    f"AGB file : {AGB_PATH}  ({agb.shape[0]:,} units x {agb.shape[1]} cols)",
    f"SOC file : {SOC_PATH}  ({soc.shape[0]:,} units x {soc.shape[1]} cols)",
    f"Merged   : {df.shape[0]:,} units x {df.shape[1]} cols  (join key: Type)",
    "",
    "-- Column Schema (AGB) " + "-" * 42,
    f"  {list(agb.columns)}",
    "",
    "-- Column Schema (SOC) " + "-" * 42,
    f"  {list(soc.columns)}",
    "",
    "-- Engineered Features " + "-" * 42,
    "  typology_class      : prefix of Type (Delta, Estuarine, etc.)",
    "  total_AGB_MgC       : Mean_AGB_Secure + Mean_AGB_Restor",
    "  total_SOC_MgC       : Mean_SOC_Secure + Mean_SOC_Restor",
    "  total_carbon_MgC    : total_AGB + total_SOC  [PRIMARY TARGET]",
    "  carbon_density      : total_carbon / AGB  (relative density)",
    "  restoration_efficiency: (AGB_Restor + SOC_Restor) / AGB",
    "  AGB_uncertainty     : Hi_AGB_Secure - Low_AGB_Secure",
    "  SOC_uncertainty     : Hi_SOC_Secure - Low_SOC_Secure",
    "",
    "-- Global Target Statistics (total_carbon_MgC) " + "-" * 18,
    f"  Mean  : {df['total_carbon_MgC'].mean():>15,.2f} MgC",
    f"  Median: {df['total_carbon_MgC'].median():>15,.2f} MgC",
    f"  Std   : {df['total_carbon_MgC'].std():>15,.2f} MgC",
    f"  Min   : {df['total_carbon_MgC'].min():>15,.2f} MgC",
    f"  Max   : {df['total_carbon_MgC'].max():>15,.2f} MgC",
    "",
    "-- Typology Class Breakdown " + "-" * 37,
    typ_stats.to_string(index=False),
    "",
    "-- India Reference (Country_Statistics.csv) " + "-" * 21,
    f"  Mangrove area 2020 : {india_data['area_km2']:,.2f} km2  (Rank #{india_data['area_rank']} globally)",
    f"  Gross loss         : {india_data['gross_loss_km2']:,.2f} km2",
    f"  Restorable area    : {india_data['restorable_km2']:,.2f} km2 ({india_data['proportion_restorable']:.2f}%)",
    f"  Restoration score  : {india_data['restoration_score']:.4f}",
    f"  AGB carbon (sec.)  : {india_data['mean_AGB_secure_MgC']:,.2f} MgC",
    f"  AGB carbon (rest.) : {india_data['mean_AGB_restor_MgC']:,.2f} MgC",
    f"  SOC carbon (sec.)  : {india_data['mean_SOC_secure_MgC']:,.2f} MgC",
    f"  SOC carbon (rest.) : {india_data['mean_SOC_restor_MgC']:,.2f} MgC",
    f"  Total carbon stock : {india_data['total_carbon_MgC']:,.2f} MgC",
    "",
    "  Indian mangrove zones covered by this dataset:",
    "  Sundarbans (West Bengal), Bhitarkanika (Odisha),",
    "  Coringa (Andhra Pradesh), Pichavaram (Tamil Nadu),",
    "  Gulf of Kutch (Gujarat), Andaman & Nicobar Islands",
    "=" * 65,
]

report_text = "\n".join(lines)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"[OK] EDA summary saved -> {REPORT}")

# Export merged df for use by train_model.py
merged_path = os.path.join("data", "merged_agb_soc.csv")
df.to_csv(merged_path, index=False)
print(f"[OK] Merged dataset saved -> {merged_path}  ({len(df):,} rows)")
print("\nEDA complete.\n")
