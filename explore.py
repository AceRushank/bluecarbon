"""
explore.py
----------
Step 1 of the Blue Carbon MRV Pipeline.

Loads the mangrove carbon dataset, prints basic statistics, and
produces a scatter plot of NDVI vs Carbon Stock to visually confirm
the relationship before training any model.

Run:
    python explore.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "mangrove_carbon_samples.csv")
OUTPUT_DIR  = "outputs"
PLOT_PATH   = os.path.join(OUTPUT_DIR, "ndvi_vs_carbon.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Blue Carbon MRV — Data Exploration")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✔  Loaded {len(df)} samples from '{DATA_PATH}'")
print(f"   Columns : {list(df.columns)}\n")


# ── 2. Basic statistics ───────────────────────────────────────────────────────
print("─" * 60)
print("Descriptive Statistics")
print("─" * 60)

# Select only numeric feature columns for display
numeric_cols = ["B2_blue", "B3_green", "B4_red", "B8_nir", "B11_swir",
                "NDVI", "carbon_stock_tC_ha"]
print(df[numeric_cols].describe().round(4).to_string())

# Pearson correlation of each band with the carbon stock target
print("\n─" * 60)
print("Pearson Correlation with carbon_stock_tC_ha")
print("─" * 60)
corr = df[numeric_cols].corr()["carbon_stock_tC_ha"].drop("carbon_stock_tC_ha")
for feat, val in corr.sort_values(key=abs, ascending=False).items():
    direction = "↑ positive" if val > 0 else "↓ negative"
    print(f"  {feat:<12}: {val:+.4f}  ({direction})")


# ── 3. Scatter plot: NDVI vs Carbon Stock ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

scatter = ax.scatter(
    df["NDVI"],
    df["carbon_stock_tC_ha"],
    c=df["carbon_stock_tC_ha"],       # colour-code by carbon stock
    cmap="YlGn",
    s=100,
    edgecolors="#ffffff44",
    linewidths=0.5,
    zorder=3,
)

# Colour-bar as a mini legend
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Carbon Stock (tC/ha)", color="#cccccc", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#cccccc")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

# Axis labels & title
ax.set_xlabel("NDVI", color="#cccccc", fontsize=12)
ax.set_ylabel("Carbon Stock (tC/ha)", color="#cccccc", fontsize=12)
ax.set_title("NDVI vs Mangrove Carbon Stock\n(Bhitarkanika, Odisha — Sentinel-2)",
             color="#ffffff", fontsize=13, pad=14)

# Grid styling
ax.grid(color="#2a2d3e", linestyle="--", linewidth=0.6, zorder=1)
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")

# Annotate each site
for _, row in df.iterrows():
    ax.annotate(
        row["site_id"],
        xy=(row["NDVI"], row["carbon_stock_tC_ha"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color="#aaaacc",
    )

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, facecolor=fig.get_facecolor())
print(f"\n✔  Scatter plot saved → {PLOT_PATH}")
plt.show()

print("\nExploration complete.\n")
