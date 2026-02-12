import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =====================================================
# 1️⃣ LOAD 5 FILES & MERGE INTO ONE DATAFRAME
# =====================================================

files = {
    2005: "Africa_Country_AnnualMean_NO2_2005_FINAL.csv",
    2010: "Africa_Country_AnnualMean_NO2_2010_FINAL.csv",
    2015: "Africa_Country_AnnualMean_NO2_2015_FINAL.csv",
    2020: "Africa_Country_AnnualMean_NO2_2020_FINAL.csv",
    2025: "Africa_Country_AnnualMean_NO2_2025_FINAL.csv"
}

all_data = []

for year, file in files.items():
    df = pd.read_csv(file)

    # Tìm cột NO2 (vì mỗi file tên khác năm)
    no2_col = [c for c in df.columns if "Annual_Mean_NO2" in c][0]

    # Rename thành tên chung
    df = df.rename(columns={no2_col: "Mean_NO2"})

    df["Year"] = year

    all_data.append(df[["ISO3", "Mean_NO2", "Year"]])

df_all = pd.concat(all_data, ignore_index=True)

# =====================================================
# 2️⃣ COMPUTE TREND (SLOPE) FOR EACH COUNTRY
# =====================================================

def compute_slope(group):
    x = group["Year"].values.astype(float)
    y = np.log10(group["Mean_NO2"].values.astype(float))

    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - (sum_x)**2

    slope = numerator / denominator

    return slope


trend_df = (
    df_all
    .groupby("ISO3")
    .apply(compute_slope)
    .reset_index()
)

trend_df.columns = ["ISO3", "NO2_trend"]

# =====================================================
# 3️⃣ LOAD MAP & MERGE
# =====================================================

url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
world = gpd.read_file(url)

merged = world.merge(trend_df, left_on="ISO_A3", right_on="ISO3")

africa = merged[merged["CONTINENT"] == "Africa"]

# =====================================================
# 4️⃣ PLOT TREND MAP
# =====================================================

fig, ax = plt.subplots(figsize=(12,10))

norm = TwoSlopeNorm(
    vmin=africa["NO2_trend"].min(),
    vcenter=0,
    vmax=africa["NO2_trend"].max()
)

africa.plot(
    column="NO2_trend",
    cmap="RdBu_r",          # 🔵⚪🔴
    norm=norm,
    legend=True,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

ax.set_title("NO₂ Trend by Country (2005–2025)\n(log-scale slope)", fontsize=14)
ax.axis("off")

plt.savefig("NO2_country_trend_map.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# 5️⃣ PRINT SLOPE SUMMARY
# =====================================================

print("\nTrend Summary:")
print("Positive trend countries:",
      (trend_df["NO2_trend"] > 0).sum())

print("Negative trend countries:",
      (trend_df["NO2_trend"] < 0).sum())

print("Mean slope:", trend_df["NO2_trend"].mean())
