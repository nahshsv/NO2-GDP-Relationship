import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import re
from matplotlib.colors import TwoSlopeNorm

# =====================================================
# 🔥 CONFIG – CHỈNH NĂM Ở ĐÂY
# =====================================================

selected_years = [2005, 2006,2007,2008,2009,2010,2011,2012,2013,2014, 2015, 2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
# Ví dụ:
# selected_years = list(range(2005, 2026))
# selected_years = [2005, 2025]

folder_path = "csv_china"

# =====================================================
# 1️⃣ LOAD SELECTED YEARS
# =====================================================

all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".csv") and "Africa_Country_AnnualMean_NO2_" in file:

        year_match = re.search(r"(\d{4})", file)
        if not year_match:
            continue

        year = int(year_match.group(1))

        # 🔥 Skip years not selected
        if year not in selected_years:
            continue

        file_path = os.path.join(folder_path, file)

        df = pd.read_csv(file_path)
        df = df.rename(columns={"Annual_Mean_NO2": "Mean_NO2"})
        df["Year"] = year

        all_data.append(df[["ISO3", "Mean_NO2", "Year"]])

if len(all_data) == 0:
    raise ValueError("No matching year files found. Check selected_years.")

df_all = pd.concat(all_data, ignore_index=True)

print("Years loaded:", sorted(df_all["Year"].unique()))

# =====================================================
# 2️⃣ COMPUTE TREND (LOG10 REGRESSION)
# =====================================================

def compute_slope(group):
    group = group.sort_values("Year")

    x = group["Year"].values.astype(float)
    y = group["Mean_NO2"].values.astype(float)

    if len(x) < 2:
        return np.nan

    y_log = np.log10(y + 1)

    slope, _ = np.polyfit(x, y_log, 1)
    return slope

trend_df = (
    df_all
    .groupby("ISO3")[["Year", "Mean_NO2"]]
    .apply(compute_slope)
    .reset_index(name="NO2_trend")
)

trend_df = trend_df.dropna()

# =====================================================
# 3️⃣ LOAD MAP
# =====================================================

url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
world = gpd.read_file(url)

merged = world.merge(trend_df, left_on="ISO_A3", right_on="ISO3")
africa = merged[merged["ISO_A3"] == "CHN"]

# =====================================================
# 4️⃣ PLOT TREND MAP (BLUE-NEGATIVE, RED-POSITIVE)
# =====================================================

fig, ax = plt.subplots(figsize=(12,10))

abs_max = np.max(np.abs(africa["NO2_trend"]))

norm = TwoSlopeNorm(
    vmin=-abs_max,
    vcenter=0,
    vmax=abs_max
)

africa.plot(
    column="NO2_trend",
    cmap="bwr",
    norm=norm,
    legend=True,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

ax.set_title(
    f"NO₂ Trend by Country ({min(selected_years)}–{max(selected_years)})\n"
    "Blue = Decrease, White = No Change, Red = Increase",
    fontsize=14
)

ax.axis("off")

plt.savefig("NO2_country_trend_map.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# 5️⃣ SUMMARY
# =====================================================

print("\nTrend Summary")
print("Positive trend countries:", (trend_df["NO2_trend"] > 0).sum())
print("Negative trend countries:", (trend_df["NO2_trend"] < 0).sum())
print("Mean slope:", trend_df["NO2_trend"].mean())
