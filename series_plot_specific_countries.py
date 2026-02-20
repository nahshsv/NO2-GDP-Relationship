import pandas as pd
import glob
import matplotlib.pyplot as plt
import re

# 🔥 Only use these years
target_years = [2005, 2006,2007,2008,2009,2010,2011,2012,2013,2014, 2015, 2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]

# 🔥 Select representative countries
# selected_countries = ["ZAF", "NGA", "ETH", "GHA", "KEN","SWZ"]
selected_countries=["CHN"]

files = sorted(glob.glob("csv_china/Africa_Country_AnnualMean_NO2_*.csv"))

all_data = []

for f in files:
    year = int(re.search(r"(\d{4})", f).group(1))

    if year not in target_years:
        continue

    df = pd.read_csv(f)
    df["Year"] = year
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Keep only selected years and countries
df_all = df_all[
    (df_all["Year"].isin(target_years)) &
    (df_all["ISO3"].isin(selected_countries))
]

df_all = df_all.sort_values(["ISO3", "Year"])

# ==========================
# PLOT SELECTED COUNTRIES
# ==========================

plt.figure(figsize=(10,6))

for iso, group in df_all.groupby("ISO3"):
    group = group.sort_values("Year")

    plt.plot(group["Year"],
             group["Annual_Mean_NO2"],
             marker="o",
             linewidth=2,
             label=iso)

plt.xticks(target_years)

plt.title("NO₂ Time Series for Selected African Countries\n(2005–2025)")
plt.xlabel("Year")
plt.ylabel("NO₂")
plt.legend()
plt.grid(True)

plt.show()
