import os
import glob
import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
import numpy as np
import re

# ==============================
# CONFIG
# ==============================

input_folder = "monthly_means"
output_folder = "csv_china"
os.makedirs(output_folder, exist_ok=True)

# 🔥 ONLY PROCESS THESE YEARS
target_years = ["2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014",
"2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022","2023","2024", "2025"]

# ==============================
# LOAD COUNTRY SHAPE
# ==============================

ne_url = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_cultural/ne_10m_admin_0_countries.zip"
)

world = gpd.read_file(ne_url).reset_index(drop=True)

# iso_list = [
#     "AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM",
#     "COD","COG","CIV","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA",
#     "GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT","MUS",
#     "MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
#     "ZAF","SSD","SDN","TZA","TGO","UGA","ZMB","ZWE"
# ]
iso_list=["CHN"]
# ==============================
# GROUP MONTHLY FILES BY YEAR
# ==============================

nc_files = sorted(glob.glob(os.path.join(input_folder, "NO2_monthly_mean_*.nc")))

year_groups = {}

for file in nc_files:
    filename = os.path.basename(file)
    match = re.search(r"(\d{4})_(\d{2})", filename)
    if not match:
        continue

    year = match.group(1)

    # 🔥 Skip years not in target list
    if year not in target_years:
        continue

    year_groups.setdefault(year, []).append(file)

# ==============================
# PROCESS EACH TARGET YEAR
# ==============================

for YEAR in target_years:

    if YEAR not in year_groups:
        print(f"⚠️ No monthly files found for {YEAR}")
        continue

    print(f"\nProcessing YEAR: {YEAR}")

    files = sorted(year_groups[YEAR])

    # --------------------------
    # 1️⃣ STACK MONTHLY GRIDS
    # --------------------------

    ds_list = [xr.open_dataset(f)["no2"] for f in files]

    no2_stack = xr.concat(ds_list, dim="time")

    # --------------------------
    # 2️⃣ GRID-BY-GRID ANNUAL MEAN
    # --------------------------

    annual_grid = no2_stack.mean(dim="time", skipna=True)

    # --------------------------
    # 3️⃣ COUNTRY MASK
    # --------------------------

    mask = regionmask.mask_geopandas(
        world,
        annual_grid.lon,
        annual_grid.lat
    )

    lon2d, lat2d = np.meshgrid(
        annual_grid.lon.values,
        annual_grid.lat.values
    )

    results = []

    # --------------------------
    # 4️⃣ COUNTRY AREA MEAN
    # --------------------------

    for iso in iso_list:

        match = world[world["ISO_A3"] == iso]
        if match.empty:
            continue

        idx = match.index[0]
        cname = match.iloc[0]["NAME"]

        country_grid = annual_grid.where(mask == idx)

        if country_grid.count().item() > 0:
            mean_val = float(
                country_grid.mean(dim=["lat", "lon"], skipna=True).values
            )
            method = "grid_annual_area_mean"
        else:
            centroid = match.iloc[0].geometry.centroid
            clon, clat = centroid.x, centroid.y

            dist2 = (lon2d - clon)**2 + (lat2d - clat)**2
            iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)

            mean_val = float(annual_grid.values[iy, ix])
            method = "nearest_grid"

        results.append({
            "ISO3": iso,
            "Country": cname,
            "Annual_Mean_NO2": mean_val,
            "Method": method
        })

    for ds in ds_list:
        ds.close()

    # --------------------------
    # SAVE CSV
    # --------------------------

    output_csv = os.path.join(
        output_folder,
        f"Africa_Country_AnnualMean_NO2_{YEAR}.csv"
    )

    pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")

print("\n✅ DONE – Only selected 5 years processed")
