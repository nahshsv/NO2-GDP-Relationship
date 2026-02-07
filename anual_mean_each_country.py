import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
import numpy as np

# ==============================
# 1️⃣ LOAD ANNUAL MEAN GRID DATA (2005)
# ==============================
ds = xr.open_dataset("NO2_annual_mean_2005.nc")
no2 = ds["no2"].where(ds["no2"] > 0)

# ==============================
# 2️⃣ LOAD COUNTRY SHAPES (10m – FULL)
# ==============================
ne_url = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_cultural/ne_10m_admin_0_countries.zip"
)

world = gpd.read_file(ne_url).reset_index(drop=True)

# ==============================
# 3️⃣ CREATE MASK FOR ALL COUNTRIES
# ==============================
mask = regionmask.mask_geopandas(
    world,
    no2.lon,
    no2.lat
)

# ==============================
# 4️⃣ ISO3 LIST (YOUR COUNTRIES)
# ==============================
iso_list = [
    "AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM",
    "COD","COG","CIV","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA",
    "GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT","MUS",
    "MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM",
    "ZAF","SSD","SDN","TZA","TGO","UGA","ZMB","ZWE"
]

# ==============================
# 5️⃣ PREP GRID COORDS (FOR NEAREST)
# ==============================
lon2d, lat2d = np.meshgrid(no2.lon.values, no2.lat.values)

# ==============================
# 6️⃣ COMPUTE COUNTRY MEAN (ISO-BASED)
# ==============================
results = []

for iso in iso_list:

    match = world[world["ISO_A3"] == iso]

    if match.empty:
        print(f"⚠️ ISO not found in shapefile: {iso}")
        continue

    idx = match.index[0]
    cname = match.iloc[0]["NAME"]

    no2_country = no2.where(mask == idx)

    # ---- CASE 1: country HAS grid ----
    if no2_country.count().item() > 0:
        mean_val = float(
            no2_country.mean(dim=["lat", "lon"], skipna=True).values
        )
        method = "area_mean"

    # ---- CASE 2: small island → nearest grid ----
    else:
        centroid = match.iloc[0].geometry.centroid
        clon, clat = centroid.x, centroid.y

        dist2 = (lon2d - clon)**2 + (lat2d - clat)**2
        iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)

        mean_val = float(no2.values[iy, ix])
        method = "nearest_grid"

    results.append({
        "ISO3": iso,
        "Country": cname,
        "Annual_Mean_NO2_2005": mean_val,
        "Method": method
    })

# ==============================
# 7️⃣ SAVE TABLE
# ==============================
df = pd.DataFrame(results)
df.to_csv("Africa_Country_AnnualMean_NO2_2005_FINAL.csv", index=False)

print("✅ DONE – ISO-based country mean NO₂ (2005)")
print(df)
