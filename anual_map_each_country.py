import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

YEAR=2025
# =============================
# 1. Load high resolution map
# =============================
url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
world = gpd.read_file(url)

# =============================
# 2. Load NO2 data
# =============================
df = pd.read_csv(f"Africa_Country_AnnualMean_NO2_{YEAR}_FINAL.csv")

# =============================
# 3. Merge
# =============================
merged = world.merge(df, left_on='ISO_A3', right_on='ISO3')

# Filter Africa only (recommended)
africa = merged[merged['CONTINENT'] == 'Africa']

# =============================
# 4. Apply LOG transform
# =============================
africa['NO2_log'] = np.log10(africa[f'Annual_Mean_NO2_{YEAR}'])

# =============================
# 5. Plot
# =============================
fig, ax = plt.subplots(figsize=(12,10))

africa.plot(
    column='NO2_log',
    cmap='OrRd',
    scheme='quantiles',
    k=9,                     # 🔥 tăng lên 9 mức màu
    legend=True,
    edgecolor='black',
    linewidth=0.6,
    ax=ax
)

ax.set_title(f"Annual Mean NO₂ {YEAR} - Log Scale", fontsize=14)
ax.axis('off')

# 🔥 Save trước khi show
plt.savefig(
    f"NO2_Africa_{YEAR}_log_quantiles.png",
    dpi=300,              # độ phân giải cao
    bbox_inches='tight'   # không bị dư viền trắng
)

plt.show()
