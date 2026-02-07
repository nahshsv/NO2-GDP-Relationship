import xarray as xr
import glob
import os

year = 2005
folder = f"data"

files = sorted(glob.glob(os.path.join(folder, "*.nc")))

if len(files) != 12:
    raise ValueError(f"{year} does not have 12 monthly files!")

monthly_maps = []

for f in files:
    ds = xr.open_dataset(f)
    no2 = ds["no2"].where(ds["no2"] > 0)
    monthly_maps.append(no2)

# Stack manually → tạo dimension "month"
annual_mean_2005 = xr.concat(monthly_maps, dim="month").mean(dim="month")

# Save
annual_mean_2005.to_netcdf(f"NO2_annual_mean_{year}.nc")

print(f"✅ Saved NO2_annual_mean_{year}.nc")
