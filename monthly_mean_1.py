import os
import glob
import numpy as np
import h5py
import xarray as xr
import re

# ==========================
# CONFIG
# ==========================

input_root = "NO2_data"
output_root = "monthly_means"
dataset_path = "HDFEOS/GRIDS/ColumnAmountNO2/Data Fields/ColumnAmountNO2Trop"

os.makedirs(output_root, exist_ok=True)

# ==========================
# LOOP THROUGH YEARS
# ==========================

year_folders = sorted(os.listdir(input_root))

for year in year_folders:

    year_path = os.path.join(input_root, year)

    if not os.path.isdir(year_path):
        continue

    print(f"\nProcessing YEAR: {year}")

    he5_files = sorted(glob.glob(os.path.join(year_path, "*.he5")))

    if len(he5_files) == 0:
        print("No HE5 files found. Skipping.")
        continue

    # ==========================
    # GROUP FILES BY MONTH
    # ==========================

    monthly_groups = {}

    for file in he5_files:
        filename = os.path.basename(file)

        # Extract year & month from filename
        match = re.search(r"(\d{4})m(\d{2})\d{2}", filename)

        if not match:
            continue

        month = match.group(2)

        monthly_groups.setdefault(month, []).append(file)

    # ==========================
    # PROCESS EACH MONTH
    # ==========================

    for month, files in monthly_groups.items():

        print(f"  Processing Month: {month} ({len(files)} files)")

        sum_data = None
        valid_count = None

        for file in files:
            try:
                with h5py.File(file, "r") as f:

                    if dataset_path not in f:
                        continue

                    data = f[dataset_path][:].astype(np.float64)

                    # Replace fill values
                    data[data < -1e20] = np.nan

                    if sum_data is None:
                        sum_data = np.zeros_like(data)
                        valid_count = np.zeros_like(data)

                    valid_mask = ~np.isnan(data)

                    sum_data[valid_mask] += data[valid_mask]
                    valid_count[valid_mask] += 1

            except Exception:
                print(f"    Skipping corrupted file: {file}")
                continue

        if sum_data is None:
            continue

        # Compute monthly mean grid-by-grid
        monthly_mean = np.divide(
            sum_data,
            valid_count,
            out=np.full_like(sum_data, np.nan),
            where=valid_count != 0
        )

        # ==========================
        # CREATE LAT LON
        # ==========================

        nlat, nlon = monthly_mean.shape

        lat = np.linspace(-89.875, 89.875, nlat)
        lon = np.linspace(-179.875, 179.875, nlon)

        # ==========================
        # SAVE NETCDF
        # ==========================

        output_file = os.path.join(
            output_root,
            f"NO2_monthly_mean_{year}_{month}.nc"
        )

        ds = xr.Dataset(
            {
                "no2": (["lat", "lon"], monthly_mean)
            },
            coords={
                "lat": lat,
                "lon": lon
            }
        )

        ds.to_netcdf(output_file)

        print(f"    Saved: {output_file}")

print("\nAll years processed successfully!")
