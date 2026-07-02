"""Download ERA5 hourly single-level SST (daily 00:00) for 1980 and convert to CSV.

Tropical Pacific box using the 0-360 longitude convention so West < East:
    area = [North, West, South, East] = [30, 120, -30, 280]  (280 == 80W)
Output: NetCDF (zip) + long-format CSV + metadata JSON under data/ERA5_sst_pacific_daily/
"""

import os
import json
import zipfile
import glob

import pandas as pd
import xarray as xr
import cdsapi

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ERA5_sst_pacific_daily"))
os.makedirs(DATA_DIR, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"
YEAR = 1980
ZIP_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_daily_{YEAR}.zip")
CSV_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_daily_{YEAR}.csv")
META_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_daily_{YEAR}_metadata.json")

request = {
    "product_type": ["reanalysis"],
    "variable": ["sea_surface_temperature"],
    "year": ["1980"],
    "month": ["01", "02", "03", "04", "05", "06",
               "07", "08", "09", "10", "11", "12"],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [30, 120, -30, 280],  # [N, W, S, E]  (280 = 80W, keeps W < E)
}


def download():
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 0:
        print(f"Already downloaded: {ZIP_PATH}")
        return ZIP_PATH
    client = cdsapi.Client()
    client.retrieve(DATASET, request, ZIP_PATH)
    print(f"Downloaded: {ZIP_PATH}")
    return ZIP_PATH


def open_era5(path):
    """Open an ERA5 download that may be a single NetCDF or a zip of NetCDFs."""
    if zipfile.is_zipfile(path):
        extract_dir = path + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
        parts = [xr.open_dataset(f) for f in nc_files]
        ds = xr.combine_by_coords(parts) if len(parts) > 1 else parts[0]
    else:
        ds = xr.open_dataset(path)

    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for c in ("number", "expver"):
        if c in ds.coords:
            ds = ds.drop_vars(c)
    # Ensure 0-360 longitudes so West < East, sorted ascending
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
    return ds


def to_csv_and_metadata(path):
    ds = open_era5(path).sortby("time")

    df = ds.to_dataframe().reset_index()
    lead = [c for c in ("time", "latitude", "longitude") if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV saved: {CSV_PATH} ({len(df)} rows)")

    times = pd.to_datetime(ds.time.values)
    metadata = {
        "dataset": DATASET,
        "region": {"name": "Tropical Pacific", "north": 30, "west": 120, "south": -30, "east": 280},
        "grid": {"resolution": "native (~0.25 deg)",
                  "n_lat": int(ds.sizes["latitude"]),
                  "n_lon": int(ds.sizes["longitude"])},
        "latitude_range": [float(ds.latitude.min()), float(ds.latitude.max())],
        "longitude_range": [float(ds.longitude.min()), float(ds.longitude.max())],
        "time_range": [str(times[0]), str(times[-1])],
        "n_time": int(ds.sizes["time"]),
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "variables": {
            v: {
                "dims": list(ds[v].dims),
                "shape": list(ds[v].shape),
                "units": ds[v].attrs.get("units", ""),
                "long_name": ds[v].attrs.get("long_name", ds[v].attrs.get("standard_name", "")),
            }
            for v in ds.data_vars
        },
        "csv_rows": int(len(df)),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved: {META_PATH}")


if __name__ == "__main__":
    print(f"Output dir : {DATA_DIR}")
    print(f"Dataset    : {DATASET}  (hourly single-levels, daily 00:00)")
    print(f"Year       : {YEAR}\n")
    zip_path = download()
    to_csv_and_metadata(zip_path)
    print("\nDone.")
