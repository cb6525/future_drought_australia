import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path
import cartopy
cartopy.config['data_dir'] = '/g/data/xp65/public/apps/cartopy-data'
cartopy.config['pre_existing_data_dir'] = '/g/data/xp65/public/apps/cartopy-data'
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# --- Config ---
base = "/g/data/w97/amu561/CABLE_AWRA_comparison/Drought_metrics"
dataset = "NARCLIM"           # ['BARPA', 'CMIP6', 'CORDEX_CSIRO', 'CORDEX_Qld', 'NARCLIM']
experiment = "historical"   # ['historical', 'ssp126', 'ssp370', 'ssp585']
variable   = os.environ["VARIABLE"]
drought_threshold = 'Perc_15'
baseline = 'Baseline_1970_2005'
scale = "Scale_3"           # ['Scale_3', 'Scale_12']
# length of time period for historical and future period:
time_min_hist = "1960"; time_max_hist = "2014";
time_min_fut = "2051"; time_max_fut = "2100"

# --- Build path ---
path = Path(base) / dataset / experiment / variable / drought_threshold / baseline / scale

# --- Load all models ---
datasets = {}
for gcm_dir in sorted(path.iterdir()):
    if not gcm_dir.is_dir():
        continue
    nc_files = list(gcm_dir.rglob("*.nc"))
    if not nc_files:
        print(f"Warning: no .nc file found in {gcm_dir.name}")
        continue
    # Group files by their immediate parent folder structure
    # Works for both flat (GCM/*.nc) and nested (GCM/ensemble/RCM/*.nc)
    from itertools import groupby
    file_groups = {}
    for f in nc_files:
        # Build key from all subdirs between gcm_dir and the file
        relative_parts = f.relative_to(gcm_dir).parts[:-1]  # exclude filename
        if relative_parts:
            # Nested: e.g. ACCESS-CM2_r2i1p1f1_CCAMoc-v2112
            key = gcm_dir.name + "_" + "_".join(relative_parts)
        else:
            # Flat: e.g. ACCESS-CM2
            key = gcm_dir.name
        file_groups.setdefault(key, []).append(f)

    for key, files in file_groups.items():
        datasets[key] = xr.open_mfdataset(sorted(files), combine="by_coords")
        print(f"Loaded: {key}")


experiment_fut = os.environ["EXPERIMENT_FUT"]

# --- Build path ---
path_future = Path(base) / dataset / experiment_fut / variable / drought_threshold / baseline / scale

# --- Load all models ---
datasets_future = {}
for gcm_dir in sorted(path_future.iterdir()):
    if not gcm_dir.is_dir():
        continue
    nc_files = list(gcm_dir.rglob("*.nc"))
    if not nc_files:
        print(f"Warning: no .nc file found in {gcm_dir.name}")
        continue
    # Group files by their immediate parent folder structure
    # Works for both flat (GCM/*.nc) and nested (GCM/ensemble/RCM/*.nc)
    from itertools import groupby
    file_groups = {}
    for f in nc_files:
        # Build key from all subdirs between gcm_dir and the file
        relative_parts = f.relative_to(gcm_dir).parts[:-1]  # exclude filename
        if relative_parts:
            # Nested: e.g. ACCESS-CM2_r2i1p1f1_CCAMoc-v2112
            key = gcm_dir.name + "_" + "_".join(relative_parts)
        else:
            # Flat: e.g. ACCESS-CM2
            key = gcm_dir.name
        file_groups.setdefault(key, []).append(f)

    for key, files in file_groups.items():
        datasets_future[key] = xr.open_mfdataset(sorted(files), combine="by_coords")
        print(f"Loaded: {key}")

# Check the calendars
for name, ds in datasets.items():
    cal = ds.time.encoding.get("calendar", ds.time.dt.calendar)
    if cal != "proleptic_gregorian":
        print(name)
        datasets[name] = ds.sel(time=slice(time_min_hist, time_max_hist)).convert_calendar("standard", align_on = "year")
    else:
        datasets[name] = ds.sel(time=slice(time_min_hist, time_max_hist))
for name, ds in datasets_future.items():
    cal = ds.time.encoding.get("calendar", ds.time.dt.calendar)
    if cal != "proleptic_gregorian":
        print(name)
        datasets_future[name] = ds.sel(time=slice(time_min_fut, time_max_fut)).convert_calendar("standard", align_on = "year")
    else:
        datasets_future[name] = ds.sel(time=slice(time_min_fut, time_max_fut))

# Determine the time under drought, i.e. the proportion of time that is under drought for each of the driving models
time_under_drought = {}
for name, ds in datasets.items():
    time_under_drought[name] = ds.timing.sum(dim="time", min_count=1) / len(ds.timing.time)
# Determine the time under drought for the future projections
time_under_drought_future = {}
for name, ds in datasets_future.items():
    time_under_drought_future[name] = ds.timing.sum(dim="time", min_count=1) / len(ds.timing.time)

# Combine the different drivingGCM datasets
time_under_drought_all = xr.concat(list(time_under_drought.values()),dim="forcing_gcm",  join="override",   # Forcibly uses the lat/lon of the very first dataset
    coords="minimal"   # Prevents copying duplicate lat/lon coordinates per GCM
    ).assign_coords(forcing_gcm=list(time_under_drought.keys()))
time_under_drought_all_future = xr.concat(list(time_under_drought_future.values()),dim="forcing_gcm",  join="override",   # Forcibly uses the lat/lon of the very first dataset
    coords="minimal"   # Prevents copying duplicate lat/lon coordinates per GCM
    ).assign_coords(forcing_gcm=list(time_under_drought_future.keys()))

# Calculate the mean for the different driving GCMS
time_under_drought_all_mmm = time_under_drought_all.mean(dim="forcing_gcm")
time_under_drought_all_mmm_future = time_under_drought_all_future.mean(dim="forcing_gcm")

# Calculate the difference between the historical period and the future projections
time_under_drought_all_diff = time_under_drought_all_mmm_future - time_under_drought_all_mmm

ax = plt.subplot(projection=ccrs.PlateCarree())
time_under_drought_all_diff.plot.pcolormesh(x="lon", y="lat", ax=ax, cmap="RdBu_r", levels=np.linspace(-0.2, 0.2, 21),
                                           cbar_kwargs={
                                               "label": "Time under drought (%)",
                                               "shrink":0.6,
                                               "pad": 0.02,
                                               "location":"bottom", 
                                           }, add_colorbar=True)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.bottom_labels = False
gl.right_labels = False
ax.set_title(f"difference {experiment_fut} - {experiment} mean - {dataset}, variable =  {variable}", fontsize=13, pad=10)

gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

plt.tight_layout()

plt.savefig(f"/g/data/dt6/cb6525/graphs/time_under_drought/difference_{experiment_fut}-{experiment}_{dataset}_time_under_drought_{variable}.png", dpi=300, bbox_inches="tight")