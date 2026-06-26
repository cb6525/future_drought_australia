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
dataset = "CORDEX_Qld"           # ['BARPA', 'CMIP6', 'CORDEX_CSIRO', 'CORDEX_Qld', 'NARCLIM']
experiment = os.environ["EXPERIMENT"]
variable   = os.environ["VARIABLE"]
drought_threshold = 'Perc_15'
baseline = 'Baseline_1970_2005'
scale = "Scale_3"           # ['Scale_3', 'Scale_12']

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

# Ensure that the historical period is always the same, thus starting in 1960
if experiment == "historical":
    time_min = "1960"; time_max = "2014"
else:
    time_min = "2015"; time_max = "2100"

# Check the calndar and convert the calendar to "proleptic_gregorian" in case it is different
for name, ds in datasets.items():
    cal = ds.time.encoding.get("calendar", ds.time.dt.calendar)
    if cal != "proleptic_gregorian":
        datasets[name] = ds.sel(time=slice(time_min, time_max)).convert_calendar("standard", align_on = "year")
    else:
        datasets[name] = ds.sel(time=slice(time_min, time_max))


# Determine the time under drought, i.e. the proportion of time that is under drought for each of the driving models
time_under_drought = {}
for name, ds in datasets.items():
    time_under_drought[name] = ds.timing.sum(dim="time", min_count=1) / len(ds.timing.time)
time_under_drought_all = xr.concat(list(time_under_drought.values()),dim="forcing_gcm",  join="override",   # Forcibly uses the lat/lon of the very first dataset
    coords="minimal"   # Prevents copying duplicate lat/lon coordinates per GCM
    ).assign_coords(forcing_gcm=list(time_under_drought.keys()))

# Determine the multi-model mean (MMM) for the corresponding RGM 
time_under_drought_all_mmm = time_under_drought_all.mean(dim="forcing_gcm")

# Plot the MMM for the corresponding RCM
ax = plt.subplot(projection=ccrs.PlateCarree())
time_under_drought_all_mmm.plot.pcolormesh(x="lon", y="lat", ax=ax, cmap="YlOrBr",levels= np.linspace(0,.4,21),
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
ax.set_title(f"{experiment} mean - {dataset}, variable =  {variable}", fontsize=13, pad=10)

gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

plt.tight_layout()

plt.savefig(f"/g/data/dt6/cb6525/graphs/time_under_drought/{experiment}_{dataset}_time_under_drought_{variable}.png", dpi=300, bbox_inches="tight")