---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
  main_language: python
kernelspec:
  display_name: Python 3
  name: python3
---

# Sweep to Raster

In this notebook we show the production of a maximum reflectivity composite from 3 neighboring radars for projected as well as geographic raster targets.

```{code-cell} python
import tempfile
import warnings

import cmweather
import xarray as xr
import xradar as xd
import matplotlib.pyplot as plt

import wradlib as wrl
import wradlib_data

warnings.filterwarnings("ignore")
```

## Get radar data

First, we import measurements from three belgian radars. This is done using {func}`xradar:xradar.io.backends.odim.open_odim_datatree`.

```{code-cell} python
filenames = ["bejab.pvol.hdf", "bewid.pvol.hdf", "behel.pvol.hdf"]
paths = [wradlib_data.DATASETS.fetch(f"hdf5/{f}") for f in filenames]
volumes = [xd.io.open_odim_datatree(p) for p in paths]
```

## Projected coordinate raster

Please see {func}`wradlib.georef.raster.create_raster_xarray` for details.

```{code-cell} python
europe_crs = 3035
bounds = [3614000, 2783000, 4200000, 3338000]
resolution = 1000
raster_projected = wrl.georef.create_raster_xarray(
    crs=europe_crs,
    bounds=bounds,
    resolution=resolution,
)
display(raster_projected)
```

## Geographic coordinate raster

Please see {func}`wradlib.georef.raster.create_raster_geographic` for details.

```{code-cell} python
bounds = [0, 48, 9, 53]
resolution = 1000
raster_geographic = wrl.georef.create_raster_geographic(
    bounds=bounds, resolution=resolution, resolution_in_meters=True
)
display(raster_geographic)
```

## Transform with Sweep to Raster

This transforms the lowest sweep into a raster for each radar. See {func}`wradlib.comp.sweep_to_raster` for details.

```{code-cell} python
raster = {}
raster["projected"] = raster_projected
raster["geographic"] = raster_geographic
rasters = {}
swp = 0
fig, axes = plt.subplots(3, 2, figsize=(13, 10))
for p, name in enumerate(["projected", "geographic"]):
    metadata = xd.model.required_sweep_metadata_vars
    rasters_radar = []
    for volume in volumes:
        sweep = volume[f"sweep_{swp}"].to_dataset()
        sweep = sweep[["DBZH"] + list(metadata)]
        sweep = sweep.sel(range=slice(0, 200e3))
        sweep = sweep.wrl.georef.georeference()
        raster_radar = sweep.wrl.comp.sweep_to_raster(raster[name])
        rasters_radar.append(raster_radar.drop_vars("crs_wkt"))

    for i, raster_radar in enumerate(rasters_radar):
        ax = axes[i, p]
        raster_radar["DBZH"].plot(ax=ax, vmin=0, vmax=50, cmap="HomeyerRainbow")
        ax.set_aspect("equal", "box")
        ax.set_title(f"{name} - Radar {i}")
    rasters[name] = rasters_radar

fig.tight_layout()

display(rasters["projected"])
display(rasters["geographic"])
```

## Combine rasters

```{code-cell} python
rasters = {k: xr.concat(v, dim="sweep") for k, v in rasters.items()}
display(rasters["projected"])
display(rasters["geographic"])
````

## Calculate maximum reflectivity

```{code-cell} python
rasters = {k: v.max(dim="sweep", keep_attrs=True) for k, v in rasters.items()}
display(rasters["projected"])
display(rasters["geographic"])
```

## Plot results

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, (name, raster) in enumerate(rasters.items()):
    raster["DBZH"].plot(ax=axes[i], vmin=0, vmax=50, cmap="HomeyerRainbow")
    axes[i].set_title(name)
fig.tight_layout()
```
