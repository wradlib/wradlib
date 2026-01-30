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

# ToGrid

In this notebook we show the production of a reflectivity composite from 3 neighboring radars to a common cartesian grid, using the sampling volume size as a quality criterion.

```{code-cell} python3
import tempfile
import warnings

import cmweather
import numpy as np
import pyproj
import xarray as xr
import matplotlib.pyplot as plt

import wradlib as wrl
import wradlib_data

warnings.filterwarnings("ignore")
```

## Get radar data

First, we import measurements from three belgian radars. This is done using {func}`xarray:xarray.open_dataset` using {class}`xradar:xradar.io.backends.odim.OdimBackendEntrypoint`.

```{code-cell} python3
filenames = ["bejab.pvol.hdf", "bewid.pvol.hdf", "behel.pvol.hdf"]
paths = {f.split(".")[0][2:]: wradlib_data.DATASETS.fetch(f"hdf5/{f}") for f in filenames}
ctree = xr.DataTree()
for radar, filename in paths.items():
    ctree[radar] = xr.open_dataset(filename, engine="odim", group="sweep_0").chunk().wrl.georef.georeference().set_coords("sweep_mode")
display(ctree)
```

## Plot Overview

```{code-cell} python3
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
ax = axs.flat[0]
ctree["jab"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Jabbeke")
ax = axs.flat[1]
ctree["wid"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Wideumont")
ax = axs.flat[2]
ctree["hel"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Helchteren")
fig.tight_layout()
```

## Georeference UTM

```{code-cell} python3
proj_utm = pyproj.CRS.from_epsg(32632)
kwargs = dict(crs=proj_utm)
for radar in ctree.children:
    ctree[radar].ds = ctree[radar].ds.wrl.georef.georeference(crs=proj_utm)
display(ctree)
```

```{code-cell} python3
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
ax = axs.flat[0]
ctree["jab"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Jabbeke")
ax = axs.flat[1]
ctree["wid"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Wideumont")
ax = axs.flat[2]
ctree["hel"].ds.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60)
ax.set_aspect("equal")
ax.set_title("Radar Helchteren")
fig.tight_layout()
```

## Calculate Quality

```{code-cell} python3
for radar in ctree.children:
    rng = ctree[radar].ds.range
    azimuth = ctree[radar].ds.azimuth
    rscale = rng.diff("range").median().values
    qual = rng.wrl.qual.pulse_volume(rscale, 1.0).expand_dims(azimuth=ctree[radar].ds.azimuth)
    ctree[radar].ds = ctree[radar].ds.assign(QUAL=qual)
display(ctree)
```

## Gridding

First, we create a Datset with cartesian coordinates, which can hold our three radars.
We are using {meth}`wradlib.comp.CompMethods.togrid` to interpolate our sweep data to the cartesian domain.
As an interpolator we use {class}`wradlib.ipol.Nearest`.

```{code-cell} python3
xmin, xmax, ymin, ymax = ctree.wrl.util.bbox()
x = np.linspace(xmin, xmax + 1000.0, 1000)
y = np.linspace(ymin, ymax + 1000.0, 1000)
cart = xr.Dataset(coords={"x": (["x"], x), "y": (["y"], y)})
display(cart)
```

```{code-cell} python3
gtree = xr.DataTree()
for radar in ctree.children:
    gridded = ctree[radar].ds.wrl.comp.togrid(cart)
    gtree[radar] = gridded
display(gtree)
```

```{code-cell} python3
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
ax = axs.flat[0]
gtree["jab"].ds.DBZH.plot(ax=ax, vmin=0, vmax=60, cmap="HomeyerRainbow")
ax.set_aspect("equal")
ax.set_title("Radar Jabbeke")
ax = axs.flat[1]
gtree["wid"].ds.DBZH.plot(ax=ax, vmin=0, vmax=60, cmap="HomeyerRainbow")
ax.set_aspect("equal")
ax.set_title("Radar Wideumont")
ax = axs.flat[2]
gtree["hel"].ds.DBZH.plot(ax=ax, vmin=0, vmax=60, cmap="HomeyerRainbow")
ax.set_aspect("equal")
ax.set_title("Radar Helchteren")
fig.tight_layout()
```

```{code-cell} python3
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
ax = axs.flat[0]
gtree["jab"].ds.QUAL.plot(ax=ax)
ax.set_aspect("equal")
ax.set_title("Radar Jabbeke")
ax = axs.flat[1]
gtree["wid"].ds.QUAL.plot(ax=ax)
ax.set_aspect("equal")
ax.set_title("Radar Wideumont")
ax = axs.flat[2]
gtree["hel"].ds.QUAL.plot(ax=ax)
ax.set_aspect("equal")
ax.set_title("Radar Helchteren")
fig.tight_layout()
```

## Compositing

Before compositing we combine the three radar grids as well as the quality grids into one Dataset, respectively.

```{code-cell} python3
radars = xr.DataArray(gtree.children, dims="radar")
radargrids = xr.concat([gtree[radar].ds.DBZH for radar in gtree.children], dim=radars)
# normalizing Quality between 1. and 0.
qualitygrids = xr.concat([1. - (gtree[radar].ds.QUAL / gtree[radar].ds.QUAL.max()) for radar in gtree.children], dim=radars)

display(radargrids)
display(qualitygrids)
```
Then we finally can call {meth}`wradlib.comp.CompMethods.compose_weighted` to create the final output.

```{code-cell} python3
composite = radargrids.wrl.comp.compose_weighted(qualitygrids)
display(composite)
```

## Plot Result

```{code-cell} python3
composite.where(composite>0.1).plot(cmap="HomeyerRainbow", vmin=0, vmax=60)
```
