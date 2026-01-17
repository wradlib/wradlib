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

# Interpolate data on polar coordinates to cartesian coordinates


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import wradlib as wrl
import wradlib_data
import warnings
import xarray as xr
import cmweather

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Read polar and cartesian datas

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("geo/bonn_new.tif")
dem = xr.open_dataset(filename, engine="rasterio")

fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
swp = xr.open_dataset(fname, engine="gamic", group="sweep_0")
```
## Prepare polar data set

In order to be on the same coordinates, we georeference our radar sweep accordingly.

```{code-cell} python
swp = swp.wrl.georef.georeference(crs=dem.spatial_ref)
swp = swp.set_coords("sweep_mode")
swp = swp.isel(range=slice(0, 100))
display(swp)
```
## Prepare cartesian data set

Inspect the data set a little

```{code-cell} python
display(dem)
```

Extract dem band and crop grid. We do this here to prevent memory issues when running this on CI. When applying this to your workflows, just use your normal grid resolution.

```{code-cell} python
order = 1
band = (
    dem
    .wrl.util.crop(swp, pad=order)
    .isel(band=0)["band_data"]
)
display(band)
```

## nearest

```{code-cell} python
band_xy = band.chunk()
nearest = swp.wrl.ipol.polar_to_cart(band_xy, method="nearest")
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = "HomeyerRainbow"
vmin, vmax = 0.0, 60.0

pm = swp.DBZH.wrl.vis.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title("Original DEM")

pm = nearest.DBZH.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_aspect(nearest.wrl.util.aspect())
ax2.set_title("Nearest")
plt.tight_layout()
```

## inverse distance

```{code-cell} python
band_xy = band.chunk()
inverse_distance = swp.wrl.ipol.polar_to_cart(band_xy, method="inverse_distance", k=4, p=2)
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = "HomeyerRainbow"
vmin, vmax = 0.0, 60.0

pm = swp.DBZH.wrl.vis.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title("Original DEM")

pm = inverse_distance.DBZH.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_aspect(inverse_distance.wrl.util.aspect())
ax2.set_title("Inverse Distance")

plt.tight_layout()
```
