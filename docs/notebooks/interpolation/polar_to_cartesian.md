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

# Interpolate data from polar to cartesian coordinates

The {{wradlib}} {mod}`wradlib.ipol` module implements several interpolator schemes, many of them based on {class}`scipy:scipy.spatial.cKDTree` class. In this notebook its shown how they are applied to {class}`xarray:xarray.Dataset` or {class}`xarray:xarray.DataArray`.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import wradlib as wrl
import wradlib_data
import warnings
import xarray as xr
import cmweather

warnings.filterwarnings("ignore")
```

## Import polar and cartesian data

```{code-cell} python
filename1 = wradlib_data.DATASETS.fetch("geo/bonn_new.tif")
dem = xr.open_dataset(filename1, engine="rasterio")

filename2 = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
swp = xr.open_dataset(filename2, engine="gamic", group="sweep_0")
```
## Preprocess polar data

In order to be on the same coordinates, we georeference the radar sweep coordinates accordingly.

```{code-cell} python
swp = swp.wrl.georef.georeference(crs=dem.spatial_ref)
swp = swp.set_coords("sweep_mode")
swp = swp.isel(range=slice(0, 100))
display(swp)
```
## Preprocess cartesian data

First, inspect the data set a little.

```{code-cell} python
display(dem)
```

Extract dem band and crop grid.

```{code-cell} python
band = (
    dem
    .wrl.util.crop(swp, pad=0)
    .isel(band=0)["band_data"]
)
display(band)
```

## Nearest neighbour interpolation

Please check with {class}`scipy:scipy.spatial.cKDTree` for kwargs for tree initialization and {meth}`scipy:scipy.spatial.cKDTree.query` for kwargs for querying the tree.

```{code-cell} python
band_xy = band.chunk()
nearest = swp.wrl.ipol.interpolate(band_xy, method="nearest")
display(nearest)
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = "HomeyerRainbow"
vmin, vmax = 0.0, 60.0

pm = swp.DBZH.wrl.vis.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title("Original PPI")

pm = nearest.DBZH.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_aspect(nearest.wrl.util.aspect())
ax2.set_title("Nearest")
plt.tight_layout()
```

## Inverse distance

```{code-cell} python
band_xy = band.chunk()
inverse_distance = swp.wrl.ipol.interpolate(band_xy, method="inverse_distance", k=4, idw_p=2)
display(inverse_distance)
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = "HomeyerRainbow"
vmin, vmax = 0.0, 60.0

pm = swp.DBZH.wrl.vis.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title("Original PPI")

pm = inverse_distance.DBZH.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_aspect(inverse_distance.wrl.util.aspect())
ax2.set_title("Inverse Distance")

plt.tight_layout()
```

## Precompute KDTree dataset

In the above examples the KDTree is computed and queried within the function. If your geometry is fixed you can precompute the KDTree and use it in subsequent computations.

```{code-cell} python
band_xy = band.chunk()
mapping = swp.wrl.ipol.get_mapping(band_xy, k=4)
display(mapping)
```

```{code-cell} python
band_xy = band.chunk()
nearest = swp.wrl.ipol.interpolate(mapping, method="nearest")
inverse_distance = swp.wrl.ipol.interpolate(mapping, method="inverse_distance", idw_p=2)
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = "HomeyerRainbow"
vmin, vmax = 0.0, 60.0

pm = nearest.DBZH.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_aspect(nearest.wrl.util.aspect())
ax1.set_title("Nearest")

pm = inverse_distance.DBZH.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_aspect(inverse_distance.wrl.util.aspect())
ax2.set_title("Inverse Distance")

plt.tight_layout()
```
