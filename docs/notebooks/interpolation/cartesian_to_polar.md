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

# Interpolate data on cartesian coordinates to polar coordinates

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

## Read a cartesian data set

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("geo/bonn_new.tif")
print(filename)
```

```{code-cell} python
dem = xr.open_dataset(filename, engine="rasterio")
```

Inspect the data set a little

```{code-cell} python
display(dem)
```

Extract dem band and coarsen grid. We do this here to prevent memory issues when running this on CI. When applying this to your workflows, just use your normal grid resolution.

```{code-cell} python
 band = dem.coarsen(x=10, y=10, boundary="trim").mean()["band_data"].isel(band=0)
 display(band)
```

## Read a polar data set

In order to be on the same coordinates, we georeference our radar sweep accordingly.

```{code-cell} python
fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
swp = xr.open_dataset(fname, engine="gamic", group="sweep_0")
swp = swp.wrl.georef.georeference(crs=dem.spatial_ref)
swp = swp.set_coords("sweep_mode")
display(swp)
```

## griddata

This is slow for large input arrays. Please check with {func}`scipy.interpolate.griddata` for kwarg distribution.

```{code-cell} python
band_xy = band.chunk()
swp = swp.isel(range=slice(0,100))
band_polar_nearest = band_xy.wrl.ipol.griddata(swp, method="nearest")
band_polar_linear = band_xy.wrl.ipol.griddata(swp, method="linear")
band_polar_cubic = band_xy.wrl.ipol.griddata(swp, method="cubic")
```

```{code-cell} python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
cmap = "terrain"
vmin = 0.0

band_crop = band_xy.wrl.util.crop(swp)
pm = band_crop.plot(ax=ax1, add_colorbar=False, cmap=cmap, vmin=vmin)
ax1.set_aspect(band_crop.wrl.util.aspect())
# add colorbar manually
fig.colorbar(pm, ax=ax1, shrink=0.66)
ax1.set_title("Original DEM")

pm = band_polar_nearest.wrl.vis.plot(ax=ax2, cmap=cmap, vmin=vmin)
ax2.set_title("Nearest")
pm = band_polar_linear.wrl.vis.plot(ax=ax3, cmap=cmap, vmin=vmin)
ax3.set_title("Linear")
pm = band_polar_cubic.wrl.vis.plot(ax=ax4, cmap=cmap, vmin=vmin)
ax4.set_title("Cubic")

plt.tight_layout()
```

## map_coordinates

Please check with {func}`scipy.ndimage.map_coordinates` for kwarg distribution.

```{code-cell} python
band_xy = band.chunk()
swp = swp.isel(range=slice(0,100))
band_polar_nearest = band_xy.wrl.ipol.map_coordinates(swp, order=0)
band_polar_linear = band_xy.wrl.ipol.map_coordinates(swp, order=1)
band_polar_quadratic = band_xy.wrl.ipol.map_coordinates(swp, order=2)
band_polar_cubic = band_xy.wrl.ipol.map_coordinates(swp, order=3)
band_polar_quartic = band_xy.wrl.ipol.map_coordinates(swp, order=4)
band_polar_quintic = band_xy.wrl.ipol.map_coordinates(swp, order=5)
```

```{code-cell} python
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))


pm = band_polar_nearest.wrl.vis.plot(ax=ax1, cmap=cmap, vmin=vmin)
ax1.set_title("Nearest")
pm = band_polar_linear.wrl.vis.plot(ax=ax2, cmap=cmap, vmin=vmin)
ax2.set_title("Linear")
pm = band_polar_quadratic.wrl.vis.plot(ax=ax3, cmap=cmap, vmin=vmin)
ax3.set_title("Quadratic")
pm = band_polar_cubic.wrl.vis.plot(ax=ax4, cmap=cmap, vmin=vmin)
ax4.set_title("Cubic")
pm = band_polar_quartic.wrl.vis.plot(ax=ax5, cmap=cmap, vmin=vmin)
ax5.set_title("Quartic")
pm = band_polar_quintic.wrl.vis.plot(ax=ax6, cmap=cmap, vmin=vmin)
ax6.set_title("Quintic")

plt.tight_layout()
```
