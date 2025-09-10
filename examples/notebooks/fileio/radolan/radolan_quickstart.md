---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.17.3
---

```{raw-cell}
:tags: [hide-cell]
This notebook is part of the wradlib documentation: https://docs.wradlib.org.

Copyright (c) wradlib developers.
Distributed under the MIT License. See LICENSE.txt for more info.
```

# RADOLAN Quick Start

Import modules, filter warnings to avoid cluttering output with DeprecationWarnings and use matplotlib inline or interactive mode if running in ipython or python respectively.

```{code-cell} python
import os
import wradlib as wrl
import wradlib_data
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
import xarray as xr
```

## Normal reader

All RADOLAN composite products can be read by the following function:

```
data, metadata = wradlib.io.read_radolan_composite("mydrive:/path/to/my/file/filename")
```

Here, ``data`` is a two dimensional integer or float array of shape (number of rows, number of columns). ``metadata`` is a dictionary which provides metadata from the files header section, e.g. using the keys `producttype`, `datetime`, `intervalseconds`, `nodataflag`.

The [RADOLAN Grid](radolan_grid) coordinates can be calculated with {func}`wradlib.georef.get_radolan_grid`.

With the following code snippet the RW-product is shown in the [Polar Stereographic Projection](radolan_grid#Polar-Stereographic-Projection).

```{code-cell} python
# load radolan files
rw_filename = wradlib_data.DATASETS.fetch(
    "radolan/misc/raa01-rw_10000-1408102050-dwd---bin.gz"
)
rwdata, rwattrs = wrl.io.read_radolan_composite(rw_filename)
# print the available attributes
print("RW Attributes:", rwattrs)
```

```{code-cell} python
# do some masking
sec = rwattrs["secondary"]
rwdata.flat[sec] = -9999
rwdata = np.ma.masked_equal(rwdata, -9999)
```

```{code-cell} python
# Get coordinates
radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
x = radolan_grid_xy[:, :, 0]
y = radolan_grid_xy[:, :, 1]
```

```{code-cell} python
# plot function
plt.pcolormesh(x, y, rwdata, cmap="viridis")
cb = plt.colorbar(shrink=0.75)
cb.set_label("mm * h-1")
plt.title("RADOLAN RW Product Polar Stereo \n" + rwattrs["datetime"].isoformat())
plt.grid(color="r")
```

A much more comprehensive section using several RADOLAN composites is shown in chapter [RADOLAN Product Showcase](radolan_showcase).


## RADOLAN Xarray backend


From wradlib version 1.10.0 a RADOLAN xarray backend is available. RADOLAN data will be imported into an `xarray.Dataset` with attached coordinates.

```{code-cell} python
# load radolan files
rw_filename = wradlib_data.DATASETS.fetch(
    "radolan/misc/raa01-rw_10000-1408102050-dwd---bin.gz"
)
ds = wrl.io.open_radolan_dataset(rw_filename)
# print the xarray dataset
ds
```

### Simple Plot

```{code-cell} python
ds.RW.plot()
```

### Simple selection

```{code-cell} python
ds.RW.sel(x=slice(-100000, 100000), y=slice(-4400000, -4200000)).plot()
```

### Map plot using `cartopy`

```{code-cell} python
import cartopy.crs as ccrs

map_proj = ccrs.Stereographic(
    true_scale_latitude=60.0, central_latitude=90.0, central_longitude=10.0
)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 8))
ds.RW.plot(subplot_kws=dict(projection=map_proj))
ax = plt.gca()
ax.gridlines(draw_labels=True, y_inline=False)
```

## Open multiple files

```{code-cell} python
# load radolan files
flist = [
    "radolan/misc/raa01-sf_10000-1305270050-dwd---bin.gz",
    "radolan/misc/raa01-sf_10000-1305280050-dwd---bin.gz",
]
sf_filenames = [wradlib_data.DATASETS.fetch(f) for f in flist]
ds = wrl.io.open_radolan_mfdataset(sf_filenames)
# print the xarray dataset
ds
```

```{code-cell} python
fig = plt.figure(figsize=(10, 5))
ds.SF.plot(col="time")
```

## Use `xr.open_dataset` and `xr.open_mfdataset`

```{code-cell} python
rw_filename = wradlib_data.DATASETS.fetch(
    "radolan/misc/raa01-rw_10000-1408102050-dwd---bin.gz"
)
ds = xr.open_dataset(rw_filename, engine="radolan")
ds
```

```{code-cell} python
sf_filename = os.path.join(
    wrl.util.get_wradlib_data_path(), "radolan/misc/raa01-sf_10000-1305*"
)
ds = xr.open_mfdataset(sf_filename, engine="radolan")
ds
```
