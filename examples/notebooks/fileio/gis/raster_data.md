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

# Export a dataset in GIS-compatible format

In this notebook, we demonstrate how to export a gridded dataset in GeoTIFF and ESRI ASCII format. This will be exemplified using RADOLAN data from the German Weather Service.

You have two options for output:

- `rioxarray.to_raster`
- builtin GDAL functionality

```{code-cell} python
import matplotlib.pyplot as plt
import os
import wradlib as wrl
import wradlib_data
import xarray as xr
import numpy as np
import warnings
from pyproj.crs import CRS

warnings.filterwarnings("ignore")

try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Step 1: Read the original data

```{code-cell} python
# We will export this RADOLAN dataset to a GIS compatible format
wdir = wradlib_data.DATASETS.abspath / "radolan/grid/"
# create output-folder if not exists
wdir.mkdir(parents=True, exist_ok=True)

filename = "radolan/misc/raa01-sf_10000-1408102050-dwd---bin.gz"
filename = wradlib_data.DATASETS.fetch(filename)
ds = xr.open_dataset(filename, engine="radolan")
display(ds)
```

```{code-cell} python
# This is the RADOLAN projection
proj_osr = wrl.georef.create_osr("dwd-radolan")
crs = CRS.from_wkt(proj_osr.ExportToWkt(["FORMAT=WKT2_2018"]))
print(proj_osr)
```

## Step 2a (output with rioxarray)


drop encoding

```{code-cell} python
ds.SF.encoding = {}
```

```{code-cell} python
ds = ds.rio.write_crs(crs)
ds.SF.rio.to_raster(wdir / "geotiff_rio.tif", driver="GTiff")
```

```{code-cell} python
ds.SF.rio.to_raster(
    wdir / "aaigrid_rio.asc",
    driver="AAIGrid",
    profile_kwargs=dict(options=["DECIMAL_PRECISION=2"]),
)
```

## Step 2b: (output with GDAL)

### Get the projected coordinates of the RADOLAN grid

```{code-cell} python
# Get projected RADOLAN coordinates for corner definition
xy_raw = wrl.georef.get_radolan_grid(900, 900)
xy_raw.shape
```

### Check Origin and Row/Column Order

We know, that {func}`wradlib.io.read_radolan_composite` returns a 2D-array (rows, cols) with the origin in the lower left corner. Same applies to {func}`wradlib.georef.get_radolan_grid`. For the next step, we need to flip the data and the coords up-down. The coordinate corner points also need to be adjusted from lower left corner to upper right corner.

```{code-cell} python
data, xy = wrl.georef.set_raster_origin(ds.SF.values, xy_raw, "upper")
print(data.shape)
```

### Export as GeoTIFF

For RADOLAN grids, this projection will probably not be recognized by
ESRI ArcGIS.

```{code-cell} python
# create 3 bands
data = np.stack((data, data + 100, data + 1000), axis=0)
print(data.shape)
gds = wrl.georef.create_raster_dataset(data, xy, crs=proj_osr)
wrl.io.write_raster_dataset(wdir / "geotiff.tif", gds, driver="GTiff")
```

### Export as ESRI ASCII file (aka Arc/Info ASCII Grid)

```{code-cell} python
# Export to Arc/Info ASCII Grid format (aka ESRI grid)
#     It should be possible to import this to most conventional
# GIS software.
# only use first band
proj_esri = proj_osr.Clone()
proj_esri.MorphToESRI()
ds = wrl.georef.create_raster_dataset(data[0], xy, crs=proj_esri)
wrl.io.write_raster_dataset(
    wdir / "aaigrid.asc", ds, driver="AAIGrid", options=["DECIMAL_PRECISION=2"]
)
```

## Step 3a: Read with xarray/rioxarray

```{code-cell} python
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
with xr.open_dataset(wdir / "geotiff.tif") as ds1:
    display(ds1)
    ds1.sel(band=1).band_data.plot(ax=ax1)
ax2 = fig.add_subplot(122)
with xr.open_dataset(wdir / "geotiff_rio.tif") as ds2:
    display(ds2)
    ds2.sel(band=1).band_data.plot(ax=ax2)
```

```{code-cell} python
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
with xr.open_dataset(wdir / "aaigrid.asc") as ds1:
    display(ds1)
    ds1.sel(band=1).band_data.plot(ax=ax1)
ax2 = fig.add_subplot(122)
with xr.open_dataset(wdir / "aaigrid_rio.asc") as ds2:
    display(ds2)
    ds2.sel(band=1).band_data.plot(ax=ax2)
```

## Step 3b: Read with GDAL

```{code-cell} python
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ds1 = wrl.io.open_raster(wdir / "geotiff.tif")
data1, xy1, proj1 = wrl.georef.extract_raster_dataset(ds1, nodata=-9999.0)
ax1.pcolormesh(xy1[..., 0], xy1[..., 1], data1[0])

ax2 = fig.add_subplot(122)
ds2 = wrl.io.open_raster(wdir / "geotiff_rio.tif")
data2, xy2, proj2 = wrl.georef.extract_raster_dataset(ds2, nodata=-9999.0)
ax2.pcolormesh(xy2[..., 0], xy2[..., 1], data2)
```

```{code-cell} python
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ds1 = wrl.io.open_raster(wdir / "aaigrid.asc")
data1, xy1, proj1 = wrl.georef.extract_raster_dataset(ds1, nodata=-9999.0)
ax1.pcolormesh(xy1[..., 0], xy1[..., 1], data1)

ax2 = fig.add_subplot(122)
ds2 = wrl.io.open_raster(wdir / "aaigrid_rio.asc")
data2, xy2, proj2 = wrl.georef.extract_raster_dataset(ds2, nodata=-9999.0)
ax2.pcolormesh(xy2[..., 0], xy2[..., 1], data2)
```
