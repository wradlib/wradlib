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

# Production of a maximum reflectivity composite

```{code-cell} python
import os
import numpy as np
import wradlib
import xarray
import xradar
import matplotlib.pyplot as plt
```

Read volume reflectivity measurements from the three belgian radars

```{code-cell} python
from wradlib_data import DATASETS

filenames = ["bejab.pvol.hdf", "bewid.pvol.hdf", "behel.pvol.hdf"]
paths = [DATASETS.fetch(f"hdf5/{f}") for f in filenames]
volumes = [xradar.io.backends.odim.open_odim_datatree(p) for p in paths]
```

Define a raster dataset with a window including the 3 radars, a pixel size of 1km and the standard European projection.

```{code-cell} python
crs = wradlib.georef.epsg_to_osr(3035)
bounds = [0, 8, 48, 53]
bounds = wradlib.georef.project_bounds(bounds, crs)
print(bounds)
size = 1000
raster = wradlib.georef.create_raster_xarray(crs, bounds, size)
```

Define a geographic raster dataset with a window including the 3 radars, and an approximate pixel size of 1km.

```{code-cell} python
crs = wradlib.georef.epsg_to_osr(3035)
bounds = [0, 8, 48, 53]
size = 1000
raster2 = wradlib.georef.create_raster_geographic(bounds, size, size_in_meters=True)
```

Combine lowest radar sweep into a raster image for each radar

```{code-cell} python
# raster = raster2
metadata = xradar.model.required_sweep_metadata_vars
rasters = []
for volume in volumes:
    sweep = volume["sweep_0"].to_dataset()
    sweep = sweep[["DBZH"] + list(metadata)]
    sweep = sweep.sel(range=slice(0, 200e3))
    raster_sweep = wradlib.comp.sweep_to_raster(sweep, raster)
    rasters.append(raster_sweep)

for raster in rasters:
    raster = raster.drop_vars("spatial_ref")
    raster["DBZH"].plot(vmin=0, vmax=50)
    plt.axis("equal")
    plt.show()
```

Take the maximum value from the 3 rasters

```{code-cell} python
rasters_concat = xarray.concat(rasters, dim="sweep")
comp = rasters_concat.max(dim="sweep", keep_attrs=True)
comp["DBZH"].plot(vmin=0, vmax=50)
with open("comp.nc", "wb") as f:
    comp.to_netcdf(f)
!gdalinfo comp.nc
ds = xarray.open_dataset("comp.nc", engine="rasterio")
comp = comp.drop_vars("spatial_ref")
plt.axis("equal")
plt.show()
```
