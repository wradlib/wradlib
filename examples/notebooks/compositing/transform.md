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

# Comparaison of transformation methods

```{code-cell} python
import time

import matplotlib.pyplot as plt
import wradlib
import xarray
import xradar
```

Get a radar sweep with reflectivity measurements and metadata

```{code-cell} python
from wradlib_data import DATASETS

filename = "hdf5/71_20181220_060628.pvol.h5"
filename = DATASETS.fetch(filename)
volume = xradar.io.open_odim_datatree(filename)
sweep = volume["sweep_0"].ds
metadata = xradar.model.required_sweep_metadata_vars
sweep = sweep[["DBZH"] + list(metadata)]
```

Interpolate radar sweep into a Cartesian grid using nearest neighbor method

```{code-cell} python
window = [-200e3, 200e3, -200e3, 200e3]
size = 1000
lon = float(sweep.longitude.values)
lat = float(sweep.latitude.values)
crs = wradlib.georef.get_radar_projection((lon, lat))
raster = wradlib.georef.create_raster_xarray(crs, window, size)
grid = raster
sweep = sweep.wrl.georef.georeference(crs=crs)
tic = time.time()
comp1 = sweep.DBZH.wrl.comp.togrid(
    grid, radius=250e3, center=(lon, lat), interpol=wradlib.ipol.Nearest
)
toc = time.time() - tic
print(f"Time elapsed: {toc}")
```

Combine radar sweep into a raster image by taking the mean

```{code-cell} python
tic = time.time()
transform = wradlib.comp.transform_binned(sweep, raster)
comp2 = wradlib.comp.sweep_to_raster(sweep, raster)
toc = time.time() - tic
print(f"Time elapsed: {toc}")
```

Compare the methods at close range

```{code-cell} python
sweep.DBZH.plot(x="x", y="y", cmap="PuBuGn", vmin=10, vmax=60)
plt.xlim(-10e3, 10e3)
plt.ylim(-10e3, 10e3)
plt.suptitle("Sweep")
plt.show()

comp1.plot.pcolormesh(cmap="PuBuGn", vmin=10, vmax=60)
plt.xlim(-10e3, 10e3)
plt.ylim(-10e3, 10e3)
plt.suptitle("Grid (nearest)")
plt.show()

comp2 = comp2.drop_vars("spatial_ref")
comp2["DBZH"].plot.pcolormesh(cmap="PuBuGn", vmin=10, vmax=60)
plt.xlim(-10e3, 10e3)
plt.ylim(-10e3, 10e3)
plt.suptitle("Raster (binned)")
plt.show()
```
