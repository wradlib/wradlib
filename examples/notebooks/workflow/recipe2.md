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

# Recipe #2: Reading and visualizing an ODIM_H5 polar volume


This recipe shows how extract the polar volume data from an ODIM_H5 hdf5 file (KNMI example file from OPERA), construct a 3-dimensional Cartesian volume and produce a diagnostic plot. The challenge for this file is that for each elevation angle, the scan strategy is different.

```{code-cell} python
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

```{code-cell} python
import datetime as dt
from osgeo import osr

# read the data (sample file in WRADLIB_DATA)
filename = wradlib_data.DATASETS.fetch("hdf5/knmi_polar_volume.h5")

raw_dt = xd.io.open_odim_datatree(filename)
display(raw_dt)
```

```{code-cell} python
proj = osr.SpatialReference()
proj.ImportFromEPSG(32632)
for key in list(raw_dt.children):
    if "sweep" in key:
        raw_dt[key].ds = raw_dt[key].ds.wrl.georef.georeference(crs=proj)
```

```{code-cell} python
swp_list = []
for key in list(raw_dt.children):
    if "sweep" in key:
        ds = raw_dt[key].ds
        xyz = (
            xr.concat(
                [
                    ds.coords["x"].reset_coords(drop=True),
                    ds.coords["y"].reset_coords(drop=True),
                    ds.coords["z"].reset_coords(drop=True),
                ],
                "xyz",
            )
            .stack(npoints=("azimuth", "range"))
            .transpose(..., "xyz")
        )
        swp_list.append(xyz)
xyz = xr.concat(swp_list, "npoints")
```

```{code-cell} python
swp_list[0]
```

```{code-cell} python
data_list = []
for key in list(raw_dt.children):
    if "sweep" in key:
        ds = raw_dt[key].ds
        data = ds.DBZH.stack(npoints=("azimuth", "range"))
        data_list.append(data)
data = xr.concat(data_list, "npoints")
```

```{code-cell} python
# generate 3-D Cartesian target grid coordinates
sitecoords = (raw_dt.longitude.values, raw_dt.latitude.values, raw_dt.altitude.values)
maxrange = 200000.0
minelev = 0.1
maxelev = 25.0
maxalt = 5000.0
horiz_res = 2000.0
vert_res = 250.0
trgxyz, trgshape = wrl.vpr.make_3d_grid(
    sitecoords, proj, maxrange, maxalt, horiz_res, vert_res
)
```

```{code-cell} python
# interpolate to Cartesian 3-D volume grid
tstart = dt.datetime.now()
gridder = wrl.vpr.CAPPI(
    xyz.values,
    trgxyz,
    # gridshape=trgshape,
    maxrange=maxrange,
    minelev=minelev,
    maxelev=maxelev,
)
vol = np.ma.masked_invalid(gridder(data.values).reshape(trgshape))
print("3-D interpolation took:", dt.datetime.now() - tstart)
```

```{code-cell} python
# diagnostic plot
trgx = trgxyz[:, 0].reshape(trgshape)[0, 0, :]
trgy = trgxyz[:, 1].reshape(trgshape)[0, :, 0]
trgz = trgxyz[:, 2].reshape(trgshape)[:, 0, 0]
wrl.vis.plot_max_plan_and_vert(
    trgx,
    trgy,
    trgz,
    vol,
    unit="dBZH",
    levels=range(-32, 60),
    cmap="turbo",
)
```

```{note}
In order to run the recipe code, you need to extract the sample data into a directory pointed to by environment variable ``WRADLIB_DATA``.
```
