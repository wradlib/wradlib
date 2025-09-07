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

# Load ODIM_H5 Volume data from German Weather Service

In this example, we obtain and read the last 2 hours of available volumetric radar data from German Weather Service available at [opendata.dwd.de](https://opendata.dwd.de). Finally we do some plotting.

This retrieves 24 timesteps of the 10 sweeps (moments DBZH and VRADH) of the DWD volume scan of a distinct radar. This amounts to 480 data files which are combined into one volumetric Cf/Radial2 like xarray powered structure.

Exports to single file Odim_H5 and Cf/Radial2 format are shown at the end of this tutorial.

```{warning}
The following code is based on [xarray](https://docs.xarray.dev) and [xradar](https://docs.openradarscience.org/projects/xradar). It claims multiple data files and presents them in a ``DataTree``.
```

```{code-cell} python
import wradlib as wrl
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd

try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

```{code-cell} python
import urllib3
import os
import io
import glob
import shutil
import datetime
```

```{code-cell} python
from html.parser import HTMLParser


class DWDHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        self.links.append(attrs[0][1])


parser = DWDHTMLParser()
```

# Download data from opendata.dwd.de

```{code-cell} python
radar = "ESS"
DBZH = "sweep_vol_z"
VRADH = "sweep_vol_v"

import certifi

opendata_url1 = f"https://opendata.dwd.de/weather/radar/sites/{DBZH}/{radar.lower()}/hdf5/filter_polarimetric/"

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
response = http.request("GET", opendata_url1).data.decode("utf-8")

parser.links = []
parser.feed(response)
filelist1 = parser.links[1:]

filelist1.sort(key=lambda x: x.split("-")[2])
filelist1.reverse()

opendata_url2 = f"https://opendata.dwd.de/weather/radar/sites/{VRADH}/{radar.lower()}/hdf5/filter_polarimetric/"

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
response = http.request("GET", opendata_url2).data.decode("utf-8")

parser.links = []
parser.feed(response)
filelist2 = parser.links[1:]

filelist2.sort(key=lambda x: x.split("-")[2])
filelist2.reverse()
```

## Clean up local folder

```{code-cell} python
flist = glob.glob("ras07*")
for f in flist:
    os.remove(f)
```

## Download latest 24 volumes to current directory

```{code-cell} python
for f in filelist1[: 10 * 25]:
    with http.request(
        "GET", os.path.join(opendata_url1, f), preload_content=False
    ) as r, open(f, "wb") as out:
        shutil.copyfileobj(r, out)

for f in filelist2[: 10 * 25]:
    with http.request(
        "GET", os.path.join(opendata_url2, f), preload_content=False
    ) as r, open(f, "wb") as out:
        shutil.copyfileobj(r, out)
```

```{code-cell} python
volume_reflectivity = glob.glob("ras07*_dbzh_*")
volume_velocity = glob.glob("ras07*_vradh_*")
```

```{code-cell} python
volume_reflectivity = np.array(
    sorted(volume_reflectivity, key=lambda x: x.split("-")[2])
)
volume_velocity = np.array(sorted(volume_velocity, key=lambda x: x.split("-")[2]))
```

```{code-cell} python
volume_reflectivity = volume_reflectivity.reshape(-1, 10).T
volume_velocity = volume_velocity.reshape(-1, 10).T
```

## Read the data into xarray powered structure

```{code-cell} python
dsl = []
reindex_angle = dict(
    tolerance=1.0, start_angle=0, stop_angle=360, angle_res=1.0, direction=1
)
for i, (r, v) in enumerate(zip(volume_reflectivity, volume_velocity)):
    ds0 = [
        xr.open_dataset(
            r0,
            engine="odim",
            group="sweep_0",
            reindex_angle=reindex_angle,
            fix_second_angle=True,
        )
        for r0 in r
    ]
    ds0 = [r0.assign_coords(sweep_mode=r0.sweep_mode.min()) for r0 in ds0]
    ds1 = [
        xr.open_dataset(
            v0,
            engine="odim",
            group="sweep_0",
            reindex_angle=reindex_angle,
            fix_second_angle=True,
        )
        for v0 in v
    ]
    ds1 = [r1.assign_coords(sweep_mode=r1.sweep_mode.min()) for r1 in ds1]
    ds2 = [
        xr.merge([r0, v0], compat="no_conflicts").assign(
            volume_time=r0.time.min().dt.floor("5min")
        )
        for r0, v0 in zip(ds0, ds1)
    ]
    ds2 = [r2.wrl.georef.georeference() for r2 in ds2]
    ds = xr.concat(ds2, "volume_time")
    dsl.append(ds)
```

```{code-cell} python
vt_min = np.array([dsl[i].volume_time.min().values for i in range(10)])
vt_max = np.array([dsl[i].volume_time.max().values for i in range(10)])
if not all((vt_min == vt_min[0])):
    dsl = [ds.sel(volume_time=slice(vt_min.max(), vt_max.min())) for ds in dsl]
else:
    dsl = [ds.isel(volume_time=slice(1, -1)) for ds in dsl]
dsl = sorted(dsl, key=lambda ds: ds.time.min().values)
```

```{code-cell} python
dtree = {"/": xd.io.backends.common._get_required_root_dataset(dsl, optional=False)}
for i, swp in enumerate(dsl):
    dsl[i]["sweep_number"] = i
dtree = xd.io.backends.common._attach_sweep_groups(dtree, dsl)
```

```{code-cell} python
vol = xr.DataTree.from_dict(dtree)
```

```{code-cell} python
display(vol)
```

## Inspect structure
### Root Group

```{code-cell} python
vol.root
```

### Sweep Groups

```{code-cell} python
vol["sweep_0"]
```

## plot sweeps
### DBZH

```{code-cell} python
vol["sweep_0"].isel(volume_time=0)
```

```{code-cell} python
vol.match("sweep*")
```

```{code-cell} python
swp = vol["sweep_0"].isel(volume_time=0).ds
swp.sweep_fixed_angle.values
```

```{code-cell} python
fig, gs = plt.subplots(
    4, 3, figsize=(20, 30), sharex=True, sharey=True, constrained_layout=True
)

for i, grp in enumerate(vol.match("sweep_*")):
    ax = gs.flat[i]
    swp = vol[grp].isel(volume_time=0).ds
    swp.DBZH.wrl.vis.plot(ax=ax, fig=fig)
    ax.set_title(swp.sweep_fixed_angle.values)

fig.delaxes(gs.flat[-2])
fig.delaxes(gs.flat[-1])
```

### VRADH

```{code-cell} python
fig, gs = plt.subplots(
    4, 3, figsize=(20, 30), sharex=True, sharey=True, constrained_layout=True
)

for i, grp in enumerate(vol.match("sweep_*")):
    ax = gs.flat[i]
    swp = vol[grp].isel(volume_time=0).ds
    swp.VRADH.wrl.vis.plot(ax=ax, fig=fig)
    ax.set_title(swp.sweep_fixed_angle.values)

fig.delaxes(gs.flat[-2])
fig.delaxes(gs.flat[-1])
```

### Plot single sweep using cartopy

```{code-cell} python
vol0 = vol.isel(volume_time=0)
swp = vol0["sweep_9"].ds
```

```{code-cell} python
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

map_trans = ccrs.AzimuthalEquidistant(
    central_latitude=swp.latitude.values,
    central_longitude=swp.longitude.values,
)
```

```{code-cell} python
map_proj = ccrs.AzimuthalEquidistant(
    central_latitude=swp.latitude.values,
    central_longitude=swp.longitude.values,
)

pm = swp.DBZH.wrl.georef.georeference().wrl.vis.plot(crs=map_proj)
ax = plt.gca()
ax.gridlines(crs=map_proj)
print(ax)
```

```{code-cell} python
map_proj = ccrs.Mercator(central_longitude=swp.longitude.values)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=map_proj)
pm = swp.DBZH.wrl.georef.georeference().wrl.vis.plot(ax=ax)
ax.gridlines(draw_labels=True)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 8))
proj = ccrs.AzimuthalEquidistant(
    central_latitude=swp.latitude.values,
    central_longitude=swp.longitude.values,
)
ax = fig.add_subplot(111, projection=proj)
pm = swp.DBZH.wrl.georef.georeference().wrl.vis.plot(ax=ax)
ax.gridlines()
```

### Inspect radar moments

The DataArrays can be accessed by key or by attribute. Each DataArray inherits dimensions and coordinates of it's parent dataset. There are attributes connected which are defined by Cf/Radial and/or ODIM_H5 standard.

```{code-cell} python
vol["sweep_9"].isel(volume_time=0).ds.DBZH
```

```{code-cell} python
vol["sweep_9"].isel(volume_time=0).ds.sweep_mode
```

## Plot Quasi Vertical Profile

```{code-cell} python
ts = vol["sweep_9"]
ts
```

```{code-cell} python
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ts.ds.DBZH.median("azimuth").plot(x="volume_time", vmin=-10, vmax=30, ax=ax)
ax.set_title(f"{np.datetime_as_string(ts.ds.time[0][0].values, unit='D')}")
ax.set_ylim(0, 20000)
```

## Export to OdimH5

This exports the radar volume at given timestep including all moments into one ODIM_H5 compliant data file.

```{code-cell} python
xd.io.to_odim(vol0, "dwd_odim.h5", source="RAD:DWD")
```

## Export to Cf/Radial2

This exports the radar volume at given timestep including all moments into one Cf/Radial2 compliant data file.

```{code-cell} python
xd.io.to_cfradial2(vol0, "dwd_cfradial2.nc")
```

## Import again


```{code-cell} python
vol1 = xd.io.open_odim_datatree("dwd_odim.h5")
display(vol1)
```

```{code-cell} python
vol2 = xr.open_datatree("dwd_cfradial2.nc")
display(vol2)
```
