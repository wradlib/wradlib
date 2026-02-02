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

```{code-cell} python3
import datetime as dt
import urllib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
from IPython.display import display

warnings.filterwarnings("ignore")
```

# 2D Hydrometeor Classification

The hydrometeorclassification code is based on the paper by {cite}`Zrnic2001` utilizing 2D trapezoidal membership functions based on the paper by {cite}`Straka2000` adapted by {cite}`Evaristo2013` for X-Band.

## Precipitation Types

```{code-cell} python3
pr_types = wrl.classify.pr_types
for k, v in pr_types.items():
    print(str(k) + " - ".join(v))
```

## Membership Functions

```{code-cell} python3
filename = wradlib_data.DATASETS.fetch("misc/msf_xband_v1.nc")
msf = xr.open_dataset(filename)
display(msf)
```

## Use Sounding Data

### Retrieve Sounding Data

To get the temperature as additional discriminator we use radiosonde data from the [University of Wyoming](http://weather.uwyo.edu/upperair/sounding.html).

The function {func}`wradlib.io.misc.get_radiosonde` tries to find the next next available radiosonde measurement on the given date.

```{code-cell} python3
rs_time = dt.datetime(2014, 6, 10, 12, 0)

try:
    rs_data, rs_meta = wrl.io.get_radiosonde(10410, rs_time)
except (urllib.error.HTTPError, urllib.error.URLError):
    dataf = wradlib_data.DATASETS.fetch("misc/radiosonde_10410_20140610_1200.h5")
    rs_data, _ = wrl.io.from_hdf5(dataf)
    metaf = wradlib_data.DATASETS.fetch("misc/radiosonde_10410_20140610_1200.json")
    with open(metaf, "r") as infile:
        import json

        rs_meta = json.load(infile)
rs_meta
```

### Extract Temperature and Height

```{code-cell} python3
stemp = rs_data["TEMP"]
sheight = rs_data["HGHT"]
# remove nans
idx = np.isfinite(stemp)
stemp = stemp[idx]
sheight = sheight[idx]
```

### Create DataArray

```{code-cell} python3
stemp_da = xr.DataArray(
    data=stemp,
    dims=["height"],
    coords=dict(
        height=(["height"], sheight),
    ),
    attrs=dict(
        description="Temperature.",
        units="degC",
    ),
)
display(stemp_da)
```

### Interpolate to higher resolution

```{code-cell} python3
hmax = 30000.0
ht = np.arange(0.0, hmax)
itemp_da = stemp_da.interp({"height": ht})
display(itemp_da)
```

### Fix Temperature below first measurement

```{code-cell} python3
itemp_da = itemp_da.bfill(dim="height")
```

### Plot Temperature Profile

```{code-cell} python3
fig = plt.figure(figsize=(5, 10))
ax = fig.add_subplot(111)
itemp_da.plot(y="height", ax=ax, marker="o", zorder=0, c="r")
stemp_da.to_dataset(name="stemp").plot.scatter(
    x="stemp", y="height", ax=ax, marker="o", c="b", zorder=1
)
ax.grid(True)
```

## Prepare Radar Data

### Load Radar Data

```{code-cell} python3
# read the radar volume scan
filename = "hdf5/2014-06-09--185000.rhi.mvol"
filename = wradlib_data.DATASETS.fetch(filename)
```

### Extract data for georeferencing

```{code-cell} python3
swp = xr.open_dataset(
    filename, engine="gamic", group="sweep_0", chunks={}
)
swp = xd.util.remove_duplicate_rays(swp)
swp = xd.util.reindex_angle(
    swp, start_angle=0, stop_angle=90, angle_res=0.2, direction=1
)
display(swp)
```

### Get Heights of Radar Bins

```{code-cell} python3
swp = swp.wrl.georef.georeference()
display(swp)
```

### Plot RHI of Heights

```{code-cell} python3
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)
cmap = mpl.cm.viridis
swp.z.plot(x="gr", y="z", ax=ax, cbar_kwargs=dict(label="Height [m]"))
ax.set_xlabel("Range [m]")
ax.set_ylabel("Height [m]")
ax.grid(True)
plt.show()
```

### Get Index into High Res Height Array

```{code-cell} python3
def merge_radar_profile(rds, cds):
    cds = cds.interp({"height": rds.z}, method="linear")
    rds = rds.assign({"TEMP": cds})
    return rds
```

```{code-cell} python3
hmc_ds = swp.pipe(merge_radar_profile, itemp_da)
display(hmc_ds)
```

```{code-cell} python3
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
hmc_ds.TEMP.plot(
    x="gr",
    y="z",
    cmap=cmap,
    ax=ax,
    add_colorbar=True,
    cbar_kwargs=dict(label="Temperature [°C]"),
)
ax.set_xlabel("Range [m]")
ax.set_ylabel("Range [m]")
ax.set_aspect("equal")
ax.set_ylim(0, 30000)
plt.show()
```

## HMC Workflow


### Setup Independent Observable $Z_H$
Retrieve membership function values based on independent observable

```{code-cell} python3
%%time
msf_val = msf.wrl.classify.msf_index_indep(swp.DBZH)
display(msf_val)
```

### Fuzzyfication

```{code-cell} python3
%%time
fu = msf_val.wrl.classify.fuzzyfi(
    hmc_ds, dict(ZH="DBZH", ZDR="ZDR", RHO="RHOHV", KDP="KDP", TEMP="TEMP")
)
```

### Probability

```{code-cell} python3
# weights dataset
w = xr.Dataset(dict(ZH=2.0, ZDR=1.0, RHO=1.0, KDP=1.0, TEMP=1.0))
display(w)
```

```{code-cell} python3
%%time
prob = fu.wrl.classify.probability(w).compute()
display(prob)
```

```{code-cell} python3
# prob = prob.compute()
```

### Classification

```{code-cell} python3
cl_res = prob.wrl.classify.classify(threshold=0.0)
display(cl_res)
```

### Compute

```{code-cell} python3
%%time
cl_res = cl_res.compute()
cl_res = cl_res.assign_coords(sweep_mode="rhi")
display(cl_res)
```

## HMC Results


### Plot Probability of HMC Types

```{code-cell} python3
prob = prob.assign_coords(hmc=np.array(list(pr_types.values())).T[1][:11])
prob = prob.where(prob > 0)
prob.plot(x="gr", y="z", col="hmc", col_wrap=4, cbar_kwargs=dict(label="Probability"))
```

### Plot maximum  probability

```{code-cell} python3
fig = plt.figure(figsize=(10, 6))
cmap = "cubehelix"
kwargs=dict(cbar_kwargs=dict(label="Probability"))
print(kwargs)
im = cl_res.max("hmc").wrl.vis.plot(
    ax=111,
    crs={"angular_spacing": 20.0, "radial_spacing": 12.0, "latmin": 2.5},
    cmap=cmap,
    fig=fig,
    **kwargs,
)
cgax = plt.gca()
cgax.set_xlim(0, 40000)
cgax.set_ylim(0, 14000)
t = cgax.set_title("Hydrometeorclassification", y=1.05)

caax = cgax.parasites[0]
caax.set_xlabel("Range [m]")
caax.set_ylabel("Range [m]")
plt.show()
```

### Plot classification result

```{code-cell} python3
bounds = np.arange(-0.5, prob.shape[0] + 0.6, 1)
ticks = np.arange(0, prob.shape[0] + 1)
cmap = mpl.cm.get_cmap("cubehelix", len(ticks))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
```

```{code-cell} python3
hydro = cl_res.argmax("hmc")
hydro.attrs = dict(long_name="Hydrometeorclassification")
hydro = hydro.assign_coords(sweep_mode="rhi")
```

```{code-cell} python3
fig = plt.figure(figsize=(10, 8))
im = hydro.wrl.vis.plot(
    ax=111,
    crs={"angular_spacing": 20.0, "radial_spacing": 12.0, "latmin": 2.5},
    norm=norm,
    cmap=cmap,
    fig=fig,
    add_colorbar=False,
)
cgax = plt.gca()
caax = cgax.parasites[0]
paax = cgax.parasites[1]

cbar = plt.colorbar(im, ticks=ticks, ax=cgax, fraction=0.046, norm=norm, pad=0.05)
cbar.set_label("Hydrometeorclass")
caax.set_xlabel("Range [km]")
caax.set_ylabel("Range [km]")
labels = [pr_types[i][1] for i, _ in enumerate(pr_types)]
labels = cbar.ax.set_yticklabels(labels)
t = cgax.set_title("Hydrometeorclassification", y=1.05)
cgax.set_xlim(0, 40000)
cgax.set_ylim(0, 14000)
plt.tight_layout()
```
