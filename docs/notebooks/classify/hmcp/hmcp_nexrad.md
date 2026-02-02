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

# Hydrometeor partitioning ratio retrievals for Ground Radar

In this notebook, measurements from NEXRAD's KDDC ground radar are used to derive Hydrometeor Partitioning Ratios (HPR) following {cite}`Pejcic2025`. This requires the horizontal reflectivity, differential reflectivity, specific differential phase, cross correlation coefficient, temperature information and rain type. The temperature information is derived from sounding and a rain type classification is applied following {cite}`Park2009`. The HPRs for the different hydrometeor classes are then presented.

```{code-cell} python
import datetime as dt
import urllib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
from IPython.display import display
from scipy import spatial

warnings.filterwarnings("ignore")
```

## Read centroids, covariances and weights

```{code-cell} python
cdp_file = wradlib_data.DATASETS.fetch("misc/hmcp_centroids_dp.nc")
with xr.open_dataset(cdp_file) as cdp:
    pass
cdp
```

```{code-cell} python
weights_file = wradlib_data.DATASETS.fetch("misc/hmcp_weights.nc")
with xr.open_dataset(weights_file) as cw:
    pass
cw
```

## Read polarimetric radar observations

```{code-cell} python
volume = wradlib_data.DATASETS.fetch("netcdf/KDDC_2018_0625_051138_min.cf")
gr_data = xd.io.open_cfradial1_datatree(volume)
gr_data
```

## Get Temperature Profile

We would need the temperature of each radar bin. For that, we use Sounding Data. We also set the max_height to 30km and interpolate the vertical profile with a resolution of 1m.

```{code-cell} python
rs_time = dt.datetime.fromisoformat(
    str(gr_data.time_coverage_start.values.item().decode())
)
wmoid = 72451

try:
    rs_ds = wrl.io.get_radiosonde(
        wmoid, rs_time, cols=np.arange(13), xarray=True, max_height=30000.0, res=1.0
    )
except (urllib.error.HTTPError, urllib.error.URLError):
    print("service down")
    dataf = wradlib_data.DATASETS.fetch("misc/radiosonde_72451_20180625_0000.h5")
    rs_data, _ = wrl.io.from_hdf5(dataf)
    metaf = wradlib_data.DATASETS.fetch("misc/radiosonde_72451_20180625_0000.json")
    with open(metaf, "r") as infile:
        import json

        rs_meta = json.load(infile)
    rs_ds = wrl.io.radiosonde_to_xarray(
        rs_data, meta=rs_meta, max_height=30000.0, res=1.0
    )
```

```{code-cell} python
display(rs_ds)
```

## Plot Temperature Profile

```{code-cell} python
fig = plt.figure(figsize=(5, 10))
ax = fig.add_subplot(111)
rs_ds.TEMP.plot(y="HGHT", ax=ax, zorder=0, c="r")
ax.grid(True)
```

## get freezing level height

We need to obtain the freezing level height, which is needed for an ad-hoc retrieval of raintype.

```{code-cell} python
fl = np.abs(rs_ds).argmin("HGHT").TEMP
display(fl)
```

## georeference DataTree

For the interpolation of the temperature sounding data onto the radar sweeps, we need the xyz coordinates of the sweeps.

```{code-cell} python
gr_data2 = gr_data.xradar.georeference()
```

```{code-cell} python
gr_data2["sweep_0"]
```

## Interpolate Temperature onto sweeps

The following function interpolates the vertical temperature profile onto the radar sweeps.

```{code-cell} python
def merge_radar_profile(rds, cds):
    if "z" in rds.coords:
        cds = cds.interp({"HGHT": rds.z}, method="linear")
        rds = rds.assign({"TEMP": cds})
    return rds


gr_data3 = gr_data2.map_over_datasets(merge_radar_profile, rs_ds.TEMP)
```

```{code-cell} python
gr_data3["sweep_1"].TEMP.plot(x="x", y="y")
```

## Ad-hoc retrieval of raintype


The following algorithm of raintype estimation is derived after {cite}`Park2009`.

1. keep all radar bins >= 45 dBZ
1. keep all radar bins > 30 dBZ and height > fl + 1600m
1. combine 1 and 2
1. iterate over x,y pairs and fetch from whole tree to set as convective.

```{code-cell} python
def mask_data(rds, fl):
    if "z" in rds.coords:
        # Thresholding and smoothing (Park et al.)
        # -----------------------------------------
        xwin_zh = 5
        rds = rds.where(rds.RH > 0.8)
        rds["CZ"] = rds.CZ.rolling(
            range=xwin_zh, min_periods=xwin_zh // 2, center=True
        ).mean(skipna=True)
        mask = (rds.CZ >= 45) | ((rds.CZ > 30) & (rds.z > (fl + 1600)))
        rds = rds.assign(mask=mask)
    return rds


gr_data4 = gr_data3.map_over_datasets(mask_data, fl)
```

## Extract xyz bin coordinates

This iterates over the whole DataTree and extracts the RainType-mask as 1-dimensional array. This keeps only valid values.

```{code-cell} python
def get_xyz(tree):
    swp_list = []
    for key in list(tree.children):
        if "sweep" in key:
            ds = tree[key].ds.stack(npoints=("azimuth", "range"))
            ds = ds.reset_coords().where(ds.mask, drop=True)
            swp_list.append(ds.mask)
    return xr.concat(swp_list, "npoints")
```

## Interpolation of RainType mask

This interpolates the RainType for all sweeps, to get a vertically consistent RainType.
For this a KDTree is created containing the valid values from above, which is used for the Nearest interpolator.
The ROI (maxdist) is assumed to be the current range resolution, but can be specified as keyword argument.

```{code-cell} python
%%time

kwargs = dict(balanced_tree=True)
xyz = get_xyz(gr_data4)
src = np.vstack([xyz.x.values, xyz.y.values]).T
kdtree = spatial.KDTree(src, **kwargs)


def get_range_res(rng):
    return rng.range.diff("range").median("range").values


def ipol_mask(swp, xyz, kdtree, maxdist=None):
    if "z" in swp.coords:
        if maxdist is None:
            maxdist = swp.range.attrs.get(
                "meters_between_gates", get_range_res(swp.range)
            )
        trg = np.vstack([swp.x.values.ravel(), swp.y.values.ravel()]).T
        nn = wrl.ipol.Nearest(kdtree, trg)
        out = nn(xyz.values, maxdist=maxdist).reshape(swp.x.shape)
        swp = swp.assign(rt=(swp.x.dims, out))
        swp["rt"] = xr.where(swp["rt"] == 1, 2, 1)
    return swp


gr_data5 = gr_data4.map_over_datasets(ipol_mask, xyz, kdtree)
```

```{code-cell} python
gr_data5["sweep_0"].rt.plot(x="x", y="y")
```

## ZDR Offset retrieval

The ZDR offset was retrieved following {cite}`Ryzhkov2019` (6.2.3 Z-ZDR Consistency in Light Rain, pp. 153-156).

```{code-cell} python
zdr_offset = 0.5
```

## Extract sweep 2 for further processing

```{code-cell} python
swp = gr_data5["sweep_2"].ds
swp
```

```{code-cell} python
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
swpp = swp[["CZ", "DR", "KD", "RH"]]
display(swpp)

LVL = [
    np.arange(10, 57.5, 2.5),
    np.array([-1, -0.5, -0.25, -0.1, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 3]),
    np.array(
        [-0.5, -0.1, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 2, 3, 4]
    ),  # np.arange(-0.5,2, 0.2),
    np.arange(0.9, 1.01, 0.01),
]

for i, var in enumerate(swpp.data_vars.values()):
    cbar_kwargs = {
        "extend": "neither",
        "label": "",
        "pad": 0.01,
        "ticks": LVL[i],
    }
    ax = axs.flat[i]
    var.dropna("range", how="all").plot(
        x="x",
        y="y",
        ax=ax,
        cmap="HomeyerRainbow",
        levels=LVL[i],
        cbar_kwargs=cbar_kwargs,
    )
    ax.set_title(var.attrs["long_name"])

plt.tight_layout()
```

## Combine observations into xr.DataArray

Use the mapping to bind the existing variable names to the needed names.

```{code-cell} python
# mapping observations
obs_mapping = {
    "ZH": "CZ",
    "ZDR": "DR",
    "KDP": "KD",
    "RHO": "RH",
    "RT": "rt",
    "TEMP": "TEMP",
}
polars = wrl.classify.create_gr_observations(swp, obs_mapping)
polars
```

## Calculate hydrometeor partitioning ratios (HPR)

This uses the loaded weights and centroids to retrieve the hydrometeor partitioning ratio from the observations.

```{code-cell} python
hmpr = wrl.classify.calculate_hmpr(polars, cw.weights, cdp)
```

## Plotting all Hydrometeor-Classes

For better plotting we transfrom to 100% and drop NaN data.

```{code-cell} python
hmpr = hmpr.dropna("range", how="all") * 100
hpr_bins = [0, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

hmpr.plot(
    col="hmc",
    col_wrap=3,
    x="x",
    y="y",
    cmap="HomeyerRainbow",
    levels=hpr_bins,
    cbar_kwargs={"ticks": hpr_bins},
)
```
