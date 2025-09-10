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

# Hydrometeor partitioning ratio retrievals for GPM

In this notebook, GPM Dual Frequency Radar (DPR) measurements are used to derive Hydrometeor Partitioning Ratios (HPR) according to Pejcic et al 2025 (in review). This requires the measured Ku-band reflectivity, the dual-frequency ratios (Ku-band - Ka-band) and the DPR temperature and rain type information. The HPRs for the different hydrometeor classes are then presented.

```{code-cell} python
import wradlib as wrl
import wradlib_data
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()


import glob
import xarray as xr
import scipy
import pandas as dp
from dask.diagnostics import ProgressBar
```

## Read dual-frequency satellite observations (GPM)

```{code-cell} python
path_gpm = wradlib_data.DATASETS.fetch(
    "gpm/2A-CS-VP-24.GPM.DPR.V9-20211125.20180625-S050710-E051028.024557.V07A.HDF5"
)
# Read GPM data
sr_data = wrl.io.open_gpm_dataset(path_gpm, group="FS").chunk(nray=1)
sr_data = sr_data.set_coords(["Longitude", "Latitude"])
sr_data = xr.decode_cf(sr_data)
```

## Plot GPM overpass

```{code-cell} python
plt.figure(figsize=(5, 4))
sr_data.zFactorFinalNearSurface.isel(nfreq=0).plot(
    x="Longitude",
    y="Latitude",
    vmin=0,
    vmax=40,
    cmap="turbo",
)
```

## Assign coordinates

```{code-cell} python
sr_data = sr_data.set_coords("height")
sr_data = sr_data.assign_coords(nbin=sr_data.nbin.data)
sr_data = sr_data.assign_coords(nscan=sr_data.nscan.data)
sr_data = sr_data.assign_coords(nray=sr_data.nray.data)
```

## Plot overview along track

```{code-cell} python
zlvl = np.arange(10, 57.5, 2.5)
zlvl2 = np.arange(10, 57.5, 5)
dpr_lvl = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])

ff = 10
lw = 2.5
x1, x2 = -101, -98.5
y1, y2 = 0, 15000

fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # , sharey='row', sharex='col'

# Ku_measured
KU = sr_data.zFactorMeasured.sel(nfreq=0, nray=19)
plot = KU.plot(
    ax=axs[0],
    x="Longitude",
    y="height",
    cmap="HomeyerRainbow",
    levels=zlvl,
    cbar_kwargs={"extend": "neither", "label": "", "pad": 0.01, "ticks": zlvl2},
    xlim=(x1, x2),
    ylim=(y1, y2),
)

colorbar = plot.colorbar
colorbar.ax.tick_params(labelsize=ff)

# Ka_measured
KA = sr_data.zFactorMeasured.sel(nfreq=1, nray=19)
plot = KA.plot(
    ax=axs[1],
    x="Longitude",
    y="height",
    cmap="HomeyerRainbow",
    levels=zlvl,
    cbar_kwargs={"extend": "neither", "label": "", "pad": 0.01, "ticks": zlvl2},
    xlim=(x1, x2),
    ylim=(y1, y2),
)

colorbar = plot.colorbar
colorbar.ax.tick_params(labelsize=ff)


# DFR_measured
DFR = sr_data.zFactorMeasured.sel(nfreq=0, nray=19) - sr_data.zFactorMeasured.sel(
    nfreq=1, nray=19
)

plot = DFR.plot(
    ax=axs[2],
    x="Longitude",
    y="height",
    cmap="HomeyerRainbow",
    levels=dpr_lvl,
    cbar_kwargs={"extend": "neither", "label": "", "pad": 0.01, "ticks": dpr_lvl},
    xlim=(x1, x2),
    ylim=(y1, y2),
)

colorbar = plot.colorbar
colorbar.ax.tick_params(labelsize=ff)

T = [r"$Z_m^{K_u}$ in dBZ", r"$Z_m^{K_a}$ in dBZ", r"$DFR_m^{K_u-K_a}$ in dB"]
for i in range(len(T)):
    axs[i].set_title("", fontsize=ff)
    axs[i].set_title(T[i], fontsize=ff, loc="right")
    axs[i].set_ylabel("Height in m", fontsize=ff)
    axs[i].set_xlabel("Longitude in deg", fontsize=ff)
    axs[i].grid(ls=":", zorder=-100)
    axs[i].tick_params(axis="both", labelsize=ff)
```

```{code-cell} python
# centroids and covariances
cdp_file = wradlib_data.DATASETS.fetch("misc/hmcp_centroids_df.nc")
with xr.open_dataset(cdp_file) as cdp:
    cdp
cdp
```

```{code-cell} python
# weights
weights_file = wradlib_data.DATASETS.fetch("misc/hmcp_weights.nc")
with xr.open_dataset(weights_file) as cw:
    display(cw)
    pass
```

```{code-cell} python
with ProgressBar():
    obs = wrl.classify.create_gpm_observations(sr_data)
obs
```

```{code-cell} python
%%time
with ProgressBar():
    hmpr = wrl.classify.calculate_hmpr(obs, cw.weights, cdp)  # .compute()
hmpr
```

```{code-cell} python
hmpr = hmpr.chunk(hmc=1, nray=1)
hmpr
```

```{code-cell} python
with ProgressBar():
    hmpr_sel = hmpr.sel(nray=19) * 100
    hmpr_sel = hmpr_sel.compute()
hmpr_sel
```

```{code-cell} python
hpr_bins = [0, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
x1, x2 = -101, -98.5
y1, y2 = 0, 15000
with ProgressBar():
    hmpr_sel.plot(
        col="hmc",
        col_wrap=3,
        x="Longitude",
        y="height",
        cmap="HomeyerRainbow",
        levels=hpr_bins,
        xlim=(x1, x2),
        ylim=(y1, y2),
        cbar_kwargs={"ticks": hpr_bins},
    )
```
