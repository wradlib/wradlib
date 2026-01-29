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

# Heuristic clutter detection based on distribution properties ("histo cut")

Detects areas with anomalously low or high average reflectivity or precipitation. It is recommended to use long term average or sums (months to year).

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr

warnings.filterwarnings("ignore")
```

## Load rainfall data

This imports annual rainfall data from DWD radar Feldberg.

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("misc/annual_rainfall_fbg.gz")
yearsum = np.loadtxt(filename)
site = (47.875, 8.004, 1489.6)
r = np.arange(0, 128000, 1000) + 500
az = np.arange(0, 360, 1) + 0.5

fbg = wrl.georef.create_xarray_dataarray(yearsum, r=r, phi=az, site=site).wrl.georef.georeference()
display(fbg)
```

## Apply histo-cut filter

Depending on your data and climate you can parameterize the upper and lower frequency percentage with the kwargs `upper_frequency`/`lower_frequency`. For European ODIM_H5 data these values have been found to be in the order of 0.05 {cite}`Overeem2023`. The current default is 0.01 for both values.

```{code-cell} python
clutter = fbg.wrl.classify.histo_cut()
display(clutter)
```

## Plot results

```{code-cell} python
fig = plt.figure(figsize=(14, 12))
ax1 = fig.add_subplot(221)
pm = np.log(fbg).wrl.vis.plot(ax=ax1)
ax1.set_title("Logarithm of annual precipitation sum")
ax2 = fig.add_subplot(222)
cmap2 = plt.get_cmap("bwr", 2)
pm = clutter.where(clutter > 0).wrl.vis.plot(ax=ax2, cmap=cmap2)
ax2.set_title("Map of execptionally low and high values\n(clutter and beam blockage)")
ax3 = fig.add_subplot(223)
pm = xr.where(clutter == 1, 1, 0).wrl.vis.plot(ax=ax3, cmap="binary")
ax3.set_title("Map of execptionally high values\n(clutter)")
ax4 = fig.add_subplot(224)
pm = xr.where(clutter == 2, 1, 0).wrl.vis.plot(ax=ax4, cmap="binary")
ax4.set_title("Map of execptionally low values\n(beam blockage)")
fig.tight_layout()
```
