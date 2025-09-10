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

# Heuristic clutter detection based on distribution properties ("histo cut")

Detects areas with anomalously low or high average reflectivity or precipitation. It is recommended to use long term average or sums (months to year).

```{code-cell} python
import wradlib as wrl
import wradlib_data
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Load annual rainfall accumulation example (from DWD radar Feldberg)

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("misc/annual_rainfall_fbg.gz")
yearsum = np.loadtxt(filename)
```

## Apply histo-cut filter to retrieve boolean array that highlights clutter as well as beam blockage

Depending on your data and climate you can parameterize the upper and lower frequency percentage with the kwargs `upper_frequency`/`lower_frequency`. For European ODIM_H5 data these values have been found to be in the order of 0.05 in [EURADCLIM: The European climatological high-resolution gauge-adjusted radar precipitation dataset](https://essd.copernicus.org/preprints/essd-2022-334/). The current default is 0.01 for both values.

```{code-cell} python
mask = wrl.classify.histo_cut(yearsum)
```

## Plot results

```{code-cell} python
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(221)
yearsum = wrl.georef.create_xarray_dataarray(yearsum).wrl.georef.georeference()
pm = np.log(yearsum).wrl.vis.plot(ax=ax)
plt.title("Logarithm of annual precipitation sum")
plt.colorbar(pm, shrink=0.75)
ax = fig.add_subplot(222)
mask = wrl.georef.create_xarray_dataarray(mask).wrl.georef.georeference()
pm = mask.wrl.vis.plot(ax=ax)
plt.title("Map of execptionally low and high values\n(clutter and beam blockage)")
plt.colorbar(pm, shrink=0.75)
ax = fig.add_subplot(223)
pm = mask.where(mask == 1).wrl.vis.plot(ax=ax)
plt.title("Map of execptionally high values\n(clutter)")
plt.colorbar(pm, shrink=0.75)
ax = fig.add_subplot(224)
pm = mask.where(mask == 2).wrl.vis.plot(ax=ax)
plt.title("Map of execptionally low values\n(beam blockage)")
plt.colorbar(pm, shrink=0.75)
```
