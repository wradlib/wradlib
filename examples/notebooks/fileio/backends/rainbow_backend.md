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

# xarray Rainbow5 backend

In this example, we read Rainbow5 data files using the xradar `rainbow` xarray backend.

```{code-cell} python
import glob
import gzip
import io
import wradlib as wrl
import wradlib_data
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import xradar as xd
import xarray as xr

try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Load Rainbow5 Volume Data

```{code-cell} python
fpath = "rainbow/2013051000000600dBZ.vol"
f = wradlib_data.DATASETS.fetch(fpath)
vol = xd.io.open_rainbow_datatree(f, reindex_angle=False)
```

## Inspect RadarVolume

```{code-cell} python
display(vol)
```

## Inspect root group

The `sweep` dimension contains the number of scans in this radar volume. Further the dataset consists of variables (location coordinates, time_coverage) and attributes (Conventions, metadata).

```{code-cell} python
vol.root
```

## Inspect sweep group(s)

The sweep-groups can be accessed via their respective keys. The dimensions consist of `range` and `time` with added coordinates `azimuth`, `elevation`, `range` and `time`. There will be variables like radar moments (DBZH etc.) and sweep-dependent metadata (like `fixed_angle`, `sweep_mode` etc.).

```{code-cell} python
display(vol["sweep_0"])
```

## Georeferencing

```{code-cell} python
swp = vol["sweep_0"].ds.copy()
swp = swp.assign_coords(sweep_mode=swp.sweep_mode)
swp = swp.wrl.georef.georeference()
```

## Inspect radar moments

The DataArrays can be accessed by key or by attribute. Each DataArray has dimensions and coordinates of its parent dataset.

```{code-cell} python
display(swp.DBZH)
```

## Create simple plot

Using xarray features a simple plot can be created like this. Note the `sortby('time')` method, which sorts the radials by time.

For more details on plotting radar data see under [Visualization](../../visualisation/plotting).

```{code-cell} python
swp.DBZH.sortby("time").plot(x="range", y="time", add_labels=False)
```

```{code-cell} python
fig = plt.figure(figsize=(5, 5))
pm = swp.DBZH.wrl.vis.plot(crs={"latmin": 3e3}, fig=fig)
```

## Retrieve explicit group

```{code-cell} python
swp_b = xr.open_dataset(
    f, engine="rainbow", group="sweep_5", backend_kwargs=dict(reindex_angle=False)
)
display(swp_b)
```
