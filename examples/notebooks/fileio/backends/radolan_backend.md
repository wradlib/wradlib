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

# xarray RADOLAN backend

In this example, we read RADOLAN data files using the xarray `radolan` backend.

```{code-cell} python
import glob
import os
import wradlib as wrl
import wradlib_data
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Load RADOLAN Data

```{code-cell} python
fpath = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
f = wradlib_data.DATASETS.fetch(fpath)
comp = wrl.io.open_radolan_dataset(f)
```

## Inspect Data

```{code-cell} python
display(comp)
```

## Inspect RADOLAN moments

The DataArrays can be accessed by key or by attribute. Each DataArray has dimensions and coordinates of its parent dataset.

```{code-cell} python
display(comp.RW)
```

## Create simple plot

Using xarray features a simple plot can be created like this.

```{code-cell} python
comp.RW.plot(x="x", y="y", add_labels=False)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
comp.RW.plot(x="x", y="y", ax=ax)
```

## Mask some values

```{code-cell} python
ds = comp["RW"].where(comp["RW"] >= 1)
ds.plot()
```

## Use `xr.open_dataset`


```{code-cell} python
comp2 = xr.open_dataset(f, engine="radolan")
display(comp2)
```

## Use `xr.open_mfdataset` to retrieve timeseries

```{code-cell} python
flist = [
    "radolan/misc/raa01-sf_10000-1305270050-dwd---bin.gz",
    "radolan/misc/raa01-sf_10000-1305280050-dwd---bin.gz",
]
flist = [wradlib_data.DATASETS.fetch(f) for f in flist]
```

```{code-cell} python
comp3 = xr.open_mfdataset(flist, engine="radolan")
display(comp3)
```
