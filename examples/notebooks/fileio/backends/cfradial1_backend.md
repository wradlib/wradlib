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

# xarray CfRadial1 backend

In this example, we read CfRadial1 data files using the `xradar` `cfradial1` backend.

Data is also exported to ODIM_H5 and CfRadial2.

```{code-cell} python
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

## Load CfRadial1 Volume Data

We use the functionality provided now by [xradar](https://docs.openradarscience.org/projects/xradar/en/stable/) to read the CfRadial1 data into a DataTree.

```{code-cell} python
fpath = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
f = wradlib_data.DATASETS.fetch(fpath)
vol = xd.io.open_cfradial1_datatree(f)
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

``sweep_mode`` is assigned coordinate, as we need it available on the DataArray.

```{code-cell} python
swp = vol["sweep_0"].ds
swp = swp.assign_coords(sweep_mode=swp.sweep_mode)
swp = swp.wrl.georef.georeference()
display(swp)
```

## Inspect radar moments

The DataArrays can be accessed by key or by attribute. Each DataArray has dimensions and coordinates of its parent dataset. There are attributes connected which are defined by Cf/Radial standard.

```{code-cell} python
display(swp.DBZ)
```

## Create simple plot

Using xarray features a simple plot can be created like this. Note the `sortby('time')` method, which sorts the radials by time.

For more details on plotting radar data see under [Visualization](../../visualisation/plotting).

```{code-cell} python
swp.DBZ.sortby("time").plot(x="range", y="time", add_labels=False)
```

```{code-cell} python
fig = plt.figure(figsize=(5, 5))
pm = swp.DBZ.wrl.vis.plot(crs={"latmin": 3e3}, fig=fig)
```

## Use `xr.open_dataset` to retrieve explicit group

```{warning}
Since $\omega radlib$ version 2.0 all xarray backend related functionality is imported from [xradar](https://github.com/openradar/xradar)-package.
```

```{code-cell} python
swp_b = xr.open_dataset(
    f, engine="cfradial1", group="sweep_1", backend_kwargs=dict(reindex_angle=False)
)
display(swp_b)
```
