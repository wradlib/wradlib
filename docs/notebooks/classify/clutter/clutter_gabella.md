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

# Clutter detection using the Gabella approach

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr

warnings.filterwarnings("ignore")
```

## Read the data

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
swp = xr.open_dataset(filename, engine="gamic", group="sweep_0")
swp = swp.set_coords("sweep_mode")
swp = swp.wrl.georef.georeference()
display(swp)
```

## Apply filter

```{code-cell} python
clmap = swp.DBTH.wrl.classify.filter_gabella(
    wsize=5, thrsnorain=0.0, tr1=6.0, n_p=8, tr2=1.3
)
clmap
```

## Plot results

```{code-cell} python
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
pm = swp.DBTH.wrl.vis.plot(ax=ax1)
ax1.set_title("Reflectivity")
ax2 = fig.add_subplot(122)
pm = clmap.wrl.vis.plot(ax=ax2)
ax2.set_title("Cluttermap")
fig.tight_layout()
```
