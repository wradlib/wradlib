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

# Quick-view a sweep in polar coordinates


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import wradlib as wrl
import wradlib_data
import warnings
import xarray as xr
import cmweather

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Read a polar data set in ODIM_H5 from the Koninklijk Nederlands Meteorologisch Instituut (KNMI)

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("hdf5/knmi_polar_volume.h5")
print(filename)
```

```{code-cell} python
swp = xr.open_dataset(filename, engine="odim", group="sweep_0")
```

Inspect the data set a little

```{code-cell} python
display(swp)
```

## The simplest plot

```{code-cell} python
swp = swp.wrl.georef.georeference()
swp = swp.set_coords("sweep_mode")
pm = swp.DBZH.wrl.vis.plot()
txt = plt.title("Simple PPI")
```
