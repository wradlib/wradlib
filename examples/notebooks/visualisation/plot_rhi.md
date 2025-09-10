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

# Quick-view a RHI sweep in polar or cartesian reference systems

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import wradlib as wrl
import wradlib_data
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## Read a RHI polar data set from University Bonn XBand radar

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("hdf5/2014-06-09--185000.rhi.mvol")
data1, metadata = wrl.io.read_gamic_hdf5(filename)
img = data1["SCAN0"]["ZH"]["data"]
# mask data array for better presentation
mask_ind = np.where(img <= np.nanmin(img))
img[mask_ind] = np.nan
img = np.ma.array(img, mask=np.isnan(img))

r = metadata["SCAN0"]["r"]
th = metadata["SCAN0"]["el"]
print(th.shape)
az = metadata["SCAN0"]["az"]
site = (
    metadata["VOL"]["Longitude"],
    metadata["VOL"]["Latitude"],
    metadata["VOL"]["Height"],
)
img = wrl.georef.create_xarray_dataarray(
    img, r=r, phi=az, theta=th, site=site, dim0="elevation", sweep_mode="rhi"
)
img
```

Inspect the data set a little

```{code-cell} python
print("Shape of polar array: %r\n" % (img.shape,))
print("Some meta data of the RHI file:")
print("\tdatetime: %r" % (metadata["SCAN0"]["Time"],))
```

## The simplest way to plot this dataset

```{code-cell} python
img = img.wrl.georef.georeference()
pm = img.wrl.vis.plot()
txt = plt.title("Simple RHI - Rays/Bins")
# plt.gca().set_xlim(0,100000)
# plt.gca().set_ylim(0,100000)
```

```{code-cell} python
pm = img.wrl.vis.plot()
plt.gca().set_ylim(0, 15000)
txt = plt.title("Simple RHI - Rays/Bins - with ylimits")
```

```{code-cell} python
pm = img.wrl.vis.plot(crs="cg")
plt.gca().set_title("Curvelineart Grid RHI", y=1.0, pad=20)
```

## More decorations and annotations

You can annotate these plots by using standard matplotlib methods.

```{code-cell} python
pm = img.wrl.vis.plot()
ax = plt.gca()
ylabel = ax.set_xlabel("Ground Range [m]")
ylabel = ax.set_ylabel("Height [m]")
title = ax.set_title("RHI manipulations/colorbar", y=1, pad=20)
# you can now also zoom - either programmatically or interactively
xlim = ax.set_xlim(25000, 40000)
ylim = ax.set_ylim(0, 15000)
# as the function returns the axes- and 'mappable'-objects colorbar needs, adding a colorbar is easy
cb = plt.colorbar(pm, ax=ax)
```
