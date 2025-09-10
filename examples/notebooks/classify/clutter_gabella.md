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

# Clutter detection using the Gabella approach

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

## Read the data

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("misc/polar_dBZ_fbg.gz")
data = np.loadtxt(filename)
data = wrl.georef.create_xarray_dataarray(data, rf=0.001).wrl.georef.georeference()
data
```

## Apply filter

```{code-cell} python
clmap = data.wrl.classify.filter_gabella(
    wsize=5, thrsnorain=0.0, tr1=6.0, n_p=8, tr2=1.3
)
clmap
```

## Plot results

```{code-cell} python
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
pm = data.wrl.vis.plot(ax=ax1)
ax1.set_title("Reflectivity")
ax2 = fig.add_subplot(122)
pm = clmap.wrl.vis.plot(ax=ax2)
ax2.set_title("Cluttermap")
```
