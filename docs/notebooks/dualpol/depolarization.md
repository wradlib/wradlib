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

# Depolarization

This notebook show how to calculate depolarization ratio for weather radars which do not provide this moment. This has been extensively discussed in {cite}`Melnikov2013`, {cite}`Ryzhkov2017` and {cite}`Kilambi2018`.

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr

warnings.filterwarnings("ignore")
```

# Open radar sweep

```{code-cell} python
filename2 = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
swp = xr.open_dataset(filename2, engine="gamic", group="sweep_0")
display(swp)
```

# Preprocess polar data

We georeference the radar sweep coordinates for better visualization.

```{code-cell} python
swp = swp.wrl.georef.georeference()
swp = swp.set_coords("sweep_mode")
display(swp)
```

# Calculate Depolarization

Please see ({meth}`~wradlib.dp.DPMethods.depolarization`) for details.

```{code-cell} python
dpr =  swp.wrl.dp.depolarization(zdr="ZDR", rho="RHOHV")
plt.figure()
dpr.wrl.vis.plot()
plt.gca().set_title("Depolarization")
plt.tight_layout()
```
