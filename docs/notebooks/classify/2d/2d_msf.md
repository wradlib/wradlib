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

```{code-cell} python3
import datetime as dt
import urllib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
from IPython.display import display

warnings.filterwarnings("ignore")
```

# 2D Membershipfunctions

## Precipitation Types

```{code-cell} python3
pr_types = wrl.classify.pr_types
for k, v in pr_types.items():
    print(str(k) + " - ".join(v))
```

## Load 2D Membership Functions

```{code-cell} python3
filename = wradlib_data.DATASETS.fetch("misc/msf_xband_v1.nc")
msf = xr.open_dataset(filename)
display(msf)
```

## Plot 2D Membership Functions

```{code-cell} python3
minmax = [(-10, 100), (-1, 6), (0.0, 1.0), (-5, 35), (-65, 45)]

for i, pr in enumerate(pr_types.values()):
    if pr[0] == "NP":
        continue
    fig = plt.figure(figsize=(10, 8))
    t = fig.suptitle(" - ".join(pr))
    t.set_y(1.02)
    hmc = msf.sel(hmc=pr[0])
    for k, p in enumerate(hmc.data_vars.values()):
        p = p.where(p != 0)
        ax = fig.add_subplot(3, 2, k + 1)
        p.sel(trapezoid=0).plot(x="idp", c="k", lw=1.0, ax=ax)
        p.sel(trapezoid=1).plot(x="idp", c="k", lw=2.0, ax=ax)
        p.sel(trapezoid=2).plot(x="idp", c="k", lw=2.0, ax=ax)
        p.sel(trapezoid=3).plot(x="idp", c="k", lw=1.0, ax=ax)
        ax.set_xlim((hmc.idp.min(), hmc.idp.max()))
        ax.margins(x=0.05, y=0.05)
        t = ax.set_title(f"{p.long_name}")
        ax.set_ylim(minmax[k])
    fig.tight_layout()
plt.show()
```
