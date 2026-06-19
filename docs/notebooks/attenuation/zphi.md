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

# Specific Attenuation via ZPHI method

Weather radar reflectivity measurements are affected by propagation effects, most importantly attenuation due to hydrometeors along the radar beam. In X-band and C-band radar applications, this can lead to significant underestimation of reflectivity at long range and in heavy precipitation.

The ZPHI method provides a physically constrained approach to estimate specific attenuation from the joint evolution of reflectivity ({math}`Z_H`) and differential phase shift ({math}`\Phi_{DP}`). It exploits the fact that {math}`\Phi_{DP}` is a path-integrated quantity that is not directly affected by attenuation and can therefore be used as a robust constraint.

In this notebook, we demonstrate a complete ZPHI processing chain including:

- estimation of total differential phase shift ({math}`\Delta \Phi_{DP}^{tot}`)
- reconstruction of {math}`\Phi_{DP}^{cal}` from {math}`K_{DP}` (when needed)
- retrieval of specific attenuation ({math}`A_H` / {math}`A_V`)

The workflow follows established formulations from {cite}`Testud2000`, {cite}`Ryzhkov2014`, and {cite}`Diederich2015`.

```{code-cell} ipython3
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

## Read BoXPol data

```{code-cell} ipython3
fname = wradlib_data.DATASETS.fetch("hdf5/2014-08-10--182000.ppi.mvol")
with xr.open_dataset(fname, engine="gamic", group="sweep_0") as swp:
    swp = swp.set_coords("sweep_mode").wrl.georef.georeference()
    mask = swp.RHOHV > 0.9
    phidp = swp.PHIDP.where(mask)
    dbz = swp.DBZH.where(mask)
```

## Overview Plot

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
dbz.wrl.vis.plot(vmin=0, vmax=60, ax=ax1)
phidp.wrl.vis.plot(vmin=-100, vmax=20, ax=ax2)
plt.title("Raw Reflectivity (dBZ)")
```

## Calculate specific attenuation AH

```{code-cell} ipython3
alpha = 0.28
ah = wrl.atten.specific_attenuation_zphi(phidp, dbz, alpha=alpha, b=0.78, rng=2000.)
```

## Plot specific attenuation AH

```{code-cell} ipython3
ah.wrl.vis.plot(vmax=2)
```

## Derive {math}`K_{DP}`

```{code-cell} ipython3
kdp_ah = ah.fillna(0) / alpha
kdp_ah.attrs = swp.KDP.attrs
```

```{code-cell} ipython3
kdp_ah.wrl.vis.plot(vmax=5)
```

## Recalculate {math}`\Phi_{DP}^{cal}`

```{code-cell} ipython3
phical = kdp_ah.wrl.dp.phidp_from_kdp()
phical = phical.rename("PHIDP_AH")
phical.attrs = swp.PHIDP.attrs
display(phical)
```

## Plot {math}`\Phi_{DP}^{cal}`

```{code-cell} ipython3
phical.wrl.vis.plot()
```
