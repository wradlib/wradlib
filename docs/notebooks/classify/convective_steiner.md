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

# Convective/Stratiform Classification (Steiner)

This notebook shows the convective/stratiform classification of a single radar sweep according to {cite}`Steiner1995`. Echoes are flagged as convective if they either exceed an absolute intensity threshold or are sufficiently peaked above their local background. Detected convective cores are expanded by a background-dependent radius.

```{code-cell} python3
import warnings

import matplotlib.pyplot as plt
import xarray as xr
import wradlib as wrl
import wradlib_data

warnings.filterwarnings("ignore")
```

## Get radar data

```{code-cell} python3
filename = wradlib_data.DATASETS.fetch("hdf5/bewid.pvol.hdf")
swp = (
    xr.open_dataset(filename, engine="odim", group="sweep_0")
    .wrl.georef.georeference()
    .set_coords("sweep_mode")
)
display(swp)
```

## Classify

The accessor {meth}`wradlib.classify.ClassifyMethods.convective_steiner` returns a boolean mask which is `True` for convective echo. The range gate size is taken from the sweep's `range` coordinate, all other parameters are Steiner's defaults.

For plotting we convert the mask to `1` (convective) and `0` (stratiform), masking gates without data.

```{code-cell} python3
conv = swp.DBZH.wrl.classify.convective_steiner()
cls = xr.where(conv, 1, 0).where(swp.DBZH > -32)  # -32 dBZ is ODIM nodata
display(cls)
```

## Plot Result

```{code-cell} python3
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
ax = axs.flat[0]
swp.DBZH.wrl.vis.plot(ax=ax, vmin=0, vmax=60, cmap="HomeyerRainbow")
ax.set_aspect("equal")
ax.set_title("Reflectivity")
ax = axs.flat[1]
cls.wrl.vis.plot(
    ax=ax,
    vmin=0,
    vmax=1,
    cmap="bwr",
    cbar_kwargs=dict(label="0 = stratiform, 1 = convective"),
)
ax.set_aspect("equal")
ax.set_title("Steiner classification")
fig.tight_layout()
```
