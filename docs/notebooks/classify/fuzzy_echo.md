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

# Fuzzy echo classification from dual-pol moments

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wradlib
import xarray as xr
from IPython.display import display
from wradlib.util import get_wradlib_data_file

warnings.filterwarnings("ignore")
```

## Setting the file paths

```{code-cell} python
rhofile = get_wradlib_data_file("netcdf/TAG-20120801-140046-02-R.nc")
phifile = get_wradlib_data_file("netcdf/TAG-20120801-140046-02-P.nc")
reffile = get_wradlib_data_file("netcdf/TAG-20120801-140046-02-Z.nc")
dopfile = get_wradlib_data_file("netcdf/TAG-20120801-140046-02-V.nc")
zdrfile = get_wradlib_data_file("netcdf/TAG-20120801-140046-02-D.nc")
mapfile = get_wradlib_data_file("hdf5/TAG_cmap_sweeps_0204050607.hdf5")
```

## Read the data

This reads teh moments and a precomputed static clutter map. The data is organized as a dictionary.

```{code-cell} python

dat = {}
dat["rho"], attrs_rho = wradlib.io.read_edge_netcdf(rhofile)
dat["phi"], attrs_phi = wradlib.io.read_edge_netcdf(phifile)
dat["ref"], attrs_ref = wradlib.io.read_edge_netcdf(reffile)
dat["dop"], attrs_dop = wradlib.io.read_edge_netcdf(dopfile)
dat["zdr"], attrs_zdr = wradlib.io.read_edge_netcdf(zdrfile)
dat["map"] = wradlib.io.from_hdf5(mapfile)[0][0]

dat = {k: (["azimuth", "range"], v) for k, v in dat.items()}
```

Create an xarray.Dataset holding the data.

```{code-cell} python
az, rng = dat["rho"][1].shape
swp = xr.Dataset(dat, coords={"azimuth": np.arange(az), "range": np.arange(rng)})
swp = swp.assign_coords(
    dict(
        longitude=7,
        latitude=53,
        altitude=0,
        elevation=1,
        sweep_mode="azimuth_surveillance",
    )
)
swp = swp.wrl.georef.georeference()
display(swp)
```

## Identify non-meteorological echoes

By defining weights for the used moments we can apply the final classifier. The algorithm returns the probability of meteorological echo.

See {cite}`Crisologo2014` and {cite}`Vulpiani2012` for details.

```{code-cell} python
moments = dict(rho="rho", phi="phi", dop="dop", zdr="zdr", map="map")
weights = {"zdr": 0.4, "rho": 0.4, "rho2": 0.4, "phi": 0.1, "dop": 0.1, "map": 0.5}
prob, nanmask = swp.wrl.classify.classify_echo_fuzzy(moments, weights=weights)
thresh = 0.5
cmap = xr.where(prob < thresh, True, False)
```

## Plot classification results

```{code-cell} python
fig = plt.figure(figsize=(12, 5))

# Horizontal reflectivity
ax = plt.subplot(121, aspect="equal")
pm = swp.ref.plot(x="x", y="y", ax=ax, cbar_kwargs=dict(label="dBZ"))
ax = wradlib.vis.plot_ppi_crosshair(site=(0, 0, 0), ranges=[80, 160, 240])
ax.set_xlim(-240, 240)
ax.set_ylim(-240, 240)
ax.set_xlabel("# bins from radar")
ax.set_ylabel("# bins from radar")
ax.set_title("Reflectivity")

# Echo classification
ax = plt.subplot(122, aspect="equal")
pm = cmap.where(~np.isnan(swp.ref)).plot(
    x="x",
    y="y",
    ax=ax,
    cmap="bwr",
    cbar_kwargs=dict(label="meterol. echo=0 - non-meteorol. echo=1"),
)
ax = wradlib.vis.plot_ppi_crosshair(site=(0, 0, 0), ranges=[80, 160, 240])
ax.set_xlim(-240, 240)
ax.set_ylim(-240, 240)
ax.set_xlabel("# bins from radar")
ax.set_ylabel("# bins from radar")
ax.set_title("Classification")
fig.tight_layout()
```
