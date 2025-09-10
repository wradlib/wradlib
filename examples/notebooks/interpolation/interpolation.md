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

# How to use wradlib's ipol module for interpolation tasks?

```{code-cell} python
import wradlib as wrl
import wradlib_data
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

## 1-dimensional example

Includes Nearest Neighbours, Inverse Distance Weighting, and Ordinary Kriging.

```{code-cell} python
# Synthetic observations
xsrc = np.arange(10)[:, None]
vals = np.sin(xsrc).ravel()

# Define target coordinates
xtrg = np.linspace(0, 20, 100)[:, None]

# Set up interpolation objects
#   IDW
idw = wrl.ipol.Idw(xsrc, xtrg)
#   Nearest Neighbours
nn = wrl.ipol.Nearest(xsrc, xtrg)
#   Linear
ok = wrl.ipol.OrdinaryKriging(xsrc, xtrg)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(xsrc.ravel(), vals, "bo", label="Observation")
plt.plot(xtrg.ravel(), idw(vals), "r-", label="IDW interpolation")
plt.plot(xtrg.ravel(), nn(vals), "k-", label="Nearest Neighbour interpolation")
plt.plot(xtrg.ravel(), ok(vals), "g-", label="Ordinary Kriging")
plt.xlabel("Distance", fontsize="large")
plt.ylabel("Value", fontsize="large")
plt.legend(loc="lower right")
```

## 2-dimensional example

Includes Nearest Neighbours, Inverse Distance Weighting, Linear Interpolation, and Ordinary Kriging.

```{code-cell} python
# Synthetic observations and source coordinates
src = np.vstack((np.array([4, 7, 3, 15]), np.array([8, 18, 17, 3]))).transpose()
np.random.seed(1319622840)
vals = np.random.uniform(size=len(src))

# Target coordinates
xtrg = np.linspace(0, 20, 40)
ytrg = np.linspace(0, 20, 40)
trg = np.meshgrid(xtrg, ytrg)
trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T

# Interpolation objects
idw = wrl.ipol.Idw(src, trg)
nn = wrl.ipol.Nearest(src, trg)
linear = wrl.ipol.Linear(src, trg)
ok = wrl.ipol.OrdinaryKriging(src, trg)


# Subplot layout
def gridplot(interpolated, title=""):
    pm = ax.pcolormesh(xtrg, ytrg, interpolated.reshape((len(xtrg), len(ytrg))))
    plt.axis("tight")
    ax.scatter(src[:, 0], src[:, 1], facecolor="None", s=50, marker="s")
    plt.title(title)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")


# Plot results
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(221, aspect="equal")
gridplot(idw(vals), "IDW")
ax = fig.add_subplot(222, aspect="equal")
gridplot(nn(vals), "Nearest Neighbours")
ax = fig.add_subplot(223, aspect="equal")
gridplot(np.ma.masked_invalid(linear(vals)), "Linear interpolation")
ax = fig.add_subplot(224, aspect="equal")
gridplot(ok(vals), "Ordinary Kriging")
plt.tight_layout()
```

## Using the convenience function ipol.interpolation in order to deal with missing values

**(1)** Exemplified for one dimension in space and two dimensions of the source value array (could e.g. be two time steps).

```{code-cell} python
# Synthetic observations (e.g. two time steps)
src = np.arange(10)[:, None]
vals = np.hstack((1.0 + np.sin(src), 5.0 + 2.0 * np.sin(src)))
# Target coordinates
trg = np.linspace(0, 20, 100)[:, None]
# Here we introduce missing values in the second dimension of the source value array
vals[3:5, 1] = np.nan
# interpolation using the convenience function "interpolate"
idw_result = wrl.ipol.interpolate(src, trg, vals, wrl.ipol.Idw, nnearest=4)
nn_result = wrl.ipol.interpolate(src, trg, vals, wrl.ipol.Nearest)
# Plot results
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
pl1 = ax.plot(trg, idw_result, "b-", label="IDW")
pl2 = ax.plot(trg, nn_result, "k-", label="Nearest Neighbour")
pl3 = ax.plot(src, vals, "ro", label="Observations")
```

**(2)** Exemplified for two dimensions in space and two dimensions of the source value array (e.g. time steps), containing also NaN values (here we only use IDW interpolation)

```{code-cell} python
# Just a helper function for repeated subplots
def plotall(ax, trgx, trgy, src, interp, pts, title, vmin, vmax):
    ix = np.where(np.isfinite(pts))
    ax.pcolormesh(
        trgx, trgy, interp.reshape((len(trgx), len(trgy))), vmin=vmin, vmax=vmax
    )
    ax.scatter(
        src[ix, 0].ravel(),
        src[ix, 1].ravel(),
        c=pts.ravel()[ix],
        s=20,
        marker="s",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    plt.axis("tight")
```

```{code-cell} python
# Synthetic observations
src = np.vstack((np.array([4, 7, 3, 15]), np.array([8, 18, 17, 3]))).T
np.random.seed(1319622840 + 1)
vals = np.round(np.random.uniform(size=(len(src), 2)), 1)

# Target coordinates
trgx = np.linspace(0, 20, 100)
trgy = np.linspace(0, 20, 100)
trg = np.meshgrid(trgx, trgy)
trg = np.vstack((trg[0].ravel(), trg[1].ravel())).transpose()

result = wrl.ipol.interpolate(src, trg, vals, wrl.ipol.Idw, nnearest=4)

# Now introduce NaNs in the observations
vals_with_nan = vals.copy()
vals_with_nan[1, 0] = np.nan
vals_with_nan[1:3, 1] = np.nan
result_with_nan = wrl.ipol.interpolate(
    src, trg, vals_with_nan, wrl.ipol.Idw, nnearest=4
)
vmin = np.concatenate((vals.ravel(), result.ravel())).min()
vmax = np.concatenate((vals.ravel(), result.ravel())).max()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(221)
plotall(ax, trgx, trgy, src, result[:, 0], vals[:, 0], "1st dim: no NaNs", vmin, vmax)
ax = fig.add_subplot(222)
plotall(ax, trgx, trgy, src, result[:, 1], vals[:, 1], "2nd dim: no NaNs", vmin, vmax)
ax = fig.add_subplot(223)
plotall(
    ax,
    trgx,
    trgy,
    src,
    result_with_nan[:, 0],
    vals_with_nan[:, 0],
    "1st dim: one NaN",
    vmin,
    vmax,
)
ax = fig.add_subplot(224)
plotall(
    ax,
    trgx,
    trgy,
    src,
    result_with_nan[:, 1],
    vals_with_nan[:, 1],
    "2nd dim: two NaN",
    vmin,
    vmax,
)
plt.tight_layout()
```

## How to use interpolation for gridding data in polar coordinates?


Read polar coordinates and corresponding rainfall intensity from file

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("misc/bin_coords_tur.gz")
src = np.loadtxt(filename)

filename = wradlib_data.DATASETS.fetch("misc/polar_R_tur.gz")
vals = np.loadtxt(filename)
```

```{code-cell} python
src.shape
```

Define target grid coordinates

```{code-cell} python
xtrg = np.linspace(src[:, 0].min(), src[:, 0].max(), 200)
ytrg = np.linspace(src[:, 1].min(), src[:, 1].max(), 200)
trg = np.meshgrid(xtrg, ytrg)
trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T
```

Linear Interpolation

```{code-cell} python
ip_lin = wrl.ipol.Linear(src, trg)
result_lin = ip_lin(vals.ravel(), fill_value=np.nan)
```

IDW interpolation

```{code-cell} python
ip_near = wrl.ipol.Nearest(src, trg)
maxdist = trg[1, 0] - trg[0, 0]
result_near = ip_near(vals.ravel(), maxdist=maxdist)
```

Plot results

```{code-cell} python
fig = plt.figure(figsize=(15, 6))
fig.subplots_adjust(wspace=0.4)
ax = fig.add_subplot(131, aspect="equal")
vals = wrl.georef.create_xarray_dataarray(vals).wrl.georef.georeference()
vals.wrl.vis.plot(ax=ax)
ax = fig.add_subplot(132, aspect="equal")
plt.pcolormesh(xtrg, ytrg, result_lin.reshape((len(xtrg), len(ytrg))))
ax = fig.add_subplot(133, aspect="equal")
plt.pcolormesh(xtrg, ytrg, result_near.reshape((len(xtrg), len(ytrg))))
```
