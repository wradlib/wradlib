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

# Example for georeferencing a radar dataset

```{code-cell} python
import wradlib as wrl
import xarray as xr
import xradar as xd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

**1st step:** Compute centroid coordinates and vertices of all radar bins in WGS84 (longitude and latitude).

```{code-cell} python
swp = (
    xd.model.create_sweep_dataset(rng=1000)
    .swap_dims(time="azimuth")
    .isel(range=slice(0, 100))
)
swp = swp.assign_coords(sweep_mode="azimuthal_surveillance")
swp = swp.wrl.georef.georeference()
swp
```

We can now generate the polygon vertices of the radar bins - with **each vertex in lon/lat coordinates**.

```{code-cell} python
proj_wgs84 = wrl.georef.epsg_to_osr(4326)
polygons = swp.wrl.georef.spherical_to_polyvert(crs=proj_wgs84, keep_attrs=True)
polygons
```

... or we can compute the corresponding centroids of all bins -  - with **each centroid in lon/lat coordinates**.

```{code-cell} python
centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_wgs84, keep_attrs=True)
centroids
```

In order to understand how vertices and centroids correspond, we can plot them together.

```{code-cell} python
fig = plt.figure(figsize=(16, 16))
site = (polygons.longitude.values, polygons.latitude.values)

aspect = (centroids[..., 0].max() - centroids[..., 0].min()) / (
    centroids[..., 1].max() - centroids[..., 1].min()
)
ax = fig.add_subplot(121, aspect=aspect)
polycoll = mpl.collections.PolyCollection(
    polygons.isel(xy=slice(0, 2)), closed=True, facecolors="None", linewidth=0.1
)
ax.add_collection(polycoll, autolim=True)
# ax.plot(centroids[..., 0], centroids[..., 1], 'r+')
plt.title("Zoom in\n(only possible for interactive plots).")
ax.add_patch(
    Rectangle(
        (site[0] + 0.25, site[1] + 0.25),
        0.2,
        0.2 / aspect,
        edgecolor="red",
        facecolor="None",
        zorder=3,
    )
)
plt.xlim(centroids[..., 0].min(), centroids[..., 0].max())
plt.ylim(centroids[..., 1].min(), centroids[..., 1].max())

ax = fig.add_subplot(122, aspect=aspect)
polycoll = mpl.collections.PolyCollection(
    polygons.isel(xy=slice(0, 2)), closed=True, facecolors="None"
)
ax.add_collection(polycoll, autolim=True)
ax.plot(centroids[..., 0], centroids[..., 1], "r+")
plt.title("Zoom into red box of left plot")
plt.xlim(site[0] + 0.25, site[0] + 0.25 + 0.2)
plt.ylim(site[1] + 0.25, site[1] + 0.25 + 0.2 / aspect)
```

**2nd step:** Reproject the centroid coordinates to Gauss-Krueger Zone 3 (i.e. EPSG-Code 31467).

```{code-cell} python
centroids_xyz = centroids.assign_coords(xyz=["x", "y", "z"]).to_dataset("xyz")
centroids_xyz
```

```{code-cell} python
proj_gk3 = wrl.georef.epsg_to_osr(31467)
centroids_xyz = centroids_xyz.wrl.georef.reproject(trg_crs=proj_gk3)
centroids_xyz
```
