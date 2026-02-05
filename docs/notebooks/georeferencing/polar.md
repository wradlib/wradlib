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

# Spherical/Polar Data

```{code-cell} python3
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pyproj
import wradlib as wrl
import xradar as xd

warnings.filterwarnings("ignore")
```

## Artificial Dataset

```{code-cell} python3
swp = (
    xd.model.create_sweep_dataset(rng=150)
    .swap_dims(time="azimuth")
)
swp = swp.assign_coords(sweep_mode="azimuthal_surveillance")
display(swp)
```

## Spherical to XYZ

With {func}`wradlib.georef.GeorefMethods.spherical_to_xyz` we can transform spherical coordinates (r, phi, theta) to cartesian coordinates (x, y, z) centered at radar site (aeqd).
It takes the shortening of the great circle distance with increasing elevation angle as well as the resulting increase in height into account.

```{code-cell} python3
xyz, aeqd = swp.wrl.georef.spherical_to_xyz()
display(xyz)
```

With a little xarray magic we can introduce our calculated x,y,z as coordinates.

```{code-cell} python3
xyz_coords = xyz.assign_coords(xyz=["x", "y", "z"]).to_dataset("xyz").set_coords(["x", "y", "z"])
display(xyz_coords)
```

```{code-cell} python3
xyz_coords.z.plot(x="x", y="y")
```

## Spherical to Proj

With {func}`wradlib.georef.GeorefMethods.spherical_to_proj` we can transform spherical coordinates (r, phi, theta) to projected coordinates (x, y, z) centered at radar site in a given projection.
It takes the shortening of the great circle distance with increasing elevation angle as well as the resulting increase in height into account.

```{code-cell} python3
crs = pyproj.CRS.from_epsg(2056)
xyz_proj = swp.wrl.georef.spherical_to_proj(crs=crs)
display(xyz_proj)
```

```{code-cell} python3
xyz_proj_coords = xyz_proj.assign_coords(xyz=["x", "y", "z"]).to_dataset("xyz").set_coords(["x", "y", "z"])
display(xyz_proj_coords)
```

```{code-cell} python3
xyz_proj_coords.z.plot(x="x", y="y")
```

## Spherical to Polyvert

With {func}`wradlib.georef.GeorefMethods.spherical_to_polyvert` we can generate 3D polygon vertices directly from spherical coordinates (r, phi, theta). It generates the polygon vertices by simply connecting the corners of the radar bins.

```{code-cell} python3
poly, aeqd = swp.wrl.georef.spherical_to_polyvert()
display(poly)
```

Let's cut a bit...

```{code-cell} python3
aspect = swp.wrl.georef.georeference().wrl.util.aspect()
poly_crop = poly.where(((poly[..., 0] > -2e3) & (poly[..., 0] < 3e3)) & ((poly[..., 1] > -2e3) & (poly[..., 1] < 3e3)), drop=True)
display(poly_crop)
```

... and plot the remaining polygons.

```{code-cell} python3
fig = plt.figure(figsize=(8, 8))
site = (poly.longitude.values, poly.latitude.values)
ax = fig.add_subplot(111, aspect=aspect)
polycoll = mpl.collections.PolyCollection(
    poly_crop.isel(xy=slice(0, 2)), closed=True, facecolors="None", linewidth=0.1
)
ax.add_collection(polycoll, autolim=True)
ax.set_title("Polygons")
ax.autoscale_view()
```

## Spherical to Centroids

With {func}`wradlib.georef.GeorefMethods.spherical_to_centroids` we can generate 3-D centroids directly from spherical coordinates (r, phi, theta).

```{code-cell} python3
cent, aeqd = swp.wrl.georef.spherical_to_centroids()
cent = cent.assign_coords(xyz=["x", "y", "z"]).to_dataset("xyz").set_coords(["x", "y", "z"])
display(cent)
```

```{code-cell} python3
aspect = swp.wrl.georef.georeference().wrl.util.aspect()
cent_crop = cent.where(((cent.x > -2e3) & (cent.x < 3e3)) & ((cent.y > -2e3) & (cent.y < 3e3)), drop=True)
display(cent_crop)
```

```{code-cell} python3
fig = plt.figure(figsize=(8, 6))
site = (cent.longitude.values, cent.latitude.values)
ax = fig.add_subplot(111, aspect=aspect)
cent_crop.plot.scatter(x="x", y="y", marker=".", hue="z")
ax.set_title("Centroids")
fig.tight_layout()
```

## Georeference

This function adds georeference data (x,y,z) directly to a given {class}`xarray:xarray.DataArray`/{class}`xarray:xarray.Dataset`.

```{code-cell} python3
swp_aeqd = swp.wrl.georef.georeference()
display(swp_aeqd)
```

```{code-cell} python3
crs = pyproj.CRS.from_epsg(2056)
swp_ch = swp.wrl.georef.georeference(crs=crs)
display(swp_ch)
```

```{code-cell} python3
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121)
swp_aeqd.z.plot(x="x", y="y", ax=ax1)
ax1.set_title(swp_aeqd.xradar.get_crs().name)
ax2 = fig.add_subplot(122)
swp_ch.z.plot(x="x", y="y", ax=ax2)
ax2.set_title(swp_ch.xradar.get_crs().name)
fig.tight_layout()
```
