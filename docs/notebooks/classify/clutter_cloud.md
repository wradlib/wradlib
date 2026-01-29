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

# Clutter detection - Satellite Cloud Images

In this notebook we show howto identify clutter based on collocated cloudtype.

```{code-cell} python3
import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
from IPython.display import display
from osgeo import osr
```

## Read Satellite Cloud Image

The cloud data from MSG3 satellite has some issue with the georeferencing. We need to apply a little fix, correct for that. The fix just adds the proper projection and coordinate objects to the dataset.

```{code-cell} ipython3
import warnings
import pyproj
from pyproj import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

filename = wradlib_data.DATASETS.fetch("hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5")

def fix_georef(ds):
    attrs = ds.attrs
    gt = attrs['GEOTRANSFORM_GDAL_TABLE'].split(',')

    transform = Affine(
        float(gt[1]), float(gt[2]), float(attrs["XGEO_UP_LEFT"]),
        float(gt[4]), float(gt[5]), float(attrs["YGEO_UP_LEFT"])
    )

    crs = pyproj.CRS.from_proj4(attrs["PROJECTION"])
    ds = ds.rio.write_crs(crs, inplace=False)
    ds = ds.rio.write_transform(transform, inplace=False)

    ny, nx = ds.sizes[ds.rio.y_dim], ds.sizes[ds.rio.x_dim]

    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e

    ds = ds.drop_vars([ds.rio.x_dim, ds.rio.y_dim], errors="ignore")
    ds = ds.assign_coords(
        x=(ds.rio.x_dim, xs),
        y=(ds.rio.y_dim, ys),
    )

    return ds

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("once", NotGeoreferencedWarning)
    sat = xr.open_dataset(filename, engine="rasterio", variable="CT")

    for _w in w:
        if isinstance(_w.message, NotGeoreferencedWarning):
            sat = sat.pipe(fix_georef)
            break

display(sat)
```

## Read radar data

We read radar data from Belgium into a DataTree.

```{code-cell} python3
# read the radar volume scan
filename = wradlib_data.DATASETS.fetch("hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf")
pvol = xd.io.open_odim_datatree(filename)
display(pvol)
```

## Georeference

Here we add two different georeferenced coordinates, ``xp``, ``yp``, ``zp`` for the radar azimuthal equidistant projection, and  ``x``, ``y``, ``z`` for the satellite data projection.

```{code-cell} python3
pvol1 = pvol.match("sweep*")
vol = []
for sweep in pvol1.values():
    swp = sweep.to_dataset().pipe(wrl.georef.georeference)
    swp = swp.rename(x="xp", y="yp", z="zp")
    swp = swp.pipe(wrl.georef.georeference, crs=sat.spatial_ref)
    vol.append(swp)
vol = xr.concat(vol, dim="tilt")
vol = vol.assign_coords(sweep_mode=vol.sweep_mode)
display(vol)
```

## Construct collocated satellite data

Here we interpolate the satellite data into the radar grid (nearest neighbour) and assign it to the volume. See {meth}`wradlib.ipol.IpolMethods.interpolate` and {doc}`../interpolation/cartesian_to_polar`.

```{code-cell} python3
ct = sat.isel(band=0).CT.wrl.ipol.interpolate(vol, method="map_coordinates_nearest")
vol = vol.assign(CT=ct)
display(vol)
```

```{code-cell} python3
print("Top-left CT center (x/y):", sat.x[0].values, sat.y[0].values)
print("Top-left radar (x/y):", vol.isel(tilt=0).x[0,0].values, vol.isel(tilt=0).y[0,0].values)
```

## Estimate localisation errors

```{code-cell} python3
timelag = 9 * 60
wind = 10
error = np.absolute(timelag) * wind
```

## Identify clutter based on collocated cloudtype

Then, we call {meth}`wradlib.classify.ClassifyMethods.filter_cloudtype` to derive the clutter map. The it is assiged to the volume.

```{code-cell} python3
clutter = vol.DBZH.wrl.classify.filter_cloudtype(vol.CT, smoothing=error)
vol = vol.assign(CMAP=clutter)
display(vol)
```

## Plot the results

```{code-cell} python3
fig = plt.figure(figsize=(16, 8))

tilt = 0

ax = fig.add_subplot(131)
pm = vol.DBZH[tilt].wrl.vis.plot(x="xp", y="yp", ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Radar reflectivity")

ax = fig.add_subplot(132)
pm = vol.CT[tilt].wrl.vis.plot(x="xp", y="yp", ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Satellite cloud classification")

ax = fig.add_subplot(133)
pm = vol.CMAP[tilt].wrl.vis.plot(x="xp", y="yp", ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Detected clutter")

fig.tight_layout()
```
