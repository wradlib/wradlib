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

# Zonal Statistics - Quickstart

Zonal statistics can be used to compute e.g. the areal average precipitation over a catchment.

Here, we show a brief example using RADOLAN composite data from the German Weather Service (DWD).

```{code-cell} python
import wradlib as wrl
import wradlib_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
from osgeo import osr
```

```{code-cell} python
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
```

## Preparing the RADOLAN data

Preparing the radar composite data includes to
- read the data,
- georeference the data in native RADOLAN projection,
- reproject the data to UTM zone 32 projection.

```{code-cell} python
# Read and preprocess the RADOLAN data
fpath = "radolan/misc/raa01-sf_10000-1406100050-dwd---bin.gz"
f = wradlib_data.DATASETS.fetch(fpath)
ds = wrl.io.open_radolan_dataset(f)
```

```{code-cell} python
gridres = ds.x.diff("x")[0].values
gridres
```

```{code-cell} python
# This is the native RADOLAN projection
# (polar stereographic projection)
# create radolan projection osr object
if ds.attrs["formatversion"] >= 5:
    proj_stereo = wrl.georef.create_osr("dwd-radolan-wgs84")
else:
    proj_stereo = wrl.georef.create_osr("dwd-radolan-sphere")

# This is our target projection (UTM Zone 32)
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(32632)

# This is the source projection of the shape data
proj_gk2 = osr.SpatialReference()
proj_gk2.ImportFromEPSG(31466)
```

```{code-cell} python
# Get RADOLAN grid coordinates - center coordinates
x_rad, y_rad = np.meshgrid(ds.x, ds.y)
grid_xy_radolan = np.stack([x_rad, y_rad], axis=-1)
# Reproject the RADOLAN coordinates
xy = wrl.georef.reproject(grid_xy_radolan, src_crs=proj_stereo, trg_crs=proj_utm)
# assign as coordinates
ds = ds.assign_coords(
    {
        "xc": (
            ["y", "x"],
            xy[..., 0],
            dict(long_name="UTM Zone 32 Easting", units="m"),
        ),
        "yc": (
            ["y", "x"],
            xy[..., 1],
            dict(long_name="UTM Zone 32 Northing", units="m"),
        ),
    }
)
```

## Fix shapefile without projection information

As an example it is shown how to fix a shapefile with missing projection information.

```{code-cell} python
from osgeo import ogr, osr, gdal
import os

# Shape Source Projection
proj_gk2 = osr.SpatialReference()
proj_gk2.ImportFromEPSG(31466)

# This is our target projection (UTM Zone 32)
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(32632)

flist = ["shapefiles/agger/agger_merge.shx", "shapefiles/agger/agger_merge.dbf"]
[wradlib_data.DATASETS.fetch(f) for f in flist]
shpfile = wradlib_data.DATASETS.fetch("shapefiles/agger/agger_merge.shp")
dst_shpfile = shpfile[:-4] + "_gk2.shp"


def transform_shapefile(src, dst, trg_crs, dst_driver="ESRI Shapefile", src_crs=None):
    # remove destination file, if exists
    driver = ogr.GetDriverByName(dst_driver)
    if os.path.exists(dst_shpfile):
        driver.DeleteDataSource(dst_shpfile)

    # create the output layer
    dst_ds = driver.CreateDataSource(dst)
    dst_lyr = dst_ds.CreateLayer("", trg_crs, geom_type=ogr.wkbPolygon)

    # get the input layer
    src_ds = gdal.OpenEx(src)
    src_lyr = src_ds.GetLayer()

    # transform - reproject
    wrl.georef.ogr_reproject_layer(src_lyr, dst_lyr, trg_crs, src_crs=src_crs)

    # unlock files
    dst_ds = None
    src_ds = None


transform_shapefile(shpfile, dst_shpfile, proj_gk2, src_crs=proj_gk2)
```

## Import catchment boundaries from ESRI shapefile

### Create trg VectorSource

This shows how to load data in a specific projection and project it on the fly to another projection. Here gk2 -> utm32.

```{code-cell} python
shpfile = wradlib_data.DATASETS.fetch("shapefiles/agger/agger_merge_gk2.shp")
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg")
print(f"Found {len(trg)} sub-catchments in shapefile.")
```

```{code-cell} python
# check projection
print(trg.crs)
```

## Clip subgrid from RADOLAN grid

This is just to speed up the computation (so we don't have to deal with the full grid).

```{code-cell} python
bbox = trg.extent
buffer = 5000.0
bbox = dict(
    left=bbox[0] - buffer,
    right=bbox[1] + buffer,
    bottom=bbox[2] - buffer,
    top=bbox[3] + buffer,
)
print(bbox)
```

```{code-cell} python
ds
```

```{code-cell} python
(
    ((ds.yc > bbox["bottom"]) & (ds.yc < bbox["top"]))
    | ((ds.xc > bbox["left"]) & (ds.xc < bbox["right"]))
).plot()
```

```{code-cell} python
ds_clip = ds.where(
    (
        ((ds.yc > bbox["bottom"]) & (ds.yc < bbox["top"]))
        & ((ds.xc > bbox["left"]) & (ds.xc < bbox["right"]))
    ),
    drop=True,
)
display(ds_clip)
```

```{code-cell} python
ds_clip.SF.plot()
```

## Compute the average precipitation for each catchment

To compute the zonal average, we have to understand the the grid cells as *polygons* defined by a set of *vertices*.

```{code-cell} python
# Create vertices for each grid cell
# (MUST BE DONE IN NATIVE RADOLAN COORDINATES)
grid_x, grid_y = np.meshgrid(ds_clip.x, ds_clip.y)
grdverts = wrl.zonalstats.grid_centers_to_vertices(grid_x, grid_y, gridres, gridres)
```

### Create src VectorSource

This shows how to load data via numpy arrays in a given source projection and project it on the fly to a needed target projection

```{code-cell} python
src = wrl.io.VectorSource(grdverts, trg_crs=proj_utm, name="src", src_crs=proj_stereo)
```

Now we create the `ZonalDataPoly` class instance providing `src`  and `trg` VectorSources. Based on the overlap of these polygons with the catchment area, we can then compute a *weighted average*.

```{code-cell} python
# This object collects our source and target data
#   and computes the intersections
zd = wrl.zonalstats.ZonalDataPoly(src, trg=trg, crs=proj_utm)
# zd = wrl.zonalstats.ZonalDataPoly(grdverts, shpfile, srs=proj_utm)

# This object can actually compute the statistics
obj = wrl.zonalstats.ZonalStatsPoly(zd)

# We just call this object with any set of radar data
avg = obj.mean(ds_clip.SF.values.ravel())
```

## Plot results in map using matplotlib

We now plot the data using matplotlib accessors of `geopandas` (vector) and `xarray` (raster).

```{code-cell} python
from matplotlib.colors import from_levels_and_colors

# Create discrete colormap
levels = np.arange(0, 30, 2.5)
colors = plt.cm.inferno(np.linspace(0, 1, len(levels)))
mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

fig = plt.figure(figsize=(10, 10))

# Average rainfall sum
ax = fig.add_subplot(121, aspect="equal")
obj.zdata.trg.geo.plot(
    column="mean",
    ax=ax,
    cmap=mycmap,
    norm=mynorm,
    edgecolor="white",
    lw=0.5,
    legend=True,
    legend_kwds=dict(orientation="horizontal", pad=0.05),
)
plt.xlabel("UTM Zone 32 Easting (m)")
plt.ylabel("UTM Zone 32 Northing (m)")
plt.title("Catchment areal average")
bbox = obj.zdata.trg.extent
plt.xlim(bbox[0] - 5000, bbox[1] + 5000)
plt.ylim(bbox[2] - 5000, bbox[3] + 5000)
plt.grid()

# Original radar data
ax = fig.add_subplot(122, aspect="equal")
pm = ds_clip.SF.plot(
    x="xc",
    y="yc",
    cmap=mycmap,
    norm=mynorm,
    ax=ax,
    cbar_kwargs=dict(orientation="horizontal", pad=0.05),
)
obj.zdata.trg.geo.plot(ax=ax, facecolor="None", edgecolor="white")
plt.title("RADOLAN rain depth")
plt.grid(color="white")
plt.tight_layout()
```

## Plot results in interactive map using geopandas

Interactive mapmaking is easy using `geopandas`:

```{code-cell} python
fmap = obj.zdata.trg.geo.explore(column="mean")
fmap
```

## Save time by reading the weights from a file

The computational expensive part is the computation of intersections and weights. You only need to do it *once*.

You can dump the results on disk and read them from there when required. Let's do a little benchmark:

```{code-cell} python
import datetime as dt

# dump to file
zd.dump_vector("test_zonal_poly_cart")

t1 = dt.datetime.now()
# Create instance of type ZonalStatsPoly from zonal data file
obj = wrl.zonalstats.ZonalStatsPoly("test_zonal_poly_cart")
t2 = dt.datetime.now()

# Create instance of type ZonalStatsPoly from scratch
src = wrl.io.VectorSource(grdverts, trg_crs=proj_utm, name="src", src_crs=proj_stereo)
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg")
zd = wrl.zonalstats.ZonalDataPoly(src, trg=trg)
obj = wrl.zonalstats.ZonalStatsPoly(zd)
t3 = dt.datetime.now()

# Calling the object
avg = obj.mean(ds_clip.SF.values.ravel())
t4 = dt.datetime.now()

print("\nCreate object from file: %f seconds" % (t2 - t1).total_seconds())
print("Create object from sratch: %f seconds" % (t3 - t2).total_seconds())
print("Compute stats: %f seconds" % (t4 - t3).total_seconds())
```

### Calculate Variance and Plot Result

```{code-cell} python
var = obj.var(ds_clip.SF.values.ravel())
```

```{code-cell} python
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
obj.zdata.trg.geo.plot(
    column="var",
    ax=ax,
    cmap=mycmap,
    norm=mynorm,
    edgecolor="white",
    lw=0.5,
    legend=True,
    legend_kwds=dict(orientation="horizontal", pad=0.1),
)
plt.xlabel("UTM Zone 32 Easting (m)")
plt.ylabel("UTM Zone 32 Northing (m)")
plt.title("Catchment areal variance")
bbox = obj.zdata.trg.extent
plt.xlim(bbox[0] - 5000, bbox[1] + 5000)
plt.ylim(bbox[2] - 5000, bbox[3] + 5000)
plt.grid()
```
