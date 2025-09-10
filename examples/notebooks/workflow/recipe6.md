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

# Recipe 6: Zonal Statistics - Polar Grid


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
import xarray as xr
```

## Setup Examples

```{code-cell} python
def testplot(
    ds,
    obj,
    col="mean",
    levels=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100],
    title="",
):
    """Quick test plot layout for this example file"""
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

    radolevels = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    radocolors = plt.cm.viridis(np.linspace(0, 1, len(radolevels)))
    radocmap, radonorm = from_levels_and_colors(radolevels, radocolors, extend="max")

    fig = plt.figure(figsize=(10, 16))

    # Average rainfall sum
    ax = fig.add_subplot(211, aspect="equal")
    obj.zdata.trg.geo.plot(
        column=col,
        ax=ax,
        cmap=mycmap,
        norm=mynorm,
        edgecolor="white",
        lw=0.5,
        legend=True,
        legend_kwds=dict(shrink=0.5),
    )
    ax.autoscale()
    plt.xlabel("UTM Zone 32 Easting")
    plt.ylabel("UTM Zone 32 Northing")
    plt.title(title)
    plt.draw()

    # Original radar data
    ax1 = fig.add_subplot(212, aspect="equal")
    pm = ds.plot(
        x="xc",
        y="yc",
        cmap=radocmap,
        norm=radonorm,
        ax=ax1,
        cbar_kwargs=dict(shrink=0.5),
    )
    obj.zdata.trg.geo.plot(ax=ax1, facecolor="None", edgecolor="white")
    plt.xlabel("UTM Zone 32 Easting")
    plt.ylabel("UTM Zone 32 Northing")
    plt.title("Original radar rain sums")
    plt.draw()
    plt.tight_layout()
```

```{code-cell} python
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
import datetime as dt
from osgeo import osr
```

```{code-cell} python
# check for GEOS enabled GDAL
if not wrl.util.has_geos():
    print("NO GEOS support within GDAL, aborting...")
    exit(0)
```

```{code-cell} python
# create radolan projection osr object
proj_stereo = wrl.georef.create_osr("dwd-radolan")

# create UTM Zone 32 projection osr object
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(32632)

# Source projection of the shape data (in GK2)
proj_gk2 = osr.SpatialReference()
proj_gk2.ImportFromEPSG(31466)
```

```{code-cell} python
def create_center_coords(ds, crs=None):
    # create polar grid centroids in GK2
    center = wrl.georef.spherical_to_centroids(
        ds.range.values,
        ds.azimuth.values,
        0.5,
        (ds.longitude.values, ds.latitude.values, ds.altitude.values),
        crs=crs,
    )
    ds = ds.assign_coords(
        {
            "xc": (["azimuth", "range"], center[..., 0]),
            "yc": (["azimuth", "range"], center[..., 1]),
            "zc": (["azimuth", "range"], center[..., 2]),
        }
    )
    return ds
```

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("hdf5/rainsum_boxpol_20140609.h5")
ds = xr.open_dataset(filename)
ds = ds.rename_dims({"phony_dim_0": "azimuth", "phony_dim_1": "range"})
ds = ds.assign_coords(
    {
        "latitude": ds.data.Latitude,
        "longitude": ds.data.Longitude,
        "altitude": 99.5,
        "azimuth": ds.data.az,
        "range": ds.data.r,
        "sweep_mode": "azimuth_surveillance",
        "elevation": 0.5,
    }
)

ds = ds.wrl.georef.georeference(crs=proj_utm)
ds = ds.pipe(create_center_coords, crs=proj_utm)
display(ds)
```

```{code-cell} python
def write_prj(filename, proj):
    with open(filename, "w") as f:
        f.write(proj.ExportToWkt())
```

```{code-cell} python
flist = ["shapefiles/agger/agger_merge.shx", "shapefiles/agger/agger_merge.dbf"]
[wradlib_data.DATASETS.fetch(f) for f in flist]
# reshape
shpfile = wradlib_data.DATASETS.fetch("shapefiles/agger/agger_merge.shp")
write_prj(shpfile[:-3] + "prj", proj_gk2)
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg")

bbox = trg.extent

# create catchment bounding box
buffer = 5000.0
bbox = dict(
    left=bbox[0] - buffer,
    right=bbox[1] + buffer,
    bottom=bbox[2] - buffer,
    top=bbox[3] + buffer,
)
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
radar_utmc = np.dstack([ds_clip.xc, ds_clip.yc]).reshape(-1, 2)
radar_utmc.shape
```

## Zonal Stats Polar Grid - Points

```{code-cell} python
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg", src_crs=proj_gk2)
src = wrl.io.VectorSource(radar_utmc, trg_crs=proj_utm, name="src")
```

```{code-cell} python
###########################################################################
# Approach #1: Assign grid points to each polygon and compute the average.
#
# - Uses matplotlib.path.Path
# - Each point is weighted equally (assumption: polygon >> grid cell)
# - this is quick, but theoretically dirty
# - for polar grids a range-area dependency has to be taken into account
###########################################################################

t1 = dt.datetime.now()

# Create instance of type ZonalDataPoint from source grid and
# catchment array
zd = wrl.zonalstats.ZonalDataPoint(src, trg=trg, crs=proj_utm, buf=500.0)
# dump to file
zd.dump_vector("test_zonal_points")
# Create instance of type ZonalStatsPoint from zonal data object
obj1 = wrl.zonalstats.ZonalStatsPoint(zd)

isecs1 = obj1.zdata.isecs
t2 = dt.datetime.now()

t3 = dt.datetime.now()

# Create instance of type ZonalStatsPoint from zonal data file
obj1 = wrl.zonalstats.ZonalStatsPoint("test_zonal_points")
# Compute stats for target polygons
avg1 = obj1.mean(ds_clip.data.values.ravel())
var1 = obj1.var(ds_clip.data.values.ravel())

t4 = dt.datetime.now()

print("Approach #1 computation time:")
print("\tCreate object from scratch: %f seconds" % (t2 - t1).total_seconds())
print("\tCreate object from dumped file: %f seconds" % (t4 - t3).total_seconds())
print("\tCompute stats using object: %f seconds" % (t3 - t2).total_seconds())
```

```{code-cell} python
src1 = wrl.io.VectorSource(radar_utmc, trg_crs=proj_utm, name="src")
trg1 = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg", src_crs=proj_gk2)

# Just a test for plotting results with zero buffer
zd = wrl.zonalstats.ZonalDataPoint(src1, trg=trg1, buf=0)
# Create instance of type ZonalStatsPoint from zonal data object
obj2 = wrl.zonalstats.ZonalStatsPoint(zd)
obj2.zdata.trg.set_attribute("mean", avg1)
obj2.zdata.trg.set_attribute("var", var1)

isecs2 = obj2.zdata.isecs
```

```{code-cell} python
# Illustrate results for an example catchment i
i = 0  # try e.g. 5, 2
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, aspect="equal")

# Target polygon patches
trg_patch = obj2.zdata.trg.get_data_by_idx([i], mode="geo")
trg_patch.plot(ax=ax, facecolor="None", edgecolor="black", linewidth=2)
trg_patch = obj1.zdata.trg.get_data_by_idx([i], mode="geo")
trg_patch.plot(ax=ax, facecolor="None", edgecolor="grey", linewidth=2)

# pips
sources = obj1.zdata.src.geo
sources.plot(ax=ax, label="all points", c="grey", markersize=200)
isecs1 = obj2.zdata.dst.get_data_by_att(attr="trg_index", value=[i], mode="geo")
isecs1.plot(ax=ax, label="buffer=0 m", c="green", markersize=200)
isecs2 = obj1.zdata.dst.get_data_by_att(attr="trg_index", value=[i], mode="geo")
isecs2.plot(ax=ax, label="buffer=500 m", c="red", markersize=50)

cat = trg.get_data_by_idx([i])[0]
bbox = wrl.zonalstats.get_bbox(cat[..., 0], cat[..., 1])
plt.xlim(bbox["left"] - 2000, bbox["right"] + 2000)
plt.ylim(bbox["bottom"] - 2000, bbox["top"] + 2000)
plt.legend()
plt.title("Catchment #%d: Points considered for stats" % i)
```

```{code-cell} python
# Plot average rainfall and original data
testplot(
    ds_clip.data, obj2, col="mean", title="Catchment rainfall mean (ZonalStatsPoint)"
)
```

```{code-cell} python
testplot(
    ds_clip.data,
    obj2,
    col="var",
    levels=np.arange(0, 20, 1.0),
    title="Catchment rainfall variance (ZonalStatsPoint)",
)
```

## Zonal Stats Polar Grid - Polygons

```{code-cell} python
radar_utm = wrl.georef.spherical_to_polyvert(
    ds.range.values,
    ds.azimuth.values,
    0.5,
    (ds.longitude.values, ds.latitude.values, ds.altitude.values),
    crs=proj_utm,
)
radar_utm.shape = (360, 1000, 5, 3)
ds = ds.assign_coords(
    {
        "xp": (["azimuth", "range", "verts"], radar_utm[..., 0]),
        "yp": (["azimuth", "range", "verts"], radar_utm[..., 1]),
        "zp": (["azimuth", "range", "verts"], radar_utm[..., 2]),
    }
)
display(ds)
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg", src_crs=proj_gk2)
bbox = trg.extent

# create catchment bounding box
buffer = 5000.0
bbox = dict(
    left=bbox[0] - buffer,
    right=bbox[1] + buffer,
    bottom=bbox[2] - buffer,
    top=bbox[3] + buffer,
)
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
radar_utm = np.stack([ds_clip.xp, ds_clip.yp], axis=-1).reshape(-1, 5, 2)
print(radar_utm.shape)
src = wrl.io.VectorSource(radar_utm, trg_crs=proj_utm, name="src")
trg = wrl.io.VectorSource(shpfile, trg_crs=proj_utm, name="trg", src_crs=proj_gk2)
```

```{code-cell} python
###########################################################################
# Approach #2: Compute weighted mean based on fraction of source polygons
# in target polygons
#
# - This is more accurate (no assumptions), but probably slower...
###########################################################################

t1 = dt.datetime.now()

# Create instance of type ZonalDataPoly from source grid and
# catchment array
zd = wrl.zonalstats.ZonalDataPoly(src, trg=trg, crs=proj_utm)
# dump to file
zd.dump_vector("test_zonal_poly")
# Create instance of type ZonalStatsPoint from zonal data object
obj3 = wrl.zonalstats.ZonalStatsPoly(zd)

obj3.zdata.dump_vector("test_zonal_poly")
t2 = dt.datetime.now()


t3 = dt.datetime.now()

# Create instance of type ZonalStatsPoly from zonal data file
obj4 = wrl.zonalstats.ZonalStatsPoly("test_zonal_poly")

avg3 = obj4.mean(ds_clip.data.values.ravel())
var3 = obj4.var(ds_clip.data.values.ravel())


t4 = dt.datetime.now()

print("Approach #2 computation time:")
print("\tCreate object from scratch: %f seconds" % (t2 - t1).total_seconds())
print("\tCreate object from dumped file: %f seconds" % (t4 - t3).total_seconds())
print("\tCompute stats using object: %f seconds" % (t3 - t2).total_seconds())

obj4.zdata.trg.dump_raster(
    "test_zonal_hdr.nc", driver="netCDF", attr="mean", pixel_size=100.0
)

obj4.zdata.trg.dump_vector("test_zonal_shp")
obj4.zdata.trg.dump_vector("test_zonal_json.geojson", driver="GeoJSON")
```

```{code-cell} python
# Plot average rainfall and original data
testplot(
    ds_clip.data,
    obj4,
    col="mean",
    title="Catchment rainfall mean (PolarZonalStatsPoly)",
)
```

```{code-cell} python
testplot(
    ds_clip.data,
    obj4,
    col="var",
    levels=np.arange(0, 20, 1.0),
    title="Catchment rainfall variance (PolarZonalStatsPoly)",
)
```

```{code-cell} python
# Illustrate results for an example catchment i
i = 0  # try e.g. 5, 2
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, aspect="equal")

# Grid cell patches
src_index = obj3.zdata.get_source_index(i)
trg_patch = obj3.zdata.src.get_data_by_idx(src_index, mode="geo")
trg_patch.plot(ax=ax, facecolor="None", edgecolor="black")

# Target polygon patches
trg_patch = obj3.zdata.trg.get_data_by_idx([i], mode="geo")
trg_patch.plot(ax=ax, facecolor="None", edgecolor="red", linewidth=2)

# intersections
isecs1 = obj3.zdata.dst.get_data_by_att(attr="trg_index", value=[i], mode="geo")
isecs1.plot(column="src_index", ax=ax, cmap=plt.cm.plasma, alpha=0.5)

cat = trg.get_data_by_idx([i])[0]
bbox = wrl.zonalstats.get_bbox(cat[..., 0], cat[..., 1])
plt.xlim(bbox["left"] - 2000, bbox["right"] + 2000)
plt.ylim(bbox["bottom"] - 2000, bbox["top"] + 2000)
plt.legend()
plt.title("Catchment #%d: Polygons considered for stats" % i)
```

```{code-cell} python
# Compare estimates
maxlim = np.max(np.concatenate((avg1, avg3)))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, aspect="equal")
plt.scatter(avg1, avg3, edgecolor="None", alpha=0.5)
plt.xlabel("Average of points in or close to polygon (mm)")
plt.ylabel("Area-weighted average (mm)")
plt.xlim(0, maxlim)
plt.ylim(0, maxlim)
plt.plot([-1, maxlim + 1], [-1, maxlim + 1], color="black")
plt.show()
```
