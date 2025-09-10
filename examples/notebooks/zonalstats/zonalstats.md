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

# Zonal Statistics - Overview

```{toctree}
:hidden:
:maxdepth: 2
Quickstart <zonalstats_quickstart>
Cartesian Grid <../workflow/recipe5>
Polar Grid <../workflow/recipe6>
```

The {mod}`wradlib.zonalstats` module provides classes and functions for calculation of zonal statistics for data on arbitrary grids and projections.

It provides classes for:

- managing georeferenced data (grid points or grid polygons, zonal polygons),
- calculation of geographic intersections and managing resulting vector data
- calculation of zonal statistics and managing result data as vector attributes
- output to vector and raster files available within ogr/gdal

```{code-cell} python
import wradlib as wrl
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

## VectorSource


The {class}`wradlib.io.gdal.VectorSource` class handles point or polygon vector data by wrapping ogr.DataSource with special functions. It's just a wrapper around the {class}`wradlib.io.gdal.VectorSource`  class.

The following example shows how to create different DataSource objects:

```{code-cell} python
from osgeo import osr

# create gauss-krueger2 srs object
proj_gk2 = osr.SpatialReference()
proj_gk2.ImportFromEPSG(31466)

# Setting up DataSource
box0 = np.array(
    [
        [2600000.0, 5630000.0],
        [2600000.0, 5640000.0],
        [2610000.0, 5640000.0],
        [2610000.0, 5630000.0],
        [2600000.0, 5630000.0],
    ]
)
box1 = np.array(
    [
        [2610000.0, 5630000.0],
        [2610000.0, 5640000.0],
        [2620000.0, 5640000.0],
        [2620000.0, 5630000.0],
        [2610000.0, 5630000.0],
    ]
)
box2 = np.array(
    [
        [2600000.0, 5640000.0],
        [2600000.0, 5650000.0],
        [2610000.0, 5650000.0],
        [2610000.0, 5640000.0],
        [2600000.0, 5640000.0],
    ]
)
box3 = np.array(
    [
        [2610000.0, 5640000.0],
        [2610000.0, 5650000.0],
        [2620000.0, 5650000.0],
        [2620000.0, 5640000.0],
        [2610000.0, 5640000.0],
    ]
)

point0 = np.array(wrl.georef.get_centroid(box0))
point1 = np.array(wrl.georef.get_centroid(box1))
point2 = np.array(wrl.georef.get_centroid(box2))
point3 = np.array(wrl.georef.get_centroid(box3))

# creates Polygons in Datasource
poly = wrl.io.VectorSource(
    np.array([box0, box1, box2, box3]), trg_crs=proj_gk2, name="poly"
)

# creates Points in Datasource
point = wrl.io.VectorSource(
    np.vstack((point0, point1, point2, point3)), trg_crs=proj_gk2, name="point"
)
```

Let's have a look at the data, which will be exported as numpy arrays. The property ``data`` exports all available data:

```{code-cell} python
print(poly.data)
print(point.data)
```

Currently data can also be retrieved by:

- index - {func}`wradlib.io.gdal.VectorSource.get_data_by_idx`,
- attribute - {func}`wradlib.io.gdal.VectorSource.get_data_by_att` and
- geometry - {func}`wradlib.io.gdal.VectorSource.get_data_by_geom`.

Now, with the DataSource being created, we can add/set attribute data of the features:

```{code-cell} python
# add attribute
poly.set_attribute("mean", np.array([10.1, 20.2, 30.3, 40.4]))
point.set_attribute("mean", np.array([10.1, 20.2, 30.3, 40.4]))
```

Attributes associated with features can also be retrieved:

```{code-cell} python
# get attributes
print(poly.get_attributes(["mean"]))
# get attributes filtered
print(poly.get_attributes(["mean"], filt=("index", 2)))
```

Finally, we can export the contained data to OGR/GDAL supported [vector](https://gdal.org/ogr_formats.html) and [raster](https://gdal.org/formats_list.html) files:

```{code-cell} python
# dump as 'ESRI Shapefile', default
poly.dump_vector("test_poly.shp")
point.dump_vector("test_point.shp")
# dump as 'GeoJSON'
poly.dump_vector("test_poly.geojson", driver="GeoJSON")
point.dump_vector("test_point.geojson", driver="GeoJSON")
# dump as 'GTiff', default
poly.dump_raster("test_poly_raster.tif", attr="mean", pixel_size=100.0)
# dump as 'netCDF'
poly.dump_raster("test_poly_raster.nc", driver="netCDF", attr="mean", pixel_size=100.0)
```

## ZonalData


ZonalData is usually available as georeferenced regular gridded data. Here the {class}`wradlib.zonalstats.ZonalDataBase` class manages the grid data, the zonal data (target polygons) and the intersection data of source grid and target polygons.
Because the calculation of intersection is different for point grids and polygon grids, we have subclasses {class}`wradlib.zonalstats.ZonalDataPoly` and {class}`wradlib.zonalstats.ZonalDataPoint`.

Basically, {class}`wradlib.zonalstats.ZonalDataBase` encapsulates three {class}`wradlib.zonalstats.DataSource` objects:

- source grid (points/polygons)
- target polygons
- destination (intersection) (points/polygons)

The destination DataSource object is created from the provided source grid and target polygons at initialisation time.


As an example the creation of a {class}`wradlib.zonalstats.ZonalDataPoly` class instance is shown:

```{code-cell} python
# setup test grid and catchment
lon = 7.071664
lat = 50.730521
alt = 0
r = np.array(range(50, 100 * 1000 + 50, 100))
a = np.array(range(0, 90, 1))
rays = a.shape[0]
bins = r.shape[0]

# setup OSR objects
proj_utm = osr.SpatialReference()
proj_utm.ImportFromEPSG(32632)

# create polar grid polygon vertices in UTM
radar_utm = wrl.georef.spherical_to_polyvert(r, a, 0, (lon, lat, alt), crs=proj_utm)
radar_utm = radar_utm[..., 0:2]
# reshape
radar_utm.shape = (rays * bins, 5, 2)

box0 = np.array(
    [
        [390000.0, 5630000.0],
        [390000.0, 5640000.0],
        [400000.0, 5640000.0],
        [400000.0, 5630000.0],
        [390000.0, 5630000.0],
    ]
)

box1 = np.array(
    [
        [400000.0, 5630000.0],
        [400000.0, 5640000.0],
        [410000.0, 5640000.0],
        [410000.0, 5630000.0],
        [400000.0, 5630000.0],
    ]
)

targets = np.array([box0, box1])

zdpoly = wrl.zonalstats.ZonalDataPoly(radar_utm, trg=targets, crs=proj_utm)
```

When calculating the intersection, also weights are calculated for every source grid feature and attributed to the destination features.

With the property ``isecs`` it is possible to retrieve the intersection geometries as numpy array, further get-functions add to the functionality:

```{code-cell} python
# get intersections as numpy array
isecs = zdpoly.isecs
# get intersections for target polygon 0
isec0 = zdpoly.get_isec(0)
# get source indices referring to target polygon 0
ind0 = zdpoly.get_source_index(0)

print(isecs.shape, isec0.shape, ind0.shape)
```

There are import/export functions using [ESRI-Shapfile Format](https://de.wikipedia.org/wiki/Shapefile) as data format. Next export and import is shown:

```{code-cell} python
zdpoly.dump_vector("test_zdpoly")
zdpoly_new = wrl.zonalstats.ZonalDataPoly("test_zdpoly")
```

## ZonalStats

For ZonalStats the {class}`wradlib.zonalstats.ZonalStatsBase` class and the two subclasses {class}`wradlib.zonalstats.ZonalStatsPoly` and {class}`wradlib.zonalstats.ZonalStatsPoint` are available. ZonalStatsBase encapsulates one ZonalData object. Properties for simple access of ZonalData, intersection indices and weights are provided. The following code will add ``mean`` and ``var`` attributes to the target DataSource:

```{code-cell} python
# create ZonalStatsPoly instance
gc = wrl.zonalstats.ZonalStatsPoly(zdpoly_new)
# create some artificial data for processing using the features indices
count = radar_utm.shape[0]
data = 1000000.0 / np.array(range(count))
# calculate mean and variance
mean = gc.mean(data)
var = gc.var(data)

print("Average:", mean)
print("Variance:", var)
```

Next we can export the resulting zonal statistics to vector and raster files:

```{code-cell} python
# export to vector GeoJSON
gc.zdata.trg.dump_vector("test_zonal_json.geojson", driver="GeoJSON")
# export 'mean' to raster netCDF
gc.zdata.trg.dump_raster(
    "test_zonal_hdr.nc", driver="netCDF", attr="mean", pixel_size=100.0
)
```

The ZonalStats classes can also be used without any ZonalData by instantiating with precalculated index and weight values. Be sure to use matching ix, w and data arrays:

```{code-cell} python
# get ix, and weight arrays
ix = gc.ix
w = gc.w
# instantiate new ZonlaStats object
gc1 = wrl.zonalstats.ZonalStatsPoly(ix=ix, w=w)
# caclulate statistics
avg = gc1.mean(data)
var = gc1.var(data)

print("Average:", avg)
print("Variance:", var)
```
