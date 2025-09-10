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

# Vector Source


The {class}`wradlib.io.VectorSource` class is designed to conveniently handle Vector Data (eg. shapefiles). It originates from the {mod}`wradlib.zonalstats` module but moved to {mod}`wradlib.io.gdal` for better visibility.

- managing georeferenced data (grid points or grid polygons, zonal polygons),
- output to vector and raster files available within ogr/gdal
- geopandas dataframe connector

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

The {class}`wradlib.io.VectorSource` class handles point or polygon vector data by wrapping ogr.DataSource with special functions.

The following example shows how to create different VectorSource objects:

```{code-cell} python
from osgeo import osr

# create gk2 projection osr object
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

```{code-cell} python
print(poly)
```

Let's have a look at the data, which will be exported as numpy arrays. The property ``data`` exports all available data as numpy arrays:


## numpy access

```{code-cell} python
print(poly.data)
print(point.data)
```

## geopandas access

```{code-cell} python
poly.geo.explore()
```

```{code-cell} python
point.geo.loc[slice(0, 2)]
```

```{code-cell} python
point.geo.loc[[0, 1, 3]]
```

```{code-cell} python
point.geo.query("index in (0, 2)")
```

```{code-cell} python
fig = plt.figure()
ax = fig.add_subplot(111)
poly.geo.plot(column="index", ax=ax)
point.geo.plot(ax=ax)
```

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

Currently data can also be retrieved by:

- index - {func}`wradlib.io.gdal.VectorSource.get_data_by_idx`,
- attribute - {func}`wradlib.io.gdal.VectorSource.get_data_by_att` and
- geometry - {func}`wradlib.io.gdal.VectorSource.get_data_by_geom`.


Using the property `mode` the output type can be set permanently.


## get_data_by_idx

```{code-cell} python
point.get_data_by_idx([0, 2])
```

```{code-cell} python
point.get_data_by_idx([0, 2], mode="geo")
```

## get_data_by_att

```{code-cell} python
point.get_data_by_att("index", [0, 2])
```

```{code-cell} python
point.get_data_by_att("index", [0, 2], mode="geo")
```

## get_data_by_geom

```{code-cell} python
# get OGR.Geometry
geom0 = poly.get_data_by_idx([0], mode="ogr")[0]
# get geopandas Geometry
geom1 = poly.get_data_by_idx([0], mode="geo")
```

```{code-cell} python
point.get_data_by_geom(geom=geom0)
```

```{code-cell} python
point.get_data_by_geom(geom=geom0, mode="ogr")
```

```{code-cell} python
point.get_data_by_geom(geom=geom1, mode="geo")
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

## reload geojson

```{code-cell} python
point2 = wrl.io.VectorSource("test_point.geojson")
poly2 = wrl.io.VectorSource("test_poly.geojson")
fig = plt.figure()
ax = fig.add_subplot(111)
poly2.geo.plot(column="index", ax=ax)
point2.geo.plot(ax=ax)
```

## reload raster geotiff

```{code-cell} python
import xarray as xr

ds = xr.open_dataset("test_poly_raster.tif")
ds.band_data[0].plot()
```

## reload raster netcdf

```{code-cell} python
ds = xr.open_dataset("test_poly_raster.nc")
ds.Band1.plot()
```
