**********
Zonalstats
**********

The :py:mod:`.zonalstats` module provides classes and functions for calculation of zonal statistics for data on arbitrary grids and projections.

It provides classes for:

    * managing georeferenced data (grid points or grid polygons, zonal polygons),
    * calculation of geographic intersections and managing resulting vector data
    * calculation of zonal statistics and managing result data as vector attributes
    * output to vector and raster files available within ogr/gdal

DataSource
==========

The :py:class:`.DataSource` class handles point or polygon vector data by wrapping ogr.DataSource with special functions.

The following example shows how to create different DataSource objects::

    import wradlib as wrl
    import numpy as np
    from osgeo import osr

    # create gk zone 2 projection osr object
    proj_gk2 = osr.SpatialReference()
    proj_gk2.ImportFromEPSG(31466)

    # Setting up DataSource
    box0 = np.array([[2600000., 5630000.],[2600000., 5640000.],
                     [2610000., 5640000.],[2610000., 5630000.],
                     [2600000., 5630000.]])
    box1 = np.array([[2610000., 5630000.],[2610000., 5640000.],
                     [2620000., 5640000.],[2620000., 5630000.],
                     [2610000., 5630000.]])
    box2 = np.array([[2600000., 5640000.],[2600000., 5650000.],
                     [2610000., 5650000.],[2610000., 5640000.],
                     [2600000., 5640000.]])
    box3 = np.array([[2610000., 5640000.],[2610000., 5650000.],
                     [2620000., 5650000.],[2620000., 5640000.],
                     [2610000., 5640000.]])

    point0 = np.array(wrl.zonalstats.get_centroid(box0))
    point1 = np.array(wrl.zonalstats.get_centroid(box1))
    point2 = np.array(wrl.zonalstats.get_centroid(box2))
    point3 = np.array(wrl.zonalstats.get_centroid(box3))

    # creates Polygons in Datasource
    poly = wrl.zonalstats.DataSource(np.array([box0, box1]), srs=proj_gk2, name='poly')

    # creates Points in Datasource
    point = wrl.zonalstats.DataSource(np.vstack((point0, point1, point2, point3)),
                                      srs=proj_gk2, name='point')


Let's have a look at the data, which will be exported as numpy arrays. The property ``data`` exports all available data::

    print(poly.data)
    print(point.data)

Currently data can also be retrieved by:

* index (:func:`.DataSource.get_data_by_idx()`),
* attribute (:func:`.DataSource.get_data_by_att()`) and
* geometry (:func:`.DataSource.get_data_by_geom()`).

Now, with the DataSource being created, we can add/set attribute data of the features::

    # add attribute
    poly.set_attribute('mean', np.array([10.1, 20.2, 30.3, 40.4]))
    point.set_attribute('mean', np.array([10.1, 20.2, 30.3, 40.4]))


Attributes associated with features can also be retrieved::

    # get attributes
    print(poly.get_attributes(['mean'])
    # get attributes filtered
    print(poly.get_attributes(['mean'], filt=('index', 2)))


Finally, we can export the contained data to OGR/GDAL supported `vector <http://www.gdal.org/ogr_formats.html>`_ and `raster <http://www.gdal.org/formats_list.html>`_ files::

    # dump as 'ESRI Shapefile', default
    poly.dump_vector('test_poly.shp')
    point.dump_vector('test_point.shp')
    # dump as 'GeoJSON'
    poly.dump_vector('test_poly.geojson', 'GeoJSON')
    point.dump_vector('test_point.geojson', 'GeoJSON')
    # dump as 'GTiff', default
    poly.dump_raster('test_poly_raster.tif', attr='mean', pixel_size=100.)
    # dump as 'netCDF'
    poly.dump_raster('test_poly_raster.nc', 'netCDF', attr='mean', pixel_size=100.)


ZonalData
=========

ZonalData is usually available as georeferenced regular gridded data. Here the :py:class:`.ZonalDataBase` class manages the grid data, the zonal data (target polygons) and the intersection data of source grid and target polygons.
Because the calculation of intersection is different for point grids and polygon grids, we have subclasses :py:class:`.ZonalDataPoly` and :py:class:`.ZonalDataPoint`.

Basically, :py:class:`.ZonalDataBase` encapsulates three :py:class:`.DataSource` objects:

    * source grid (points/polygons)
    * target polygons
    * destination (intersection) (points/polygons)

The destination DataSource object is created from the provided source grid and target polygons at initialisation time.

As an example the creation of a :py:class:`.ZonalDataPoly` class instance is shown::

    # setup test grid and catchment
    lon = 7.071664
    lat = 50.730521
    r = np.array(range(50, 100*1000 + 50 , 100))
    a = np.array(range(0, 90, 1))
    rays = a.shape[0]
    bins = r.shape[0]
    # create polar grid polygon vertices in lat,lon
    radar_ll = wrl.georef.polar2polyvert(r, a, (lon, lat))

    # setup OSR objects
    proj_gk = osr.SpatialReference()
    proj_gk.ImportFromEPSG(31466)
    proj_ll = osr.SpatialReference()
    proj_ll.ImportFromEPSG(4326)

    # project ll grids to GK2
    radar_gk = wrl.georef.reproject(radar_ll, projection_source=proj_ll,
                                projection_target=proj_gk)

    # reshape
    radar_gk.shape = (rays * bins, 5, 2)

    box0 = np.array([[2600000., 5630000.],[2600000., 5640000.],
                     [2610000., 5640000.],[2610000., 5630000.],
                     [2600000., 5630000.]])

    box1 = np.array([[2610000., 5630000.],[2610000., 5640000.],
                     [2620000., 5640000.],[2620000., 5630000.],
                     [2610000., 5630000.]])

    targets = np.array([box0, box1])

    zdpoly = wrl.zonalstats.ZonalDataPoly(radar_gk, targets, srs=proj_gk)


When calculationg the intersection, also weights are calculated for every source grid feature and attributed to the destination features.

With the property ``isecs`` it is possible to retrieve the intersection geometries as numpy array, further get-functions add to the functionality::

    # get intersections as numpy array
    isecs = zdpoly.isecs()
    # get intersections for target polygon 0
    isec0 = zdpoly.get_isec(0)
    # get source indices referring to target polygon 0
    ind0 = zdpoly.get_source_index(0)


There are import/export functions using ``ESRI Shapefile`` as data format. Next export and import is shown::

    zdpoly.dump_vector('test_zdpoly')
    zdpoly_new = wrl.zonalstats.ZonalDataPoly('test_zdpoly')


ZonalStats
==========

For ZonalStats the :py:class:`.ZonalStatsBase` class and the two subclasse :py:class:`.GridCellsToPoly` and :py:class:`.GridPointsToPoly` are available. ZonalStatsBase encapsulates one ZonalData object. Properties for simple access of ZonalData, intersection indices and weights are provided::

    # create GridCellsToPoly instance
    gc = wrl.zonalstats.GridCellsToPoly(zdpoly_new)
    # create some artificial data for processing using the features indices
    count = radar_gk.shape[0]
    data = 1000000. / np.array(range(count))
    # calculate mean and variance
    mean = gc.mean(data)
    var = gc.var(data)


This will add ``mean`` and ``var`` attributes to the target DataSource. This can be used to export the resulting zonal statistics to vector and raster files::

    # export to vector GeoJSON
    gc.zdata.trg.dump_vector('test_zonal_json.geojson', 'GeoJSON')
    # export 'mean' to raster netCDF
    gc.zdata.trg.dump_raster('test_zonal_hdr.nc', 'netCDF', 'mean', pixel_size=100.)


The ZonalStats classes can also be used without any ZonalData by instantiating with precalculated index and weight values. Be sure to use matching ix, w and data arrays::

    # get ix, and weight arrays
    ix = gc.ix
    w = gc.w
    # instantiate new ZonlaStats object
    gc1 = wrl.zonalstats.GridCellsToPoly(ix=ix, w=w)
    # caclulate statistics
    avg = gc1.mean(data)
    var = gc1.var(data)


Examples
========

`This extensive example <https://bitbucket.org/wradlib/wradlib/src/default/examples/tutorial_zonal_statistics.py>`_ shows creation of Zonal Statistics from RADOLAN Data:

    .. plot:: pyplots/tutorial_zonal_statistics.py
        :include-source: False

`This second example <https://bitbucket.org/wradlib/wradlib/src/default/examples/tutorial_zonal_statistics_polar.py>`_ shows creation of Zonal Statistics from Radar Data:

    .. plot:: pyplots/tutorial_zonal_statistics_polar.py
        :include-source: False