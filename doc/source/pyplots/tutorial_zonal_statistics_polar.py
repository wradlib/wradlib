#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:        tutorial_zonal_statistics_polar.py
# Purpose:
#
# Author:      Maik Heistermann, Kai Muehlbauer
#
# Created:     26.08.2015
# Copyright:   (c) heistermann, muehlbauer 2015
# Licence:     The MIT License
# -----------------------------------------------------------------------------

from osgeo import osr
import wradlib
import pylab as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
import datetime as dt


def testplot(cats, catsavg, xy, data,
             levels=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100],
             title=""):
    """Quick test plot layout for this example file
    """
    colors = plt.cm.spectral(np.linspace(0, 1, len(levels)))
    mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

    radolevels = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    radocolors = plt.cm.spectral(np.linspace(0, 1, len(radolevels)))
    radocmap, radonorm = from_levels_and_colors(radolevels, radocolors,
                                                extend="max")

    fig = plt.figure(figsize=(14, 8))

    # Average rainfall sum
    ax = fig.add_subplot(121, aspect="equal")
    coll = PatchCollection(cats, array=catsavg, cmap=mycmap, norm=mynorm,
                           edgecolors='white', lw=0.5)
    ax.add_collection(coll)
    ax.autoscale()
    plt.colorbar(coll, ax=ax, shrink=0.5)
    plt.xlabel("GK2 Easting")
    plt.ylabel("GK2 Northing")
    plt.title(title)
    plt.draw()

    # Original radar data
    ax1 = fig.add_subplot(122, aspect="equal")
    pm = plt.pcolormesh(xy[:, :, 0], xy[:, :, 1], np.ma.masked_invalid(data),
                        cmap=radocmap, norm=radonorm)
    coll = PatchCollection(cats, facecolor='None', edgecolor='white', lw=0.5)
    ax1.add_collection(coll)
    cb = plt.colorbar(pm, ax=ax1, shrink=0.5)
    cb.set_label("(mm/h)")
    plt.xlabel("GK2 Easting")
    plt.ylabel("GK2 Northing")
    plt.title("Original radar rain sums")
    plt.draw()
    plt.tight_layout()


def ex_tutorial_zonal_statistics_polar():
    filename = 'hdf5/rainsum_boxpol_20140609.h5'
    filename = wradlib.util.get_wradlib_data_file(filename)
    data, attrib = wradlib.io.from_hdf5(filename)

    # get Lat, Lon, range, azimuth, rays, bins out of radar data
    lat1 = attrib['Latitude']
    lon1 = attrib['Longitude']
    r1 = attrib['r']
    a1 = attrib['az']

    rays = a1.shape[0]
    bins = r1.shape[0]

    # create polar grid polygon vertices in lat,lon
    radar_ll = wradlib.georef.polar2polyvert(r1, a1, (lon1, lat1))

    # create polar grid centroids in lat,lon
    rlon, rlat = wradlib.georef.polar2centroids(r1, a1, (lon1, lat1))
    radar_llc = np.dstack((rlon, rlat))

    # setup OSR objects
    proj_gk = osr.SpatialReference()
    proj_gk.ImportFromEPSG(31466)
    proj_ll = osr.SpatialReference()
    proj_ll.ImportFromEPSG(4326)

    # project ll grids to GK2
    radar_gk = wradlib.georef.reproject(radar_ll, projection_source=proj_ll,
                                        projection_target=proj_gk)
    radar_gkc = wradlib.georef.reproject(radar_llc, projection_source=proj_ll,
                                         projection_target=proj_gk)

    # reshape
    radar_gk.shape = (rays, bins, 5, 2)
    radar_gkc.shape = (rays, bins, 2)

    shpfile = wradlib.util.get_wradlib_data_file(
        'shapefiles/agger/agger_merge.shp')
    dataset, inLayer = wradlib.io.open_shape(shpfile)
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer)

    # create synthetic box
    box = np.array([[2600000., 5630000.], [2600000., 5640000.],
                    [2610000., 5640000.], [2610000., 5630000.],
                    [2600000., 5630000.]])
    l = list(cats)
    l.append(box)
    cats = np.array(l)
    bbox = inLayer.GetExtent()

    # create catchment bounding box
    buffer = 5000.
    bbox = dict(left=bbox[0] - buffer, right=bbox[1] + buffer,
                bottom=bbox[2] - buffer, top=bbox[3] + buffer)

    mask, shape = wradlib.zonalstats.mask_from_bbox(radar_gkc[..., 0],
                                                    radar_gkc[..., 1],
                                                    bbox,
                                                    polar=True)

    radar_gkc_ = radar_gkc[mask, :]
    radar_gk_ = radar_gk[mask]
    data_ = data[mask]

    ###########################################################################
    # Approach #1: Assign grid points to each polygon and compute the average.
    #
    # - Uses matplotlib.path.Path
    # - Each point is weighted equally (assumption: polygon >> grid cell)
    # - this is quick, but theoretically dirty
    # - for polar grids a range-area dependency has to be taken into account
    ###########################################################################

    t1 = dt.datetime.now()

    try:
        # Create instance of type GridPointsToPoly from zonal data file
        obj1 = wradlib.zonalstats.GridPointsToPoly('test_zonal_points')
    except:
        # Create instance of type ZonalDataPoint from source grid and
        # catchment array
        zd = wradlib.zonalstats.ZonalDataPoint(radar_gkc_, cats, srs=proj_gk,
                                               buf=500.)
        # dump to file
        zd.dump_vector('test_zonal_points')
        # Create instance of type GridPointsToPoly from zonal data object
        obj1 = wradlib.zonalstats.GridPointsToPoly(zd)

    isecs1 = obj1.zdata.isecs
    t2 = dt.datetime.now()

    # Compute stats for target polygons
    avg1 = obj1.mean(data_.ravel())
    var1 = obj1.var(data_.ravel())

    t3 = dt.datetime.now()

    print("Approach #1 (create object) takes: %f "
          "seconds" % (t2 - t1).total_seconds())
    print("Approach #1 (compute average) takes: %f "
          "seconds" % (t3 - t2).total_seconds())

    # Just a test for plotting results with zero buffer
    zd = wradlib.zonalstats.ZonalDataPoint(radar_gkc_, cats, buf=0)
    # Create instance of type GridPointsToPoly from zonal data object
    obj2 = wradlib.zonalstats.GridPointsToPoly(zd)
    isecs2 = obj2.zdata.isecs

    # Illustrate results for an example catchment i
    i = 0  # try e.g. 6, 12
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    # Target polygon patches
    trg_patches = [patches.Polygon(item, True) for item in obj1.zdata.trg.data]
    trg_patch = [trg_patches[i]]
    p = PatchCollection(trg_patch, facecolor="None", edgecolor="black",
                        linewidth=2)
    ax.add_collection(p)

    # pips
    sources = obj1.zdata.src.data
    plt.scatter(sources[:, 0], sources[:, 1], s=200, c="grey",
                edgecolor="None", label="all points")
    plt.scatter(isecs2[i][:, 0], isecs2[i][:, 1], s=200, c="green",
                edgecolor="None", label="buffer=0 m")
    plt.scatter(isecs1[i][:, 0], isecs1[i][:, 1], s=50, c="red",
                edgecolor="None", label="buffer=500 m")
    bbox = wradlib.zonalstats.get_bbox(cats[i][:, 0], cats[i][:, 1])
    plt.xlim(bbox["left"] - 2000, bbox["right"] + 2000)
    plt.ylim(bbox["bottom"] - 2000, bbox["top"] + 2000)
    plt.legend()
    plt.title("Catchment #%d: Points considered for stats" % i)

    # Plot average rainfall and original data
    testplot(trg_patches, avg1, radar_gkc, data,
             title="Catchment rainfall mean (GridPointsToPoly)")
    testplot(trg_patches, var1, radar_gkc, data, levels=np.arange(0, 20, 1.0),
             title="Catchment rainfall variance (GridPointsToPoly)")

    ###########################################################################
    # Approach #2: Compute weighted mean based on fraction of source polygons
    # in target polygons
    #
    # - This is more accurate (no assumptions), but probably slower...
    ###########################################################################

    t1 = dt.datetime.now()

    try:
        # Create instance of type GridCellsToPoly from zonal data file
        obj3 = wradlib.zonalstats.GridCellsToPoly('test_zonal_poly')
    except Exception as e:
        print(e)
        # Create instance of type ZonalDataPoly from source grid and
        # catchment array
        zd = wradlib.zonalstats.ZonalDataPoly(radar_gk_, cats, srs=proj_gk)
        # dump to file
        zd.dump_vector('test_zonal_poly')
        # Create instance of type GridPointsToPoly from zonal data object
        obj3 = wradlib.zonalstats.GridCellsToPoly(zd)

    obj3.zdata.dump_vector('test_zonal_poly')
    t2 = dt.datetime.now()

    avg3 = obj3.mean(data_.ravel())
    var3 = obj3.var(data_.ravel())

    obj3.zdata.trg.dump_raster('test_zonal_hdr.nc', 'netCDF', 'mean',
                               pixel_size=100.)

    obj3.zdata.trg.dump_vector('test_zonal_shp')
    obj3.zdata.trg.dump_vector('test_zonal_json.geojson', 'GeoJSON')

    t3 = dt.datetime.now()

    print("Approach #2 (create object) takes: %f "
          "seconds" % (t2 - t1).total_seconds())
    print("Approach #2 (compute average) takes: %f "
          "seconds" % (t3 - t2).total_seconds())

    # Target polygon patches
    trg_patches = [patches.Polygon(item, True) for item in obj3.zdata.trg.data]

    # Plot average rainfall and original data
    testplot(trg_patches, avg3, radar_gkc, data,
             title="Catchment rainfall mean (PolarGridCellsToPoly)")
    testplot(trg_patches, var3, radar_gkc, data, levels=np.arange(0, 20, 1.0),
             title="Catchment rainfall variance (PolarGridCellsToPoly)")

    # Illustrate results for an example catchment i
    i = 0  # try any index between 0 and 13
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    # Grid cell patches
    src_index = obj3.zdata.get_source_index(i)
    grd_patches = [patches.Polygon(item) for item in
                   obj3.zdata.src.get_data_by_idx(src_index)]
    p = PatchCollection(grd_patches, facecolor="None", edgecolor="black")
    ax.add_collection(p)

    # Target polygon patches
    trg_patch = [trg_patches[i]]
    p = PatchCollection(trg_patch, facecolor="None", edgecolor="red",
                        linewidth=2)
    ax.add_collection(p)

    # View the actual intersections
    t1 = dt.datetime.now()
    isecs = obj3.zdata.get_isec(i)
    isec_patches = wradlib.zonalstats.numpy_to_pathpatch(isecs)
    colors = 100 * np.linspace(0, 1., len(isec_patches))
    p = PatchCollection(isec_patches, cmap=plt.cm.jet, alpha=0.5)
    p.set_array(np.array(colors))
    ax.add_collection(p)

    bbox = wradlib.zonalstats.get_bbox(cats[i][:, 0], cats[i][:, 1])
    plt.xlim(bbox["left"] - 2000, bbox["right"] + 2000)
    plt.ylim(bbox["bottom"] - 2000, bbox["top"] + 2000)
    plt.draw()

    t2 = dt.datetime.now()
    print("plot intersection takes: %f seconds" % (t2 - t1).total_seconds())

    # Compare estimates
    maxlim = np.max(np.concatenate((avg1, avg3)))
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect="equal")
    plt.scatter(avg1, avg3, edgecolor="None", alpha=0.5)
    plt.xlabel("Average of points in or close to polygon (mm)")
    plt.ylabel("Area-weighted average (mm)")
    plt.xlim(0, maxlim)
    plt.ylim(0, maxlim)
    plt.plot([-1, maxlim + 1], [-1, maxlim + 1], color="black")
    plt.show()


# =======================================================
if __name__ == '__main__':
    ex_tutorial_zonal_statistics_polar()
