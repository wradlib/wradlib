# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:05:48 2015

@author: heistermann
"""

from osgeo import osr
import wradlib
import pylab as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
import datetime as dt

   
def testplot(cats, catsavg, xy, data, levels = [0,1,2,3,4,5,10,15,20,25,30,40,50,100], title=""):
    """Quick test plot layout for this example file
    """
    colors = plt.cm.spectral(np.linspace(0,1,len(levels)) )    
    mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

    radolevels = [0,1,2,3,4,5,10,15,20,25,30,40,50,100]
    radocolors = plt.cm.spectral(np.linspace(0,1,len(radolevels)) )    
    radocmap, radonorm = from_levels_and_colors(radolevels, radocolors, extend="max")

    fig = plt.figure(figsize=(14,8))
    # Average rainfall sum
    ax = fig.add_subplot(121, aspect="equal")
    wradlib.vis.add_lines(ax, cats, color='white', lw=0.5)
    coll = PolyCollection(cats, array=catsavg, cmap=mycmap, norm=mynorm, edgecolors='none')
    ax.add_collection(coll)
    ax.autoscale()
    cb = plt.colorbar(coll, ax=ax, shrink=0.5)
    plt.xlabel("GK4 Easting")
    plt.ylabel("GK4 Northing")
    plt.title(title)
    plt.draw()
    # Original RADOLAN data
    ax1 = fig.add_subplot(122, aspect="equal")
    pm = plt.pcolormesh(xy[:, :, 0], xy[:, :, 1], np.ma.masked_invalid(data), cmap=radocmap, norm=radonorm)
    wradlib.vis.add_lines(ax1, cats, color='white', lw=0.5)
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    cb = plt.colorbar(pm, ax=ax1, shrink=0.5)
    cb.set_label("(mm/h)")
    plt.xlabel("GK4 Easting")
    plt.ylabel("GK4 Northing")
    plt.title("Original RADOLAN rain sums")
    plt.draw()
    plt.tight_layout()


if __name__ == '__main__':

    # Get RADOLAN grid coordinates
    grid_xy_radolan = wradlib.georef.get_radolan_grid(900, 900)
    x_radolan = grid_xy_radolan[:, :, 0]
    y_radolan = grid_xy_radolan[:, :, 1]
    
    # create radolan projection osr object
    proj_stereo = wradlib.georef.create_osr("dwd-radolan")

    # create Gauss Krueger zone 4 projection osr object
    proj_gk = osr.SpatialReference()
    proj_gk.ImportFromEPSG(31468)

    # transform radolan polar stereographic projection to GK4
    xy = wradlib.georef.reproject(grid_xy_radolan,
                                  projection_source=proj_stereo,
                                  projection_target=proj_gk)

    # Open shapefile (already in GK4)
    shpfile = "data/freiberger_mulde/freiberger_mulde.shp"
    dataset, inLayer = wradlib.io.open_shape(shpfile)
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='GWKZ')

    # Read and prepare the actual data (RADOLAN)
    f = "data/radolan/raa01-sf_10000-1305280050-dwd---bin.gz"
    data, attrs = wradlib.io.read_RADOLAN_composite(f, missing=np.nan)
    sec = attrs['secondary']
    data.flat[sec] = np.nan
    
    # Reduce grid size using a bounding box (to enhancing performance)
    bbox = inLayer.GetExtent()
    buffer = 5000.
    bbox = dict(left=bbox[0]-buffer, right=bbox[1]+buffer, bottom=bbox[2]-buffer, top=bbox[3]+buffer)
    mask, shape = wradlib.zonalstats.mask_from_bbox(xy[...,0],xy[...,1], bbox)
    xy_ = np.vstack((xy[...,0][mask].ravel(),xy[...,1][mask].ravel())).T
    data_ = data[mask]
    
    ###########################################################################
    # Approach #1: Assign grid points to each polygon and compute the average.
    # 
    # - Uses matplotlib.path.Path
    # - Each point is weighted equally (assumption: polygon >> grid cell)
    # - this is quick, but theoretically dirty     
    ###########################################################################

    t1 = dt.datetime.now()

    # Create instances of type GridPointsToPoly (one instance for each target polygon)
    obj1 = wradlib.zonalstats.GridPointsToPoly(xy_, cats, buffer=500.)

    t2 = dt.datetime.now()

    # Compute stats for target polygons
    avg1 =  obj1.mean( data_.ravel() )
    var1 =  obj1.var( data_.ravel() )

    t3 = dt.datetime.now()

    print "Approach #1 (create object) takes: %f seconds" % (t2 - t1).total_seconds()
    print "Approach #1 (compute average) takes: %f seconds" % (t3 - t2).total_seconds()
    
    # Just a test for plotting results with zero buffer
    obj2 = wradlib.zonalstats.GridPointsToPoly(xy_, cats, buffer=0.)    

    # Illustrate results for an example catchment i
    i = 100 # try e.g. 48, 100 
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    # Target polygon patches
    trg_patches = [patches.Polygon(item, True) for item in [cats[i]] ]
    p = PatchCollection(trg_patches, facecolor="None", edgecolor="black", linewidth=2)
    ax.add_collection(p)
    # pips
    plt.scatter(xy_[:,0], xy_[:,1], s=200, c="grey", edgecolor="None", label="all points")
    plt.scatter(xy_[obj2.ix[i],0], xy_[obj2.ix[i],1], s=200, c="green", edgecolor="None", label="buffer=0 m")
    plt.scatter(xy_[obj1.ix[i],0], xy_[obj1.ix[i],1], s=50, c="red", edgecolor="None", label="buffer=500 m")
    bbox = wradlib.zonalstats.get_bbox(cats[i][:,0], cats[i][:,1])
    plt.xlim(bbox["left"]-2000, bbox["right"]+2000)
    plt.ylim(bbox["bottom"]-2000, bbox["top"]+2000)
    plt.legend()
    plt.title("Catchment #%d: Points considered for stats" % i)

    # Plot average rainfall and original data
    testplot(cats, avg1, xy, data, title="Catchment rainfall mean (GridPointsToPoly)")
    testplot(cats, var1, xy, data, levels = np.arange(0,4.2,0.2), title="Catchment rainfall variance (GridPointsToPoly)")    
   

    ###########################################################################
    # Approach #2: Compute weighted mean based on fraction of source polygons in target polygons
    # 
    # - This is more accurate (no assumptions), but probably slower...
    ###########################################################################

    t1 = dt.datetime.now()

    # Create vertices for each grid cell (MUST BE DONE IN NATIVE RADOLAN COORDINATES)
    grdverts = wradlib.zonalstats.grid_centers_to_vertices(x_radolan[mask],y_radolan[mask],1.,1.)
    # And reproject to Cartesian reference system (here: GK4)
    grdverts = wradlib.georef.reproject(grdverts,
                                  projection_source=proj_stereo,
                                  projection_target=proj_gk)

    # Create instances of type GridCellsToPoly (one instance for each target polygon)
    obj3 = wradlib.zonalstats.GridCellsToPoly(grdverts, cats)

    t2 = dt.datetime.now()

    # Compute stats for target polygons
    avg3 =  obj3.mean( data_.ravel() )
    var3 =  obj3.var( data_.ravel() )

    t3 = dt.datetime.now()

    print "Approach #2 (create object) takes: %f seconds" % (t2 - t1).total_seconds()
    print "Approach #2 (compute average) takes: %f seconds" % (t3 - t2).total_seconds()
    
    # Plot average rainfall and original data
    testplot(cats, avg3, xy, data, title="Catchment rainfall mean (GridCellsToPoly)")
    testplot(cats, var3, xy, data, levels = np.arange(0,4.2,0.2), title="Catchment rainfall variance (GridCellsToPoly)")    
    

    # Illustrate results for an example catchment i
    i = 100 # try any index between 0 and 429 
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    # Grid cell patches
    grd_patches = [patches.Polygon(item, True) for item in grdverts[obj3.ix[i]] ]
    p = PatchCollection(grd_patches, facecolor="None", edgecolor="black")
    ax.add_collection(p)
    # Target polygon patches
    trg_patches = [patches.Polygon(item, True) for item in [cats[i]] ]
    p = PatchCollection(trg_patches, facecolor="None", edgecolor="red", linewidth=2)
    ax.add_collection(p)
    # View the actual intersections 
    isecs = obj3._get_intersection(cats[i])
    isec_patches = [patches.Polygon(item, True) for item in isecs ]
    colors = 100*np.linspace(0,1.,len(isecs))
    p = PatchCollection(isec_patches, cmap=plt.cm.jet, alpha=0.5)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    bbox = wradlib.zonalstats.get_bbox(cats[i][:,0], cats[i][:,1])
    plt.xlim(bbox["left"]-2000, bbox["right"]+2000)
    plt.ylim(bbox["bottom"]-2000, bbox["top"]+2000)
    plt.draw()
    

    # Compare estimates
    maxlim = np.max(np.concatenate((avg1, avg3)))
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, aspect="equal")
    plt.scatter(avg1, avg3, edgecolor="None", alpha=0.5)
    plt.xlabel("Average of points in or close to polygon (mm)")
    plt.ylabel("Area-weighted average (mm)")
    plt.xlim(0, maxlim)
    plt.ylim(0, maxlim)
    plt.plot([-1,maxlim+1], [-1,maxlim+1], color="black")
    
    ###########################################################################
    # DUMP
    ###########################################################################

#    # First a simple (illustrative) example
#    #   Create grid
#    x = np.arange(0,6)
#    y = np.arange(10,16)
#    x, y = np.meshgrid(x,y)
#    #   Build vertices from grid center points
#    srcs = wradlib.zonalstats.grid_centers_to_vertices(x,y,1.,1.)
#    #   Example polygon vertices
#    trgs = np.array([ [[1.7,10.],
#                          [3. , 9.8],
#                          [3. , 12.],
#                          [2. , 12.5],
#                          [1.7, 10.]],
#                       
#                         [[4. , 11.],
#                          [5. , 12.],
#                          [5  , 14.8],
#                          [4. , 15.],
#                          [4. , 11.]],
#
#                         [[1. , 13.],
#                          [2. , 13.],
#                          [2. , 14.],
#                          [1. , 14.],
#                          [1. , 13.]],
#
#                         [[2.7, 13.7],
#                          [3.3, 13.7],
#                          [3.3, 14.3],
#                          [2.7, 14.3],
#                          [2.7, 13.7]] 
#                    ])
#    
#    # Iterate over all target polygons
#    intersecs, areas = [], []
#    for trg in trgs:
#        isecs_ = []
#        areas_ = []
#        for src in srcs:
#            tmp = wradlib.zonalstats.intersect(src, trg)            
#            if not tmp[0]==None:
#                isecs_.append(tmp[0])
#                areas_.append(tmp[1] )
#        intersecs.append(np.array(isecs_))
#        areas.append(np.array(areas_))
#
#                     
#    # Plot the simple example
#    fig = plt.figure()
#    ax = fig.add_subplot(111, aspect="equal")
#    # Grid cell patches
#    grd_patches = [patches.Polygon(item, True) for item in srcs ]
#    p = PatchCollection(grd_patches, facecolor="None", edgecolor="black")
#    ax.add_collection(p)
#    # Target polygon patches
#    trg_patches = [patches.Polygon(item, True) for item in trgs ]
#    p = PatchCollection(trg_patches, facecolor="None", edgecolor="red", linewidth=2)
#    ax.add_collection(p)
#    # Intersectin patches
#    for isec in intersecs:
#        colors = 100*np.linspace(0,1.,len(isec))
#        isec_patches = [patches.Polygon(item, True) for item in isec ]
#        p = PatchCollection(isec_patches, cmap=plt.cm.jet, alpha=0.5)
#        p.set_array(np.array(colors))
#        ax.add_collection(p)
#    ax.set_xlim(-1,6)
#    ax.set_ylim(9,16)
#    plt.draw()
