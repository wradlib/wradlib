# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:05:48 2015

@author: heistermann
"""

from osgeo import ogr, osr
import wradlib
import pylab as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
from scipy.spatial import cKDTree
import datetime as dt


def mask_from_bbox(x, y, bbox):
    """Return index array based on spatial selection from a bounding box.
    """
    ny, nx = x.shape
    
    ix = np.arange(x.size).reshape(x.shape)

    # Find bbox corners
    #    Plant a tree
    tree = cKDTree(np.vstack((x.ravel(),y.ravel())).transpose())
    # find lower left corner index
    dists, ixll = tree.query([bbox["left"], bbox["bottom"]], k=1)
    ill, jll = np.array(np.where(ix==ixll))[:,0]
    ill = (ixll / nx)-1
    jll = (ixll % nx)-1
    # find lower left corner index
    dists, ixur = tree.query([bbox["right"],bbox["top"]], k=1)
    iur, jur = np.array(np.where(ix==ixur))[:,0]
    iur = (ixur / nx)+1
    jur = (ixur % nx)+1
    
    mask = np.repeat(False, ix.size).reshape(ix.shape)
    if iur>ill:
        mask[ill:iur,jll:jur] = True
        shape = (iur-ill, jur-jll)
    else:
        mask[iur:ill,jll:jur] = True
        shape = (ill-iur, jur-jll)
    
    return mask, shape
        

def points_in_polygon(polygon, points, buffer=0.):
    """Select points inside polygon
    """
    mpath = Path( polygon )
    return  mpath.contains_points(points, radius=-buffer)


def subset_points(pts, bbox, buffer=0.):
    """Subset a large set of points by polygon bbox
    """
    x = pts[:,0]
    y = pts[:,1]
    return np.where(
            (x >= bbox["left"]  -buffer) & \
            (x <= bbox["right"] +buffer) & \
            (y >= bbox["bottom"]-buffer) & \
            (y <= bbox["top"]   +buffer) )[0]
        

def get_bbox(x,y, buffer=0.):
    """Return dictionary of bbox
    """
    return dict(left=np.min(x), 
                right=np.max(x), 
                bottom=np.min(y), 
                top=np.max(y))
    
    
def get_centroid(polyg):
    """Return centroid of a polygon
    
    Parameters
    ----------
    polyg : ndarray of shape (num vertices, 2) or ogr.Geometry object
    """
    if not type(polyg) == ogr.Geometry:
        polyg = polyg_to_ogr(polyg)
    return polyg.Centroid().GetPoint()[0:2]



def grid_centers_to_vertices(X, Y, dx, dy):
    """Produces array of vertices from grid's center point coordinates.
    
    .. warning:: This has to be done in the "native" grid projection. 
                 Once you reprojected the coordinates, this trivial function
                 cannot be used to compute vertices from center points.

    Parameters
    ----------   
    X : 2-d array of x coordinates (same shape as the actual 2-D grid)
    Y : 2-d array of y coordinates (same shape as the actual 2-D grid)
    dx : grid spacing in x direction
    dy : grid spacing in y direction
    
    Returns
    -------
    out : 3-d array of vertices for each grid cell of shape (n grid points, 
          5, 2)  
    
    """
    left    = X - dx/2
    right   = X + dy/2    
    bottom  = Y - dy/2
    top     = Y + dy/2
    
    verts = np.vstack(( [left.ravel() ,bottom.ravel()],
                        [right.ravel(),bottom.ravel()],
                        [right.ravel(),top.ravel()],
                        [left.ravel() ,top.ravel()],
                        [left.ravel() ,bottom.ravel()]) ).T.reshape((-1,5,2))
    
    return verts


def polyg_to_ogr(vert):
    """Convert a polygon vertex to gdal/ogr polygon geometry.
    
    Using JSON as a vehicle to efficiently deal with numpy arrays.
    
    """
    str = {"type":"Polygon", "coordinates":[vert.tolist()]}.__repr__()
    
    return ogr.CreateGeometryFromJson(str)


def ogr_to_polyg(ogrobj):
    """Backconvert a gdal/ogr Polygon geometry to a numpy vertex array.
    
    Using JSON as a vehicle to efficiently deal with numpy arrays.
    
    """
    jsonobj = eval(ogrobj.ExportToJson())
    return np.array(jsonobj["coordinates"][0])


def intersect(src, trg):
    """Return intersection and its area from target and source vertex.
    
    Parameters
    ----------
    src : numpy array of shape (n corners, 2) or ogr.Geometry
    trg : numpy array of shape (n corners, 2) or ogr.Geometry
    
    Returns
    -------
    out : intersection, area of intersection
    
    """
    # Convert to ogr if necessary
    if not type(trg) == ogr.Geometry:
        trg = polyg_to_ogr(trg)
    if not type(src) == ogr.Geometry:    
        src = polyg_to_ogr(src)
    isec = trg.Intersection(src)
    if isec.GetGeometryName()=="POLYGON":
        return ogr_to_polyg(isec), isec.Area()         
    else:
        return None, 0.


def could_intersect(src, trg):
    """Roughly checks for intersection between polygons in src and trg.
    
    This function should be used to filter polygons from src for which an 
    intersection with trg *might* be possible. It simply uses the spatial 
    bounding box (extent/envelope) of trg and checks whetehr any corner points
    of src fall within. Beware that this does not mean that the polygons in 
    fact intersect. This function is just to speed up the computation of true
    intersections by preselection.
    
    Parameters
    ----------
    src : array polygons
    trg : numpy array of polygon vertices of shape (n vertices, 2)
    
    Returns
    -------
    out : Boolean array of same length as src
        ith element is True if at least on vertex of the ith element of src 
        is within the bounding box of trg
        
    """
    bbox = get_bbox(trg[:,0], trg[:,1])

    right_of_left = np.any(src[...,0]>=bbox["left"], axis=1)
    left_of_right = np.any(src[...,0]<=bbox["right"], axis=1)
    above_bottom  = np.any(src[...,1]>=bbox["bottom"], axis=1)
    below_top     = np.any(src[...,1]<=bbox["top"], axis=1)
    
    return right_of_left & left_of_right & above_bottom & below_top


class ZonalMeanBase():
    """
    ZonalMeanBase(src, trg)

    The base class for computing 2-dimensional zonal average for target 
    polygons from source points or polygons. Provides the basic design 
    for all other classes.

    Parameters
    ----------
    src : sequence of source points or polygons 
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """

    def __init__(self, src, trg, **kwargs):
        self.src = self._check_src(src)
        self.trg = self._check_trg(trg)
        self.ix, self.w = self.get_weights(**kwargs)
    def _get_weights(self, **kwargs):
        """TODO This is the key method that needs to be filled for any inheriting class.
        """
        pass
    def _check_src(self, src):
        """TODO Basic check of source elements (sequence of points or polygons).

        """
        return src
    def _check_trg(self, trg):
        """TODO Basic check of target elements (sequence of polygons).

        """
        return trg
    def _check_vals(self, vals):
        """TODO Basic check of target elements (sequence of polygons).

        """
        assert len(vals)==len(self.src), "Argment vals must be of length %d" % len(self.src)
        return vals
    def __call__(self, vals):
        """
        Evaluate zonal statistics for values given at the source points.

        Parameters
        ----------
        vals : 1-d ndarray of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        return np.array( [np.sum( vals[self.ix[i]] * self.w[i]) for i in xrange(len(self.trg))] )
        

class GridCellsToPoly(ZonalMeanBase):
    """Compute weighted average for target polygons based on areal weights.   

    Parameters
    ----------
    src : sequence of source points or polygons 
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """
    def get_weights(self, **kwargs):
        """
        """
        # Conversion to ogr Geometries (only once)
        ogr_srcs = np.array([polyg_to_ogr(item) for item in self.src])
        ogr_trgs = np.array([polyg_to_ogr(item) for item in self.trg])
        
        ix, w = [], []
        for i in xrange( len(self.trg) ):
            # Pre-select grid vertices to increase performance
            ix_ = could_intersect(self.src, self.trg[i])
            areas = np.array([ intersect(ogr_src, ogr_trgs[i])[1] for ogr_src in ogr_srcs[ix_] ])
            w.append(areas / np.sum(areas))
            ix.append(ix_)
        
        return ix, w
            

class GridPointsToPoly(ZonalMeanBase):
    """Compute zonal average from all points in or close to the target polygon.

    Parameters
    ----------
    src : sequence of source points or polygons 
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)
    
    Keyword arguments
    -----------------
    buffer : 

    """
    def get_weights(self, **kwargs):
        """
        """
        ix, w = [], []
        for i in xrange( len(self.trg) ):
            # Pre-selection to increase performance 
            ix1 = subset_points(self.src, get_bbox(self.trg[i][:,0],self.trg[i][:,1]), buffer=kwargs["buffer"])
            ix2 = ix1[points_in_polygon(self.trg[i], self.src[ix1,:], buffer=kwargs["buffer"])]
            if len(ix2)==0:
                # No points in target polygon? Find the closest point to provide a value
                #    Polygon centroid
                centroid = get_centroid(self.trg[i])
                tree = cKDTree(self.src)
                distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
                ix2 = np.array([ixnext])
            w.append( np.ones(len(ix2)) / len(ix2 ) )
            ix.append(ix2)
        
        return ix, w

    
def testplot(cats, catsavg, xy, data):
    """Quick test plot layout for this example file
    """
    levels = [0,1,2,3,4,5,10,15,20,25,30,40,50,100]
    colors = plt.cm.spectral(np.linspace(0,1,len(levels)) )    
    mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

    fig = plt.figure(figsize=(14,8))
    # Average rainfall sum
    ax = fig.add_subplot(121, aspect="equal")
    wradlib.vis.add_lines(ax, cats, color='white', lw=0.5)
    coll = PolyCollection(cats, array=catsavg, cmap=mycmap, norm=mynorm, edgecolors='none')
    ax.add_collection(coll)
    ax.autoscale()
    cb = plt.colorbar(coll, ax=ax, shrink=0.5)
    cb.set_label("(mm/h)")
    plt.xlabel("GK4 Easting")
    plt.ylabel("GK4 Northing")
    plt.title("Areal average rain sums")
    plt.draw()
    # Original RADOLAN data
    ax1 = fig.add_subplot(122, aspect="equal")
    pm = plt.pcolormesh(xy[:, :, 0], xy[:, :, 1], np.ma.masked_invalid(data), cmap=mycmap, norm=mynorm)
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
    
    # Reduce grid size using a bounding box (for enhancing performance)
    bbox = inLayer.GetExtent()
    buffer = 5000.
    bbox = dict(left=bbox[0]-buffer, right=bbox[1]+buffer, bottom=bbox[2]-buffer, top=bbox[3]+buffer)
    mask, shape = mask_from_bbox(xy[...,0],xy[...,1], bbox)
    xy_ = np.vstack((xy[...,0][mask].ravel(),xy[...,1][mask].ravel())).T
    data_ = data[mask]
    
    ###########################################################################
    # Approach #1: Assign grid points to each polygon and compute the average.
    # 
    # - Uses matplotlib.path.Path
    # - Each point is weighted equally (assumption: polygon >> grid cell)     
    ###########################################################################

    t1 = dt.datetime.now()
    
    # Create object    
    myobj1 = GridPointsToPoly(xy_, cats, buffer=500.)
    
    t2 = dt.datetime.now()
    
    # Compute averages for target polygons
    avg1 = myobj1( data_.ravel() )

    t3 = dt.datetime.now()

    print "Approach #2 (create object) takes: %f seconds" % (t2 - t1).total_seconds()
    print "Approach #2 (compute average) takes: %f seconds" % (t3 - t2).total_seconds()

    # Plot average rainfall and original data
    testplot(cats, avg1, xy, data)    

    

    ###########################################################################
    # Approach #2: Compute weighted mean based on fraction of source polygons in target polygons
    # 
    # - This is more accurate (no assumptions), but probably slower...
    ###########################################################################

    # First a simple (illustrative) example
    #   Create grid
    x = np.arange(0,6)
    y = np.arange(10,16)
    x, y = np.meshgrid(x,y)
    #   Build vertices from grid center points
    srcs = grid_centers_to_vertices(x,y,1.,1.)
    #   Example polygon vertices
    trgs = np.array([ [[1.7,10.],
                          [3. , 9.8],
                          [3. , 12.],
                          [2. , 12.5],
                          [1.7, 10.]],
                       
                         [[4. , 11.],
                          [5. , 12.],
                          [5  , 14.8],
                          [4. , 15.],
                          [4. , 11.]],

                         [[1. , 13.],
                          [2. , 13.],
                          [2. , 14.],
                          [1. , 14.],
                          [1. , 13.]],

                         [[2.7, 13.7],
                          [3.3, 13.7],
                          [3.3, 14.3],
                          [2.7, 14.3],
                          [2.7, 13.7]] 
                    ])
    
    # Iterate over all target polygons
    intersecs, areas = [], []
    for trg in trgs:
        isecs_ = []
        areas_ = []
        for src in srcs:
            tmp = intersect(src, trg)            
            if not tmp[0]==None:
                isecs_.append(tmp[0])
                areas_.append(tmp[1] )
        intersecs.append(np.array(isecs_))
        areas.append(np.array(areas_))

                     
    # Plot the simple example
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    # Grid cell patches
    grd_patches = [patches.Polygon(item, True) for item in srcs ]
    p = PatchCollection(grd_patches, facecolor="None", edgecolor="black")
    ax.add_collection(p)
    # Target polygon patches
    trg_patches = [patches.Polygon(item, True) for item in trgs ]
    p = PatchCollection(trg_patches, facecolor="None", edgecolor="red", linewidth=2)
    ax.add_collection(p)
    # Intersectin patches
    for isec in intersecs:
        colors = 100*np.linspace(0,1.,len(isec))
        isec_patches = [patches.Polygon(item, True) for item in isec ]
        p = PatchCollection(isec_patches, cmap=plt.cm.jet, alpha=0.5)
        p.set_array(np.array(colors))
        ax.add_collection(p)
    ax.set_xlim(-1,6)
    ax.set_ylim(9,16)
    plt.draw()

    ###########################################################################
    # Approach #2: Real world example
    ###########################################################################

    t1 = dt.datetime.now()

    # Create vertices for each grid cell (MUST BE DONE IN NATIVE RADOLAN COORDINATES)
    grdverts = grid_centers_to_vertices(x_radolan[mask],y_radolan[mask],1.,1.)
    # And reproject to Cartesian reference system (here: GK4)
    grdverts = wradlib.georef.reproject(grdverts,
                                  projection_source=proj_stereo,
                                  projection_target=proj_gk)

    # Create object of type GridCellsToPoly
    myobj2 = GridCellsToPoly(grdverts, cats)

    t2 = dt.datetime.now()

    # Compute averages for target polygons
    avg2 = myobj2( data_.ravel() )

    t3 = dt.datetime.now()

    print "Approach #2 (create object) takes: %f seconds" % (t2 - t1).total_seconds()
    print "Approach #2 (compute average) takes: %f seconds" % (t3 - t2).total_seconds()
    
    # Plot average rainfall and original data
    testplot(cats, avg2, xy, data) 
    

    # Compare estimates
    maxlim = np.max(np.concatenate((avg1, avg2)))
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, aspect="equal")
    plt.scatter(avg1, avg2, edgecolor="None", alpha=0.5)
    plt.xlabel("Average of points in or close to polygon (mm)")
    plt.ylabel("Area-weighted average (mm)")
    plt.xlim(0, maxlim)
    plt.ylim(0, maxlim)
    plt.plot([-1,maxlim+1], [-1,maxlim+1], color="black")


###############################################################################
# DUMP
###############################################################################

#    tstart = dt.datetime.now()    
#    
#    # Assign points to polygons (we need to do this only ONCE) 
#    pips = []  # these are those which we consider inside or close to our polygon
#    for cat in cats:
#        # Pre-selection to increase performance 
#        ixix = points_in_polygon(cat, xy_, buffer=500.)
#        if len(ixix)==0:
#            # For very small catchments: increase buffer size
#            ixix = points_in_polygon(cat, xy_, buffer=1000.)
#        pips.append( ixix )
#    
#    tend = dt.datetime.now()
#    print "Approach #1a (assign points) takes: %f seconds" % (tend - tstart).total_seconds()
#
#
#    ###########################################################################
#    # Approach #1b: Assign grid points to each polygon and compute the average
#    # 
#    # - same as approach #1a, but speed up vai preselecting points using a bbox
#    ###########################################################################
#    tstart = dt.datetime.now()    
#    
#    # Assign points to polygons (we need to do this only ONCE) 
#    pips = []  # these are those which we consider inside or close to our polygon
#    for cat in cats:
#        # Pre-selection to increase performance 
#        ix = subset_points(xy_, get_bbox(cat[:,0],cat[:,1]), buffer=500.)
#        ixix = ix[points_in_polygon(cat, xy_[ix,:], buffer=500.)]
#        if len(ixix)==0:
#            # For very small catchments: increase buffer size
#            ix = subset_points(xy_, get_bbox(cat[:,0],cat[:,1]), buffer=1000.)
#            ixix = ix[points_in_polygon(cat, xy_[ix,:], buffer=1000.)]            
#        pips.append( ixix )
#    
#    tend = dt.datetime.now()
#    print "Approach #1b (assign points) takes: %f seconds" % (tend - tstart).total_seconds()
#       
#    # Plot polygons and grid points
#    fig = plt.figure(figsize=(10,10))
#    ax = fig.add_subplot(111, aspect="equal")
#    wradlib.vis.add_lines(ax, cats, color='black', lw=0.5)
#    plt.scatter(xy[...,0][mask], xy[...,1][mask], c="blue", edgecolor="None", s=4)
#    plt.xlim([bbox["left"]-buffer, bbox["right"]+buffer])
#    plt.ylim([bbox["bottom"]-buffer, bbox["top"]+buffer])
#    # show associated points for some arbitrarily selected polygons
#    for i in xrange(0, len(pips), 15):
#        plt.scatter(xy_[pips[i],0], xy_[pips[i],1], c="red", edgecolor="None", s=8)
#    plt.tight_layout()
#    
#        
#    tstart = dt.datetime.now()    
#    # Now compute the average areal rainfall based on the point assignments
#    avg1 = np.array([])
#    for i, cat in enumerate(cats):
#        if len(pips[i])>0:
#            avg1 = np.append(avg1, np.nanmean(data_.ravel()[pips[i]]) )
#        else:
#            avg1 = np.append(avg1, np.nan )
#
#    tend = dt.datetime.now()
#    print "Approach #1 (average rainfall) takes: %f seconds" % (tend - tstart).total_seconds()
#    
#    # Plot average rainfall and original data
#    testplot(cats, avg1, xy, data)    
#          
#    

    
    # Conversion to ogr Geometries (only once)
#    srcs = np.array([polyg_to_ogr(item) for item in grdverts])
#    trgs = np.array([polyg_to_ogr(item) for item in cats]    )
#    
#    intersecs, areas, ixs = [], [], []
#    for i, trg in enumerate(trgs):
#        # Pre-select grid vertices to increase performance
#        ix = could_intersect(grdverts, cats[i])
#        isecs_ = []
#        areas_ = []
#        for src in srcs[ix]:
#            tmp = intersect(src, trg)
#            areas_.append(tmp[1] )            
#            if not tmp[0]==None:
#                isecs_.append(tmp[0])
#                
#        intersecs.append(np.array(isecs_))
#        areas.append(np.array(areas_))
#        ixs.append(ix)
#        
#    avg2 = np.array([ np.sum(areas[i] * data_.ravel()[ixs[i]] / np.sum(areas[i])) for i in xrange(len(cats)) ])
    
  




    
