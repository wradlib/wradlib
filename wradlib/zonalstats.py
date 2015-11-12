#-------------------------------------------------------------------------------
# Name:        zonalstats
# Purpose:
#
# Author:      Maik Heistermann, Kai Muehlbauer
#
# Created:     12.11.2015
# Copyright:   (c) Maik Heistermann, Kai Muehlbauer 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Zonal Statistics
^^^^^^^^^^^^^^^^

.. versionadded:: 0.7.0

This module supports you in computing statistics over spatial zones. A typical
application would be to compute mean areal precipitation for a catchment by using 
precipitation estimates from a radar grid in polar coordinates, or from precipitation
estimates in a Cartesian grid.

The general usage is similar to the ipol and adjustment modules: You have to 
create an instance of a class by using the spatial information of your source and
target objects (e.g. radar bins and catchment polygons). You can then compute
zonal statistics for your target objects by calling the instance with an array of
values (one for each source object). Typically, creating the instance will be
computationally expensive, but only has to be done once (as long as the geometries
do not change). Calling the objects with actual data, however, will be very fast.

..note:: Right now we only support a limited set of 2-dimensional zonal statistics. 
         In the future, we plan to extend this to three dimensions.


.. currentmodule:: wradlib.zonalstats

.. autosummary::
   :nosignatures:
   :toctree: generated/

   GridCellsToPoly
   GridPointsToPoly
   
"""

from osgeo import ogr
from matplotlib.path import Path
import numpy as np
from scipy.spatial import cKDTree


class ZonalStatsBase():
    """Base class for all 2-dimensional zonal statistics.
    
    .. versionadded:: 0.7.0

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
    def get_weights(self, **kwargs):
        """This is the key method that needs to be filled for any inheriting class.
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
        

class GridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.
    
    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points or polygons 
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """
    def get_weights(self, **kwargs):
        """
        """
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
            

class GridPointsToPoly(ZonalStatsBase):
    """Compute zonal average from all points in or close to the target polygon.
    
    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points or polygons 
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)
    
    Keyword arguments
    -----------------
    buffer : float (same unit as coordiantes)
             Points will be considered inside the target if they are contained in the buffer.

    """
    def get_weights(self, **kwargs):
        """
        """
        ix, w = [], []
        for trg in self.trg:
            # Pre-selection to increase performance
            ix_ = self.get_points_in_target(trg, **kwargs)
            if len(ix_)==0:
                # No points in target polygon? Find the closest point to provide a value
                ix_ = self.get_point_next_to_target(trg, **kwargs)
            w.append( np.ones(len(ix_)) / len(ix_ ) )
            ix.append(ix_)        
        return ix, w
    def get_points_in_target(self, trg, **kwargs):
        """Helper method that can also be used to return intermediary results.
        """
        # Pre-selection to increase performance 
        ix1 = subset_points(self.src, get_bbox(trg[:,0],trg[:,1]), buffer=kwargs["buffer"])
        ix2 = ix1[points_in_polygon(trg, self.src[ix1,:], buffer=kwargs["buffer"])]
        return ix2
    def get_point_next_to_target(self, trg, **kwargs):
        """Computes the target centroid and finds the closest point from src.
        """
        centroid = get_centroid(trg)
        tree = cKDTree(self.src)
        distnext, ixnext = tree.query([centroid[0], centroid[1]], k=1)
        return np.array([ixnext])


def polyg_to_ogr(vert):
    """Convert a polygon vertex to gdal/ogr polygon geometry.
    
    .. versionadded:: 0.7.0
    
    Using JSON as a vehicle to efficiently deal with numpy arrays.
    
    Parameters
    ----------
    vert : a numpy array of polygon vertices of shape (num vertices, 2)
    
    Returns
    -------
    out : an ogr Geometry object of type POLYGON
    
    """
    str = {"type":"Polygon", "coordinates":[vert.tolist()]}.__repr__()
    
    return ogr.CreateGeometryFromJson(str)


def ogr_to_polyg(ogrobj):
    """Backconvert a gdal/ogr Polygon geometry to a numpy vertex array.
    
    .. versionadded:: 0.7.0
    
    Using JSON as a vehicle to efficiently deal with numpy arrays.
    
    Parameters
    ----------
    ogrobsj : an ogr Geometry object of type POLYGON
    
    Returns
    -------
    out : a numpy array of polygon vertices of shape (num vertices, 2)
    
    """
    jsonobj = eval(ogrobj.ExportToJson())
    return np.array(jsonobj["coordinates"][0])


def intersect(src, trg):
    """Return intersection and its area from target and source vertex.
    
    .. versionadded:: 0.7.0
    
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
    
    .. versionadded:: 0.7.0
    
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


def mask_from_bbox(x, y, bbox):
    """Return 2-d index array based on spatial selection from a bounding box.
    
    Use this function to create a 2-d boolean mask from 2-d arrays of grids points.
    
    .. versionadded:: 0.7.0
    
    Parameters
    ----------
    x : nd array of shape (num rows, num columns)
        x (Cartesian) coordinates
    y : nd array of shape (num rows, num columns)
        y (Cartesian) coordinates
    bbox : dictionary with keys "left", "right", "bottom", "top" 
        These must refer to the same Cartesian reference system as x and y
        
    Returns
    -------
    out : mask, shape
          mask is a boolean array that is True if the point is inside the bbox
          shape is the shape of the True subgrid
    
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
    """Select points inside or close to polygon.
    
    .. versionadded:: 0.7.0
    
    Parameters
    ----------
    polygon : ndarray of polygon vertices of shape (num vertices, 2)
    points : nd array of point coordinates of shape (num points, 2)
    buffer : neighbourhood around polygon borders in which a point will be considered as inside the polygon.
    
    Returns
    -------
    out : index array indicating the points that are located inside the polygon.
    
    """
    mpath = Path( polygon )
    return  mpath.contains_points(points, radius=-buffer)


def subset_points(pts, bbox, buffer=0.):
    """Subset a large set of points by polygon bbox.
    
    .. versionadded:: 0.7.0
    
    Parameters
    ----------
    pts : ndarray of points of shape (num points, 2)
    bbox : dictionary with keys "left", "right", "bottom", "top" 
        These must refer to the same Cartesian reference system as x and y
    buffer : extends the bbox in all directions, default value = 0

    Returns
    -------
    out : index array indicating the points that are located inside the bbox.

    
    """
    x = pts[:,0]
    y = pts[:,1]
    return np.where(
            (x >= bbox["left"]  -buffer) & \
            (x <= bbox["right"] +buffer) & \
            (y >= bbox["bottom"]-buffer) & \
            (y <= bbox["top"]   +buffer) )[0]
        

def get_bbox(x, y):
    """Return bbox dictionary that represents the extent of the points.
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
    
    Returns
    -------
    out : x and y coordinate of the centroid
    
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


if __name__ == '__main__':
    print 'wradlib: Calling module <zonalstats> as main...'
