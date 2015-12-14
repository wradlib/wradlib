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
from scipy.spatial import cKDTree, Delaunay


class ZonalStatsBase():
    """Base class for all 2-dimensional zonal statistics.

    .. versionadded:: 0.7.0

    The base class for computing 2-dimensional zonal statistics for target
    polygons from source points or polygons. Provides the basic design
    for all other classes.
    
    If no source points or polygons can be associated to a target polygon (e.g.
    no intersection), the zonal statistic for that target will be NaN.

    Parameters
    ----------
    src : sequence of source points or polygons
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """

    #def __init__(self, src, trg=None, idx=None, weights=None, **kwargs):
    def __init__(self, src, trg=None, ix=None, w=None, **kwargs):
        self.src = self._check_src(src, **kwargs)
        self.test = None
        self._ix = []
        self._w = []

        if trg is not None:
            for i, item in enumerate(trg):
                self.add_target(item, **kwargs)
        else:
            if ix is not None:
                if w is not None:
                    for _ix, _w in zip(ix,w):
                        self.add_idx_weights(_ix, _w)
                else:
                    print("ix and w are complementary parameters and must both be given")
                    raise TypeError

    def add_target(self, trg, **kwargs):
        ix, w = self.get_weights(trg, **kwargs)
        self.add_idx_weights(ix, w)

    def add_idx_weights(self, ix, w):
        self.ix = self.ix + [ix]
        self.w = self.w + [w]

    @property
    def ix(self):
        return self._ix

    @ix.setter
    def ix(self, value):
        self._ix = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    def get_weights(self, trg, **kwargs):
        """This is the key method that needs to be filled for any inheriting class.
        """
        pass

    def check_empty(self):
        """
        """
        isempty = np.repeat(False, len(self.w))
        for i, weights in enumerate(self.w):
            if np.sum(weights)==0 or np.isnan(np.sum(weights)):
                isempty[i] = True
        return isempty
        
    def _check_src(self, src, **kwargs):
        """TODO Basic check of source elements (sequence of points or polygons).

        """
        return src

    def _check_trg(self, trg):
        """TODO Basic check of target elements (sequence of polygons).

        """
        return np.array(trg)
    def _check_vals(self, vals):
        """TODO Basic check of target elements (sequence of polygons).

        """
        assert len(vals)==len(self.src), "Argment vals must be of length %d" % len(self.src)
        return vals

    def mean(self, vals):
        """
        Evaluate (weighted) zonal mean for values given at the source points.

        Parameters
        ----------
        vals : 1-d ndarray of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        self.isempty = self.check_empty()
        out = np.zeros(len(self.ix))*np.nan
        out[~self.isempty] =  np.array( [np.average( vals[self.ix[i]], weights=self.w[i] ) \
                                        for i in np.arange(len(self.ix))[~self.isempty]] )
        return out
            

    def var(self, vals):
        """
        Evaluate (weighted) zonal variance for values given at the source points.

        Parameters
        ----------
        vals : 1-d ndarray of type float with the same length as self.src
            Values at the source element for which to compute zonal statistics

        """
        self._check_vals(vals)
        mean = self.mean(vals)
        out = np.zeros(len(self.ix))*np.nan
        out[~self.isempty] = np.array( [np.average( (vals[self.ix[i]] - mean[i])**2, weights=self.w[i]) \
                                       for i in np.arange(len(self.ix))[~self.isempty]] )
        return out


class PolarGridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points or polygons
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """
    def _check_src(self, src, **kwargs):
        """TODO Basic check of source elements (sequence of points or polygons).

        """
        src = np.array(src)
        self.shape = kwargs.get('shape', src.shape)
        # Test
        #self.tree = cKDTree(src[:,0:4,:].reshape((-1,2), order='F'))
        # reshaping
        self.ogr_srcs = np.array([polyg_to_ogr(item) for item in src.reshape((-1,) + src.shape[-2:],)])
        self.ogr_srcs_area = np.array([item.Area() for item in self.ogr_srcs])
        return src

    def get_weights(self, trg, **kwargs):
        """
        """

        ogr_trg = polyg_to_ogr(trg)

        # precalculate points within convex hull to speed things up
        # uses scipy.spatial.Delaunay, seemes faster than path-method
        hull = Delaunay(trg)

        # just check two polygon points of source array
        simplex = hull.find_simplex(self.src[...,0:2,:])

        # set shape according source shape
        simplex.shape = self.shape[0:2] + (2,)

        pip_hull = simplex >= 0

        # stack the 4 associated src polygon points together
        pip_hull = np.dstack((pip_hull, np.roll(pip_hull, -1, axis=0)))

        # flatten all but last dimensions
        pip_hull = pip_hull.reshape((-1,4))

        # Test
        # find possible neighbours which do intersect but don't have points inside
        #dn, ixn = self.tree.query(trg, k=2)
        #uind = np.unravel_index(np.unique(np.squeeze(np.array([ixn]))), pip_hull.shape)
        #pip_hull[uind] = True

        # get indices from `any` source polygon points contained
        # in target polygon
        ix0_ = np.where(np.any(pip_hull, axis=1) == True)[0]

        # checks if all 4 source polygon points inside target polygon
        #ix2_ = np.where(np.all(pip_hull, axis=1) == True)[0]

        # checks if src polygon is fully contained within target polygon
        # slower, but more precise
        ix2_ = ix0_[np.array([ogr_src.Within(ogr_trg) for ogr_src in self.ogr_srcs[ix0_]])]

        # get indices of source polygons which are not fully contained
        # in target polygon
        ix1_ = np.setdiff1d(ix0_, ix2_, assume_unique=True)

        # calculate intersection area of not fully contained source polygons
        # here we could also get the interscetion-poly vertexes, leave that for later
        areas = np.array([intersect(ogr_src, ogr_trg)[1] for ogr_src in self.ogr_srcs[ix1_]])

        # fetch precalculated areas and append
        areas = np.append(areas, self.ogr_srcs_area[ix2_])

        w = areas / np.sum(areas)
        ix = np.append(ix1_, ix2_)

        return ix, w

    def _get_intersection(self, trg, **kwargs):
        """Just a toy function if you want to inspect the intersection polygons of a specific target.
        """
        ogr_trg  = polyg_to_ogr(trg)
        ix = could_intersect(self.src, trg)
        intersecs = []
        for ogr_src in self.ogr_srcs[ix]:
            tmp = intersect(ogr_src, ogr_trg)[0]
            if not tmp==None:
                intersecs.append( tmp )
        return intersecs

    def _get_intersection_by_idx(self, trg, idx, **kwargs):
        """Just a toy function if you want to inspect the intersection polygons of a specific target.
        """
        ogr_trg  = polyg_to_ogr(trg)
        ix = self.ix[idx]
        intersecs = []
        for ogr_src in self.ogr_srcs[ix]:
            tmp = intersect(ogr_src, ogr_trg)[0]
            if not tmp==None:
                intersecs.append( tmp )
        return intersecs


class GridCellsToPoly(ZonalStatsBase):
    """Compute weighted average for target polygons based on areal weights.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    src : sequence of source points or polygons
    trg : sequence of target polygons - each item is an ndarray of shape (num vertices, 2)

    """
    def _check_src(self, src):
        """TODO Basic check of source elements (sequence of points or polygons).

        """
        src = np.array(src)
        self.ogr_srcs = np.array([polyg_to_ogr(item) for item in src])
        self.ogr_srcs_area = np.array([item.Area() for item in self.ogr_srcs])
        return src

    def get_weights(self, trg, **kwargs):
        """
        """
        ogr_trgs = polyg_to_ogr(trg)

        ix_ = could_intersect(self.src, trg)
        areas = np.array([intersect(ogr_src, ogr_trgs)[1] for ogr_src in self.ogr_srcs[ix_]])
        w = areas / np.sum(areas)

        return ix_, w

    def _get_intersection(self, trg, **kwargs):
        """Just a toy function if you want to inspect the intersection polygons of a specific target.
        """
        ogr_trg  = polyg_to_ogr(trg)
        ix = could_intersect(self.src, trg)
        intersecs = []
        for ogr_src in self.ogr_srcs[ix]:
            tmp = intersect(ogr_src, ogr_trg)[0]
            if not tmp==None:
                intersecs.append( tmp )
        return intersecs

    def _get_intersection_by_idx(self, trg, idx, **kwargs):
        """Just a toy function if you want to inspect the intersection polygons of a specific target.
        """
        ogr_trg  = polyg_to_ogr(trg)
        ix = self.ix[idx]
        intersecs = []
        for ogr_src in self.ogr_srcs[ix]:
            tmp = intersect(ogr_src, ogr_trg)[0]
            if not tmp==None:
                intersecs.append( tmp )
        return intersecs


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
    def get_weights(self, trg, **kwargs):
        """
        """
        # Pre-selection to increase performance
        ix_ = self.get_points_in_target(trg, **kwargs)
        if len(ix_)==0:
            # No points in target polygon? Find the closest point to provide a value
            ix_ = self.get_point_next_to_target(trg, **kwargs)
        w = np.ones(len(ix_)) / len(ix_ )
        return ix_, w

    def get_points_in_target(self, trg, **kwargs):
        """Helper method that can also be used to return intermediary results.
        """
        buffer = kwargs.get('buffer', 0.)
        polar = kwargs.get('polar', False)
        if polar:
            ix2 = np.where(points_in_polygon(trg, self.src, buffer=buffer))[0]
        else:
            # Pre-selection to increase performance
            ix1 = subset_points(self.src, get_bbox(trg[:,0],trg[:,1]), buffer=buffer)
            ix2 = ix1[points_in_polygon(trg, self.src[ix1,:], buffer=buffer)]
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
    out : a nested list of polygon vertices of shape (num vertices, 2)

    """
    jsonobj = eval(ogrobj.ExportToJson())

    return jsonobj['coordinates']


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
    if isec.GetGeometryName() in ["POLYGON", "MULTIPOLYGON"]:
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


def mask_from_bbox(x, y, bbox, polar=False):
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
    polar : x, y are aligned polar (azimuth x range)

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
    ill = (ixll / nx)-1
    jll = (ixll % nx)-1
    # find upper right corner index
    dists, ixur = tree.query([bbox["right"],bbox["top"]], k=1)
    iur = (ixur / nx)+1
    jur = (ixur % nx)+1

    # for polar grids we need all 4 corners
    if polar:
        # find upper left corner index
        dists, ixul = tree.query([bbox["left"], bbox["top"]], k=1)
        iul = (ixul / nx)-1
        jul = (ixul % nx)-1
        # find lower right corner index
        dists, ixlr = tree.query([bbox["right"],bbox["bottom"]], k=1)
        ilr = (ixlr / nx)+1
        jlr = (ixlr % nx)+1

    mask = np.repeat(False, ix.size).reshape(ix.shape)

    # for polar grids we have to handle the azimuth carefully
    if polar:
        # ranges are not problematic, just get min and max
        jmin = min(jll, jul, jur, jlr)
        jmax = max(jll, jul, jur, jlr)

        # azimuth array for angle_between calculation
        ax = np.array([[ill, ilr],
                       [ill, iur],
                       [iul, ilr],
                       [iul, iur]])

        # this calculates the angles between 4 azimuth and returns indices
        # of the greatest angle
        ar = angle_between(ax[:,0], ax[:,1])
        maxind = np.argmax(ar)
        imin, imax = ax[maxind,:]

        # if catchment extends over zero angle
        if imin > imax:
            mask[:imax, jmin:jmax] = True
            mask[imin:, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax-jmin)
        else:
            mask[imin:imax, jmin:jmax] = True
            shape = (int(ar[maxind]), jmax-jmin)

    else:

        if iur>ill:
            mask[ill:iur,jll:jur] = True
            shape = (iur-ill, jur-jll)
        else:
            mask[iur:ill,jll:jur] = True
            shape = (ill-iur, jur-jll)

    return mask, shape


def angle_between(source_angle, target_angle):
    """Return angle between source and target radial angle
    """
    sin1 = np.sin(np.radians(target_angle)-np.radians(source_angle))
    cos1 = np.cos(np.radians(target_angle)-np.radians(source_angle))
    return np.rad2deg(np.arctan2(sin1, cos1))


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
