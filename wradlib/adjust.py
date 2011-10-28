#-------------------------------------------------------------------------------
# Name:         adjust
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Adjustment
^^^^^^^^^^

Adjusting remotely sensed spatial data by ground truth (gage observations)

The main objective of this module is the adjustment of radar-based QPE
by rain gage observations. However, this module can also be applied to adjust
satellite rainfall by rain gage observations, remotely sensed soil moisture
patterns by ground truthing moisture sensors or any spatial point pattern
which ought to be adjusted by selcted point measurements.

Basically, we only need two data sources:

- point observations (e.g. rain gage observations)

- set of (potentially irregular) unadjusted point values (e.g. remotely sensed rainfall)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   AdjustAdd
   Raw_at_obs

"""

# site packages
import numpy as np
from scipy.spatial import cKDTree

# wradlib modules
import wradlib.ipol as ipol


class AdjustBase(ipol.IpolBase):
    """
    The basic adjustment class

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.

    """
    def __init__(self, obs_coords, raw_coords, nnear_raws=9, stat='median', nnear_idw=6, p_idw=2.):
        self.obs_coords = self._make_coord_arrays(obs_coords)
        self.raw_coords = self._make_coord_arrays(raw_coords)
        self.get_raw_at_obs = Raw_at_obs(obs_coords, raw_coords, nnear=nnear_raws, stat=stat)
        self.ip = ipol.Idw(src=obs_coords, trg=raw_coords, nnearest=nnear_idw, p=p_idw)
    def _check_shape(self, obs, raw):
        """
        Check consistency of the input data obs and raw with the shapes of the coordinates
        """
        print 'TODO WARNING: fill in _check_shape method'

class AdjustAdd(AdjustBase):
    """
    Gage adjustment using an additive error model

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear_raws : integer
        defaults to 9
    stat : string
        defaults to 'median'
    nnear_idw : integer
        defaults to 6
    p_idw : float
        defaults to 2.

    Notes
    -----
    Inherits from AdjustBase

    """

    def __call__(self, obs, raw):
        """
        Return the field of raw values adjusted by obs

        Parameters
        ----------
        obs : array of float
            observations
        raw : array of float
            raw unadjusted field

        """
        # checking input shape consistency
        self._check_shape(obs, raw)
        # computing the error
        error = obs - self.get_raw_at_obs(raw)
        # interpolate error field
        error = self.ip(error)
        # add error field to raw and cut negatives to zero
        return np.where( (raw + error)<0., 0., raw + error)



class Raw_at_obs():
    """
    Get the raw values in the neighbourhood of the observation points

    Parameters
    ----------
    obs_coords : array of float
        coordinate pairs of observations points
    raw_coords : array of float
        coordinate pairs of raw (unadjusted) field
    nnear: integer
        number of neighbours which should be considered in the vicinity of each point in obs
    stat: string
        function name
    """
    def __init__(self, obs_coords, raw_coords, nnear=9, stat='median'):
        self.statfunc = _get_statfunc(stat)
        self.raw_ix = _get_neighbours_ix(obs_coords, raw_coords, nnear)

    def __call__(self, raw, obs=None):
        """
        Returns the values of raw at the observation locations

        Parameters
        ----------
        raw : array of float
            raw values

        """
        # get the values of the raw neighbours of obs
        raw_neighbs = raw[self.raw_ix]
        # and summarize the values of these neighbours by using a statistics option
        return self.statfunc(raw_neighbs)


def get_raw_at_obs(obs_coords, raw_coords, obs, raw, nnear=9, stat='median'):
    """
    Get the raw values in the neighbourhood of the observation points

    Parameters
    ----------

    obs_coords :

    raw: Datset of raw values (which shall be adjusted by obs)
    nnear: number of neighbours which should be considered in the vicinity of each point in obs
    stat: a numpy statistical function which should be used to summarize the values of raw in the neighbourshood of obs
    """
    # get the values of the raw neighbours of obs
    raw_neighbs = _get_neighbours(obs_coords, raw_coords, raw, nnear)
    # and summarize the values of these neighbours by using a statistics option
    return _get_statfunc(stat)(raw_neighbs)


def _get_neighbours_ix(obs_coords, raw_coords, nnear):
    """
    Returns <nnear> neighbour indices per <obs_coords> coordinate pair

    Parameters
    ----------
    obs_coords : array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    nnear : integer
        number of neighbours to be selected per coordinate pair of obs_coords

    """
    # plant a tree
    tree = cKDTree(raw_coords)
    # return nearest neighbour indices
    return tree.query(obs_coords, k=nnear)[1]



def _get_neighbours(obs_coords, raw_coords, raw, nnear):
    """
    Returns <nnear> neighbour values per <obs_coords> coordinate pair

    Parameters
    ----------
    obs_coords : array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    raw : array of float of shape (num_points,...)
        this is the data corresponding to the coordinate pairs raw_coords
    nnear : integer
        number of neighbours to be selected per coordinate pair of obs_coords

    """
    # plant a tree
    tree = cKDTree(raw_coords)
    # retrieve nearest neighbour indices
    ix = tree.query(obs_coords, k=nnear)[1]
    # return the values of the nearest neighbours
    return raw[ix]

def _get_statfunc(funcname):
    """
    Returns a function that corresponds to parameter <funcname>

    Parameters
    ----------
    funcname : string
        a name of a numpy function OR another option known by _get_statfunc
        Potential options: 'mean', 'median', 'best'

    """
    try:
        # first try to find a numpy function which corresponds to <funcname>
        func = getattr(np,funcname)
        def newfunc(x):
            return func(x, axis=1)
    except:
        try:
            # then try to find a function in this module with name funcname
            if funcname=='best':
                newfunc=best
        except:
            # if no function can be found, raise an Exception
            raise NameError('Unkown function name option: '+funcname)
    return newfunc


def best(x, y):
    """
    Find the values of y which corresponds best to x

    If x is an array, the comparison is carried out for each element of x

    Parameters
    ----------
    x : float or 1-d array of float
    y : array of float

    Returns
    -------
    output : 1-d array of float with length len(y)

    """
    if type(x)==np.ndarray:
        assert x.ndim==1, 'x must be a 1-d array of floats or a float.'
        assert len(x)==len(y), 'Length of x and y must be equal.'
    if type(y)==np.ndarray:
        assert y.ndim<=2, 'y must be 1-d or 2-d array of floats.'
    else:
        raise ValueError('y must be 1-d or 2-d array of floats.')
    x = np.array(x).reshape((-1,1))
    if y.ndim==1:
        y = np.array(y).reshape((1,-1))
        axis = None
    else:
        axis = 0
    return y[np.argmin(np.abs(x-y), axis=axis)]




if __name__ == '__main__':
##    print 'wradlib: Calling module <adjust> as main...'
##    x = np.array([1., 5., 10.])
##    x=10.
##    y = np.array([1., 10., 40.])
##    print best(x,y)
    num_raw = 100
    raw_coords = np.meshgrid(np.linspace(0,100,num_raw), np.linspace(0,100,num_raw))
    raw_coords = np.vstack((raw_coords[0].ravel(), raw_coords[1].ravel())).transpose()
    raw = np.abs(np.sin(0.1*raw_coords).sum(axis=1))
    obs_ix = np.random.uniform(low=0, high=num_raw**2, size=50).astype('i4')
    obs_coords = raw_coords[obs_ix]
    obs = raw[obs_ix]+np.random.uniform(low=-1., high=1, size=len(obs_ix))
    obs = np.abs(obs)

    # adjustment
    adjuster = AdjustAdd(obs_coords, raw_coords, stat='mean', p_idw=2.)
    result = adjuster(obs, raw)

    import pylab as pl
    maxval = np.max(np.concatenate((raw, obs, result)).ravel())
    fig = pl.figure()
    # unadjusted
    ax = fig.add_subplot(221, aspect='equal')
    raw_plot = ax.scatter(raw_coords[:,0], raw_coords[:,1], c=raw, vmin=0, vmax=maxval, edgecolor='none')
    ax.scatter(obs_coords[:,0], obs_coords[:,1], c=obs.ravel(), marker='s', s=50, vmin=0, vmax=maxval)
    pl.colorbar(raw_plot)
    pl.title('Raw field and observations')
    # adjusted
    ax = fig.add_subplot(222, aspect='equal')
    raw_plot = ax.scatter(raw_coords[:,0], raw_coords[:,1], c=result, vmin=0, vmax=maxval, edgecolor='none')
#    ax.scatter(obs_coords[:,0], obs_coords[:,1], c=obs.ravel(), marker='s', s=50, vmin=0, vmax=maxval)
    pl.colorbar(raw_plot)
    pl.title('Adjusted field and observations')
    # scatter
    ax = fig.add_subplot(223, aspect='equal')
    ax.scatter(obs, raw[obs_ix])
    ax.plot([0,maxval],[0,maxval],'-', color='grey')
    pl.title('Scatter plot raw vs. obs')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # scatter
    ax = fig.add_subplot(224, aspect='equal')
    ax.scatter(obs, result[obs_ix])
    ax.plot([0,maxval],[0,maxval],'-', color='grey')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    pl.title('Scatter plot adjusted vs. obs')


    pl.show()
    pl.close()


