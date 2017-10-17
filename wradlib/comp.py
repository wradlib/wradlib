#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Composition
^^^^^^^^^^^

Combine data from different radar locations on one common set of locations

.. autosummary::
   :nosignatures:
   :toctree: generated/

   extract_circle
   togrid
   compose_ko
   compose_weighted

"""
import numpy as np


# from scipy.spatial import KDTree
# def extract_circle(center, radius, coords):
#     """
#     Extract the indices of coords which fall within a circle
#     defined by center and radius
#
#     Parameters
#     ----------
#     center : float
#     radius : float
#     coords : array of float with shape (numpoints,2)
#
#     Returns
#     -------
#     output : 1-darray of integers
#        index array referring to the coords array
#
#     """
#     print 'Building tree takes:'
#     t0 = dt.datetime.now()
#     tree = KDTree(coords)
#     print dt.datetime.now() - t0
#     print 'Query tree takes:'
#     t0 = dt.datetime.now()
#     ix = tree.query(center, k=len(coords), distance_upper_bound=radius)[1]
#     print dt.datetime.now() - t0
#     ix = ix[np.where(ix<len(coords))[0]]
#     return ix


def extract_circle(center, radius, coords):
    """
    Extract the indices of coords which fall within a circle
    defined by center and radius

    Parameters
    ----------
    center : float
    radius : float
    coords : array of float with shape (numpoints,2)

    Returns
    -------
    output : 1-darray of integers
        index array referring to the coords array

    """
    return np.where(((coords - center) ** 2).sum(axis=-1) < radius ** 2)[0]


def togrid(src, trg, radius, center, data, interpol, *args, **kwargs):
    """
    Interpolate data from a radar location to the composite grid or set of
    locations

    Parameters
    ----------
    src : ndarray of float of shape (numpoints, ndim)
        cartesian x / y coordinates of the radar bins
    trg : ndarray of float of shape (numpoints, ndim)
        cartesian x / y coordinates of the composite
    radius : float
        the radius of the radar circle (same units as src and trg)
    center : array of float
        the location coordinates of the radar
    data : ndarray of float
        the data that should be transferred to composite
    interpol : an interpolation class name from :meth:`wradlib.ipol`
        e.g. :class:`~wradlib.ipol.Nearest` or :class:`~wradlib.ipol.Idw`

    Other Parameters
    ----------------
    *args : arguments of Interpolator (see class documentation)

    Keyword Arguments
    -----------------
    **kwargs : keyword arguments of Interpolator (see class documentation)

    Returns
    -------
    output : ndarray of float
        data of the radar circle which is interpolated on the composite grid

    Examples
    --------

    See :ref:`notebooks/basics/wradlib_workflow.ipynb#Gridding`.

    """
    # get indices to select the subgrid from the composite grid
    ix = extract_circle(center, radius, trg)
    # interpolate on subgrid
    ip = interpol(src, trg[ix], *args, **kwargs)
    data_on_subgrid = ip(data).reshape((len(ix)))
    # create container for entire grid
    composegridshape = [len(trg)]
    composegridshape.extend(data.shape[1:])
    compose_grid = np.repeat(np.nan, len(trg) *
                             np.prod(data.shape[1:])).reshape(composegridshape)
    # push subgrid results into the large grid
    compose_grid[ix] = data_on_subgrid
    return compose_grid


def compose_ko(radargrids, qualitygrids):
    """Composes grids according to quality information using quality \
    information as a knockout criterion.

    The value of the composed pixel is taken from the radargrid whose
    quality grid has the highest value.

    Parameters
    ----------
    radargrids : list of arrays
        radar data to be composited. Each item in the list corresponds to the
        data of one radar location. All items must have the same shape.
    qualitygrids : list of arrays
        quality data to decide upon which radar site will contribute its pixel
        to the composite. Then length of this list must be the same as that
        of `radargrids`. All items must have the same shape and be aligned with
        the items in `radargrids`.


    Returns
    -------
    composite : array

    """
    # first add a fallback array for all pixels having missing values in all
    # radargrids
    radarfallback = (np.repeat(np.nan, np.prod(radargrids[0].shape))
                     .reshape(radargrids[0].shape))
    radargrids.append(radarfallback)
    radarinfo = np.array(radargrids)
    # then do the same for the quality grids
    qualityfallback = (np.repeat(-np.inf, np.prod(radargrids[0].shape))
                       .reshape(radargrids[0].shape))
    qualitygrids.append(qualityfallback)
    qualityinfo = np.array(qualitygrids)

    select = np.nanargmax(qualityinfo, axis=0)
    composite = (radarinfo.reshape((radarinfo.shape[0], -1))
                 [select.ravel(), np.arange(np.prod(radarinfo.shape[1:]))]
                 .reshape(radarinfo.shape[1:]))
    radargrids.pop()
    qualitygrids.pop()

    return composite


def compose_weighted(radargrids, qualitygrids):
    """Composes grids according to quality information using a weighted \
    averaging approach.

    The value of the composed pixel is the weighted average of all radar
    pixels with the quality values being the weights.

    Parameters
    ----------
    radargrids : list of arrays
    qualitygrids : list of arrays

    Returns
    -------
    composite : array

    Examples
    --------

    See :ref:`notebooks/workflow/recipe1.ipynb`.

    See Also
    --------
    compose_ko : for more description about the shape of the input arrays

    """
    radarinfo = np.array(radargrids)
    qualityinfo = np.array(qualitygrids)
    # overall nanmask
    nanmask = np.all(np.isnan(radarinfo), axis=0)
    # quality grids must contain values only where radarinfo does
    qualityinfo[np.isnan(radarinfo)] = np.nan

    qualityinfo /= np.nansum(qualityinfo, axis=0)

    composite = np.nansum(radarinfo * qualityinfo, axis=0)
    composite[nanmask] = np.nan

    return composite


if __name__ == '__main__':
    print('wradlib: Calling module <comp> as main...')
