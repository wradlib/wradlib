#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Composition
^^^^^^^^^^^

Combine data from different radar locations on one common set of locations

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["extract_circle", "togrid", "compose_ko", "compose_weighted"]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def extract_circle(center, radius, coords):
    """Extract the indices of ``coords`` which fall within a circle \
    defined by ``center`` and ``radius``.

    Parameters
    ----------
    center : float
    radius : float
    coords : :class:`numpy:numpy.ndarray`
        array of float with shape (numpoints, 2)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        1-darray of integers, index array referring to the ``coords`` array

    """
    return np.where(((coords - center) ** 2).sum(axis=-1) < radius ** 2)[0]


def togrid(src, trg, radius, center, data, interpol, *args, **kwargs):
    """Interpolate data from a radar location to the composite grid or set of \
    locations

    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        array of float of shape (numpoints, ndim),
        cartesian x / y coordinates of the radar bins
    trg : :class:`numpy:numpy.ndarray`
        array of float of shape (numpoints, ndim),
        cartesian x / y coordinates of the composite
    radius : float
        the radius of the radar circle (same units as src and trg)
    center : :class:`numpy:numpy.ndarray`
        array of float, the location coordinates of the radar
    data : :class:`numpy:numpy.ndarray`
        array of float, the data that should be transferred to composite
    interpol : object
        an interpolation class name from :mod:`wradlib.ipol`
        e.g. :class:`~wradlib.ipol.Nearest` or :class:`~wradlib.ipol.Idw`

    Other Parameters
    ----------------
    *args : arguments of Interpolator (see class documentation)

    Keyword Arguments
    -----------------
    **kwargs : keyword arguments of Interpolator (see class documentation)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        array of float, data of the radar circle which is interpolated on
        the composite grid

    Note
    ----
    Keyword arguments to be used while calling the interpolator can be issued as
    `call_kwargs`, eg. togrid(..., call_kwargs=dict(maxdist=10))

    Examples
    --------

    See :ref:`/notebooks/basics/wradlib_workflow.ipynb#Gridding`.

    """
    # get indices to select the subgrid from the composite grid
    ix = extract_circle(center, radius, trg)
    call_kwargs = kwargs.pop("call_kwargs", {})
    # interpolate on subgrid
    ip = interpol(src, trg[ix], *args, **kwargs)
    data_on_subgrid = ip(data, **call_kwargs).reshape((len(ix)))
    # create container for entire grid
    composegridshape = [len(trg)]
    composegridshape.extend(data.shape[1:])
    compose_grid = np.repeat(np.nan, len(trg) * np.prod(data.shape[1:])).reshape(
        composegridshape
    )
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
    composite : :class:`numpy:numpy.ndarray`

    """
    # first add a fallback array for all pixels having missing values in all
    # radargrids
    radarfallback = np.repeat(np.nan, np.prod(radargrids[0].shape)).reshape(
        radargrids[0].shape
    )
    radargrids.append(radarfallback)
    radarinfo = np.array(radargrids)
    # then do the same for the quality grids
    qualityfallback = np.repeat(-np.inf, np.prod(radargrids[0].shape)).reshape(
        radargrids[0].shape
    )
    qualitygrids.append(qualityfallback)
    qualityinfo = np.array(qualitygrids)

    select = np.nanargmax(qualityinfo, axis=0)
    composite = radarinfo.reshape((radarinfo.shape[0], -1))[
        select.ravel(), np.arange(np.prod(radarinfo.shape[1:]))
    ].reshape(radarinfo.shape[1:])
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
    composite : :class:`numpy:numpy.ndarray`

    Examples
    --------

    See :ref:`/notebooks/workflow/recipe1.ipynb`.

    See Also
    --------
    :func:`~wradlib.comp.compose_ko` : for more description about the shape of
    the input arrays

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


if __name__ == "__main__":
    print("wradlib: Calling module <comp> as main...")
