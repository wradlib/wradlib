#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Vertical Profile of Reflectivity (VPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Precipitation is 3-dimensional in space. The vertical distribution of
precipitation (and thus reflectivity) is typically non-uniform. As the height
of the radar beam increases with the distance from the radar location
(beam elevation, earth curvature), one sweep samples from different heights.
The effects of the non-uniform VPR and the different sampling heights need to
be accounted for if we are interested in the precipitation near the ground or
in defined heights. This module is intended to provide a set of tools to
account for these effects.

The first step will normally be to reference the polar volume data in a
3-dimensional Cartesian coordinate system. The three dimensional Cartesian
coordinates of the original polar volume data can be computed using
:meth:`wradlib.vpr.volcoords_from_polar`.

Then, we can create regular 3-D grids in order to analyse the vertical profile
of reflectivity or rainfall intensity. For some applications you might want
to create so-called `Constant Altitude Plan Position Indicators (CAPPI)
<https://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator>`_
in order to make radar observations at different distances from the radar more
comparable. Basically, a CAPPI is simply one slice out of a 3-D volume grid.
Analoguous, we will refer to the elements in a three dimensional Cartesian grid
as *voxels*. In wradlib, you can create
CAPPIS (:meth:`wradlib.vpr.CAPPI`) and Pseudo CAPPIs
(:meth:wradlib.vpr.PseudoCAPPI`) for different altitudes at once.

Here's an example how a set of CAPPIs can be created from synthetic polar
volume data::

    import wradlib
    import numpy as np

    # define elevation and azimuth angles, ranges, radar site coordinates,
    # projection
    elevs  = np.array([0.5,1.5,2.4,3.4,4.3,5.3,6.2,7.5,8.7,10,12,14,16.7,19.5])
    azims  = np.arange(0., 360., 1.)
    ranges = np.arange(0., 120000., 1000.)
    sitecoords = (14.924218,120.255547,500.)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(32651)

    # create Cartesian coordinates corresponding the location of the
    # polar volume bins
    polxyz  = wradlib.vpr.volcoords_from_polar(sitecoords, elevs,
                                               azims, ranges, proj)  # noqa
    poldata = wradlib.vpr.synthetic_polar_volume(polxyz)
    # this is the shape of our polar volume
    polshape = (len(elevs),len(azims),len(ranges))

    # now we define the coordinates for the 3-D grid (the CAPPI layers)
    x = np.linspace(polxyz[:,0].min(), polxyz[:,0].max(), 120)
    y = np.linspace(polxyz[:,1].min(), polxyz[:,1].max(), 120)
    z = np.arange(500.,10500.,500.)
    xyz = wradlib.util.gridaspoints(x, y, z)
    gridshape = (len(x), len(y), len(z))

    # create an instance of the CAPPI class and
    # use it to create a series of CAPPIs
    gridder = wradlib.vpr.CAPPI(polxyz, xyz, maxrange=ranges.max(),
                                gridshape=gridshape, Ipclass=wradlib.ipol.Idw)
    gridded = np.ma.masked_invalid( gridder(poldata) ).reshape(gridshape)

    # plot results
    levels = np.linspace(0,100,25)
    wradlib.vis.plot_max_plan_and_vert(x, y, z, gridded, levels=levels,
                                       cmap=pl.cm.spectral)


.. autosummary::
   :nosignatures:
   :toctree: generated/

   volcoords_from_polar
   make_3D_grid
   CartesianVolume
   CAPPI
   PseudoCAPPI

"""

import numpy as np
import scipy

from . import georef as georef
from . import ipol as ipol
from . import util as util
from . import qual as qual


class CartesianVolume():
    """Create 3-D regular volume grid in Cartesian coordinates from polar data
    with multiple elevation angles

    Parameters
    ----------
    polcoords : :func:`numpy:numpy.array` of shape (num bins, 3)
    gridcoords : :func:`numpy:numpy.array` of shape (num voxels, 3)
    gridshape : tuple
        shape of the Cartesian grid (num x, num y, num z)
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    Ipclass : an interpolation class from :mod:`wradlib.ipol`
    ipargs : keyword arguments corresponding to Ipclass

    Returns
    -------
    output : float ndarray of shape (num levels, num x coordinates,
        num y coordinates)

    """

    def __init__(self, polcoords, gridcoords, gridshape=None,
                 maxrange=None, minelev=None, maxelev=None,
                 Ipclass=ipol.Idw, **ipargs):
        # TODO: rename Ipclas to ipclass
        # radar location in Cartesian coordinates
        # TODO: pass projected radar location as argument
        # (allows processing of incomplete polar volumes)
        self.radloc = np.array([np.mean(polcoords[:, 0]),
                                np.mean(polcoords[:, 1]),
                                np.min(polcoords[:, 2])]).reshape((-1, 3))
        # Set the mask which masks the blind voxels of the 3-D volume grid
        self.mask = self._get_mask(gridcoords, polcoords, gridshape,
                                   maxrange, minelev, maxelev)
        # create an instance of the Interpolation class
        self.trgix = np.where(np.logical_not(self.mask))
        self.ip = Ipclass(src=polcoords, trg=gridcoords[self.trgix], **ipargs)

    def __call__(self, data):
        """Interpolates the polar data to 3-dimensional Cartesian coordinates

        Parameters
        ----------
        data : 1-d array of length (num radar bins in volume,)
            The length of this array must be the same as len(polcoords)

        Returns
        -------
        output : 1-d array of length (num voxels,)

        """
        # Interpolate data in 3-D
        ipdata = np.repeat(np.nan, len(self.mask))
        ipdata[self.trgix] = self.ip(data)

        return ipdata

    def _get_mask(self, gridcoords, polcoords=None, gridshape=None,
                  maxrange=None, minelev=None, maxelev=None):
        """Returns a mask (the base class only contains a dummy function which
        masks nothing)

        This method needs to be replaced for inherited classes such as CAPPI or
        PseudoCAPPI.

        Parameters
        ----------
        gridcoords :
        polcoords :
        gridshape :
        maxrange :

        Returns
        -------
        output : Boolean array of length (num voxels,)

        """
        return np.repeat(False, len(gridcoords))


class CAPPI(CartesianVolume):
    """Create a Constant Altitude Plan Position Indicator (CAPPI)

    A CAPPI gives the value of a target variable (typically reflectivity
    in dBZ, but here also other variables such as e.g. rainfall intensity) in
    a defined altitude.

    In order to create a CAPPI, you first have to create an instance of this
    class. Calling this instance with the actual polar volume data will return
    the CAPPI grid.

    Parameters
    ----------
    polcoords : :func:`numpy:numpy.array` coordinate array \
        of shape (num bins, 3)
        Represents the 3-D coordinates of the orginal radar bins
    gridcoords : :func:`numpy:numpy.array` coordinate array \
        of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    gridshape : tuple
        shape of the original polar volume (num elevation angles,
        num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    Ipclass : an interpolation class from :mod:`wradlib.ipol`
    ipargs : keyword arguments corresponding to Ipclass

    Examples
    --------
    See :ref:`notebooks/workflow/recipe2.ipynb`.
    """

    def _get_mask(self, gridcoords, polcoords, gridshape,
                  maxrange, minelev, maxelev):
        """Masks the "blind" voxels of the Cartesian 3D-volume grid
        """
        below, above, out_of_range = blindspots(self.radloc, gridcoords,
                                                minelev, maxelev, maxrange)
        return np.logical_not(np.logical_not(out_of_range) &
                              np.logical_not(below) & np.logical_not(above))


class PseudoCAPPI(CartesianVolume):
    """Create a Pseudo-CAPPI Constant Altitude Plan Position Indicator (CAPPI)

    The difference to a CAPPI (:meth:`wradlib.vpr.CAPPI` is that the blind area
    *below* and *above* the radar are not masked, but filled by interpolation.
    Only the areas beyond the *range* of the radar are masked out. As a result,
    "blind" areas below the radar are particularly filled from the lowest
    available elevation angle.

    In order to create a Pseudo CAPPI, you first have to create an instance of
    this class. Calling this instance with the actual polar volume data will
    return the Pseudo CAPPI grid.

    Parameters
    ----------
    polcoords : :func:`numpy:numpy.array` coordinate array \
        of shape (num bins, 3)
        Represents the 3-D coordinates of the orginal radar bins
    gridcoords : :func:`numpy:numpy.array` coordinate array \
        of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    gridshape : tuple
        shape of the original polar volume (num elevation angles,
        num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    Ipclass : an interpolation class from :mod:`wradlib.ipol`
    ipargs : keyword arguments corresponding to Ipclass

    """

    def _get_mask(self, gridcoords, polcoords, gridshape,
                  maxrange, minelev, maxelev):
        """Masks the "blind" voxels of the Cartesian 3D-volume grid
        """
        return np.logical_not(np.logical_not(out_of_range(self.radloc,
                                                          gridcoords,
                                                          maxrange)))


def out_of_range(center, gridcoords, maxrange):
    """Flags the region outside the radar range

    Paramters
    ---------
    center : radar location
    gridcoords : array of 3-D coordinates with shape (num voxels, 3)
    maxrange : maximum range (meters)

    Returns
    -------
    output : 1-D Boolean array of length len(gridcoords)

    """
    return ((gridcoords - center) ** 2).sum(axis=-1) > maxrange ** 2


def blindspots(center, gridcoords, minelev, maxelev, maxrange):
    """Blind regions of the radar, marked on a 3-D grid

    The radar is blind below the radar, above the radar and beyond the range.
    The function returns three boolean arrays which indicate whether (1) the
    grid node is below the radar, (2) the grid node is above the radar,
    (3) the grid node is beyond the maximum range.

    Parameters
    ----------
    center
    gridcoords
    minelev
    maxelev
    maxrange

    Returns
    -------
    output : tuple of three Boolean arrays each of length (num grid points)

    """
    # distances of 3-D grid nodes from radar site (center)
    dist_from_rad = np.sqrt(((gridcoords - center) ** 2).sum(axis=-1))
    # below the radar
    # TODO: use qual.beam_height_ft_doviak
    below = gridcoords[:, 2] < (qual.beam_height_ft(dist_from_rad, minelev) +
                                center[:, 2])
    # above the radar
    above = gridcoords[:, 2] > (qual.beam_height_ft(dist_from_rad, maxelev) +
                                center[:, 2])
    # out of range
    out_of_range = dist_from_rad > maxrange
    return below, above, out_of_range


def volcoords_from_polar(sitecoords, elevs, azimuths, ranges, proj=None):
    """
    Create Cartesian coordinates for the polar volume bins

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

    Parameters
    ----------
    sitecoords : sequence of three floats indicating the radar position
        (longitude in decimal degrees, latitude in decimal degrees,
        height a.s.l. in meters)
    elevs : sequence of elevation angles
    azimuths : sequence of azimuth angles
    ranges : sequence of ranges
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection

    Returns
    -------
    output :  :func:`numpy:numpy.array`
        (num volume bins, 3)

    Examples
    --------
    See :ref:`notebooks/workflow/recipe2.ipynb`.
    """
    # make sure that elevs is an array
    elevs = np.array([elevs]).ravel()
    # create polar grid
    el, az, r = util.meshgridN(elevs, azimuths, ranges)
    # get geographical coordinates
    lons, lats, z = georef.polar2lonlatalt_n(r, az, el,
                                             sitecoords, re=6370040.)
    # get projected horizontal coordinates
    x, y = georef.reproject(lons, lats, projection_target=proj)
    # create standard shape
    coords = np.vstack((x.ravel(), y.ravel(), z.ravel())).transpose()
    return coords


def volcoords_from_polar_irregular(sitecoords, elevs, azimuths,
                                   ranges, proj=None):
    """Create Cartesian coordinates for the polar volume bins

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

    Parameters
    ----------
    sitecoords : sequence of three floats indicating the radar position
        (longitude in decimal degrees, latitude in decimal degrees,
        height a.s.l. in meters)
    elevs : sequence of elevation angles
    azimuths : sequence of azimuth angles
    ranges : sequence of ranges
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection

    Returns
    -------
    output : :func:`numpy:numpy.array`
        (num volume bins, 3)

    """
    # check structure: Are azimuth angles and range bins the same for each
    # elevation angle?
    oneaz4all = True
    onerange4all = True
    #   check elevs array, first: must be one-dimensional
    try:
        elevs = np.array(elevs)
    except Exception:
        print("Could not create an array from argument <elevs>.")
        print("The following exception was raised:")
        raise
    assert (elevs.ndim == 1) and (elevs.dtype != np.dtype("object")), \
        "Argument <elevs> in wradlib.wolcoords_from_polar must be a 1-D array."
    # now: is there one azimuths array for all elevation angles
    # or one for each?
    try:
        azimuths = np.array(azimuths)
    except Exception:
        print("Could not create an array from argument <azimuths>.")
        print("The following exception was raised:")
        raise
    if len(azimuths) == len(elevs):
        # are the items of <azimuths> arrays themselves?
        isseq = [util.issequence(elem) for elem in azimuths]
        assert not ((False in isseq) and (True in isseq)), \
            "Argument <azimuths> contains both iterable " \
            "and non-iterable items."
        if True in isseq:
            # we expect one azimuth array for each elevation angle
            oneaz4all = False
    # now: is there one ranges array for all elevation angles or one for each?
    try:
        ranges = np.array(ranges)
    except Exception:
        print("Could not create an array from argument <ranges>.")
        print("The following exception was raised:")
        raise
    if len(ranges) == len(elevs):
        # are the items of <azimuths> arrays themselves?
        isseq = [util.issequence(elem) for elem in ranges]
        assert not ((False in isseq) and (True in isseq)), \
            "Argument <azimuths> contains both iterable " \
            "and non-iterable items."
        if True in isseq:
            # we expect one azimuth array for each elevation angle
            onerange4all = False
    if oneaz4all and onerange4all:
        # this is the simple way
        return volcoords_from_polar(sitecoords, elevs, azimuths, ranges, proj)
    # No simply way, so we need to construct the coordinates arrays for
    # each elevation angle
    # but first adapt input arrays to this task
    if onerange4all:
        ranges = np.array([ranges for i in range(len(elevs))])
    if oneaz4all:
        azimuths = np.array([azimuths for i in range(len(elevs))])
    # and second create the corresponding polar volume grid
    el = np.array([])
    az = np.array([])
    r = np.array([])
    for i, elev in enumerate(elevs):
        az_tmp, r_tmp = np.meshgrid(azimuths[i], ranges[i])
        el = np.append(el, np.repeat(elev, len(azimuths[i]) * len(ranges[i])))
        az = np.append(az, az_tmp.ravel())
        r = np.append(r, r_tmp.ravel())
    # get geographical coordinates
    lons, lats, z = georef.polar2lonlatalt_n(r, az, el,
                                             sitecoords, re=6370040.)
    # get projected horizontal coordinates
    x, y = georef.reproject(lons, lats, projection_target=proj)
    # create standard shape
    coords = np.vstack((x.ravel(), y.ravel(), z.ravel())).transpose()
    return coords


def make_3D_grid(sitecoords, proj, maxrange, maxalt, horiz_res, vert_res):
    """Generate Cartesian coordinates for a regular 3-D grid based on
    radar specs.

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

    Parameters
    ----------
    sitecoords
    proj

        .. versionadded:: 0.6.0
           using osr objects instead of PROJ.4 strings as parameter

    maxrange
    maxalt
    horiz_res
    vert_res

    Returns
    -------
    output : float array of shape (num grid points, 3), a tuple of
        3 representing the grid shape

    """
    center = georef.reproject(sitecoords[0], sitecoords[1],
                              projection_target=proj)
    # minz = sitecoords[2]
    llx = center[0] - maxrange
    lly = center[1] - maxrange
    x = np.arange(llx, llx + 2 * maxrange + horiz_res, horiz_res)
    y = np.arange(lly, lly + 2 * maxrange + horiz_res, horiz_res)
    z = np.arange(0., maxalt + vert_res, vert_res)
    xyz = util.gridaspoints(z, y, x)
    shape = (len(z), len(y), len(x))
    return xyz, shape


def synthetic_polar_volume(coords):
    """Returns a synthetic polar volume
    """
    x = coords[:, 0] * 10 / np.max(coords[:, 0])
    y = coords[:, 1] * 10 / np.max(coords[:, 1])
    z = coords[:, 2] * 10 / np.max(coords[:, 2])
    out = np.abs(np.sin(x * y * z) / (x * y * z))
    out = out * 100. / out.max()
    return out


def vpr_interpolator(data, heights, method='linear'):
    """"""
    if method.lower() == 'linear':
        return scipy.interpolate.interp1d(heights, data, kind='linear')
    if method.lower() == 'nearest':
        return scipy.interpolate.interp1d(heights, data,
                                          kind='nearest',
                                          bounds_error=False,
                                          fill_value=data[0])
    else:
        raise NotImplementedError('Method: {0:s} unkown'.format(method))


def correct_vpr(data, heights, vpr, target_height=0.):
    """"""
    return (data * vpr(target_height)) / vpr(heights)


def mean_norm_vpr_from_volume(volume, reference_idx):
    """"""
    return norm_vpr_stats(volume, reference_idx, np.mean)


def norm_vpr_stats(volume, reference_idx, stat, **kwargs):
    # tmp = volume / volume[...,reference_idx]
    tmp = volume / volume[reference_idx]
    return stat(tmp.reshape((-1, np.prod(tmp.shape[-2:]))), **kwargs)


if __name__ == '__main__':
    print('wradlib: Calling module <vpr> as main...')
