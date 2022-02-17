#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
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
:func:`wradlib.vpr.volcoords_from_polar`.

Then, we can create regular 3-D grids in order to analyse the vertical profile
of reflectivity or rainfall intensity. For some applications you might want
to create so-called `Constant Altitude Plan Position Indicators (CAPPI)
<https://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator>`_
in order to make radar observations at different distances from the radar more
comparable. Basically, a CAPPI is simply one slice out of a 3-D volume grid.
Analoguous, we will refer to the elements in a three dimensional Cartesian grid
as *voxels*. In wradlib, you can create
CAPPIS (:class:`~wradlib.vpr.CAPPI`) and Pseudo CAPPIs
(:class:`~wradlib.vpr.PseudoCAPPI`) for different altitudes at once.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "volcoords_from_polar",
    "make_3d_grid",
    "norm_vpr_stats",
    "CartesianVolume",
    "CAPPI",
    "PseudoCAPPI",
    "out_of_range",
    "blindspots",
]
__doc__ = __doc__.format("\n   ".join(__all__))
__doctest_requires__ = {"CAPPI": ["osgeo"]}

import warnings

import numpy as np

from wradlib import georef, ipol, util


class CartesianVolume:
    """Create 3-D regular volume grid in Cartesian coordinates from polar \
    data with multiple elevation angles

    Parameters
    ----------
    polcoords : :class:`numpy:numpy.ndarray`
        of shape (num bins, 3)
    gridcoords : :class:`numpy:numpy.ndarray`
        of shape (num voxels, 3)
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    minelev : float
        The minimum elevation angle of the volume (degree)
    maxelev : float
        The maximum elevation angle of the volume (degree)
    ipclass : :class:`wradlib.ipol.IpolBase`
        an interpolation class from :mod:`wradlib.ipol`
    ipargs : dict
        keyword arguments corresponding to ``ipclass``

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        float 1-d ndarray of the same length as ``gridcoords`` (num voxels,)

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.
    """

    def __init__(
        self,
        polcoords,
        gridcoords,
        gridshape=None,
        maxrange=None,
        minelev=None,
        maxelev=None,
        ipclass=ipol.Idw,
        **ipargs,
    ):
        if gridshape is not None:
            warnings.warn(
                f"``gridshape`` is not used in {self.__class__}. "
                "It will be removed in wradlib version 2.0.",
                DeprecationWarning,
            )
        # radar location in Cartesian coordinates
        # TODO: pass projected radar location as argument
        # (allows processing of incomplete polar volumes)
        self.radloc = np.array(
            [
                np.mean(polcoords[:, 0]),
                np.mean(polcoords[:, 1]),
                np.min(polcoords[:, 2]),
            ]
        ).reshape((-1, 3))
        # Set the mask which masks the blind voxels of the 3-D volume grid
        self.mask = self._get_mask(gridcoords, polcoords, maxrange, minelev, maxelev)
        # create an instance of the Interpolation class
        self.trgix = np.where(np.logical_not(self.mask))
        self.ip = ipclass(src=polcoords, trg=gridcoords[self.trgix], **ipargs)

    def __call__(self, data, **kwargs):
        """Interpolates the polar data to 3-dimensional Cartesian coordinates

        Parameters
        ----------
        data : :class:`numpy:numpy.ndarray`
            1-d array of length (num radar bins in volume,)
            The length of this array must be the same as len(polcoords)

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            1-d array of length (num voxels,)

        """
        # Interpolate data in 3-D
        ipdata = np.repeat(np.nan, len(self.mask))
        ipdata[self.trgix] = self.ip(data, **kwargs)

        return ipdata

    def _get_mask(
        self,
        gridcoords,
        polcoords=None,
        maxrange=None,
        minelev=None,
        maxelev=None,
    ):
        """Returns a mask

        (the base class only contains a dummy function which masks nothing)

        This method needs to be replaced for inherited classes such as CAPPI or
        PseudoCAPPI.

        Parameters
        ----------
        gridcoords : :class:`numpy:numpy.ndarray`
            Array of shape (num voxels, 3)
        polcoords : :class:`numpy:numpy.ndarray`
            Array of shape (num bins, 3)
        maxrange : float
            The maximum radar range
            (must be the same for each elevation angle,
            and same unit as gridcoords)
        minelev : float
            The minimum elevation angle of the volume (degree)
        maxelev : float
            The maximum elevation angle of the volume (degree)

        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            Boolean array of length (num voxels,)
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
    polcoords : :class:`numpy:numpy.ndarray`
        coordinate array of shape (num bins, 3)
        Represents the 3-D coordinates of the original radar bins
    gridcoords : :class:`numpy:numpy.ndarray`
        coordinate array of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    ipclass : :class:`wradlib.ipol.IpolBase`
        an interpolation class from :mod:`wradlib.ipol`
    ipargs : dict
        keyword arguments corresponding to ``ipclass``

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        float 1-d ndarray of the same length as ``gridcoords`` (num voxels,)

    See Also
    --------
    :func:`~wradlib.vpr.out_of_range`
    :func:`~wradlib.vpr.blindspots`

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.

    Here's an example how a set of CAPPIs can be created from synthetic polar volume data:


        >>> import wradlib
        >>> import numpy as np
        >>> from osgeo import osr
        >>> import matplotlib.pyplot as pl
        >>> pl.interactive(True)
        >>> # define elevation and azimuth angles, ranges, radar site coordinates,
        >>> # projection
        >>> elevs  = np.array([0.5,1.5,2.4,3.4,4.3,5.3,6.2,7.5,8.7,10,12,14,16.7,19.5])
        >>> azims  = np.arange(0., 360., 1.)
        >>> ranges = np.arange(0., 120000., 1000.)
        >>> sitecoords = (120.255547,14.924218,500.)
        >>> proj = osr.SpatialReference()
        >>> _ = proj.ImportFromEPSG(32651)
        >>> # create Cartesian coordinates corresponding the location of the
        >>> # polar volume bins
        >>> polxyz  = wradlib.vpr.volcoords_from_polar(sitecoords, elevs,
        ...                                            azims, ranges, proj)  # noqa
        >>> poldata = wradlib.vpr.synthetic_polar_volume(polxyz)
        >>> # this is the shape of our polar volume
        >>> polshape = (len(elevs),len(azims),len(ranges))
        >>> # now we define the coordinates for the 3-D grid (the CAPPI layers)
        >>> x = np.linspace(polxyz[:,0].min(), polxyz[:,0].max(), 120)
        >>> y = np.linspace(polxyz[:,1].min(), polxyz[:,1].max(), 120)
        >>> z = np.arange(500.,10500.,500.)
        >>> xyz = wradlib.util.gridaspoints(z, y, x)
        >>> gridshape = (len(z), len(y), len(x))
        >>> # create an instance of the CAPPI class and
        >>> # use it to create a series of CAPPIs
        >>> gridder = wradlib.vpr.CAPPI(polxyz, xyz, maxrange=ranges.max(),  # noqa
        ...                             minelev=elevs.min(), maxelev=elevs.max(),
        ...                             ipclass=wradlib.ipol.Idw)
        >>> gridded = np.ma.masked_invalid( gridder(poldata) ).reshape(gridshape)
        >>>
        >>> # plot results
        >>> levels = np.linspace(0,100,25)
        >>> wradlib.vis.plot_max_plan_and_vert(x, y, z, gridded, levels=levels,
        ...                                    cmap=pl.cm.viridis)
        >>> pl.show()
    """

    def _get_mask(self, gridcoords, polcoords, maxrange, minelev, maxelev):
        """Masks the "blind" voxels of the Cartesian 3D-volume

        For the CAPPI, blind voxels are below `minelev` and above `maxelev`
        and beyond `maxrange`.
        """
        below, above, out_of_range = blindspots(
            self.radloc, gridcoords, minelev, maxelev, maxrange
        )
        return np.logical_not(
            np.logical_not(out_of_range) & np.logical_not(below) & np.logical_not(above)
        )


class PseudoCAPPI(CartesianVolume):
    """Create a Pseudo-CAPPI Constant Altitude Plan Position Indicator (CAPPI)

    The difference to a CAPPI (:class:`wradlib.vpr.CAPPI`) is that the blind
    area *below* and *above* the radar are not masked, but filled by
    interpolation.
    Only the areas beyond the *range* of the radar are masked out. As a result,
    "blind" areas below the radar are particularly filled from the lowest
    available elevation angle.

    In order to create a Pseudo CAPPI, you first have to create an instance of
    this class. Calling this instance with the actual polar volume data will
    return the Pseudo CAPPI grid.

    Parameters
    ----------
    polcoords : :class:`numpy:numpy.ndarray`
        coordinate array of shape (num bins, 3)
        Represents the 3-D coordinates of the original radar bins
    gridcoords : :class:`numpy:numpy.ndarray`
        coordinate array of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    minelev : float
        The minimum elevation angle of the volume (degree)
    maxelev : float
        The maximum elevation angle of the volume (degree)
    ipclass : :class:`wradlib.ipol.IpolBase`
        an interpolation class from :mod:`wradlib.ipol`
    ipargs : dict
        keyword arguments corresponding to ``ipclass``

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        float 1-d ndarray of the same length as ``gridcoords`` (num voxels,)

    See Also
    --------
    :func:`~wradlib.vpr.out_of_range`

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.
    """

    def _get_mask(self, gridcoords, polcoords, maxrange, minelev, maxelev):
        """Masks the "blind" voxels of the Cartesian 3D-volume grid

        For the Pseudo CAPPI, blind voxels are only those beyond `maxrange`.
        """
        return np.logical_not(
            np.logical_not(out_of_range(self.radloc, gridcoords, maxrange))
        )


def out_of_range(center, gridcoords, maxrange):
    """Masks the region outside the radar range

    Parameters
    ---------
    center : tuple
        radar location
    gridcoords : :class:`numpy:numpy.ndarray`
        array of 3-D coordinates with shape (num voxels, 3)
    maxrange : float
        maximum range (same unit as gridcoords)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        1-D Boolean array of length len(gridcoords)

    """
    return ((gridcoords - center) ** 2).sum(axis=-1) > maxrange**2


def blindspots(center, gridcoords, minelev, maxelev, maxrange):
    """Masks blind regions of the radar, marked on a 3-D grid

    The radar is blind below the radar, above the radar and beyond the range.
    The function returns three boolean arrays which indicate whether (1) the
    grid node is below the radar, (2) the grid node is above the radar,
    (3) the grid node is beyond the maximum range.

    Parameters
    ---------
    center : tuple
        radar location
    gridcoords : :class:`numpy:numpy.ndarray`
        array of 3-D coordinates with shape (num voxels, 3)
    minelev : float
        The minimum elevation angle of the volume (degree)
    maxelev : float
        The maximum elevation angle of the volume (degree)
    maxrange : float
        maximum range (same unit as gridcoords)

    Returns
    -------
    output : tuple
        tuple of three boolean arrays (below, above, out_of_range) each of length
        (num grid points)
    """
    site_altitude = center[:, 2]
    # distances of 3-D grid nodes from radar site (center)
    dist_from_rad = np.sqrt(((gridcoords - center) ** 2).sum(axis=-1))
    # below the radar
    below = gridcoords[:, 2] < (
        georef.bin_altitude(dist_from_rad, minelev, site_altitude, re=6371000)
    )
    # above the radar
    above = gridcoords[:, 2] > (
        georef.bin_altitude(dist_from_rad, maxelev, site_altitude, re=6371000)
    )
    # out of range
    out_of_range = dist_from_rad > maxrange
    return below, above, out_of_range


def volcoords_from_polar(sitecoords, elevs, azimuths, ranges, proj=None):
    """Create Cartesian coordinates for regular polar volumes

    Parameters
    ----------
    sitecoords : tuple
        sequence of three floats indicating the radar position
        (longitude in decimal degrees, latitude in decimal degrees,
        height a.s.l. in meters)
    elevs : sequence
        sequence of elevation angles
    azimuths : sequence
        sequence of azimuth angles
    ranges : sequence
        sequence of ranges
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL OSR Spatial Reference Object describing projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array of shape (num volume bins, 3)

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.
    """
    # make sure that elevs is an array
    elevs = np.array([elevs]).ravel()
    # create polar grid
    el, az, r = util.meshgrid_n(elevs, azimuths, ranges)

    # get projected coordinates
    coords = georef.spherical_to_proj(r, az, el, sitecoords, proj=proj)
    coords = coords.reshape(-1, 3)

    return coords


def volcoords_from_polar_irregular(sitecoords, elevs, azimuths, ranges, proj=None):
    """Create Cartesian coordinates for polar volumes with irregular \
    sweep specifications

    Parameters
    ----------
    sitecoords : tuple
        sequence of three floats indicating the radar position
        (longitude in decimal degrees, latitude in decimal degrees,
        height a.s.l. in meters)
    elevs : sequence
        sequence of elevation angles
    azimuths : sequence
        sequence of azimuth angles
    ranges : sequence
        sequence of ranges
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL OSR Spatial Reference Object describing projection

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Array of shape (num volume bins, 3)

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
    assert (elevs.ndim == 1) and (
        elevs.dtype != np.dtype("object")
    ), "Argument <elevs> in wradlib.volcoords_from_polar must be a 1-D array."
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
        assert not ((False in isseq) and (True in isseq)), (
            "Argument <azimuths> contains both iterable " "and non-iterable items."
        )
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
        assert not ((False in isseq) and (True in isseq)), (
            "Argument <azimuths> contains both iterable " "and non-iterable items."
        )
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
    # get projected coordinates
    coords = georef.spherical_to_proj(r, az, el, sitecoords, proj=proj)
    coords = coords.reshape(-1, 3)

    return coords


def make_3d_grid(sitecoords, proj, maxrange, maxalt, horiz_res, vert_res, minalt=0.0):
    """Generate Cartesian coordinates for a regular 3-D grid based on \
    radar specs.

    Parameters
    ----------
    sitecoords : tuple
        Radar location coordinates in lon, lat
    proj : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL OSR Spatial Reference Object describing projection
    maxrange : float
        maximum radar range (same unit as SRS defined by ``proj``,
        typically meters)
    maxalt : float
        maximum altitude to which the 3-d grid should extent (meters)
    horiz_res : float
        horizontal resolution of the 3-d grid (same unit as
        SRS defined by ``proj``, typically meters)
    vert_res : float
        vertical resolution of the 3-d grid (meters)

    Keyword Arguments
    -----------------
    minalt : float
        minimum altitude to which the 3-d grid should extent (meters)

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`, tuple
        float array of shape (num grid points, 3), a tuple of
        3 representing the grid shape
    """
    center = georef.reproject(sitecoords[0], sitecoords[1], projection_target=proj)
    # minz = sitecoords[2]
    llx = center[0] - maxrange
    lly = center[1] - maxrange
    x = np.arange(llx, llx + 2 * maxrange + horiz_res, horiz_res)
    y = np.arange(lly, lly + 2 * maxrange + horiz_res, horiz_res)
    z = np.arange(minalt, maxalt + vert_res, vert_res)
    xyz = util.gridaspoints(z, y, x)
    shape = (len(z), len(y), len(x))
    return xyz, shape


def synthetic_polar_volume(coords):
    """Returns a totally arbitrary synthetic polar volume - just for testing

    Parameters
    ----------
    coords : :class:`numpy:numpy.ndarray`
        (num volume bins, 3), as returned by volcoords_from_polar

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        float array of shape (num volume bins, 3)
    """
    x = coords[:, 0] * 10 / np.max(coords[:, 0])
    y = coords[:, 1] * 10 / np.max(coords[:, 1])
    z = coords[:, 2] / 1000.0
    out = np.abs(np.sin(x * y)) * np.exp(-z)
    out = out * 100.0 / out.max()
    return out


def norm_vpr_stats(volume, reference_layer, stat=None, **kwargs):
    """Returns the average normalised vertical profile of a volume or \
    any other desired statistics

    Given a Cartesian 3-d ``volume`` and an arbitrary ``reference layer``
    index, the function normalises all vertical profiles represented by the
    volume and computes a static of all profiles (e.g. an average vertical
    profile using the default ``stat``).

    Parameters
    ----------
    volume : :class:`numpy:numpy.ndarray` or
        :class:`numpy:numpy.ma.MaskedArray`
        Cartesian 3-d grid with shape (num vertical layers, num x intervals,
        num y intervals)
    reference_layer : int
        This index defines the vertical layers of ``volume`` that is used to
        normalise all vertical profiles
    stat : callable
        typically a numpy statistics function (defaults to numpy.mean)
    kwargs : dict
        further keyword arguments taken by ``stat``

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray` or :py:class:`numpy:numpy.ma.MaskedArray`
        Array of shape (num vertical layers,) which provides the statistic from
        ``stat`` applied over all normalised vertical profiles (e.g. the
        mean normalised vertical profile if :py:func:`numpy:numpy.mean` is used)

    """
    if stat is None:
        stat = np.mean
    tmp = volume / volume[reference_layer]
    return stat(tmp.reshape((-1, np.prod(tmp.shape[-2:]))), axis=1, **kwargs)


if __name__ == "__main__":
    print("wradlib: Calling module <vpr> as main...")
