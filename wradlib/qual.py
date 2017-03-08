#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Data Quality
^^^^^^^^^^^^

This module will serve two purposes:

#. provide routines to create simple radar data quality related fields.
#. provide routines to decide which radar pixel to choose based on the
   competing information in different quality fields.

Data is supposed to be stored in 'aligned' arrays. Aligned here means that
all fields are structured such that in each field the data for a certain index
is representative for the same physical target.

Therefore no assumptions are made on the dimensions or shape of the input
fields except that they exhibit the numpy ndarray interface.

.. autosummary::
   :nosignatures:
   :toctree: generated/

    beam_height_ft
    beam_height_ft_doviak
    pulse_volume
    beam_block_frac
    get_bb_ratio

"""

import numpy as np


def beam_height_ft(ranges, elevations, degrees=True, re=6371000):
    """Calculates the height of a radar beam above the antenna according to
    the 4/3 (four-thirds -> ft) effective Earth radius model.
    The formula was taken from :cite:`Collier1996`.

    Parameters
    ----------
    ranges : :class:`numpy:numpy.ndarray`
        The distances of each bin from the radar [m]
    elevations : :class:`numpy:numpy.ndarray`
        The elevation angles of each bin from the radar [degrees or radians]
    degrees : bool
        If True (the default) elevation angles are given in degrees and will
        be converted to radians before calculation. If False no transformation
        will be done and elevations has to be given in radians.
    re : float
        Earth radius [m]

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Height of the beam [m]

    Note
    ----
    The shape of `elevations` and `ranges` may differ in which case numpy's
    broadcasting rules will apply and the shape of `output` will be that of
    the broadcast arrays. See the numpy documentation on how broadcasting
    works.

    """
    if degrees:
        elev = np.deg2rad(elevations)
    else:
        elev = elevations

    return ((ranges ** 2 * np.cos(elev) ** 2) /
            (2 * (4. / 3.) * re)) + ranges * np.sin(elev)


def beam_height_ft_doviak(ranges, elevations, degrees=True, re=6371000):
    """Calculates the height of a radar beam above the antenna according to
    the 4/3 (four-thirds -> ft) effective Earth radius model.
    The formula was taken from :cite:`Doviak1993`.

    Parameters
    ----------
    ranges : :class:`numpy:numpy.ndarray`
        The distances of each bin from the radar [m]
    elevations : :class:`numpy:numpy.ndarray`
        The elevation angles of each bin from the radar [degrees or radians]
    degrees : bool
        If True (the default) elevation angles are assumed to be given in
        degrees and will
        be converted to radians before calculation. If False no transformation
        will be done and `elevations` has to be given in radians.
    re : float
        Earth radius [m]

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Height of the beam [m]

    Note
    ----
    The shape of `elevations` and `ranges` may differ in which case numpy's
    broadcasting rules will apply and the shape of `output` will be that of
    the broadcast arrays. See the numpy documentation on how broadcasting
    works.

    """
    if degrees:
        elev = np.deg2rad(elevations)
    else:
        elev = elevations

    reft = (4. / 3.) * re

    return np.sqrt(ranges ** 2 + reft ** 2 +
                   2 * ranges * reft * np.sin(elev)) - reft


def pulse_volume(ranges, h, theta):
    r"""Calculates the sampling volume of the radar beam per bin depending on
    range and aperture.

    We assume a cone frustum which has the volume
    :math:`V=(\pi/3) \cdot h \cdot (R^2 + R \cdot r + r^2)`.
    R and r are the radii of the two frustum surface circles. Assuming that the
    pulse width is small compared to the range, we get
    :math:`R=r= \tan ( 0.5 \cdot \theta \cdot \pi/180 ) \cdot range`
    with theta being the aperture angle (beam width).
    Thus, the pulse volume simply becomes the volume of a cylinder with
    :math:`V=\pi \cdot h \cdot range^2 \cdot \tan(
    0.5 \cdot \theta \cdot \pi/180)^2`

    Parameters
    ----------
    ranges : array
        the distances of each bin from the radar [m]
    h : float
        pulse width (which corresponds to the range resolution [m])
    theta : float
        the aperture angle (beam width) of the radar beam [degree]

    Returns
    -------
    output : :class:`numpy:numpy.ndarray`
        Volume of radar bins at each range in `ranges` [:math:`m^3`]

    Examples
    --------

    See :ref:`notebooks/workflow/recipe1.ipynb`.

    """
    return np.pi * h * (ranges ** 2) * (np.tan(np.radians(theta/2.))) ** 2


def beam_block_frac(Th, Bh, a):
    """Partial beam blockage fraction.

    .. versionadded:: 0.6.0

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    From Bech et al. (2003), Eqn 2 and Appendix

    Parameters
    ----------
    Th : float | :class:`numpy:numpy.ndarray` of floats
        Terrain height [m]
    Bh : float | :class:`numpy:numpy.ndarray` of floats
        Beam height [m]
    a : float | :class:`numpy:numpy.ndarray` of floats
        Half power beam radius [m]

    Returns
    -------
    PBB : float
        Partial beam blockage fraction [unitless]

    Examples
    --------
    >>> PBB = beam_block_frac(Th,Bh,a) #doctest: +SKIP

    See :ref:`notebooks/beamblockage/wradlib_beamblock.ipynb`.

    Note
    ----
    This procedure uses a simplified interception function where no vertical
    gradient of refractivity is considered.  Other algorithms treat this
    more thoroughly.  However, this is accurate in most cases other than
    the super-refractive case.

    See the the half_power_radius function to calculate variable `a`.

    The heights must be the same units!
    """

    # First find the difference between the terrain and height of
    # radar beam (Bech et al. (2003), Fig.3)
    y = Th - Bh

    Numer = (y * np.sqrt(a ** 2 - y ** 2)) + \
            (a ** 2 * np.arcsin(y / a)) + (np.pi * a ** 2 / 2.)

    Denom = np.pi * a ** 2

    PBB = Numer / Denom

    return PBB


def cum_beam_block_frac(pbb):
    """Cumulative beam blockage fraction along a beam.

    Computes the cumulative beam blockage (cbb) along a beam from the partial
    beam blockage (pbb) fraction of each bin along that beam. CBB in one bin
    along a beam will always be at least as high as the maximum PBB of the
    preceeding bins.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pbb : :class:`numpy:numpy.ndarray`
        2-D array of floats of shape (num beams, num range bins)
        Partial beam blockage fraction of a bin along a beam [m]

    Returns
    -------
    cbb : :class:`numpy:numpy.ndarray`
        Array of floats of the same shape as pbb
        Cumulative partial beam blockage fraction [unitless]

    Examples
    --------
    >>> PBB = beam_block_frac(Th,Bh,a) #doctest: +SKIP
    >>> CBB = cum_beam_block_frac(PBB) #doctest: +SKIP

    See :ref:`notebooks/beamblockage/wradlib_beamblock.ipynb`.

    """

    # This is the index of the maximum PBB along each beam
    maxindex = np.nanargmax(pbb, axis=1)
    cbb = np.copy(pbb)

    # Iterate over all beams
    for ii, index in enumerate(maxindex):
        premax = 0.
        for jj in range(index):
            # Only iterate to max index to make this faster
            if pbb[ii, jj] > premax:
                cbb[ii, jj] = pbb[ii, jj]
                premax = pbb[ii, jj]
            else:
                cbb[ii, jj] = premax
        # beyond max index, everything is max anyway
        cbb[ii, index:] = pbb[ii, index]

    return cbb


def get_bb_ratio(bb_height, bb_width, quality, zp_r):
    """Returns the Bright Band ratio of each PR bin

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    This function basically applies the Bright Band (BB) information as
    provided by the corresponding PR datasets per beam, namely BB height and
    width, as well as quality flags of the PR beams. A BB ratio of <= 0
    indicates that a bin is located below the melting layer (ML), >=1
    above the ML, and in between 0 and 1 inside the ML.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    bb_height : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the PR beams' BB heights
        in meters.
    bb_width : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the PR beams' BB widths
        in meters.
    quality : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the PR beams' BB quality
        index.
    zp_r : :class:`numpy:numpy.ndarray`
        Array of PR bin altitudes of shape (nbeams, nbins).

    Returns
    -------
    ratio : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins) containing the BB ratio of
        every PR bin.
        - ratio <= 0: below ml
        - 0 < ratio < 1: between ml
        - 1 <= ratio: above ml
    ibb : :class:`numpy:numpy.ndarray`
        Boolean array containing the indices of PR bins connected to the
        BB.
    """
    # parameters for bb detection
    ibb = (bb_height > 0) & (bb_width > 0) & (quality == 1)

    # set non-bb-pixels to np.nan
    bb_height = bb_height.copy()
    bb_height[~ibb] = np.nan
    bb_width = bb_width.copy()
    bb_width[~ibb] = np.nan
    # get median of bb-pixels
    bb_height_m = np.nanmedian(bb_height)
    bb_width_m = np.nanmedian(bb_width)

    # approximation of melting layer top and bottom
    zmlt = bb_height_m + bb_width_m / 2.
    zmlb = bb_height_m - bb_width_m / 2.

    # get ratio connected to brightband height
    ratio = (zp_r - zmlb) / (zmlt - zmlb)
    ratio = np.broadcast_to(ratio, (bb_width.shape[0],) + ratio.shape)

    return ratio, ibb


if __name__ == '__main__':
    print('wradlib: Calling module <qual> as main...')
