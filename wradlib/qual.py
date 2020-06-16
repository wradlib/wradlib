#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
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

   {}
"""
__all__ = ["pulse_volume", "beam_block_frac", "cum_beam_block_frac", "get_bb_ratio"]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def pulse_volume(ranges, h, theta):
    """Calculates the sampling volume of the radar beam per bin depending on \
    range and aperture.

    We assume a cone frustum which has the volume
    :math:`V=(\\pi/3) \\cdot h \\cdot (R^2 + R \\cdot r + r^2)`.
    R and r are the radii of the two frustum surface circles. Assuming that the
    pulse width is small compared to the range, we get
    :math:`R=r= \\tan ( 0.5 \\cdot \\theta \\cdot \\pi/180 ) \\cdot range`
    with theta being the aperture angle (beam width).
    Thus, the pulse volume simply becomes the volume of a cylinder with
    :math:`V=\\pi \\cdot h \\cdot range^2 \\cdot \\tan(
    0.5 \\cdot \\theta \\cdot \\pi/180)^2`

    Parameters
    ----------
    ranges : :class:`numpy:numpy.ndarray`
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

    See :ref:`/notebooks/workflow/recipe1.ipynb`.

    """
    return np.pi * h * (ranges ** 2) * (np.tan(np.radians(theta / 2.0))) ** 2


def beam_block_frac(th, bh, a):
    """Partial beam blockage fraction.

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    From Bech et al. (2003), Eqn 2 and Appendix

    Parameters
    ----------
    th : float | :class:`numpy:numpy.ndarray` of floats
        Terrain height [m]
    bh : float | :class:`numpy:numpy.ndarray` of floats
        Beam height [m]
    a : float | :class:`numpy:numpy.ndarray` of floats
        Half power beam radius [m]

    Returns
    -------
    pbb : float
        Partial beam blockage fraction [unitless]

    Examples
    --------
    >>> pbb = beam_block_frac(th,bh,a) #doctest: +SKIP

    See :ref:`/notebooks/beamblockage/wradlib_beamblock.ipynb`.

    Note
    ----
    This procedure uses a simplified interception function where no vertical
    gradient of refractivity is considered.  Other algorithms treat this
    more thoroughly.  However, this is accurate in most cases other than
    the super-refractive case.

    See the the half_power_radius function to calculate variable `a`.

    The heights must be the same units!
    """
    isfloat = isinstance(th, float) and isinstance(bh, float) and isinstance(a, float)

    # convert to numpy array in any case
    th = np.atleast_1d(th)
    bh = np.atleast_1d(bh)
    a = np.atleast_1d(a)

    # First find the difference between the terrain and height of
    # radar beam (Bech et al. (2003), Fig.3)
    y = th - bh

    # check if beam is clear or blocked
    ya = y / a
    clear = ya < -1.0
    block = ya > 1.0

    numer = (ya * np.sqrt(a ** 2 - y ** 2)) + (a * np.arcsin(ya)) + (np.pi * a / 2.0)

    denom = np.pi * a

    pbb = numer / denom

    pbb[clear] = 0.0
    pbb[block] = 1.0

    if isfloat:
        return pbb[0]
    else:
        return pbb


def cum_beam_block_frac(pbb):
    """Cumulative beam blockage fraction along a beam.

    Computes the cumulative beam blockage (cbb) along a beam from the partial
    beam blockage (pbb) fraction of each bin along that beam. CBB in one bin
    along a beam will always be at least as high as the maximum PBB of the
    preceeding bins.

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
    >>> pbb = beam_block_frac(th, bh, a) #doctest: +SKIP
    >>> cbb = cum_beam_block_frac(pbb) #doctest: +SKIP

    See :ref:`/notebooks/beamblockage/wradlib_beamblock.ipynb`.

    """

    # This is the index of the maximum PBB along each beam
    maxindex = np.nanargmax(pbb, axis=1)
    cbb = np.copy(pbb)

    # Iterate over all beams
    for ii, index in enumerate(maxindex):
        premax = 0.0
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

    With *SR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    This function basically applies the Bright Band (BB) information as
    provided by the corresponding SR datasets per beam, namely BB height and
    width, as well as quality flags of the SR beams. A BB ratio of <= 0
    indicates that a bin is located below the melting layer (ML), >=1
    above the ML, and in between 0 and 1 inside the ML.

    Parameters
    ----------
    bb_height : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the SR beams' BB heights
        in meters.
    bb_width : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the SR beams' BB widths
        in meters.
    quality : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams) containing the SR beams' BB quality
        index.
    zp_r : :class:`numpy:numpy.ndarray`
        Array of SR bin altitudes of shape (nscans, nbeams, nbins).

    Returns
    -------
    ratio : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins) containing the BB ratio of
        every SR bin.
        - ratio <= 0: below ml
        - 0 < ratio < 1: between ml
        - 1 <= ratio: above ml
    ibb : :class:`numpy:numpy.ndarray`
        Boolean array containing the indices of SR bins connected to the
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
    zmlt = bb_height_m + bb_width_m / 2.0
    zmlb = bb_height_m - bb_width_m / 2.0

    # get ratio connected to brightband height
    ratio = (zp_r - zmlb) / (zmlt - zmlb)

    return ratio, ibb


if __name__ == "__main__":
    print("wradlib: Calling module <qual> as main...")
