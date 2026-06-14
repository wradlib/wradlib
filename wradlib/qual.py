#!/usr/bin/env python
# Copyright (c) 2011-2026, wradlib developers.
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

Therefore, no assumptions are made on the dimensions or shape of the input
fields except that they exhibit the numpy ndarray interface.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "pulse_volume",
    "beam_block_frac",
    "cum_beam_block_frac",
    "estimate_snr",
    "get_bb_ratio",
    "QualMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

from functools import singledispatch

import numpy as np
import xarray as xr
from xradar.model import sweep_vars_mapping

from wradlib.util import XarrayMethods, docstring


@singledispatch
def pulse_volume(*args, **kwargs):
    pass


@pulse_volume.register(np.ndarray)
def _pulse_volume_numpy(ranges, h, theta):
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
    See :doc:`notebooks:notebooks/workflow/recipe1`.
    """
    return np.pi * h * (ranges**2) * (np.tan(np.radians(theta / 2.0))) ** 2


@pulse_volume.register(xr.DataArray)
def _pulse_volume_xarray(obj, h, theta, **kwargs):
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
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    h : float
        pulse width (which corresponds to the range resolution [m])
    theta : float
        the aperture angle (beam width) of the radar beam [degree]

    Returns
    -------
    obj : :py:class:`xarray:xarray.Dataset`
        obj with volumes of radar bins at each range in `ranges` [:math:`m^3`].

    Examples
    --------
    See :doc:`notebooks:notebooks/workflow/recipe1`.
    """
    return _pulse_volume_numpy(obj, h, theta)


def beam_block_frac(th, bh, a):
    """Partial beam blockage fraction.

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    From Bech et al. (2003), Eqn 2 and Appendix

    Parameters
    ----------
    th : float or :class:`numpy:numpy.ndarray`
        Terrain height [m]
    bh : float or :class:`numpy:numpy.ndarray`
        Beam height [m]
    a : float or :class:`numpy:numpy.ndarray`
        Half power beam radius [m]

    Returns
    -------
    pbb : float or :class:`numpy:numpy.ndarray`
        Partial beam blockage fraction [unitless]

    Examples
    --------
    >>> pbb = beam_block_frac(th,bh,a) #doctest: +SKIP

    See :doc:`notebooks:notebooks/beamblockage/beamblockage`.

    Note
    ----
    This procedure uses a simplified interception function where no vertical
    gradient of refractivity is considered.  Other algorithms treat this
    more thoroughly.  However, this is accurate in most cases other than
    the super-refractive case.

    See the half_power_radius function to calculate variable `a`.

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

    numer = (ya * np.sqrt(a**2 - y**2)) + (a * np.arcsin(ya)) + (np.pi * a / 2.0)

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

    See :doc:`notebooks:notebooks/beamblockage/beamblockage`.

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


@singledispatch
def get_bb_ratio(*args, **kwargs):
    pass


@get_bb_ratio.register(np.ndarray)
def _get_bb_ratio_numpy(bb_height, bb_width, quality, zp_r):
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


@get_bb_ratio.register(xr.Dataset)
def _get_bb_ratio_xarray(obj):
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
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`

    Returns
    -------
    obj : :py:class:`xarray:xarray.Dataset`
        obj with bb ratio and boolean array containing the indices of SR bins connected
        to the BB.
    """
    quality = obj["qualityBB"]
    qtype = obj.get("qualityTypePrecip")
    zp_r = obj["zp"]
    bb_height = obj["heightBB"]
    bb_width = obj["widthBB"]

    if qtype is not None:
        quality = xr.where(((quality == 0) | (quality == 1)) & (qtype == 1), 1, quality)
        quality = xr.where(((quality > 1) | (quality > 2)), 2, quality)
        quality = xr.where(((quality == 1) | (quality == 2)), quality, 0)

    # parameters for bb detection
    ibb = (bb_height > 0) & (bb_width > 0) & (quality == 1)

    # set non-bb-pixels to np.nan
    bb_height_m = bb_height.where(ibb)
    bb_width_m = bb_width.where(ibb)
    # get median of bb-pixels
    bb_height_m = bb_height_m.median(skipna=True)
    bb_width_m = bb_width_m.median(skipna=True)
    # approximation of melting layer top and bottom
    zmlt = bb_height_m + bb_width_m / 2.0
    zmlb = bb_height_m - bb_width_m / 2.0

    # get ratio connected to brightband height
    ratio = (zp_r - zmlb) / (zmlt - zmlb)

    if qtype is not None:
        ibb0 = (bb_height == 0) & (bb_width == 0) & (quality == 1)
        ratio = xr.where(ibb0, 0, ratio)

    return obj.assign(bb_ratio=ratio, bb_mask=ibb)


@singledispatch
def estimate_snr(*args, **kwargs):
    pass


@estimate_snr.register(np.ndarray)
def _estimate_snr_numpy(dbz, rng, noise_level, gas_att):
    """
    Estimate radar signal-to-noise ratio (SNR) from reflectivity.

    This function reconstructs the radar SNR in dB space using reflectivity,
    range-dependent geometric spreading loss, receiver noise level, and
    gaseous attenuation.

    The formulation follows standard radar equation scaling, where reflectivity
    is converted into a received signal proxy and corrected for range and
    propagation losses.

    Parameters
    ----------
    dbz : array_like
        Equivalent radar reflectivity factor in dBZ.

    rng : array_like
        Range from radar in meters (m).

    noise_level : float or array_like
        Receiver noise level expressed in dB (consistent with dbz scaling).

    gas_att : float
        Effective gaseous attenuation coefficient in dB/km.
        This parameter represents a *two-way radar-path attenuation*
        (i.e. round-trip effective loss).

    Returns
    -------
    snr : array_like
        Estimated signal-to-noise ratio in dB.

    Notes
    -----
    The SNR is computed as:

        SNR = dbz - 20 * log10(rng_km) - noise_level - gas_att * rng_km

    where:
    - 20 * log10(rng) represents two-way geometric spreading loss
    - gas_att is an effective attenuation coefficient (dB/km),
      typically including round-trip propagation effects
    - noise_level represents receiver noise in dB

    References
    ----------
    :cite:`Doviak1993`, :cite:`Bringi2001`, :cite:`ITU_P676`
    """
    rng_km = rng * 0.001
    snr = dbz - 20 * np.log10(rng_km) - noise_level - gas_att * rng_km
    return snr


@estimate_snr.register(xr.DataArray)
def _estimate_snr_xarray(dbz, noise_level, gas_att):
    dim0 = dbz.wrl.util.dim0()
    rng = dbz.range
    snr = xr.apply_ufunc(
        _estimate_snr_numpy,
        dbz,
        rng,
        noise_level,
        gas_att,
        input_core_dims=[[dim0, "range"], ["range"], [], []],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    name = getattr(dbz, "name", "") or ""
    snr_name = "SNRV" if "DBZV" in name else "SNRH"
    snr.attrs = sweep_vars_mapping[snr_name]
    snr.name = snr.attrs["short_name"]
    return snr


class QualMethods(XarrayMethods):
    """wradlib xarray SubAccessor methods for Qual Methods."""

    @docstring(_pulse_volume_xarray)
    def pulse_volume(self, *args, **kwargs):
        if not isinstance(self, QualMethods):
            return pulse_volume(self, *args, **kwargs)
        else:
            return pulse_volume(self._obj, *args, **kwargs)

    @docstring(_get_bb_ratio_xarray)
    def get_bb_ratio(self, *args, **kwargs):
        if not isinstance(self, QualMethods):
            return get_bb_ratio(self, *args, **kwargs)
        else:
            return get_bb_ratio(self._obj, *args, **kwargs)

    @docstring(_estimate_snr_xarray)
    def estimate_snr(self, *args, **kwargs):
        if not isinstance(self, QualMethods):
            return estimate_snr(self, *args, **kwargs)
        else:
            return estimate_snr(self._obj, *args, **kwargs)


if __name__ == "__main__":
    print("wradlib: Calling module <qual> as main...")
