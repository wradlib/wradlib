#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Dual-Pol and Differential Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
--------

This module provides algorithms to process polarimetric radar moments,
namely the differential phase, :math:`Phi_{{DP}}`, and, based on successful
:math:`Phi_{{DP}}` retrieval, also the specific differential phase,
:math:`K_{{DP}}`.
Please note that the actual application of polarimetric moments is implemented
in the corresponding wradlib modules, e.g.:

    - fuzzy echo classification from polarimetric moments
      (:func:`wradlib.clutter.classify_echo_fuzzy`)
    - attenuation correction (:func:`wradlib.atten.pia_from_kdp`)
    - direct precipitation retrieval from Kdp (:func:`wradlib.trafo.kdp_to_r`)

Establishing a valid :math:`Phi_{{DP}}` profile for :math:`K_{{DP}}` retrieval
involves despeckling (linear_despeckle), phase unfolding, and iterative
retrieval of :math:`Phi_{{DP}}` form :math:`K_{{DP}}`.
The main workflow and its single steps is based on a publication by
:cite:`Vulpiani2012`. For convenience, the entire workflow has been
put together in the function :func:`wradlib.dp.process_raw_phidp_vulpiani`.

Once a valid :math:`Phi_{{DP}}` profile has been established, the
`kdp_from_phidp` functions can be used to retrieve :math:`K_{{DP}}`.

Please note that so far, the functions in this module were designed to increase
performance. This was mainly achieved by allowing the simultaneous application
of functions over multiple array dimensions. The only requirement to apply
these function is that the **range dimension must be the last dimension** of
all input arrays.


.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "process_raw_phidp_vulpiani",
    "kdp_from_phidp",
    "unfold_phi_vulpiani",
    "unfold_phi",
    "linear_despeckle",
    "texture",
    "depolarization",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import deprecation
import numpy as np
from scipy import integrate, interpolate

from wradlib import trafo, util, version


def process_raw_phidp_vulpiani(
    phidp, dr, ndespeckle=5, winlen=7, niter=2, copy=False, **kwargs
):
    """Establish consistent :math:`Phi_{DP}` profiles from raw data.

    This approach is based on :cite:`Vulpiani2012` and involves a
    two step procedure of :math:`Phi_{DP}` reconstruction.

    Processing of raw :math:`Phi_{DP}` data contains the following steps:

        - Despeckle
        - Initial :math:`K_{DP}` estimation
        - Removal of artifacts
        - Phase unfolding
        - :math:`Phi_{DP}` reconstruction using iterative estimation
          of :math:`K_{DP}`

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (n azimuth angles, n range gates)
    dr : float
        gate length in km
    ndespeckle : int
        ``ndespeckle`` parameter of :func:`~wradlib.util.despeckle`
    winlen : int
        ``winlen`` parameter of :func:`~wradlib.dp.kdp_from_phidp`
    niter : int
        Number of iterations in which :math:`Phi_{DP}` is retrieved from
        :math:`K_{DP}` and vice versa
    copy : bool
        if True, the original :math:`Phi_{DP}` array will remain unchanged

    Keyword Arguments
    -----------------
    th1 : float
        Threshold th1 from above cited paper.
    th2 : float
        Threshold th2 from above cited paper.
    th3 : float
        Threshold th3 from above cited paper.

    Returns
    -------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (..., , n azimuth angles, n range gates) reconstructed
        :math:`Phi_{DP}`
    kdp : :class:`numpy:numpy.ndarray`
        array of shape (..., , n azimuth angles, n range gates)
        ``kdp`` estimate corresponding to ``phidp`` output

    Examples
    --------

    See :ref:`/notebooks/verification/wradlib_verify_example.ipynb`.

    """
    if copy:
        phidp = phidp.copy()

    # get thresholds
    th1 = kwargs.pop("th1", -2)
    th2 = kwargs.pop("th2", 20)
    th3 = kwargs.pop("th3", -20)

    method = kwargs.pop("method", None)

    # despeckle
    phidp = util.despeckle(phidp, ndespeckle)

    # kdp retrieval first guess
    # use finite difference scheme as written in the cited paper
    kdp = kdp_from_phidp(
        phidp,
        dr=dr,
        winlen=winlen,
        method="finite_difference_vulpiani",
        skipna=False,
        **kwargs,
    )

    # try unfolding phidp
    phidp = unfold_phi_vulpiani(phidp, kdp, th=th3, winlen=winlen)

    # clean up unfolded PhiDP
    phidp[phidp > 360] = np.nan

    # kdp retrieval second guess
    # re-add given method to kwargs
    if method is not None:
        kwargs["method"] = method
    # use given (fast) derivation methods
    kdp = kdp_from_phidp(phidp, dr=dr, winlen=winlen, **kwargs)

    # find kdp values with no physical meaning like noise, backscatter differential
    # phase, nonuniform beamfilling or residual artifacts using th1 and th2
    mask = (kdp <= th1) | (kdp >= th2)
    kdp[mask] = 0

    # fill remaining NaN with zeros
    kdp = np.nan_to_num(kdp)

    # start the actual phidp/kdp iteration
    for i in range(niter):
        # phidp from kdp through integration
        phidp = 2 * integrate.cumtrapz(kdp, dx=dr, initial=0.0, axis=-1)
        # kdp from phidp by convolution
        kdp = kdp_from_phidp(phidp, dr=dr, winlen=winlen, **kwargs)

    return phidp, kdp


def unfold_phi_vulpiani(phidp, kdp, th=-20, winlen=7):
    """Alternative phase unfolding which completely relies on :math:`K_{DP}`.

    This unfolding should be used in oder to iteratively reconstruct
    :math:`Phi_{DP}` and :math:`K_{DP}` (see :cite:`Vulpiani2012`).

    Note
    ----
    :math:`Phi_{DP}` is assumed to be in the interval [-180, 180] degree.
    From experience the window for calculation of :math:`K_{DP}` should not
    be too large to catch possible phase wraps.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of floats
    kdp : :class:`numpy:numpy.ndarray`
        array of floats
    th : float
        Threshold th3 in the above citation.
    winlen : int
        Length of window to fix possible phase over-correction. Normally
        should take the value of the length of the processing window in
        the above citation.
    """
    # unfold phidp
    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))
    kdp = kdp.reshape((-1, shape[-1]))

    # check for possible phase wraps
    mask = kdp < th
    if np.any(mask):
        # setup index on last dimension
        idx = np.arange(phidp.shape[-1])[np.newaxis, :]
        # set last bin to 1 to get that index in case of no kdp < th
        mask[:, -1] = 1
        # find first occurrence of kdp < th in each ray
        amax = np.argmax(mask, axis=-1)[:, np.newaxis]
        # get maximum phase in each ray
        phimax = np.nanmax(phidp, axis=-1)[:, np.newaxis]
        # retrieve folding location mask and unfold
        foldmask = np.where(idx > amax)
        phidp[foldmask] += 360
        # retrieve checkmask for remaining "over" unfolds and fix
        # phimax + 180 is chosen, because it's half of the max phase wrap
        checkmask = np.where((idx <= amax + winlen) & (phidp > (phimax + 180.0)))
        phidp[checkmask] -= 360

    return phidp.reshape(shape)


def _fill_sweep(dat, kind="nan_to_num", fill_value=0.0):
    """Fills missing data in a 1D profile.

    Parameters
    ----------
    dat : :class:`numpy:numpy.ndarray`
        array of shape (n azimuth angles, n range gates)
    kind : str
        Defines how the filling is done.
    fill_value : float
        Fill value in areas of extrapolation.

    """
    if kind == "nan_to_num":
        return np.nan_to_num(dat)

    if not np.any(np.isnan(dat)):
        return dat

    shape = dat.shape
    dat = dat.reshape((-1, shape[-1]))

    for beam in range(len(dat)):
        invalid = np.isnan(dat[beam])
        validx = np.where(~invalid)[0]
        if len(validx) < 2:
            dat[beam, invalid] = 0.0
            continue
        f = interpolate.interp1d(
            validx,
            dat[beam, validx],
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
        )
        invalidx = np.where(invalid)[0]
        dat[beam, invalidx] = f(invalidx)
    return dat.reshape(shape)


def kdp_from_phidp(
    phidp, winlen=7, dr=1.0, method="lanczos_conv", skipna=True, **kwargs
):
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}`.

    In normal operation the method uses convolution to estimate :math:`K_{DP}`
    (the derivative of :math:`Phi_{DP}`) with Low-noise Lanczos differentiators
    (`method='lanczos_conv'`). The results are very similar to the moving window
    linear regression (`method='lstsq'`), but the former is *much* faster.

    The :math:`K_{DP}` retrieval will return NaNs in case at least one value
    in the moving window is NaN. By default, the remaining gates are treated by
    using local linear regression where possible.

    Please note that the moving window size ``winlen`` is specified as the
    number of range gates. Thus, this argument might need adjustment in case the
    range resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value ``winlen=7``
    was chosen for a range resolution of 1km.

    Uses :func:`~wradlib.util.derivate` to calculate the derivation. See for
    additional kwargs.

    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        multi-dimensional array, note that the range dimension must be the
        last dimension of the input array.
    winlen : int
        Width of the window (as number of range gates)
    dr : float
        gate length in km
    method : str
        Defaults to 'lanczos_conv'. Can also take one of 'lanczos_dot', 'lstsq',
        'cov', 'cov_nan', 'matrix_inv'.
    skipna : bool
        Defaults to True. Local Linear regression removing NaN values using
        valid neighbors > min_periods

    Keyword Arguments
    -----------------
    min_periods : int
        Minimum number of valid values in moving window for linear regression.
        Defaults to winlen // 2 + 1.

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        array of :math:`K_{DP}` with the same shape as phidp

    Examples
    --------

    >>> import wradlib
    >>> import numpy as np
    >>> import matplotlib.pyplot as pl
    >>> pl.interactive(True)
    >>> kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    >>> phidp_true = np.cumsum(kdp_true)
    >>> phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    >>> gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    >>> phidp_raw[gaps] = np.nan
    >>> kdp_re = wradlib.dp.kdp_from_phidp(phidp_raw)
    >>> line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa
    >>> pl.show()
    """
    pad_mode = kwargs.pop("pad_mode", None)
    if pad_mode is None:
        pad_mode = "reflect"
    min_periods = kwargs.pop("min_periods", winlen // 2 + 1)
    return (
        util.derivate(
            phidp,
            winlen=winlen,
            skipna=skipna,
            method=method,
            pad_mode=pad_mode,
            min_periods=min_periods,
            **kwargs,
        )
        / 2
        / dr
    )


def unfold_phi(phidp, rho, width=5, copy=False):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the fast Fortran-based implementation (RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    width : int
       Width of the analysis window
    copy : bool
       Leaves original ``phidp`` array unchanged if set to True
       (default: False)
    """
    # Check whether fast Fortran implementation is available
    speedup = util.import_optional("wradlib.speedup")

    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = util.gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r : r + 9], -1)

    phidp = speedup.f_unfold_phi(
        phidp=phidp.astype("f4"),
        rho=rho.astype("f4"),
        gradphi=gradphi.astype("f4"),
        stdarr=stdarr.astype("f4"),
        beams=beams,
        rs=rs,
        w=width,
    )

    return phidp.reshape(shape)


def unfold_phi_naive(phidp, rho, width=5, copy=False):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the slow Python-based implementation (NOT RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    width : int
       Width of the analysis window
    copy : bool
        Leaves original ``phidp`` array unchanged if set to True
        (default: False)

    """
    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = util.gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r : r + 9], -1)

    # phi_corr = np.zeros(phidp.shape)
    for beam in range(beams):

        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - width):
            if (np.sum(stdarr[beam, j : j + width] < 5) == width) and (
                np.sum(rho[beam, j : j + 5] > 0.9) == width
            ):
                break

        ref = np.mean(phidp[beam, j : j + width])
        for k in range(j + width, rs):
            if np.sum(stdarr[beam, k - width : k] < 5) and np.logical_and(
                gradphi[beam, k] > -5, gradphi[beam, k] < 20
            ):
                ref += gradphi[beam, k] * 0.5
                if phidp[beam, k] - ref < -80:
                    if phidp[beam, k] < 0:
                        phidp[beam, k] += 360
            elif phidp[beam, k] - ref < -80:
                if phidp[beam, k] < 0:
                    phidp[beam, k] += 360
    return phidp


@deprecation.deprecated(
    deprecated_in="1.7",
    removed_in="2.0",
    current_version=version.version,
    details="Use `wradlib.util.despeckle` " "instead.",
)
def linear_despeckle(data, ndespeckle=3, copy=False):
    """Remove floating pixels in between NaNs in a multi-dimensional array.

    Warning
    -------
    This function changes the original input array if argument copy is set to
    default (False).

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Note that the range dimension must be the last dimension of the
        input array.
    ndespeckle : int
        (must be either 3 or 5, 3 by default),
        Width of the window in which we check for speckle
    copy : bool
        If True, the input array will remain unchanged.

    """
    return util.despeckle(data, n=ndespeckle, copy=copy)


def texture(data):
    """Compute the texture of data.

    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        multi-dimensional array with shape (..., number of beams, number
        of range bins)

    Returns
    ------
    texture : :class:`numpy:numpy.ndarray`
        array of textures with the same shape as data

    """
    # one-element wrap-around padding for last two axes
    x = np.pad(data, [(0,)] * (data.ndim - 2) + [(1,), (1,)], mode="wrap")

    # set first and last range elements to NaN
    x[..., 0] = np.nan
    x[..., -1] = np.nan

    # get neighbours using views into padded array
    x1 = x[..., :-2, 1:-1]  # center:2
    x2 = x[..., 1:-1, :-2]  # 4
    x3 = x[..., 2:, 1:-1]  # 8
    x4 = x[..., 1:-1, 2:]  # 6
    x5 = x[..., :-2, :-2]  # 1
    x6 = x[..., :-2, 2:]  # 3
    x7 = x[..., 2:, 2:]  # 9
    x8 = x[..., 2:, :-2]  # 7

    # stack arrays
    xa = np.array([x1, x2, x3, x4, x5, x6, x7, x8])

    # get count of valid neighbouring pixels
    xa_valid_count = np.count_nonzero(~np.isnan(xa), axis=0)

    # root mean of squared differences
    rmsd = np.sqrt(np.nansum((xa - data) ** 2, axis=0) / xa_valid_count)

    # reinforce that NaN values should have NaN textures
    rmsd[np.isnan(data)] = np.nan

    return rmsd


def depolarization(zdr, rho):
    """Compute the depolarization ration.

    Compute the depolarization ration using differential
    reflectivity :math:`Z_{DR}` and crosscorrelation coefficient
    :math:`Rho_{HV}` of a radar sweep (:cite:`Kilambi2018`,
    :cite:`Melnikov2013`, :cite:`Ryzhkov2017`).

    Parameters
    ----------
    zdr : float or :class:`numpy:numpy.ndarray`
        differential reflectivity
    rho : float or :class:`numpy:numpy.ndarray`
        crosscorrelation coefficient

    Returns
    ------
    depolarization : :class:`numpy:numpy.ndarray`
        array of depolarization ratios with the same shape as input data,
        numpy broadcasting rules apply
    """
    zdr = trafo.idecibel(np.asanyarray(zdr))
    m = 2 * np.asanyarray(rho) * zdr ** 0.5

    return trafo.decibel((1 + zdr - m) / (1 + zdr + m))


if __name__ == "__main__":
    print("wradlib: Calling module <dp> as main...")
