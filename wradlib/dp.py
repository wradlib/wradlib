#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2026, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

r"""
Dual-Pol and Differential Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
--------

This module provides algorithms to process polarimetric radar moments,
namely the differential phase, :math:`\Phi_{DP}`, and, based on successful
:math:`\Phi_{DP}` retrieval, also the specific differential phase,
:math:`K_{DP}`.
Please note that the actual application of polarimetric moments is implemented
in the corresponding wradlib modules, e.g.:

    - fuzzy echo classification from polarimetric moments
      (:func:`wradlib.classify.classify_echo_fuzzy`)
    - attenuation correction (:func:`wradlib.atten.pia_from_kdp`)
    - direct precipitation retrieval from Kdp (:func:`wradlib.trafo.kdp_to_r`)

Establishing a valid :math:`\Phi_{DP}` profile for :math:`K_{DP}` retrieval
involves despeckling (:func:`wradlib.util.despeckle`), phase unfolding, and iterative
retrieval of :math:`\Phi_{DP}` form :math:`K_{DP}`.
The main workflow and its single steps is based on a publication by
:cite:`Vulpiani2012`. For convenience, the entire workflow has been
put together in the function :func:`wradlib.dp.phidp_kdp_vulpiani`.

Once a valid :math:`\Phi_{DP}` profile has been established, the
:func:`wradlib.dp.kdp_from_phidp` function can be used to retrieve :math:`K_{DP}`.

Please note that so far, the functions in this module were designed to increase
performance. This was mainly achieved by allowing the simultaneous application
of functions over multiple array dimensions. The only requirement to apply
these function is that the **range dimension must be the last dimension** of
all input arrays.
"""

_AUTOSUMMARY = r"""
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "depolarization",
    "kdp_from_phidp",
    "phidp_kdp_vulpiani",
    "texture",
    "unfold_phi",
    "unfold_phi_vulpiani",
    "system_phidp_block",
    "system_phidp_window",
    "system_phidp_first",
    "system_phidp_hist",
    "DpMethods",
]
__doc__ = (__doc__ or "") + _AUTOSUMMARY.format("\n   ".join(__all__))

from functools import singledispatch

import numpy as np
import xarray as xr
from scipy import integrate, interpolate
from xhistogram.xarray import histogram as xhist
from xradar.model import sweep_vars_mapping

from wradlib import trafo, util


@singledispatch
def phidp_kdp_vulpiani(
    obj, dr, *, ndespeckle=5, winlen=7, niter=2, copy=False, **kwargs
):
    r"""Establish consistent :math:`\Phi_{DP}` profiles from raw data.

    This approach is based on :cite:`Vulpiani2012` and involves a
    two-step procedure of :math:`\Phi_{DP}` reconstruction.

    Processing of raw :math:`\Phi_{DP}` data contains the following steps:

        - Despeckle
        - Initial :math:`K_{DP}` estimation
        - Removal of artifacts
        - Phase unfolding
        - :math:`\Phi_{DP}` reconstruction using iterative estimation
          of :math:`K_{DP}`

    Parameters
    ----------
    obj : :class:`numpy:numpy.ndarray`
        array of shape (n azimuth angles, n range gates)
    dr : float
        gate length in km
    ndespeckle : int, optional
        ``ndespeckle`` parameter of :func:`~wradlib.util.despeckle`,
        defaults to 5
    winlen : int, optional
        ``winlen`` parameter of :func:`~wradlib.dp.kdp_from_phidp`,
        defaults to 7
    niter : int, optional
        Number of iterations in which :math:`\Phi_{DP}` is retrieved from
        :math:`K_{DP}` and vice versa, defaults to 2.
    copy : bool, optional
        if True, the original :math:`\Phi_{DP}` array will remain unchanged,
        defaults to False

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
        array of shape (..., n azimuth angles, n range gates) reconstructed
        :math:`\Phi_{DP}`
    kdp : :class:`numpy:numpy.ndarray`
        array of shape (..., n azimuth angles, n range gates)
        ``kdp`` estimate corresponding to ``phidp`` output

    Examples
    --------
    See :doc:`notebooks:notebooks/verification/verification`.

    """
    if copy:
        obj = obj.copy()

    # get thresholds
    th1 = kwargs.pop("th1", -2)
    th2 = kwargs.pop("th2", 20)
    th3 = kwargs.pop("th3", -20)

    method = kwargs.pop("method", None)

    # despeckle
    phidp = util.despeckle(obj, n=ndespeckle)

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
    for _i in range(niter):
        # phidp from kdp through integration
        phidp = 2 * integrate.cumulative_trapezoid(kdp, dx=dr, initial=0.0, axis=-1)
        # kdp from phidp by convolution
        kdp = kdp_from_phidp(phidp, dr=dr, winlen=winlen, **kwargs)

    return phidp, kdp


@phidp_kdp_vulpiani.register(xr.DataArray)
def _phidp_kdp_vulpiani_xarray(obj, *, winlen=7, **kwargs):
    r"""Retrieves :math:`K_{DP}` from :math:`\Phi_{DP}`.

    Parameter
    ---------
    obj : :py:class:`xarray:xarray.DataArray`
        DataArray containing differential phase
    winlen : int
        window length

    Keyword Arguments
    -----------------
    method : str
        Defaults to 'lanczos_conv'. Can also take one of 'lanczos_dot', 'lstsq',
        'cov', 'cov_nan', 'matrix_inv'.
    skipna : bool
        Defaults to True. Local Linear regression removing NaN values using
        valid neighbors > min_periods
    min_periods : int
        Minimum number of valid values in moving window for linear regression.
        Defaults to winlen // 2 + 1.

    Returns
    -------
    phidp : :py:class:`xarray:xarray.DataArray`
        DataArray
    kdp : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    dr = obj.range.diff("range").median("range").values / 1000.0
    phidp, kdp = xr.apply_ufunc(
        phidp_kdp_vulpiani,
        obj,
        dr,
        input_core_dims=[[dim0, "range"], []],
        output_core_dims=[[dim0, "range"], [dim0, "range"]],
        dask="parallelized",
        kwargs=dict(winlen=winlen, **kwargs),
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    phidp.attrs = sweep_vars_mapping["PHIDP"]
    phidp.name = phidp.attrs["short_name"]
    kdp.attrs = sweep_vars_mapping["KDP"]
    kdp.name = kdp.attrs["short_name"]
    return phidp, kdp


@singledispatch
def unfold_phi_vulpiani(phidp, kdp, *, th=-20, winlen=7):
    r"""Alternative phase unfolding which completely relies on :math:`K_{DP}`.

    This unfolding should be used in oder to iteratively reconstruct
    :math:`\Phi_{DP}` and :math:`K_{DP}` (see :cite:`Vulpiani2012`).

    Note
    ----
    :math:`\Phi_{DP}` is assumed to be in the interval [-180, 180] degree.
    From experience the window for calculation of :math:`K_{DP}` should not
    be too large to catch possible phase wraps.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of floats
    kdp : :class:`numpy:numpy.ndarray`
        array of floats
    th : float, optional
        Threshold th3 in the above citation, defaults to -20.
    winlen : int, optional
        Length of window to fix possible phase over-correction. Normally
        should take the value of the length of the processing window in
        the above citation, defaults to 7.

    Returns
    -------
    phidp : :class:`numpy:numpy.ndarray`
        array of floats
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


@unfold_phi_vulpiani.register(xr.Dataset)
def _unfold_phi_vulpiani_xarray(obj, **kwargs):
    r"""Alternative phase unfolding which completely relies on :math:`K_{DP}`.

    This unfolding should be used in oder to iteratively reconstruct
    :math:`\Phi_{DP}` and :math:`K_{DP}` (see :cite:`Vulpiani2012`).

    Note
    ----
    :math:`\Phi_{DP}` is assumed to be in the interval [-180, 180] degree.
    From experience the window for calculation of :math:`K_{DP}` should not
    be too large to catch possible phase wraps.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.Dataset`
        Dataset

    Keyword Arguments
    -----------------
    phidp : str
        name of PhiDP
    kdp : str
        name of KDP
    th : float
        Threshold th3 in the above citation.
    winlen : int
        Length of window to fix possible phase over-correction. Normally
        should take the value of the length of the processing window in
        the above citation.

    Returns
    -------
    out : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    phidp = kwargs.pop("phidp", None)
    kdp = kwargs.pop("kdp", None)
    if phidp is None or kdp is None:
        raise TypeError("Both `phidp` and `kdp` kwargs need to be given.")
    phidp = util.get_dataarray(obj, phidp).copy(deep=True)
    kdp = util.get_dataarray(obj, kdp).copy(deep=True)
    out = xr.apply_ufunc(
        unfold_phi_vulpiani,
        phidp,
        kdp,
        input_core_dims=[[dim0, "range"], [dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.attrs = sweep_vars_mapping["PHIDP"]
    out.name = "PHIDP"

    return out


def _fill_sweep(dat, *, kind="nan_to_num", fill_value=0.0):
    """Fills missing data in a 1D profile.

    Parameters
    ----------
    dat : :class:`numpy:numpy.ndarray`
        array of shape (n azimuth angles, n range gates)

    Keyword Arguments
    -----------------
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


@singledispatch
def kdp_from_phidp(phidp, *, winlen=7, dr=1.0, method="lanczos_conv", **kwargs):
    r"""Retrieves :math:`K_{DP}` from :math:`\Phi_{DP}`.

    In normal operation the method uses convolution to estimate :math:`K_{DP}`
    (the derivative of :math:`\Phi_{DP}`) with Low-noise Lanczos differentiators
    (`method='lanczos_conv'`, :cite:`Diekema2012`). The results are very similar to the moving window
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
        multidimensional array, note that the range dimension must be the
        last dimension of the input array.
    winlen : int, optional
        Width of the window (as number of range gates), defaults to 7
    dr : float, optional
        gate length in km, defaults to 1
    method : str, optional
        Defaults to 'lanczos_conv'. Can also take one of 'lanczos_dot', 'lstsq',
        'cov', 'cov_nan', 'matrix_inv'.

    Keyword Arguments
    -----------------
    skipna : bool
        Defaults to True. Local Linear regression removing NaN values using
        valid neighbors > min_periods
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
    >>> import matplotlib.pyplot as plt
    >>> plt.interactive(True)
    >>> kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    >>> phidp_true = np.cumsum(kdp_true)
    >>> phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    >>> gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    >>> phidp_raw[gaps] = np.nan
    >>> kdp_re = wradlib.dp.kdp_from_phidp(phidp_raw)
    >>> line1 = plt.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = plt.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = plt.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = plt.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = plt.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa
    >>> plt.show()
    """
    skipna = kwargs.pop("skipna", True)
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


@kdp_from_phidp.register(xr.DataArray)
def _kdp_from_phidp_xarray(obj, *, winlen=7, **kwargs):
    r"""Retrieves :math:`K_{DP}` from :math:`\Phi_{DP}`.

    Parameter
    ---------
    obj : :py:class:`xarray:xarray.DataArray`
        DataArray containing differential phase

    Keyword Arguments
    -----------------
    winlen : int
        window length
    method : str
        Defaults to 'lanczos_conv'. Can also take one of 'lanczos_dot', 'lstsq',
        'cov', 'cov_nan', 'matrix_inv'.
    skipna : bool
        Defaults to True. Local Linear regression removing NaN values using
        valid neighbors > min_periods
    min_periods : int
        Minimum number of valid values in moving window for linear regression.
        Defaults to winlen // 2 + 1.

    Returns
    -------
    out : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    dr = obj.range.diff("range").median("range").values / 1000.0
    out = xr.apply_ufunc(
        kdp_from_phidp,
        obj,
        input_core_dims=[[dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=dict(winlen=winlen, dr=dr, **kwargs),
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.attrs = sweep_vars_mapping["KDP"]
    out.name = out.attrs["short_name"]
    return out


def phidp_from_kdp(da):
    r"""Derive PHIDP from KDP.

    Parameter
    ---------
    da : xarray.DataArray
        array with specific differential phase data
    winlen : int
        size of window in range dimension

    Return
    ------
    phi : xarray.DataArray
        DataArray with differential phase values
    """
    dr = da.range.diff("range").median("range").values / 1000.0
    out = (
        xr.apply_ufunc(
            integrate.cumulative_trapezoid,
            da,
            input_core_dims=[["range"]],
            output_core_dims=[["range"]],
            dask="parallelized",
            kwargs=dict(dx=dr, initial=0.0, axis=-1),
        )
        * 2
    )
    out.attrs = sweep_vars_mapping["PHIDP"]
    out.name = out.attrs["short_name"]
    return out


def _unfold_phi_naive(phidp, rho, gradphi, stdarr, beams, rs, w, ts, tr):
    """This is the slow Python-based implementation (NOT RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.
    """
    for beam in range(beams):
        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - w):
            if (np.sum(stdarr[beam, j : j + w] < ts) == w) and (
                np.sum(rho[beam, j : j + w] > tr) == w
            ):
                break

        ref = np.mean(phidp[beam, j : j + w])
        for k in range(j + w, rs):
            if np.sum(stdarr[beam, k - w : k] < ts) and np.logical_and(
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


@singledispatch
def unfold_phi(phidp, rho, *, width=5, copy=False, thr_sphidp=5, thr_rho=0.9):
    r"""
    Unfold differential phase (:math:`\Phi_{DP}`) along the range dimension
    using a gate-wise phase-unfolding procedure adapted from
    :cite:`Wang2009`.

    This routine detects reliable weather gates, constructs a local reference
    phase profile, and corrects wrapped negative phase values by adding 360°
    where needed.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        Differential phase array (..., nr), where the last dimension is range.
    rho : :class:`numpy:numpy.ndarray`
        Copolar correlation coefficient array with the same shape as `phidp`.
    width : int, optional
        Number of gates used for local stability and slope checks.
        Default is 5.
    thr_sphidp : float, optional
        Maximum allowed local standard deviation of :math:`\Phi_{DP}` used when
        detecting the beginning of the valid :math:`\Phi_{DP}` profile.
        Default is 5°.
    thr_rho : float, optional
        Minimum required :math:`\rho_{HV}` within the stability window.
        Default is 0.9.
    copy : bool, optional
        If True, operate on a copy of `phidp` and leave the original `phidp`
        array unchanged.

    Returns
    -------
    phidp : :class:`numpy:numpy.ndarray`
        Unfolded :math:`\Phi_{DP}` array with the same shape as the input.

    Notes
    -----
    * Accepts arbitrarily dimensioned arrays, but the last dimension must be
      range.
    * Uses the fast Fortran-based implementation if the speedup module is
      compiled.
    * The algorithm follows the logic described by :cite:`Wang2009`:

        - The beginning of the valid :math:`\Phi_{DP}` profile is
          identified using a stability criterion based on the local standard
          deviation of :math:`\Phi_{DP}` and sufficiently high
          :math:`\rho_{HV}`.
        - A reference :math:`\Phi_{DP}` is initialised from the mean phase
          over the first reliable gates.
        - At each subsequent gate, local phase variability and the local
          :math:`\Phi_{DP}` gradient are checked before updating the reference.
        - If the observed phase falls more than 80° below the reference and is
          negative, 360° is added to unfold the phase.
    """
    # Check whether fast Fortran implementation is available
    speedup = util.import_optional("wradlib.speedup")

    if util.has_import(speedup):
        func = speedup.f_unfold_phi
        dtype = "f4"
    else:
        func = _unfold_phi_naive
        dtype = "f8"

    shape = phidp.shape
    if rho.shape != shape:
        raise ValueError(
            f"`rho` ({rho.shape}) and `phidp` ({shape}) must have the same shape."
        )

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = util.gradient_from_smoothed(phidp)

    beams, rs = phidp.shape
    # TODO: Internal thresholds could also be set as configurable args.
    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r : r + 9], -1)

    phidp = func(
        phidp=phidp.astype(dtype),
        rho=rho.astype(dtype),
        gradphi=gradphi.astype(dtype),
        stdarr=stdarr.astype(dtype),
        beams=beams,
        rs=rs,
        w=width,
        ts=thr_sphidp,
        tr=thr_rho,
    )

    return phidp.reshape(shape)


@unfold_phi.register(xr.Dataset)
def _unfold_phi_xarray(obj, **kwargs):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the fast Fortran-based implementation (RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    phidp : str
        name of PhiDP data variable
    rho : str
        name of RhoHV data variable
    width : int
       Width of the analysis window

    Returns
    -------
    out : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    phidp = kwargs.pop("phidp", None)
    rho = kwargs.pop("rho", None)
    if phidp is None or rho is None:
        raise TypeError("Both `phidp` and `rho` kwargs need to be given.")
    if isinstance(phidp, str):
        phidp = obj[phidp]
    if isinstance(rho, str):
        rho = obj[rho]
    if not isinstance(phidp, xr.DataArray):
        raise TypeError("`phidp` need to be xarray.DataArray.")
    if not isinstance(rho, xr.DataArray):
        raise TypeError("`rho` need to be xarray.DataArray.")
    out = xr.apply_ufunc(
        unfold_phi,
        phidp,
        rho,
        input_core_dims=[[dim0, "range"], [dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.attrs = phidp.attrs
    out.name = phidp.name
    return out


@singledispatch
def texture(obj):
    """``wradlib.dp.texture`` is deprecated, use :func:`wradlib.util.texture` instead."""
    util.warn(
        "`wradlib.dp.texture` is deprecated. " "Use `wradlib.util.texture` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return util.texture(obj)


@texture.register(xr.Dataset)
@texture.register(xr.DataArray)
def _texture_xarray(obj):
    """``wradlib.dp.DpMethods.texture`` is deprecated, use :meth:`wradlib.util.UtilMethods.texture` instead."""
    return util.texture(obj)


@singledispatch
def depolarization(zdr, rho):
    r"""Compute the depolarization ration.

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
    m = 2 * np.asanyarray(rho) * zdr**0.5

    return trafo.decibel((1 + zdr - m) / (1 + zdr + m))


@depolarization.register(xr.Dataset)
def _depolarization_xarray(obj: xr.Dataset, **kwargs):
    r"""Compute the depolarization ration.

    Compute the depolarization ration using differential
    reflectivity :math:`Z_{DR}` and crosscorrelation coefficient
    :math:`Rho_{HV}` of a radar sweep (:cite:`Kilambi2018`,
    :cite:`Melnikov2013`, :cite:`Ryzhkov2017`).

    Parameter
    ----------
    obj : :py:class:`xarray:xarray.Dataset`

    Keyword Arguments
    -----------------
    zdr : str
        name of differential reflectivity
    rho : str
        name crosscorrelation coefficient

    Returns
    ------
    depolarization : :py:class:`xarray:xarray.DataArray`
        array of depolarization ratios with the same shape as input data,
        numpy broadcasting rules apply
    """
    core_dims = obj.wrl.util.core_dims()
    zdr = kwargs.pop("zdr", None)
    rho = kwargs.pop("rho", None)
    if zdr is None or rho is None:
        raise TypeError("Both `zdr` and `rhp` kwargs need to be given.")
    if isinstance(zdr, str):
        zdr = obj[zdr]
    if isinstance(rho, str):
        rho = obj[rho]
    if not isinstance(zdr, xr.DataArray):
        raise TypeError("`zdr` need to be xarray.DataArray.")
    if not isinstance(rho, xr.DataArray):
        raise TypeError("`rho` need to be xarray.DataArray.")
    out = xr.apply_ufunc(
        depolarization,
        zdr,
        rho,
        input_core_dims=[core_dims[0]] * 2,
        output_core_dims=[core_dims[1]],
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    attrs = {
        "standard_name": "depolarization_ratio",
        "long_name": "Depolarization Ratio",
        "units": "unitless",
    }
    out.attrs = attrs
    out.name = "DP"
    return out


def _get_range_step(obj):
    return float(obj.range[1] - obj.range[0])


def _get_bins_from_range(obj, rng):
    return int(rng / _get_range_step(obj))


def _aggregate_sysphi(phi, n_lowest_rays):
    valid_phi = phi.where(phi.notnull(), drop=True)
    return valid_phi.sortby(valid_phi)[:n_lowest_rays].median(skipna=True)


def system_phidp_block(phidp, rng, n_lowest_rays=30):
    """
    Estimate the system differential phase (system PHIDP) from contiguous
    valid PHIDP segments along radar rays.

    The algorithm searches each ray for a sequence of N consecutive valid
    PHIDP bins, where N is derived from the requested range length (`rng`).
    For each ray, the median PHIDP within the identified segment is used as
    a ray-wise system PHIDP estimate. The final system PHIDP estimate is
    computed as the median of the `n_lowest_rays` smallest ray-wise estimates.

    Parameters
    ----------
    phidp : xarray.DataArray
        Differential phase field containing a ``range`` dimension.
    rng : float
        Range length (m) used to determine the required number of consecutive
        valid bins.
    n_lowest_rays : int, optional
        Number of lowest ray-wise PHIDP estimates used to derive the final
        system PHIDP estimate. Default is 30.

    Returns
    -------
    xarray.Dataset
        Dataset containing:

        - ``sysphi_ray`` : ray-wise system PHIDP estimate.
        - ``sysphi`` : global system PHIDP estimate.
        - ``start_range`` : Start range of the selected valid segment.
        - ``stop_range`` : Stop range of the selected valid segment.
        - ``valid_bins`` : Number of valid PHIDP bins within the selected
          interval.

    Notes
    -----
    Rays that do not contain N consecutive valid PHIDP bins receive NaN
    values for the corresponding outputs.
    """

    # binary mask of valid PHIDP bins
    phib = phidp.notnull().astype(np.int8)

    # required number of consecutive bins
    N = phidp.pipe(_get_bins_from_range, rng)

    # count valid bins in rolling window
    phib_sum = phib.rolling(range=N, center=True).sum(skipna=True)

    # maximum number of valid bins per ray
    smax = phib_sum.max(dim="range", skipna=True)

    # derive selected range interval
    rstep = _get_range_step(phib_sum)
    center_range = phib_sum.idxmax(dim="range").where(smax == N)

    start_range = center_range - (N // 2) * rstep
    stop_range = start_range + N * rstep

    start_range.name = "start_range"
    stop_range.name = "stop_range"

    # select PHIDP values within identified interval
    phi = phidp.where((phidp.range >= start_range) & (phidp.range <= stop_range))
    valid_bins = phi.count("range")
    valid_bins.name = "valid_bins"

    # ray-wise estimate
    sysphi_ray = phi.median(dim="range", skipna=True)
    sysphi_ray.name = "sysphi_ray"

    # global estimate from lowest ray-wise values
    sysphi = _aggregate_sysphi(sysphi_ray, n_lowest_rays)
    sysphi.name = "sysphi"

    return xr.merge(
        [
            sysphi_ray,
            sysphi,
            start_range,
            stop_range,
            valid_bins,
        ],
        compat="no_conflicts",
    )


def system_phidp_window(phidp, rng, n_lowest_rays=30):
    """
    Estimate the system differential phase (system PHIDP) from the
    range interval with the highest valid-data coverage.

    For each ray, a rolling window of length ``rng`` is evaluated and
    the interval containing the maximum number of valid PHIDP bins is
    selected. The median PHIDP within this interval is used as a ray-wise
    system PHIDP estimate. The final system PHIDP estimate is computed as
    the median of the ``n_lowest_rays`` smallest ray-wise estimates.

    Unlike ``system_phidp_block``, this method does not require all
    bins within the selected interval to be valid. It therefore provides
    estimates for rays that contain gaps in PHIDP coverage.

    Parameters
    ----------
    phidp : xarray.DataArray
        Differential phase field containing a ``range`` dimension.
    rng : float
        Physical range length (m) used to determine the rolling window size.
    n_lowest_rays : int, optional
        Number of lowest ray-wise estimates used to derive the final
        system PHIDP estimate. Default is 30.

    Returns
    -------
    xarray.Dataset
        Dataset containing:

        - ``sysphi_ray`` : ray-wise system PHIDP estimate.
        - ``sysphi`` : global system PHIDP estimate.
        - ``start_range`` : Start range of the selected interval.
        - ``stop_range`` : Stop range of the selected interval.
        - ``valid_bins`` : Number of valid PHIDP bins within the selected
          interval.

    Notes
    -----
    The selected interval maximizes the count of valid PHIDP bins within
    the specified window length. No minimum coverage threshold is applied.
    """

    # binary mask of valid PHIDP bins
    phib = phidp.notnull().astype(np.int8)

    # window length in bins
    N = phidp.pipe(_get_bins_from_range, rng)

    # count valid bins in rolling window
    phib_sum = phib.rolling(range=N, center=True).sum(skipna=True)

    # maximum valid-bin count per ray
    valid_bins = phib_sum.max(dim="range", skipna=True)
    valid_bins.name = "valid_bins"

    # derive selected interval
    rstep = _get_range_step(phib_sum)
    center_range = phib_sum.idxmax(dim="range")

    start_range = center_range - (N // 2) * rstep
    stop_range = start_range + N * rstep

    start_range.name = "start_range"
    stop_range.name = "stop_range"

    # extract interval and compute ray-wise estimate
    phi = phidp.where((phidp.range >= start_range) & (phidp.range <= stop_range))

    sysphi_ray = phi.median(dim="range", skipna=True)
    sysphi_ray.name = "sysphi_ray"

    # derive global estimate
    sysphi = _aggregate_sysphi(sysphi_ray, n_lowest_rays)
    sysphi.name = "sysphi"

    return xr.merge(
        [
            sysphi_ray,
            sysphi,
            start_range,
            stop_range,
            valid_bins,
        ],
        compat="no_conflicts",
    )


def system_phidp_first(phidp, n_valid_bins=10, n_lowest_rays=30):
    """
    Estimate system PHIDP using the first N valid PHIDP bins along each ray.

    For each ray, the first `n_valid_bins` valid (non-NaN) PHIDP values are
    selected, regardless of whether they are contiguous in range. The median
    of these values is computed to obtain a ray-wise system PHIDP estimate.
    The final system PHIDP is the median of the `n_lowest_rays` smallest
    ray-wise estimates.

    Parameters
    ----------
    phidp : xarray.DataArray
        Differential phase field with a ``range`` dimension.
    n_valid_bins : int, optional
        Number of valid PHIDP samples to use per ray.
    n_lowest_rays : int, optional
        Number of lowest ray-wise estimates used for the final system value.

    Returns
    -------
    xarray.Dataset
        Dataset containing:
         - ``sysphi_ray`` : ray-wise system PHIDP estimate.
        - ``sysphi`` : global system PHIDP estimate.
        - ``start_range`` : Start range of the selected interval.
        - ``stop_range`` : Stop range of the selected interval.
        - ``valid_bins`` : Number of valid PHIDP bins within the selected
          interval.
    """

    # mask valid data
    phib = phidp.notnull().astype(np.int8)

    # cumulative count of valid bins along range
    phib_cumsum = phib.cumsum("range", skipna=True)

    # select first N valid bins per ray
    mask = phib.astype(bool) & (phib_cumsum <= n_valid_bins)
    phi = phidp.where(mask)

    # diagnostics
    valid_bins = phi.count("range")
    valid_bins.name = "valid_bins"

    start_range = phidp.range.where(mask).min("range", skipna=True)
    start_range.name = "start_range"

    stop_range = phidp.range.where(mask).max("range", skipna=True)
    stop_range.name = "stop_range"

    # ray-wise statistic
    sysphi_ray = phi.median("range", skipna=True)
    sysphi_ray.name = "sysphi_ray"

    # global statistic (robust low-end aggregation)
    sysphi = _aggregate_sysphi(sysphi_ray, n_lowest_rays)
    sysphi.name = "sysphi"

    return xr.merge(
        [
            sysphi_ray,
            sysphi,
            start_range,
            stop_range,
            valid_bins,
        ],
        compat="no_conflicts",
    )


def system_phidp_hist(
    phidp, bins=(-180, 180, 0.1), window=11, threshold=0.5, n_lowest_rays=30
):
    """
    Estimate the system differential phase (PHIDP) offset from PHIDP histograms.

    A histogram of PHIDP values is computed for each azimuth ray and smoothed
    along the histogram bin dimension. Two ray-wise estimates of the system
    PHIDP are derived:

    - ``sysphi_peak_ray``: location of the histogram maximum.
    - ``sysphi_first_ray``: first histogram bin exceeding a fraction of the
      normalized peak count.

    Sweep-level estimates (``sysphi_peak`` and ``sysphi_first``) are obtained
    by aggregating the lowest ``n_lowest_rays`` ray-wise estimates.

    Parameters
    ----------
    phidp : xarray.DataArray
        Differential phase field with dimensions including ``range`` and
        typically ``azimuth``.
    bins : tuple, optional
        Histogram bin specification passed to ``np.arange`` as
        ``(start, stop, step)``. Default is ``(-180, 180, 0.1)``
    window : int, optional
        Size of the moving-average smoothing window applied to the histogram
        along the bin dimension. Default is ``11``.
    threshold : float, optional
        Relative threshold applied to the normalized histogram. The first bin
        exceeding this threshold is used to derive ``sysphi_first_ray``.
        Default is ``0.5``.
    n_lowest_rays : int, optional
        Number of lowest ray-wise estimates used when aggregating the
        sweep-level system PHIDP estimate. Default is ``30``.

    Returns
    -------
    xarray.Dataset
        Dataset containing:

        - ``sysphi_hist``: PHIDP histogram for each ray.
        - ``sysphi_peak_ray``: ray-wise estimate from the histogram peak.
        - ``sysphi_first_ray``: ray-wise estimate from the threshold crossing.
        - ``sysphi_peak``: sweep-level estimate aggregated from
          ``sysphi_peak_ray``.
        - ``sysphi_first``: sweep-level estimate aggregated from
          ``sysphi_first_ray``.
    """
    sysphi_hist = xhist(phidp, dim=("range",), bins=[np.arange(*bins)])
    sysphi_hist.name = "sysphi_hist"
    hist_dim = next(dim for dim in sysphi_hist.dims if dim.endswith("_bin"))
    sysphi_hist = sysphi_hist.rename({hist_dim: "bin"})

    phi_hist_smooth = sysphi_hist.rolling(bin=window, center=True).mean()
    sysphi_peak_ray = phi_hist_smooth.idxmax("bin")
    sysphi_peak_ray.name = "sysphi_peak_ray"

    hist_max = phi_hist_smooth.max(dim="bin")
    phi_hist_norm = phi_hist_smooth / hist_max.where(hist_max > 0)
    phi_hist_thresh = phi_hist_norm.where(phi_hist_norm > threshold)

    sysphi_first_ray = phi_hist_thresh.bin.where(phi_hist_thresh.notnull()).min("bin")
    sysphi_first_ray.name = "sysphi_first_ray"

    sysphi_peak = _aggregate_sysphi(sysphi_peak_ray, n_lowest_rays)
    sysphi_peak.name = "sysphi_peak"

    sysphi_first = _aggregate_sysphi(sysphi_first_ray, n_lowest_rays)
    sysphi_first.name = "sysphi_first"

    return xr.merge(
        [
            sysphi_hist,
            sysphi_peak_ray,
            sysphi_first_ray,
            sysphi_peak,
            sysphi_first,
        ],
        compat="no_conflicts",
    )


class DpMethods(util.XarrayMethods):
    """wradlib xarray SubAccessor methods for DualPol."""

    @util.docstring(_depolarization_xarray)
    def depolarization(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return depolarization(self, *args, **kwargs)
        else:
            return depolarization(self._obj, *args, **kwargs)

    @util.docstring(_kdp_from_phidp_xarray)
    def kdp_from_phidp(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return kdp_from_phidp(self, *args, **kwargs)
        else:
            return kdp_from_phidp(self._obj, *args, **kwargs)

    @util.docstring(phidp_from_kdp)
    def phidp_from_kdp(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return phidp_from_kdp(self, *args, **kwargs)
        else:
            return phidp_from_kdp(self._obj, *args, **kwargs)

    @util.docstring(_phidp_kdp_vulpiani_xarray)
    def phidp_kdp_vulpiani(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return phidp_kdp_vulpiani(self, *args, **kwargs)
        else:
            return phidp_kdp_vulpiani(self._obj, *args, **kwargs)

    @util.docstring(_texture_xarray)
    def texture(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return texture(self, *args, **kwargs)
        else:
            return texture(self._obj, *args, **kwargs)

    @util.docstring(_unfold_phi_xarray)
    def unfold_phi(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return unfold_phi(self, *args, **kwargs)
        else:
            return unfold_phi(self._obj, *args, **kwargs)

    @util.docstring(_unfold_phi_vulpiani_xarray)
    def unfold_phi_vulpiani(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return unfold_phi_vulpiani(self, *args, **kwargs)
        else:
            return unfold_phi_vulpiani(self._obj, *args, **kwargs)

    @util.docstring(system_phidp_block)
    def system_phidp_block(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return system_phidp_block(self, *args, **kwargs)
        else:
            return system_phidp_block(self._obj, *args, **kwargs)

    @util.docstring(system_phidp_window)
    def system_phidp_window(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return system_phidp_window(self, *args, **kwargs)
        else:
            return system_phidp_window(self._obj, *args, **kwargs)

    @util.docstring(system_phidp_first)
    def system_phidp_first(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return system_phidp_first(self, *args, **kwargs)
        else:
            return system_phidp_first(self._obj, *args, **kwargs)

    @util.docstring(system_phidp_hist)
    def system_phidp_hist(self, *args, **kwargs):
        if not isinstance(self, DpMethods):
            return system_phidp_hist(self, *args, **kwargs)
        else:
            return system_phidp_hist(self._obj, *args, **kwargs)


if __name__ == "__main__":
    print("wradlib: Calling module <dp> as main...")
