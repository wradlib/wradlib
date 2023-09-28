#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
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
      (:func:`wradlib.classify.classify_echo_fuzzy`)
    - attenuation correction (:func:`wradlib.atten.pia_from_kdp`)
    - direct precipitation retrieval from Kdp (:func:`wradlib.trafo.kdp_to_r`)

Establishing a valid :math:`Phi_{{DP}}` profile for :math:`K_{{DP}}` retrieval
involves despeckling (:func:`wradlib.util.despeckle`), phase unfolding, and iterative
retrieval of :math:`Phi_{{DP}}` form :math:`K_{{DP}}`.
The main workflow and its single steps is based on a publication by
:cite:`Vulpiani2012`. For convenience, the entire workflow has been
put together in the function :func:`wradlib.dp._phidp_vulpiani`.

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
    "depolarization",
    "kdp_from_phidp",
    "phidp_kdp_vulpiani",
    "texture",
    "unfold_phi",
    "unfold_phi_vulpiani",
    "DpMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

from functools import singledispatch

import numpy as np
import xarray as xr
from scipy import integrate, interpolate
from xradar.model import sweep_vars_mapping

from wradlib import trafo, util


@singledispatch
def phidp_kdp_vulpiani(
    obj, dr, *, ndespeckle=5, winlen=7, niter=2, copy=False, **kwargs
):
    """Establish consistent :math:`Phi_{DP}` profiles from raw data.

    This approach is based on :cite:`Vulpiani2012` and involves a
    two-step procedure of :math:`Phi_{DP}` reconstruction.

    Processing of raw :math:`Phi_{DP}` data contains the following steps:

        - Despeckle
        - Initial :math:`K_{DP}` estimation
        - Removal of artifacts
        - Phase unfolding
        - :math:`Phi_{DP}` reconstruction using iterative estimation
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
        Number of iterations in which :math:`Phi_{DP}` is retrieved from
        :math:`K_{DP}` and vice versa, defaults to 2.
    copy : bool, optional
        if True, the original :math:`Phi_{DP}` array will remain unchanged,
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
        :math:`Phi_{DP}`
    kdp : :class:`numpy:numpy.ndarray`
        array of shape (..., n azimuth angles, n range gates)
        ``kdp`` estimate corresponding to ``phidp`` output

    Examples
    --------

    See :ref:`/notebooks/verification/verification.ipynb`.

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
        phidp = 2 * integrate.cumtrapz(kdp, dx=dr, initial=0.0, axis=-1)
        # kdp from phidp by convolution
        kdp = kdp_from_phidp(phidp, dr=dr, winlen=winlen, **kwargs)

    return phidp, kdp


@phidp_kdp_vulpiani.register(xr.DataArray)
def _phidp_kdp_vulpiani_xarray(obj, *, winlen=7, **kwargs):
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}`.

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
        input_core_dims=[[dim0, "range"], [None]],
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
    phidp = util.get_dataarray(obj, phidp)
    kdp = util.get_dataarray(obj, kdp)
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
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}`.

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


def _unfold_phi_naive(phidp, rho, gradphi, stdarr, beams, rs, w):
    """This is the slow Python-based implementation (NOT RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.
    """
    for beam in range(beams):
        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - w):
            if (np.sum(stdarr[beam, j : j + w] < 5) == w) and (
                np.sum(rho[beam, j : j + 5] > 0.9) == w
            ):
                break

        ref = np.mean(phidp[beam, j : j + w])
        for k in range(j + w, rs):
            if np.sum(stdarr[beam, k - w : k] < 5) and np.logical_and(
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
def unfold_phi(phidp, rho, *, width=5, copy=False):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    Uses the fast Fortran-based implementation if the speedup module is compiled.

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    width : int, optional
       Width of the analysis window, defaults to 5.
    copy : bool, optional
       Leaves original `phidp` array unchanged if set to True
       (default: False)

    Returns
    -------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (..., n azimuth angles, n range gates) reconstructed
        :math:`Phi_{DP}`
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
def texture(data):
    """Compute the texture of data.

    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        multidimensional array with shape (..., number of beams, number
        of range bins)

    Returns
    -------
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


@texture.register(xr.Dataset)
@texture.register(xr.DataArray)
def _texture_xarray(obj):
    """Compute the texture of data.

    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray`
        DataArray

    Returns
    ------
    texture : :py:class:`xarray:xarray.DataArray`
        DataArray
    """
    dim0 = obj.wrl.util.dim0()
    if isinstance(obj, xr.Dataset):
        obj, keep = util.get_apply_ufunc_variables(obj, dim0)
    out = xr.apply_ufunc(
        texture,
        obj,
        input_core_dims=[[dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    if isinstance(obj, xr.DataArray):
        attrs = obj.attrs
        standard_name = attrs["standard_name"].split("_")
        standard_name.append("texture")
        attrs["standard_name"] = "_".join(standard_name)
        attrs["long_name"] = "Texture of " + attrs["long_name"]
        attrs["units"] = "unitless"
        out.attrs = attrs
        out.name = obj.name + "_TEXTURE"
    else:
        out = xr.merge([out, keep])
    return out


@singledispatch
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
    m = 2 * np.asanyarray(rho) * zdr**0.5

    return trafo.decibel((1 + zdr - m) / (1 + zdr + m))


@depolarization.register(xr.Dataset)
def _depolarization_xarray(obj: xr.Dataset, **kwargs):
    """Compute the depolarization ration.

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
    dim0 = obj.wrl.util.dim0()
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
        input_core_dims=[[dim0, "range"], [dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
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


if __name__ == "__main__":
    print("wradlib: Calling module <dp> as main...")
