#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Dual-Pol and Differential Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
--------

This module provides algorithms to process polarimetric radar moments,
namely the differential phase, :math:`Phi_{DP}`, and, based on successful
:math:`Phi_{DP}` retrieval, also the specific differential phase,
:math:`K_{DP}`.
Please note that the actual application of polarimetric moments is implemented
in the corresponding wradlib modules, e.g.:

    - fuzzy echo classification from polarimetric moments
      (:meth:`wradlib.clutter.classify_echo_fuzzy`)
    - attenuation correction (:meth:`wradlib.atten.pia_from_kdp`)
    - direct precipitation retrieval from Kdp (:meth:`wradlib.trafo.kdp2r`)

Establishing a valid :math:`Phi_{DP}` profile for :math:`K_{DP}` retrieval
involves despeckling (linear_despeckle), phase unfolding, and iterative
retrieval of :math:`Phi_{DP}` form :math:`K_{DP}`.
The main workflow and its single steps is based on a publication by
:cite:`Vulpiani2012`. For convenience, the entire workflow has been
put together in the function :meth:`wradlib.dp.process_raw_phidp_vulpiani`.

Once a valid :math:`Phi_{DP}` profile has been established, the
`kdp_from_phidp` functions can be used to retrieve :math:`K_{DP}`.

Please note that so far, the functions in this module were designed to increase
performance. This was mainly achieved by allowing the simultaneous application
of functions over multiple array dimensions. The only requirement to apply
these function is that the **range dimension must be the last dimension** of
all input arrays.


.. autosummary::
   :nosignatures:
   :toctree: generated/

    process_raw_phidp_vulpiani
    kdp_from_phidp_finitediff
    kdp_from_phidp_linregress
    kdp_from_phidp_convolution
    kdp_from_phidp_sobel
    unfold_phi_vulpiani
    unfold_phi
    linear_despeckle
    texture

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.stats import linregress
from scipy.ndimage.filters import convolve1d
from . import util as util


def process_raw_phidp_vulpiani(phidp, dr, N_despeckle=5, L=7,
                               niter=2, copy=False):
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
    phidp : array
        array of shape (n azimuth angles, n range gates)
    dr : gate length in km
    N_despeckle : integer
        *N* parameter of function dp.linear_despeckle
    L : integer
        *L* parameter of :meth:`~wradlib.dp.kdp_from_phidp_convolution`
    niter : integer
        Number of iterations in which phidp is retrieved from kdp
        and vice versa
    copy : boolean
        if True, the original phidp array will remain unchanged

    Returns
    -------
    phidp : array of shape (n azimuth angles, n range gates)
        reconstructed phidp
    kdp : array of shape (n azimuth angles, n range gates)
        kdp estimate corresponding to phidp output

    Examples
    --------

    See :ref:`notebooks/verification/wradlib_verify_example.ipynb`.

    """
    if copy:
        phidp = phidp.copy()

    # despeckle
    phidp = linear_despeckle(phidp, N_despeckle)
    # kdp retrieval first guess
    kdp = kdp_from_phidp_convolution(phidp, dr=dr, L=L)
    # remove extreme values
    kdp[kdp > 20] = 0
    kdp[np.logical_and(kdp < -2, kdp > -20)] = 0

    # unfold phidp
    phidp = unfold_phi_vulpiani(phidp, kdp)

    # clean up unfolded PhiDP
    phidp[phidp > 360] = np.nan

    # kdp retrieval second guess
    kdp = kdp_from_phidp_convolution(phidp, dr=dr, L=L)
    kdp = _fill_sweep(kdp)

    # remove remaining extreme values
    kdp[kdp > 20] = 0
    kdp[kdp < -2] = 0

    # start the actual phidp/kdp iteration
    for i in range(niter):
        # phidp from kdp through integration
        phidp = 2 * np.cumsum(kdp, axis=-1) * dr
        # kdp from phidp by convolution
        kdp = kdp_from_phidp_convolution(phidp, dr=dr, L=L)
        # convert all NaNs to zeros (normally, this line can be assumed
        # to be redundant)
        kdp = _fill_sweep(kdp)

    return phidp, kdp


def unfold_phi_vulpiani(phidp, kdp):
    """Alternative phase unfolding which completely relies on Kdp.

    This unfolding should be used in oder to iteratively reconstruct
    phidp and Kdp (see :cite:`Vulpiani2012`).

    Parameters
    ----------
    phidp : array of floats
    kdp : array of floats

    """
    # unfold phidp
    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))
    kdp = kdp.reshape((-1, shape[-1]))

    for beam in range(len(phidp)):
        below_th3 = kdp[beam] < -20
        try:
            idx1 = np.where(below_th3)[0][2]
            phidp[beam, idx1:] += 360
        except Exception:
            pass

    return phidp.reshape(shape)


def _fill_sweep(dat, kind="nan_to_num", fill_value=0.):
    """Fills missing data in a 1d profile

    Parameters
    ----------
    dat : array of shape (n azimuth angles, n range gates)
    kind : string
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
            dat[beam, invalid] = 0.
            continue
        f = interp1d(validx, dat[beam, validx], kind=kind,
                     bounds_error=False, fill_value=fill_value)
        invalidx = np.where(invalid)[0]
        dat[beam, invalidx] = f(invalidx)
    return dat.reshape(shape)


def kdp_from_phidp_finitediff(phidp, L=7, dr=1.):
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}` by applying a moving
    window range finite difference derivative.

    See :cite:`Vulpiani2012` for details about this approach.

    Please note that the moving window size *L* is specified as the number of
    range gates. Thus, this argument might need adjustment in case the
    range resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value L=7 was
    chosen for a range resolution of 1km.

    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of
        the input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km
    """
    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."
    # Make really sure L is an integer
    L = int(L)
    kdp = np.zeros(phidp.shape)
    for r in range(int(L / 2), phidp.shape[-1] - int(L / 2)):
        kdp[..., r] = (phidp[..., r + int(L / 2)] -
                       phidp[..., r - int(L / 2)]) / (L - 1)
    return kdp / 2. / dr


def kdp_from_phidp_linregress(phidp, L=7, dr=1.):
    """Alternative :math:`K_{DP}` from :math:`Phi_{DP}` by applying a moving
    window linear regression.

    Please note that the moving window size *L* is specified as the number of
    range gates. Thus, this argument might need adjustment in case the range
    resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value L=7
    was chosen for a range resolution of 1km.

    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of the
        input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km

    Examples
    --------
    >>> import wradlib
    >>> import numpy as np
    >>> import pylab as pl
    >>> pl.interactive(True)
    >>> kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    >>> phidp_true = np.cumsum(kdp_true)
    >>> phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    >>> gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    >>> phidp_raw[gaps] = np.nan
    >>> kdp_re = wradlib.dp.kdp_from_phidp_linregress(phidp_raw)
    >>> line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa

    """
    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."

    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))

    # Make really sure L is an integer
    L = int(L)

    x = np.arange(phidp.shape[-1])
    valids = ~np.isnan(phidp)
    kdp = np.zeros(phidp.shape) * np.nan

    for beam in range(len(phidp)):
        for r in range(int(L / 2), phidp.shape[-1] - int(L / 2)):
            # iterate over gates
            ix = np.arange(r - L / 2, r + L / 2 + 1, dtype=np.int)
            if np.sum(valids[beam, ix]) < L / 2:
                # not enough valid values inside our window
                continue
            kdp[beam, r] = linregress(x[ix][valids[beam, ix]],
                                      phidp[beam, ix[valids[beam, ix]]])[0]
        # take care of the start and end of the beam
        #   start
        ix = np.arange(0, L)
        if np.sum(valids[beam, ix]) >= L / 2:
            kdp[beam, ix] = linregress(x[ix][valids[beam, ix]],
                                       phidp[beam, ix[valids[beam, ix]]])[0]
        # end
        ix = np.arange(shape[-1] - L, shape[-1])
        if np.sum(valids[beam, ix]) >= L / 2:
            kdp[beam, ix] = linregress(x[ix][valids[beam, ix]],
                                       phidp[beam, ix[valids[beam, ix]]])[0]

    # accounting for forward/backward propagation AND gate length
    return kdp.reshape(shape) / 2. / dr


def kdp_from_phidp_sobel(phidp, L=7, dr=1.):
    """Alternative :math:`K_{DP}` from :math:`Phi_{DP}` by applying a sobel
    filter where possible and linear regression otherwise.

    The results are quite similar to the moving window linear regression, but
    this is much faster, depending on the percentage of NaN values in the beam,
    though. The Sobel filter is applied everywhere but will return NaNs in case
    only one value in the moving window is NaN. The remaining NaN values are
    then dealt with by using local linear regression
    (see :meth:`~wradlib.dp.kdp_from_phidp_linregress`).

    This Sobel filter solution has been provided by Scott Collis at
    StackOverflow :cite:`Sobel-linfit`

    Please note that the moving window size *L* is specified as the number of
    range gates. Thus, this argument might need adjustment in case the range
    resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value L=7
    was chosen for a range resolution of 1km.

    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of the
        input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km

    Examples
    --------
    >>> import wradlib
    >>> import numpy as np
    >>> import pylab as pl
    >>> pl.interactive(True)
    >>> kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    >>> phidp_true = np.cumsum(kdp_true)
    >>> phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    >>> gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    >>> phidp_raw[gaps] = np.nan
    >>> kdp_re = wradlib.dp.kdp_from_phidp_linregress(phidp_raw)
    >>> line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa

    """
    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."

    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))

    # Make really sure L is an integer
    L = int(L)

    kdp = np.zeros(phidp.shape) * np.nan

    # do it fast using the sobel filter
    for beam in range(len(phidp)):
        kdp[beam, :] = sobel(phidp[beam, :], window_len=L)

    # find remaining NaN values with valid neighbours
    x = np.arange(phidp.shape[-1])
    invalidkdp = np.isnan(kdp)
    validphidp = ~np.isnan(phidp)
    kernel = np.ones(L, dtype="i4")
    # and do the slow moving window linear regression
    for beam in range(len(phidp)):
        # number of valid neighbours around one gate
        nvalid = np.convolve(validphidp[beam], kernel, "same") > L / 2
        # find those gates which have invalid Kdp AND enough valid neighbours
        nangates = np.where(invalidkdp[beam] & nvalid)[0]
        # now iterate over those
        for r in nangates:
            ix = np.arange(min(0, r - L / 2), max(shape[-1], r + L / 2 + 1))
            # check again (just to make sure...)
            if np.sum(validphidp[beam, ix]) < L / 2:
                # not enough valid values inside our window
                continue
            kdp[beam, r] = linregress(x[ix][validphidp[beam, ix]],
                                      phidp[beam,
                                            ix[validphidp[beam, ix]]])[0]
        # take care of the start and end of the beam
        #   start
        ix = np.arange(0, L)
        if np.sum(validphidp[beam, ix]) >= L / 2:
            kdp[beam, ix] = linregress(x[ix][validphidp[beam, ix]],
                                       phidp[beam,
                                             ix[validphidp[beam, ix]]])[0]
        # end
        ix = np.arange(shape[-1] - L, shape[-1])
        if np.sum(validphidp[beam, ix]) >= L / 2:
            kdp[beam, ix] = linregress(x[ix][validphidp[beam, ix]],
                                       phidp[beam,
                                             ix[validphidp[beam, ix]]])[0]

    # accounting for forward/backward propagation AND gate length
    return kdp.reshape(shape) / 2. / dr


def sobel(x, window_len=7):
    """Sobel differential filter for calculating KDP.

    This solution has been taken from StackOverflow :cite:`Sobel-linfit`

    Returns
    -------
    output : differential signal (unscaled for gate spacing)

    """
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = 2.0 * np.arange(window_len) / (window_len - 1.0) - 1.0
    w = w / (abs(w).sum())
    y = np.convolve(w, s, mode='valid')
    return (-1.0 * y[int(window_len / 2):len(x) + int(window_len / 2)] /
            (window_len / 3.0))


def kdp_from_phidp_convolution(phidp, L=7, dr=1.):
    """Alternative :math:`K_{DP}` from :math:`Phi_{DP}` by applying a
    convolution filter where possible and linear regression otherwise.

    The results are very similar to the moving window linear regression, but
    the convolution is *much* faster, depending on the percentage of NaN values
    in the beam, though.

    The convolution filter was suggested by Kai Mühlbauer (University of Bonn).

    The filter provides fast :math:`K_{DP}` retrieval but will return NaNs in
    case at least one value in the moving window is NaN. The remaining gates
    are treated by using local linear regression where possible
    (see :meth:`~wradlib.dp.kdp_from_phidp_linregress`).

    Please note that the moving window size *L* is specified as the number of
    range gates. Thus, this argument might need adjustment in case the
    range resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value L=7
    was chosen for a range resolution of 1km.

    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of the
        input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km

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
    >>> kdp_re = wradlib.dp.kdp_from_phidp_linregress(phidp_raw)
    >>> line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa
    >>> pl.show()

    """
    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."

    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))

    # Make really sure L is an integer
    L = int(L)

    window = 2. * np.arange(L) / (L - 1.0) - 1.0
    window = window / (abs(window).sum())
    window = window[::-1]
    kdp = convolve1d(phidp, window, axis=1) / (len(window) / 3.0)

    # find remaining NaN values with valid neighbours
    invalidkdp = np.isnan(kdp)
    if not np.any(invalidkdp.ravel()):
        # No NaN? Return KdP
        return kdp.reshape(shape) / 2. / dr

    # Otherwise continue
    x = np.arange(phidp.shape[-1])
    validphidp = ~np.isnan(phidp)
    kernel = np.ones(L, dtype="i4")
    # and do the slow moving window linear regression
    for beam in range(len(phidp)):
        # number of valid neighbours around one gate
        nvalid = np.convolve(validphidp[beam], kernel, "same") > L / 2
        # find those gates which have invalid Kdp AND enough valid neighbours
        nangates = np.where(invalidkdp[beam] & nvalid)[0]
        # now iterate over those
        for r in nangates:
            ix = np.arange(max(0, r - int(L / 2)),
                           min(r + int(L / 2) + 1, shape[-1]))
            # check again (just to make sure...)
            if np.sum(validphidp[beam, ix]) < L / 2:
                # not enough valid values inside our window
                continue
            kdp[beam, r] = linregress(x[ix][validphidp[beam, ix]],
                                      phidp[beam, ix[validphidp[beam, ix]]])[0]
        # take care of the start and end of the beam
        #   start
        ix = np.arange(0, L)
        if np.sum(validphidp[beam, ix]) >= 2:
            kdp[beam, 0:int(L / 2)] = linregress(x[ix][validphidp[beam, ix]],
                                                 phidp[beam,
                                                       ix[validphidp[beam,
                                                                     ix]]])[0]
        # end
        ix = np.arange(shape[-1] - L, shape[-1])
        if np.sum(validphidp[beam, ix]) >= 2:
            kdp[beam, -int(L / 2):] = linregress(x[ix][validphidp[beam, ix]],
                                                 phidp[beam,
                                                       ix[validphidp[beam,
                                                                     ix]]])[0]

    # accounting for forward/backward propagation AND gate length
    return kdp.reshape(shape) / 2. / dr


def unfold_phi(phidp, rho, width=5, copy=False):
    """
    Unfolds differential phase by adjusting values that exceeded maximum
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the fast Fortran-based implementation (RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : array of shape (...,nr) with nr being the number of range bins
    rho : array of same shape as phidp
    width : integer
       Width of the analysis window
    copy : boolean
       Leaves original phidp array unchanged if set to True (default: False)
    """
    # Check whether fast Fortran implementation is available
    speedup = util.import_optional("wradlib.speedup")

    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r:r + 9], -1)

    phidp = speedup.f_unfold_phi(phidp=phidp.astype("f4"),
                                 rho=rho.astype("f4"),
                                 gradphi=gradphi.astype("f4"),
                                 stdarr=stdarr.astype("f4"),
                                 beams=beams, rs=rs, w=width)

    return phidp.reshape(shape)


def unfold_phi_naive(phidp, rho, width=5, copy=False):
    """
    Unfolds differential phase by adjusting values that exceeded maximum
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the slow Python-based implementation (NOT RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : array of shape (...,nr) with nr being the number of range bins
    rho : array of same shape as phidp
    width : integer
       Width of the analysis window
    copy : boolean
       Leaves original phidp array unchanged if set to True (default: False)

    """
    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r:r + 9], -1)

    # phi_corr = np.zeros(phidp.shape)
    for beam in range(beams):

        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - width):
            if (np.sum(stdarr[beam, j:j + width] < 5) == width) and \
                    (np.sum(rho[beam, j:j + 5] > 0.9) == width):
                break

        ref = np.mean(phidp[beam, j:j + width])
        for k in range(j + width, rs):
            if np.sum(stdarr[beam, k - width:k] < 5) and \
                    np.logical_and(gradphi[beam, k] > -5,
                                   gradphi[beam, k] < 20):
                ref += gradphi[beam, k] * 0.5
                if phidp[beam, k] - ref < -80:
                    if phidp[beam, k] < 0:
                        phidp[beam, k] += 360
            elif phidp[beam, k] - ref < -80:
                if phidp[beam, k] < 0:
                    phidp[beam, k] += 360
    return phidp


def linear_despeckle(data, N=3, copy=False):
    """Remove floating pixels in between NaNs in a multi-dimensional array.

    Warning
    -------
    This function changes the original input array if argument copy is set to
    default (False).

    Parameters
    ----------
    data : multi-dimensional array
        Note that the range dimension must be the last dimension of the
        input array.

    N : integer (must be either 3 or 5, 3 by default)
        Width of the window in which we check for speckle

    copy : Boolean
        If True, the input array will remain unchanged.

    """
    assert N in (3, 5), \
        "Window size N for function linear_despeckle must be 3 or 5."
    if copy:
        data = data.copy()
    axis = data.ndim - 1
    arr = np.ones(data.shape, dtype="i4")
    arr[np.isnan(data)] = 0
    arr_plus1 = np.roll(arr, shift=1, axis=axis)
    arr_minus1 = np.roll(arr, shift=-1, axis=axis)
    if N == 3:
        # for a window of size 3
        test = arr + arr_plus1 + arr_minus1
        data[np.logical_and(np.logical_not(np.isnan(data)), test < 2)] = np.nan
    else:
        # for a window of size 5
        arr_plus2 = np.roll(arr, shift=2, axis=axis)
        arr_minus2 = np.roll(arr, shift=-2, axis=axis)
        test = arr + arr_plus1 + arr_minus1 + arr_plus2 + arr_minus2
        data[np.logical_and(np.logical_not(np.isnan(data)), test < 3)] = np.nan
    # remove isolated pixels at the first gate
    secondgate = np.squeeze(np.take(data, range(1, 2), data.ndim - 1))
    data[..., 0][np.isnan(secondgate)] = np.nan
    return data


def texture(data):
    """
    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`).
    NaN values in the original array have NaN textures.

    Parameters
    ----------
    data : multi-dimensional array with shape (..., number of beams, number
        of range bins)

    Returns
    ------
    texture : array of textures with the same shape as data

    """
    x1 = np.roll(data, 1, -2)  # center:2
    x2 = np.roll(data, 1, -1)  # 4
    x3 = np.roll(data, -1, -2)  # 8
    x4 = np.roll(data, -1, -1)  # 6
    x5 = np.roll(x1, 1, -1)  # 1
    x6 = np.roll(x4, 1, -2)  # 3
    x7 = np.roll(x3, -1, -1)  # 9
    x8 = np.roll(x2, -1, -2)  # 7

    # at least one NaN would give a sum of NaN
    xa = np.array([x1, x2, x3, x4, x5, x6, x7, x8])

    # get count of valid neighboring pixels
    xa_valid = np.ones(np.shape(xa))
    xa_valid[np.isnan(xa)] = 0
    # count number of valid neighbors
    xa_valid_count = np.sum(xa_valid, axis=0)

    num = np.zeros(data.shape)
    for xarr in xa:
        diff = data - xarr
        # difference of NaNs will be converted to zero
        # (to not affect the summation)
        diff[np.isnan(diff)] = 0
        # only those with valid values are considered in the summation
        num += diff ** 2

    # reinforce that NaN values should have NaN textures
    num[np.isnan(data)] = np.nan

    return np.sqrt(num / xa_valid_count)


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition".

    This function was adopted from an StackOverflow answer as proposed
    by Joe Kington in 2010 :cite:`Consecutive-values`.

    Parameters
    ----------
    condition : 1d boolean array

    Returns
    -------
    output : a 2D array where the first column is the start index of the region
        and the second column is the end index.
    """

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


# TO UTILS
def medfilt_along_axis(x, N, axis=-1):
    """Applies median filter smoothing on one axis of an N-dimensional array.
    """
    kernel_size = np.array(x.shape)
    kernel_size[:] = 1
    kernel_size[axis] = N
    return medfilt(x, kernel_size)


# TO UTILS
def gradient_along_axis(x):
    """Computes gradient along last axis of an N-dimensional array
    """
    axis = -1
    newshape = np.array(x.shape)
    newshape[axis] = 1
    diff_begin = (x[..., 1] - x[..., 0]).reshape(newshape)
    diff_end = (x[..., -1] - x[..., -2]).reshape(newshape)
    diffs = ((x - np.roll(x, 2, axis)) / 2.)
    diffs = np.append(diffs[..., 2:], diff_end, axis=axis)
    return np.insert(diffs, 0, diff_begin, axis=axis)


# TO UTILS
def gradient_from_smoothed(x, N=5):
    """Computes gradient of smoothed data along final axis of an array
    """
    return gradient_along_axis(medfilt_along_axis(x, N)).astype("f4")


if __name__ == '__main__':
    print('wradlib: Calling module <dp> as main...')
