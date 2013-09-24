# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:         dp
# Purpose:      Processing related to Dual-Pol and Differential Phase
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      20.09.2013
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Dual-Pol and Differential Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module provides algorithms to process polarimentric radar moments,
namely the differential phase, PhiDP, and, based on successful PhiDP retrieval,
also the specific differential phase, Kdp. Please note that the actual application
of polarimetric moments is implemented in the corresponding wradlib modules, e.g.:

    - fuzzy echo classification from polarimetric moments (:doc:`classify_echo_fuzzy <generated/wradlib.clutter.classify_echo_fuzzy>`)

    - attenuation correction (:doc:`pia_from_kdp <generated/wradlib.atten.pia_from_kdp>`)

    - direct precipitation retrieval from Kdp (:doc:`kdp2r <generated/wradlib.trafo.kdp2r>`)

Establishing a valid PhiDP profile for Kdp retrieval involves despeckling (linear_despeckle),
gap filling (fill_phidp), phase unfolding and smoothing. For convenience, these
steps have been combined in the function :doc:`process_raw_phidp <generated/wradlib.dp.process_raw_phidp>`.

Once a valid PhiDP profile has been established, :doc:`kdp_from_phidp <generated/wradlib.dp.kdp_from_phidp>`
can be used to retrieve Kdp.

Please note that so far, the functions in this module were designed to increase
performance. This was mainly achieved by allowing the simultaneous application
of functions over multiple array dimensions. The only requirement to apply these
function is that the **range dimension must be the last dimension** of all input arrays.

Another increase in performance was achieved by replacing the naive (Python/numpy)
implementation of the phase unfolding by a Fortan implementation using f2py
(http://cens.ioc.ee/projects/f2py2e/). f2py usually ships with numpy and should
be available via the command line. To test whether f2py is available on your
system, execute ``f2py`` on the system console. Or, alternatively, ``f2py.py``. If it is
available, you should get a bunch of help instructions. Now change to the wradlib
module directory and execute on the system console:

   ``f2py.py -c -m speedup speedup.f``



.. autosummary::
   :nosignatures:
   :toctree: generated/

    process_raw_phidp
    kdp_from_phidp
    fill_phidp
    unfold_phi
    unfold_phi_naive
    texture

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# Check whether fast Fortran implementation is available
speedupexists = True
try:
    from wradlib.speedup import f_unfold_phi
except ImportError:
    print "WARNING: To increase performance, you should try to build module <speedup>."
    print "See module documentation for details."
    speedupexists = False



def process_raw_phidp(phidp, rho, copy=False):
    """Establish consistent PhiDP profiles from raw data.

    Processing of raw PhiDP data contains the following steps:

        - Despeckle

        - Fill missing data
          (general asssumption: PhiDP is monotonically increasing along the beam)

        - Phase unfolding

        - Smoothing

    Parameters
    ----------
    phidp : array of shape (n azimuth angles, n range gates)
    rho : array of shape (n azimuth angles, n range gates)
    copy : boolean
        leaves the original phidp array untouched

    """
    if copy:
        phidp = phidp.copy()
    # despeckle
    phidp = linear_despeckle(phidp)
    phidp = fill_phidp(phidp)
    # apply unfolding
    if speedupexists:
        phidp = unfold_phi(phidp, rho)
    else:
        phidp = unfold_phi_naive(phidp, rho)
    # median filter smoothing
    phidp = medfilt_along_axis(phidp,5)
    return phidp


def kdp_from_phidp(phidp, L=7):
    """Retrieves Kdp from PhiDP by applying a moving window range derivative.

    ATTENTION: Function is designed to speed up performance by allowing to process
    multiple dimension in one step. For this purpose, the RANGE dimension needs
    to be the LAST dimension of the input array.

    Parameters
    ----------
    data : multi-dimensional array
        Note that the range dimension must be the last dimension of the input array.

    N : integer
        Width of the window

    copy : Boolean
        If True, the input array will remain unchanged.

    """
    assert (N % 2) == 1, "Window size N for function kdp_from_phidp must be an odd number."
    kdp = np.zeros(phidp.shape)
    for r in xrange(L/2, phidp.shape[-1]-L/2):
        kdp[...,r] = (phidp[...,r+L/2] - phidp[...,r-L/2]) / (2*L)
    return kdp


def unfold_phi(phidp, rho, width=5, copy=False):
    """
    Unfolds differential phase by adjusting values that exceeded maximum ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE THE RANGE.

    This is the fast Fortran-based implementation (RECOMMENDED).

    The algorithm is based on the paper of [Wang2009]_.

    Parameters
    ----------
    phidp : array of shape (...,nr) with nr being the number of range bins
    rho : array of same shape as phidp
    width : integer
       Width of the analysis window
    copy : boolean
       Leaves original phidp array unchanged if set to True (default: False)

    References
    ----------
    .. [Wang2009] Wang, Yanting, V. Chandrasekar, 2009: Algorithm for Estimation
       of the Specific Differential Phase. J. Atmos. Oceanic Technol., 26, 2565-2578.

    """
    shape = phidp.shape
    assert rho.shape==shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1,shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho   = rho.reshape((-1,shape[-1]))
    gradphi = gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in xrange(rs-9):
        stdarr[...,r] = np.std(phidp[...,r:r+9],-1)

    phidp = f_unfold_phi(phidp=phidp.astype("f4"), rho=rho.astype("f4"), gradphi=gradphi.astype("f4"), stdarr=stdarr.astype("f4"), beams=beams, rs=rs, w=width)

    return phidp.reshape(shape)


def unfold_phi_naive(phidp, rho, width=5, copy=False):
    """
    Unfolds differential phase by adjusting values that exceeded maximum ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE THE RANGE.

    This is the slow Python-based implementation (NOT RECOMMENDED).

    Parameters
    ----------
    phidp : array of shape (...,nr) with nr being the number of range bins
    rho : array of same shape as phidp
    width : integer
       Width of the analysis window
    copy : boolean
       Leaves original phidp array unchanged if set to True (default: False)

    References
    ----------
    .. [Wang2009] Wang, Yanting, V. Chandrasekar, 2009: Algorithm for Estimation
       of the Specific Differential Phase. J. Atmos. Oceanic Technol., 26, 2565-2578.

    """
    shape = phidp.shape
    assert rho.shape==shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1,shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho   = rho.reshape((-1,shape[-1]))
    gradphi = gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in xrange(rs-9):
        stdarr[...,r] = np.std(phidp[...,r:r+9],-1)

    phi_corr = np.zeros(phidp.shape)
    for beam in xrange(beams):

        if np.all(phidp[beam]==0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0,rs-width):
            if (np.sum(stdarr[beam,j:j+width] < 5) == width) and (np.sum(rho[beam,j:j+5] > 0.9) == width):
                break

        ref = np.mean(phidp[beam,j:j+width])
        for k in range(j+width,rs):
            if np.sum(stdarr[beam,k-width:k] < 5) and np.logical_and(gradphi[beam, k]>-5, gradphi[beam, k]<20):
                ref = ref + gradphi[beam, k]*0.5
                if phidp[beam,k] - ref < -80:
                    if phidp[beam,k] < 0:
                        phidp[beam,k] += 360
            elif phidp[beam,k] - ref < -80:
                if phidp[beam,k] < 0:
                    phidp[beam,k] += 360
    return phidp



##def smooth_and_gradient1d(x):
##    """
##    """
##    return np.gradient(medfilt(x, kernel_size=5)).astype("f4")
##
##
##def smooth_and_gradient2d(x):
##    """
##    """
##    return np.apply_along_axis(func1d=smooth_and_gradient1d, axis=0, arr=x)

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
    axis=-1
    newshape = np.array(x.shape)
    newshape[axis] = 1
    diff_begin = ( x[...,1] - x[...,0] ).reshape(newshape)
    diff_end = ( x[...,-1] - x[...,-2] ).reshape(newshape)
    diffs = ( (x - np.roll(x, 2, axis) ) / 2. )
    diffs = np.append(diffs[...,2:], diff_end, axis=axis)
    return np.insert(diffs, 0, diff_begin, axis=axis)

# TO UTILS
def gradient_from_smoothed(x, N=5):
    """Computes gradient of smoothed data along final axis of an array
    """
    return gradient_along_axis(medfilt_along_axis(x, N)).astype("f4")


def linear_despeckle(data, N=3, copy=False):
    """Remove floating pixels in between NaNs in a multi-dimensional array.

    ATTENTION: This function changes the original input array if argument copy is set to default (False).

    Parameters
    ----------
    data : multi-dimensional array
        Note that the range dimension must be the last dimension of the input array.

    N : integer (must be either 3 or 5, 3 by default)
        Width of the window in which we check for speckle

    copy : Boolean
        If True, the input array will remain unchanged.

    """
    assert N in (3,5), "Window size N for function linear_despeckle must be 3 or 5."
    if copy:
        data = data.copy()
    axis = data.ndim - 1
    arr  = np.ones(data.shape, dtype="i4")
    arr[np.isnan(data)] = 0
    arr_plus1  = np.roll(arr, shift=1,  axis=axis)
    arr_minus1 = np.roll(arr, shift=-1, axis=axis)
    if N==3:
        # for a window of size 3
        test = arr + arr_plus1 + arr_minus1
        data[np.logical_and( np.logical_not(np.isnan(data)), test<2)] = np.nan
    else:
        # for a window of size 5
        arr_plus2  = np.roll(arr, shift=2,  axis=axis)
        arr_minus2 = np.roll(arr, shift=-2, axis=axis)
        test = arr + arr_plus1 + arr_minus1 + arr_plus2 + arr_minus2
        data[np.logical_and( np.logical_not(np.isnan(data)), test<3)] = np.nan
    return data


##def fill_phi1d(data):
##    """
##    Fills in missing PhiDP in a 1d array. The ends are extrapolated by extending the
##    first and last values to the start and end of the range, respectively.
##
##    Parameters
##    ----------
##    data : 1d array
##
##    Returns
##    -------
##    data_filled : 1d array
##        array with filled gaps and extrapolated ends
##
##    """
##    rs = len(data)
##
##    # return zeros of there are no valid phidp values
##    if np.all(np.isnan(data)):
##        return np.zeros(rs, dtpye="f4")
##
##    # get last value and extend to end of range
##    valid_ix = np.where(np.logical_not(np.isnan(data)))[0]
##    data[0:valid_ix[0]] = data[valid_ix[0]]
##    data[valid_ix[-1]:len(data)] = data[valid_ix[-1]]
##    valid_ix = np.where(np.logical_not(np.isnan(data)))[0]
##
##    # interpolate
##    f = scipy.interpolate.interp1d(valid_ix, data[valid_ix], copy=False)
##    data = f(np.arange(rs))
##    return data
##
##def fill_phi2d(data):
##    """
##    Fills in missing PhiDP beam-wise in a 2D array. The ends are extrapolated by extending the
##    first and last values to the start and end of the range, respectively.
##
##    Parameters
##    ----------
##    data : array
##        array representing polar radar data
##
##    Returns
##    -------
##    data_filled : array
##        array with filled gaps and extrapolated ends
##
##    """
##    return np.apply_along_axis(func1d=fill_phi1d, axis=0, arr=data)


def fill_phidp(data):
    """
    Fills in missing PhiDP. The ends are extrapolated by extending the
    first and last values to the start and end of the range, respectively.

    Parameters
    ----------
    data : N-dim array with last dimension representing the range

    Returns
    -------
    out : array of same shape as phi gaps filled

    """
    shape = data.shape
    data  = data.reshape((-1,shape[-1]))
    zeros = np.zeros(data.shape[1], dtpye="f4")
    x = np.arange(data.shape[1])
    valids = np.logical_not(np.isnan(data))

    for i in xrange(data.shape[0]):
        # return zeros of there are no valid phidp values
        if np.all(np.isnan(data[i])):
            data[i] = zeros
        # interpolate
        ix = np.where(valids[i])[0]
        f = interp1d(ix, data[i,ix], copy=False, bounds_error=False)
        data = f(x)
        # find and replace remaining NaNs
        data[i, :ix[0]]  = data[i, ix[0]]
        data[i, ix[-1]:] = data[i, ix[-1]]
    return data.reshape(shape)


def texture(array):
    """
    Compute the texture of the data by comparing values with a 3x3 neighborhood (based on Gourley, 2007).
    NaN values in the original array have NaN textures.

    Parameters
    ----------
    data : 2-D data array

    Returns
    ------
    texture : 2-D array of textures

    """
    x1 = np.roll(array,1,0) # center:2
    x2 = np.roll(array,1,1) # 4
    x3 = np.roll(array,-1,0) # 8
    x4 = np.roll(array,-1,1) # 6
    x5 = np.roll(x1,1,1) # 1
    x6 = np.roll(x4,1,0) # 3
    x7 = np.roll(x3,-1,1) # 9
    x8 = np.roll(x2,-1,0) # 7

    xa = np.array([x1, x2, x3, x4, x5, x6, x7, x8]) # at least one NaN would give a sum of NaN

    # get count of valid neighboring pixels
    xa_valid = np.ones(np.shape(xa))
    xa_valid[np.isnan(xa)] = 0
    xa_valid_count = np.sum(xa_valid, axis = 0) # count number of valid neighbors

    num = np.zeros(np.shape(array))
    for xarr in xa:
        diff = array - xarr
        # difference of NaNs will be converted to zero (to not affect the summation)
        diff[np.isnan(diff)] = 0
        # only those with valid values are considered in the summation
        num += diff**2

    num[np.isnan(array)] = np.nan # reinforce that NaN values should have NaN textures

    texture = np.sqrt(num / xa_valid_count)

    return texture





if __name__ == '__main__':
    print 'wradlib: Calling module <dp> as main...'


