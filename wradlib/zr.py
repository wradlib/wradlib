#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Z-R Conversions
^^^^^^^^^^^^^^^

Module zr takes care of transforming reflectivity
into rainfall rates and vice versa

.. currentmodule:: wradlib.zr

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["z_to_r", "r_to_z", "z_to_r_enhanced"]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
from scipy import signal

from wradlib import trafo


def z_to_r(z, a=200.0, b=1.6):
    """Conversion from reflectivities to rain rates.

    Calculates rain rates from radar reflectivities using
    a power law Z/R relationship Z = a*R**b

    Parameters
    ----------
    z : float
        a float or an array of floats
        Corresponds to reflectivity Z in mm**6/m**3
    a : float
        Parameter a of the Z/R relationship
            Standard value according to Marshall-Palmer is a=200., b=1.6
    b : float
        Parameter b of the Z/R relationship
        Standard value according to Marshall-Palmer is b=1.6

    Note
    ----
    The German Weather Service uses a=256 and b=1.42 instead of
    the Marshall-Palmer defaults.

    Returns
    -------
    output : float
        a float or an array of floats
        rainfall intensity in mm/h

    """
    return (z / a) ** (1.0 / b)


def r_to_z(r, a=200.0, b=1.6):
    """Calculates reflectivity from rain rates using
    a power law Z/R relationship Z = a*R**b

    Parameters
    ----------
    r : a float or an array of floats
        Corresponds to rainfall intensity in mm/h
    a : float
        Parameter a of the Z/R relationship
            Standard value according to Marshall-Palmer is a=200., b=1.6
    b : float
        Parameter b of the Z/R relationship
        Standard value according to Marshall-Palmer is b=1.6

    Note
    ----
    The German Weather Service uses a=256 and b=1.42 instead of
    the Marshall-Palmer defaults.

    Returns
    -------
    output : a float or an array of floats
             reflectivity in mm**6/m**3

    """
    return a * r ** b


def z_to_r_enhanced(z, polar=True, shower=True):
    """Calculates rainrates from radar reflectivities using the enhanced \
    three-part Z-R-relationship used by the DWD (as of 2009)

    To be used with polar representations so that one dimension is cyclical.
    i.e. z should be of shape (nazimuths, nbins) --> the first dimension
    is the cyclical one. For DWD DX-Data z's shape is (360,128).

    Neighborhood-means are taken only for available data via fast convolution
    sums.
    Refer to the RADOLAN final report or the RADOLAN System handbook for
    details on the calculations.
    Basically, for low reflectivities an index called the shower index is
    calculated as the mean of the differences along both axis in a neighborhood
    of 3x3 pixels.
    This means:

                        +------+-----------------+
                        |      | x-direction --> |
                        +------+-----+-----+-----+
                        | | y  |  1  |  2  |  3  |
                        | | l  +-----+-----+-----+
                        | | d  |  4  |  5  |  6  |
                        | | i  +-----+-----+-----+
                        | | r  |  7  |  8  |  9  |
                        +------+-----+-----+-----+

    If 5 is the pixel in question, it's shower index is calculated as:

    .. math::

        ( &|1-2| + |2-3| + |4-5| + |5-6| + |7-8| + |8-9| + \\\\
          &|1-4| + |4-7| + |2-5| + |5-8| + |3-6| + |6-9| ) / 12.

    then, the upper line of the sum would be diffx (DIFFerences in
    X-direction), the lower line would be diffy
    (DIFFerences in Y-direction) in the code below.

    Parameters
    ----------
    z : :class:`numpy:numpy.ndarray`
        Corresponds to reflectivity Z in mm**6/m**3
        ND-array, at least 2D
    polar : bool
        defaults to to True (polar data), False for cartesian data.
    shower : bool
        output shower index, defaults to True

    Returns
    -------
    r : :class:`numpy:numpy.ndarray`
        r  - array of shape z.shape - calculated rain rates
    si : :class:`numpy:numpy.ndarray`
        si - array of shape z.shape - calculated shower index
        for control purposes. May be omitted in later versions

    """
    z = np.asanyarray(z, dtype=np.float64)
    shape = z.shape
    z = z.reshape((-1,) + shape[-2:])
    if polar:
        z0 = np.concatenate([z[:, -1:, :], z, z[:, 0:1, :]], axis=-2)
        x_ymin, x_ymax = 2, -2
        y_ymin, y_ymax = 1, -1
        x_xmin, x_xmax = None, None
        y_xmin, y_xmax = 1, -1
    else:
        z0 = z.copy()
        x_ymin, x_ymax = 1, -1
        y_ymin, y_ymax = None, None
        x_xmin, x_xmax = None, None
        y_xmin, y_xmax = 1, -1
    z0 = trafo.decibel(z0)

    # create shower index using differences and convolution sum
    diffx = np.abs(np.diff(z0, n=1, axis=-1))
    diffy = np.abs(np.diff(z0, n=1, axis=-2))
    xkernel = np.ones((1, 3, 2))
    ykernel = np.ones((1, 2, 3))
    resx = signal.convolve(diffx, xkernel, mode="full", method="direct")[
        :, x_ymin:x_ymax, x_xmin:x_xmax
    ]
    resy = signal.convolve(diffy, ykernel, mode="full", method="direct")[
        :, y_ymin:y_ymax, y_xmin:y_xmax
    ]
    si = resx + resy

    # edge cases divide by 7, everything else divide by 12
    if polar:
        si[:, :, 0] /= 7.0
        si[:, :, -1] /= 7.0
        si[:, :, 1:-1] /= 12.0
    else:
        si[:, 1:-1, 1:-1] /= 12.0
        si[:, :, 0] /= 7
        si[:, :, -1] /= 7.0
        si[:, 0, :] /= 7
        si[:, -1, :] /= 7.0

    rr = np.zeros(z.shape)
    z0_ = z0[:, y_ymin:y_ymax, :]

    # get masks
    gt44 = z0_ > 44
    bt3644 = (z0_ >= 36.5) & (z0_ <= 44.0)
    si[bt3644] = -1.0
    si[gt44] = -1.0
    mn75h = (si > 7.5) & (si < 36.5)
    mn35 = (si > -1) & (si < 3.5)
    mn75l = (si >= 3.5) & (si <= 7.5)

    # calculate rainrates according DWD
    rr[mn75l] = z_to_r(z[mn75l], a=200.0, b=1.6)
    rr[mn75h] = z_to_r(z[mn75h], a=320.0, b=1.4)
    rr[mn35] = z_to_r(z[mn35], a=125.0, b=1.4)
    rr[gt44] = z_to_r(z[gt44], a=77.0, b=1.9)
    rr[bt3644] = z_to_r(z[bt3644], a=200.0, b=1.6)

    rr.shape = shape
    si.shape = shape

    if shower:
        return rr, si
    else:
        return rr


if __name__ == "__main__":
    print("wradlib: Calling module <zr> as main...")
