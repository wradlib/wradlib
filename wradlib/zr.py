#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
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

   z2r
   r2z
   z2rEnhanced


"""
import numpy as np
import scipy.ndimage.filters as filters
from .trafo import decibel


def z2r(z, a=200., b=1.6):
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
    return (z / a) ** (1. / b)


def r2z(r, a=200., b=1.6):
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


def _z2rEnhanced(z):
    """Calculates rainrates from radar reflectivities using the enhanced
    three-part Z-R-relationship used by the DWD (as of 2009).

    This function does the actual calculations without any padding.
    Neighborhood-means are taken only for available data, reducing the number
    of elements used near the edges of the array.
    Refer to the RADOLAN final report or the RADOLAN System handbook for
    details on the calculations.
    Basically, for low reflectivities an index called the shower index is
    calculated as the mean of the differences along both axis in a neighborhood
    of 3x3 pixels.
    This means:
           x-direction -->
    y |    +---+---+---+
    | |    | 1 | 2 | 3 |
    d v    +---+---+---+
    i      | 4 | 5 | 6 |
    r      +---+---+---+
    e      | 7 | 8 | 9 |
    c      +---+---+---+
    t
    i      if 5 is the pixel in question, its shower index is calculated as
    o      ( |1-2| + |2-3| + |4-5| + |5-6| + |7-8| + |8-9| +
    n      + |1-4| + |4-7| + |2-5| + |5-8| + |3-6| + |6-9| ) / 12.
           then, the upper line of the sum would be diffx (DIFFerences in
           X-direction), the lower line would be diffy
           (DIFFerences in Y-direction) in the code below.
    """
    # get the shape of the input
    dimy = z.shape[0]
    dimx = z.shape[1]

    # calculate the decibel values from the input
    db = decibel(z)

    # set up our output arrays
    r = np.zeros(z.shape)
    si = np.zeros(z.shape)

    # calculate difference fields in x and y direction
    #  mainly for performance reasons, so that we can use numpy's efficient
    #  array operations and avoid calculating a difference more than once
    diffx = np.abs(db[:, :-1] - db[:, 1:])
    diffy = np.abs(db[:-1, :] - db[1:, :])

    # if the reflectivity is larger than 44dBZ, then there is no need to
    # calculate the shower index
    gt44 = np.where(db > 44.)
    r[gt44] = z2r(z[gt44], a=77., b=1.9)
    # the same is true for values between 36.5 and 44 dBZ
    bt3644 = np.where(np.logical_and(db >= 36.5, db <= 44.))
    r[bt3644] = z2r(z[bt3644], a=200., b=1.6)

    # now iterate over the array and look for the remaining values
    # TODO : this could be a starting point for further optimization, if we
    #        iterated only over the remaining pixels instead of all
    for i in range(dimy):
        for j in range(dimx):
            # if the reflectivity is too high, we coped with it already
            # so we can skip that one
            if db[i, j] >= 36.5:
                # just set the shower index to some impossible value so that
                # we know that there was no calculation done here
                si[i, j] = -1
                # continue with the next iteration
                continue
            else:
                # calculate the bounds of the region where we have to consider
                # the respective difference
                xmin = max(0, j - 1)
                xmax = min(dimx, j + 1)
                ymin = max(0, i - 1)
                ymax = min(dimy, i + 1)
                # in fact python is quite forgiving with upper indices
                # ours might go one index too far, so don't try to port this
                # to another programming language straigt away!
                diffxcut = diffx[ymin:ymax + 1, xmin:xmax]
                diffycut = diffy[ymin:ymax, xmin:xmax + 1]
                # calculate the mean for the current pixel
                mn = (np.sum(diffxcut) + np.sum(diffycut)) / \
                     (diffxcut.size + diffycut.size)
                # apply the three different Z/R relations
                if mn < 3.5:
                    r[i, j] = z2r(z[i, j], a=125., b=1.4)
                elif mn <= 7.5:
                    r[i, j] = z2r(z[i, j], a=200., b=1.6)
                else:
                    r[i, j] = z2r(z[i, j], a=320., b=1.4)
                # save the shower index
                si[i, j] = mn
    # return the results
    return r, si


def _z2rEnhanced_md(z):
    """multidimensional version
    assuming the two last dimensions represent an image
    This version should also be a bit more performant than the original because
    it uses less for-loops"""
    # get the shape of the input
    dimy = z.shape[-2]
    dimx = z.shape[-1]

    # calculate the decibel values from the input
    db = decibel(z)

    # set up our output arrays
    r = np.zeros(z.shape)
    si = np.zeros(z.shape)

    # calculate difference fields in x and y direction
    #  mainly for performance reasons, so that we can use numpy's efficient
    #  array operations and avoid calculating a difference more than once
    diffx = np.abs(db[..., :, :-1] - db[..., :, 1:])
    diffy = np.abs(db[..., :-1, :] - db[..., 1:, :])

    # if the reflectivity is larger than 44dBZ, then there is no need to
    # calculate the shower index
    gt44 = db > 44.
    r[gt44] = z2r(z[gt44], a=77, b=1.9)
    si[gt44] = -1
    # the same is true for values between 36.5 and 44 dBZ
    bt3644 = (db >= 36.5) & (db <= 44.)
    r[bt3644] = z2r(z[bt3644], a=200, b=1.6)
    si[bt3644] = -1

    # now iterate only over remaining pixels
    wlt36 = np.where(db < 36.5)
    # nlt36 = len(wlt36[0])
    for indices in zip(*wlt36):
        # x and y-indices
        i = indices[-2]
        j = indices[-1]
        # all remaining dimensional indices
        ridx = indices[:-2]
        # calculate the bounds of the region where we have to consider
        # the respective difference
        xmin = max(0, j - 1)
        xmax = min(dimx, j + 1)
        ymin = max(0, i - 1)
        ymax = min(dimy, i + 1)
        # in fact python is quite forgiving with upper indices
        # ours might go one index too far, so don't try to port this
        # to another programming language straigt away!
        diffxcut = diffx[ridx, ymin:ymax + 1, xmin:xmax]
        diffycut = diffy[ridx, ymin:ymax, xmin:xmax + 1]
        # calculate the mean for the current pixel
        mn = (np.sum(diffxcut) + np.sum(diffycut)) / \
             (diffxcut.size + diffycut.size)
        # apply the three different Z/R relations
        if mn < 3.5:
            r[ridx, i, j] = z2r(z[ridx, i, j], a=125, b=1.4)
        elif mn <= 7.5:
            r[ridx, i, j] = z2r(z[ridx, i, j], a=200, b=1.6)
        else:
            r[ridx, i, j] = z2r(z[ridx, i, j], a=320, b=1.4)
        # save the shower index
        si[ridx, i, j] = mn
    # return the results
    return r, si


def z2rEsifilter(data):
    """calculates the shower index for the enhanced z-r relation
    to be used as the callable for a scipy.ndimate.filters.generic_filter
    """
    if data[4] < 36.5:
        tdata = data.reshape((3, 3))
        # calculate difference fields in x and y direction
        #  mainly for performance reasons, so that we can use numpy's efficient
        #  array operations and avoid calculating a difference more than once
        diffx = np.abs(tdata[:, :-1] - tdata[:, 1:])
        diffy = np.abs(tdata[:-1, :] - tdata[1:, :])
        return np.concatenate([diffx.ravel(), diffy.ravel()]).mean()
    else:
        return -1.


def _z2rEnhanced_mdfilt(z):
    """multidimensional version
    assuming the two last dimensions represent a 2-D image
    Uses scipy.ndimage.filters.generic_filter to reduce the number of for-loops
    even more.
    """
    # get the shape of the input
    # dimy = z.shape[-2]
    # dimx = z.shape[-1]

    # calculate the decibel values from the input
    db = decibel(z)

    # set up our output arrays
    r = np.zeros(z.shape)
    size = list(z.shape)
    size[-2:] = [3, 3]
    size[:-2] = [1] * len(size[:-2])
    size = tuple(size)
    si = filters.generic_filter(db, z2rEsifilter, size=size)

    gt44 = db > 44.
    r[gt44] = z2r(z[gt44], a=77, b=1.9)
    si[gt44] = -1.
    # the same is true for values between 36.5 and 44 dBZ
    bt3644 = (db >= 36.5) & (db <= 44.)
    r[bt3644] = z2r(z[bt3644], a=200, b=1.6)
    si[bt3644] = -2.

    si1 = (si >= 0.)
    si2 = si1 & (si < 3.5)
    si3 = si1 & ~si2 & (si <= 7.5)
    si4 = si > 7.5

    r[si2] = z2r(z[si2], a=125, b=1.4)
    r[si3] = z2r(z[si3], a=200, b=1.6)
    r[si4] = z2r(z[si4], a=320, b=1.4)

    return r, si


def _z2rEnhanced_mdcorr(z, xmode='reflect', ymode='wrap'):
    """multidimensional version
    assuming the two last dimensions represent a 2-D image
    Uses scipy.ndimage.filters.correlate to reduce the number of for-loops
    even more.
    """
    # get the shape of the input
    # dimy = z.shape[-2]
    # dimx = z.shape[-1]

    # calculate the decibel values from the input
    db = decibel(z)
    # calculate the shower differences by 1-d correlation with a differencing
    # kernel
    db_diffx = np.abs(filters.correlate1d(db, [1, -1], axis=-1,
                                          mode=xmode, origin=-1))
    db_diffy = np.abs(filters.correlate1d(db, [1, -1], axis=-2,
                                          mode=ymode, origin=-1))

    diffxmode = 'wrap' if xmode == 'wrap' else 'constant'
    diffymode = 'wrap' if ymode == 'wrap' else 'constant'
    diffx_sum1 = filters.correlate1d(db_diffx, [1, 1, 1],
                                     axis=-2, mode=diffymode)
    diffxsum = filters.correlate1d(diffx_sum1, [1, 1, 0],
                                   axis=-1, mode=diffxmode)
    diffy_sum1 = filters.correlate1d(db_diffy, [1, 1, 1],
                                     axis=-1, mode=diffxmode)
    diffysum = filters.correlate1d(diffy_sum1, [1, 1, 0],
                                   axis=-2, mode=diffymode)

    divider = np.ones(db.shape) * 12.
    if xmode != 'wrap':
        divider[..., [0, -1]] = np.rint((divider[..., [0, -1]] + 1) /
                                        1.618) - 1
    if ymode != 'wrap':
        divider[..., [0, -1], :] = np.rint((divider[..., [0, -1], :] + 1) /
                                           1.618) - 1

    # the shower index is the sum of the x- and y-differences
    si = (diffxsum + diffysum) / divider

    # set up our rainfall output array
    r = np.zeros(z.shape)

    gt44 = db > 44.
    r[gt44] = z2r(z[gt44], a=77, b=1.9)
    si[gt44] = -1.
    # the same is true for values between 36.5 and 44 dBZ
    bt3644 = (db >= 36.5) & (db <= 44.)
    r[bt3644] = z2r(z[bt3644], a=200, b=1.6)
    si[bt3644] = -2.

    si1 = (si >= 0.)
    si2 = si1 & (si < 3.5)
    si3 = si1 & ~si2 & (si <= 7.5)
    si4 = si > 7.5

    r[si2] = z2r(z[si2], a=125, b=1.4)
    r[si3] = z2r(z[si3], a=200, b=1.6)
    r[si4] = z2r(z[si4], a=320, b=1.4)

    return r, si


def z2rEnhanced(z):
    """Calculates rainrates from radar reflectivities using the enhanced
    three-part Z-R-relationship used by the DWD (as of 2009)

    To be used with polar representations so that one dimension is cyclical.
    i.e. z should be of shape (nazimuths, nbins) --> the first dimension
    is the cyclical one. For DWD DX-Data z's shape is (360,128).

    Parameters
    ----------
    z : a float or an array of floats
        Corresponds to reflectivity Z in mm**6/m**3
        **must** be a 2-D array

    Returns
    -------
    r : array
        r  - array of shape z.shape - calculated rain rates
    si : array
        si - array of shape z.shape - calculated shower index
        for control purposes. May be omitted in later versions

    """
    # create a padded version of the input array
    padz = np.zeros((z.shape[0] + 2, z.shape[1]))
    # fill the center with the original data
    padz[1:-1, :] = z
    # add the last beam before the first one
    padz[0, :] = z[-1, :]
    # add the first beam after the last one
    padz[-1, :] = z[0, :]

    # do the actual calculation
    padr, padsi = _z2rEnhanced(padz)

    # return the unpadded field
    return padr[1:-1, :], padsi[1:-1, :]


if __name__ == '__main__':
    print('wradlib: Calling module <zr> as main...')
