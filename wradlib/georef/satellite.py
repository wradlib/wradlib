#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Satellite Functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["correct_parallax", "dist_from_orbit"]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np


def correct_parallax(sr_xy, nbin, drt, alpha):
    """Adjust the geo-locations of the SR pixels.

    With *SR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    The `sr_xy` coordinates of the SR beam footprints need to be in the
    azimuthal equidistant projection of the ground radar. This ensures that the
    ground radar is fixed at xy-coordinate (0, 0), and every SR bin has its
    relative xy-coordinates with respect to the ground radar site.

    Parameters
    ----------
    sr_xy : :class:`numpy:numpy.ndarray`
        Array of xy-coordinates of shape (nscans, nbeams, 2)
    nbin : int
        Number of bins along SR beam.
    drt : float
        Gate lenght of SR in meter.
    alpha: :class:`numpy:numpy.ndarray`
        Array of local zenith angles of the SR beams
        with shape (nscans, nbeams).

    Returns
    -------

    sr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins, 2).
    r_sr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to SR platform of shape (nbins).
    z_sr : :class:`numpy:numpy.ndarray`
        Array of SR bin altitudes of shape (nscans, nbeams, nbins).
    """
    # get x,y-grids
    sr_x = sr_xy[..., 0]
    sr_y = sr_xy[..., 1]

    # create range array from ground to satellite
    r_sr_inv = np.arange(nbin) * drt

    # calculate height of bin
    z_sr = r_sr_inv * np.expand_dims(np.cos(np.deg2rad(alpha)), axis=-1)
    # calculate bin ground xy-displacement length
    ds = r_sr_inv * np.expand_dims(np.sin(np.deg2rad(alpha)), axis=-1)

    # calculate x,y-differences between ground coordinate
    # and center ground coordinate [25th element]
    center = int(np.floor(len(sr_x[-1]) / 2.0))
    xdiff = sr_x - np.expand_dims(sr_x[:, center], axis=-1)
    ydiff = sr_y - np.expand_dims(sr_y[:, center], axis=-1)

    # assuming ydiff and xdiff being a triangles adjacent and
    # opposite this calculates the xy-angle of the SR scan
    ang = np.arctan2(ydiff, xdiff)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.expand_dims(np.cos(ang), axis=-1)
    dy = ds * np.expand_dims(np.sin(ang), axis=-1)

    # subtract displacement from SR ground coordinates
    sr_xp = np.expand_dims(sr_x, axis=-1) - dx
    sr_yp = np.expand_dims(sr_y, axis=-1) - dy

    return np.stack((sr_xp, sr_yp), axis=3), r_sr_inv, z_sr


def dist_from_orbit(sr_alt, alpha, beta, r_sr_inv, re):
    """Returns range distances of SR bins (in meters) as seen from the orbit

    With *SR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    Parameters
    ----------
    sr_alt : float
        SR orbit height in meters.
    alpha: :class:`numpy:numpy.ndarray`
        Array of local zenith angles of the SR beams
        with shape (nscans, nbeams).
    beta: :class:`numpy:numpy.ndarray`
        Off-Nadir scan angle with shape (nbeams).
    r_sr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to SR platform of shape (nbins).
    re : float
        earth radius [m]

    Returns
    -------
    ranges : :class:`numpy:numpy.ndarray`
        Array of shape (nbeams, nbins) of PR bin range distances from
        SR platform in orbit.
    """
    ro = (
        (re + sr_alt) * np.cos(np.radians(alpha - np.expand_dims(beta, axis=0))) - re
    ) / np.cos(np.radians(alpha))
    return np.expand_dims(ro, axis=-1) - r_sr_inv
