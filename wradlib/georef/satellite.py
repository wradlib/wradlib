#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Satellite Functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   correct_parallax
   sat2pol
   dist_from_orbit
"""
import numpy as np


def correct_parallax(pr_xy, nbin, drt, alpha):
    """Adjust the geo-locations of the PR pixels

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    The `pr_xy` coordinates of the PR beam footprints need to be in the
    azimuthal equidistant projection of the ground radar. This ensures that the
    ground radar is fixed at xy-coordinate (0, 0), and every PR bin has its
    relative xy-coordinates with respect to the ground radar site.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_xy : :class:`numpy:numpy.ndarray`
        Array of xy-coordinates of shape (nscans, nbeams, 2)
    nbin : int
        Number of bins along PR beam.
    drt : float
        Gate lenght of PR in meter.
    alpha: :class:`numpy:numpy.ndarray`
        Array of depression angles of the PR beams with shape (nbeams).

    Returns
    -------

    pr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins, 2).
    r_pr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to PR platform of shape (nbins).
    z_pr : :class:`numpy:numpy.ndarray`
        Array of PR bin altitudes of shape (nbeams, nbins).
    """
    # get x,y-grids
    pr_x = pr_xy[..., 0]
    pr_y = pr_xy[..., 1]

    # create range array from ground to sat
    r_pr_inv = np.arange(nbin) * drt

    # calculate height of bin
    z_pr = r_pr_inv * np.cos(np.deg2rad(alpha))[:, np.newaxis]
    # calculate bin ground xy-displacement length
    ds = r_pr_inv * np.sin(np.deg2rad(alpha))[:, np.newaxis]

    # calculate x,y-differences between ground coordinate
    # and center ground coordinate [25th element]
    center = int(np.floor(len(pr_x[-1]) / 2.))
    xdiff = pr_x[:, center][:, np.newaxis] - pr_x
    ydiff = pr_y[:, center][:, np.newaxis] - pr_y

    # assuming ydiff and xdiff being a triangles adjacent and
    # opposite this calculates the xy-angle of the PR scan
    ang = np.arctan2(ydiff, xdiff)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.cos(ang)[..., np.newaxis]
    dy = ds * np.sin(ang)[..., np.newaxis]

    # add displacement to PR ground coordinates
    pr_xp = dx + pr_x[..., np.newaxis]
    pr_yp = dy + pr_y[..., np.newaxis]

    return np.stack((pr_xp, pr_yp), axis=3), r_pr_inv, z_pr


def sat2pol(pr_xyz, gr_site_alt, re):
    """Returns spherical coordinates of PR bins as seen from the GR location.

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM. With *GR*, we refer to terrestrial weather radars
    (ground radars).

    For this function to work, the `pr_xyz` coordinates of the PR bins need
    to be in the azimuthal equidistant projection of the ground radar! This
    ensures that the ground radar is fixed at xy-coordinate (0, 0), and every
    PR bin has its relative xy-coordinates with respect to the ground radar
    site.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_xyz : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins, 3). Contains corrected
        PR xy coordinates in GR azimuthal equidistant projection and altitude
    gr_site_alt : float
        Altitude of the GR site (in meters)
    re : float
        Effective Earth radius at GR site (in meters)

    Returns
    -------
    r : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the slant
        distance of PR bins from GR site.
    theta: :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the elevation
        angle of PR bins as seen from GR site.
    phi : :class:`numpy:numpy.ndarray`
        Array of shape (nscans, nbeams, nbins). Contains the azimuth
        angles of PR bins as seen from GR site.
    """
    # calculate arc length
    s = np.sqrt(np.sum(pr_xyz[..., 0:2] ** 2, axis=-1))

    # calculate arc angle
    gamma = s / re

    # calculate theta (elevation-angle)
    numer = np.cos(gamma) - (re + gr_site_alt) / (re + pr_xyz[..., 2])
    denom = np.sin(gamma)
    theta = np.rad2deg(np.arctan(numer / denom))

    # calculate SlantRange r
    r = (re + pr_xyz[..., 2]) * denom / np.cos(np.deg2rad(theta))

    # calculate Azimuth phi
    phi = 90 - np.rad2deg(np.arctan2(pr_xyz[..., 1], pr_xyz[..., 0]))
    phi[phi <= 0] += 360

    return r, theta, phi


def dist_from_orbit(pr_alt, alpha, r_pr_inv):
    """Returns range distances of PR bins (in meters) as seen from the orbit

    With *PR*, we refer to precipitation radars based on space-born platforms
    such as TRMM or GPM.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    pr_alt : float
        PR orbit height in meters.
    alpha: :class:`numpy:numpy.ndarray`
       Array of depression angles of the PR beams with shape (nbeams).
    r_pr_inv : :class:`numpy:numpy.ndarray`
        Array of ranges from ground to PR platform of shape (nbins).

    Returns
    -------
    ranges : :class:`numpy:numpy.ndarray`
        Array of shape (nbeams, nbins) of PR bin range distances from
        PR platform in orbit.
    """
    return pr_alt / np.cos(np.radians(alpha))[:, np.newaxis] - r_pr_inv
