#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   bin_altitude
   bin_distance
   site_distance
   get_earth_radius
"""

import numpy as np

from .projection import get_default_projection


def bin_altitude(r, theta, sitealt, re, ke=4./3.):
    """Calculates the height of a radar bin taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the bin altitude is calculated as

    .. math::

        h = \\sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \\sin\\theta} - k_e r_e


    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    sitealt : float
        Altitude in [m] a.s.l. of the referencing radar site
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    altitude : :class:`numpy:numpy.ndarray`
        Array of heights of the radar bins in [m]

    """
    reff = ke * re
    sr = reff + sitealt
    return np.sqrt(r ** 2 + sr ** 2 +
                   2 * r * sr * np.sin(np.radians(theta))) - reff


def bin_distance(r, theta, sitealt, re, ke=4./3.):
    """Calculates great circle distance from radar site to radar bin over \
    spherical earth, taking the refractivity of the atmosphere into account.

    .. math::

        s = k_e r_e \\arctan\\left(
        \\frac{r \\cos\\theta}{r \\cos\\theta + k_e r_e + h}\\right)

    where :math:`h` would be the radar site altitude amsl.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    sitealt : float
        site altitude [m] amsl.
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : :class:`numpy:numpy.ndarray`
        Array of great circle arc distances [m]
    """
    reff = ke * re
    sr = reff + sitealt
    theta = np.radians(theta)
    return reff * np.arctan(r * np.cos(theta) / (r * np.sin(theta) + sr))


def site_distance(r, theta, binalt, re=None, ke=4./3.):
    """Calculates great circle distance from bin at certain altitude to the \
    radar site over spherical earth, taking the refractivity of the \
    atmosphere into account.

    Based on :cite:`Doviak1993` the site distance may be calculated as

    .. math::

        s = k_e r_e \\arcsin\\left(
        \\frac{r \\cos\\theta}{k_e r_e + h_n(r, \\theta, r_e, k_e)}\\right)

    where :math:`h_n` would be provided by
    :func:`~wradlib.georef.misc.bin_altitude`.

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    binalt : :class:`numpy:numpy.ndarray`
        site altitude [m] amsl. same shape as r.
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : :class:`numpy:numpy.ndarray`
        Array of great circle arc distances [m]
    """
    reff = ke * re
    return reff * np.arcsin(r * np.cos(np.radians(theta)) / (reff + binalt))


def get_earth_radius(latitude, sr=None):
    """Get the radius of the Earth (in km) for a given Spheroid model (sr) at \
    a given position.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    sr : osr object
        spatial reference
    latitude : float
        geodetic latitude in degrees

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if sr is None:
        sr = get_default_projection()
    radius_e = sr.GetSemiMajor()
    radius_p = sr.GetSemiMinor()
    latitude = np.radians(latitude)
    radius = np.sqrt((np.power(radius_e, 4) * np.power(np.cos(latitude), 2) +
                      np.power(radius_p, 4) * np.power(np.sin(latitude), 2)) /
                     (np.power(radius_e, 2) * np.power(np.cos(latitude), 2) +
                      np.power(radius_p, 2) * np.power(np.sin(latitude), 2)))
    return radius
