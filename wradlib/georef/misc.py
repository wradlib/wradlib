#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   beam_height_n
   arc_distance_n
   get_earth_radius
"""

import numpy as np

from .projection import get_default_projection

# Seitenlänge Zenit - Himmelsnordpol: 90°-phi
# Seitenlänge Himmelsnordpol - Gestirn: 90°-delta
# Seitenlänge Zenit - Gestirn: 90°-h
# Winkel Himmelsnordpol - Zenit - Gestirn: 180°-a
# Winkel Zenit - Himmelsnordpol - Gestirn: tau

# alpha - rektaszension
# delta - deklination
# theta - sternzeit
# tau = theta - alpha - stundenwinkel
# a - azimuth (von süden aus gezählt)
# h - Höhe über Horizont


def hor2aeq(a, h, phi):
    """"""
    delta = np.arcsin(np.sin(h) * np.sin(phi) - np.cos(h) *
                      np.cos(a) * np.cos(phi))
    tau = np.arcsin(np.cos(h) * np.sin(a) / np.cos(delta))
    return delta, tau


def aeq2hor(tau, delta, phi):
    """"""
    h = np.arcsin(np.cos(delta) * np.cos(tau) * np.cos(phi) +
                  np.sin(delta) * np.sin(phi))
    a = np.arcsin(np.cos(delta) * np.sin(tau) / np.cos(h))
    return a, h


def beam_height_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the height of a radar beam taking the refractivity of the
    atmosphere into account.

    Based on :cite:`Doviak1993` the beam height is calculated as

    .. math::

        h = \sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \sin\theta} - k_e r_e


    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    height : float
        height of the beam in [m]

    """
    return np.sqrt(r ** 2 + (ke * re) ** 2 +
                   2 * r * ke * re * np.sin(np.radians(theta))) - ke * re


def arc_distance_n(r, theta, re=6370040., ke=4. / 3.):
    r"""Calculates the great circle distance of a radar beam over a sphere,
    taking the refractivity of the atmosphere into account.

    Based on :cite:`Doviak1993` the arc distance may be calculated as

    .. math::

        s = k_e r_e \arcsin\left(
        \frac{r \cos\theta}{k_e r_e + h_n(r, \theta, r_e, k_e)}\right)

    where :math:`h_n` would be provided by
    :meth:`~wradlib.georef.beam_height_n`

    Parameters
    ----------
    r : :class:`numpy:numpy.ndarray`
        Array of ranges [m]
    theta : scalar or :class:`numpy:numpy.ndarray` broadcastable to the shape
        of r elevation angles in degrees with 0° at horizontal and +90°
        pointing vertically upwards from the radar
    re : float
        earth's radius [m]
    ke : float
        adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths

    Returns
    -------
    distance : float
        great circle arc distance [m]

    See Also
    --------
    beam_height_n

    """
    return ke * re * np.arcsin((r * np.cos(np.radians(theta))) /
                               (ke * re + beam_height_n(r, theta, re, ke)))


def get_earth_radius(latitude, sr=None):
    r"""
    Get the radius of the Earth (in km) for a given Spheroid model (sr) at a
    given position

    .. math::

        R^2 = \frac{a^4 \cos(f)^2 + b^4 \sin(f)^2}
        {a^2 \cos(f)^2 + b^2 \sin(f)^2}

    Parameters
    ----------
    sr : osr object
        spatial reference;
    latitude : float
        geodetic latitude in degrees;

    Returns
    -------
    radius : float
        earth radius in meter

    """
    if sr is None:
        sr = get_default_projection()
    RADIUS_E = sr.GetSemiMajor()
    RADIUS_P = sr.GetSemiMinor()
    latitude = np.radians(latitude)
    radius = np.sqrt((np.power(RADIUS_E, 4) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 4) * np.power(np.sin(latitude), 2)) /
                     (np.power(RADIUS_E, 2) * np.power(np.cos(latitude), 2) +
                      np.power(RADIUS_P, 2) * np.power(np.sin(latitude), 2)))
    return (radius)
