#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Data Transformation
^^^^^^^^^^^^^^^^^^^

Module <trafo> transforms data e.g. from RVP-units
to dBZ-values to Z-values and vice versa.

.. currentmodule:: wradlib.trafo

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "rvp_to_dbz",
    "decibel",
    "idecibel",
    "r_to_depth",
    "kdp_to_r",
    "si_to_kmh",
    "si_to_mph",
    "si_2_kts",
    "kmh_to_si",
    "mph_to_si",
    "kts_to_si",
    "KuBandToS",
    "SBandToKu",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

# CONSTANTS
meters_per_mile = 1609.344
meters_per_nautical_mile = 1852.0


class SBandToKu:
    """Class to hold coefficients for Radar Reflectivity Conversion

    From S-band (2.8GHz) to Ku-band (13.8GHz)

    See :cite:`Liao2009` for reference.
    """

    snow = np.array([0.185074, 1.01378, -0.00189212])
    rain = np.array([-1.50393, 1.07274, 0.000165393])


class KuBandToS:
    """Class to hold coefficients for Radar Reflectivity Conversion

    From Ku-band (13.8 GHz) to S-band (2.8 GHz)

    See :cite:`Cao2013` for reference.

    """

    # Rain 90% 80% 70% 60% 50% 40% 30% 20% 10% Snow
    snow = np.array(
        [
            [
                4.78e-2,
                4.12e-2,
                8.12e-2,
                1.59e-1,
                2.87e-1,
                4.93e-1,
                8.16e-1,
                1.31e0,
                2.01e0,
                2.82e0,
                1.74e-1,
            ],
            [
                1.23e-2,
                3.66e-3,
                2.00e-3,
                9.42e-4,
                5.29e-4,
                5.96e-4,
                1.22e-3,
                2.11e-3,
                3.34e-3,
                5.33e-3,
                1.35e-2,
            ],
            [
                -3.50e-4,
                1.17e-3,
                1.04e-3,
                8.16e-4,
                6.59e-4,
                5.85e-4,
                6.13e-4,
                7.01e-4,
                8.24e-4,
                1.01e-3,
                -1.38e-3,
            ],
            [
                -3.30e-5,
                -8.08e-5,
                -6.44e-5,
                -4.97e-5,
                -4.15e-5,
                -3.89e-5,
                -4.15e-5,
                -4.58e-5,
                -5.06e-5,
                -5.78e-5,
                4.74e-5,
            ],
            [
                4.27e-7,
                9.25e-7,
                7.41e-7,
                6.13e-7,
                5.80e-7,
                6.16e-7,
                7.12e-7,
                8.22e-7,
                9.39e-7,
                1.10e-6,
                0.00e0,
            ],
        ]
    )

    # Rain 90% 80% 70% 60% 50% 40% 30% 20% 10% Hail
    hail = np.array(
        [
            [
                4.78e-2,
                1.80e-1,
                1.95e-1,
                1.88e-1,
                2.36e-1,
                2.70e-1,
                2.98e-1,
                2.85e-1,
                1.75e-1,
                4.30e-2,
                8.80e-2,
            ],
            [
                1.23e-2,
                -3.73e-2,
                -3.83e-2,
                -3.29e-2,
                -3.46e-2,
                -2.94e-2,
                -2.10e-2,
                -9.96e-3,
                -8.05e-3,
                -8.27e-3,
                5.39e-2,
            ],
            [
                -3.50e-4,
                4.08e-3,
                4.14e-3,
                3.75e-3,
                3.71e-3,
                3.22e-3,
                2.44e-3,
                1.45e-3,
                1.21e-3,
                1.66e-3,
                -2.99e-4,
            ],
            [
                -3.30e-5,
                -1.59e-4,
                -1.54e-4,
                -1.39e-4,
                -1.30e-4,
                -1.12e-4,
                -8.56e-5,
                -5.33e-5,
                -4.66e-5,
                -7.19e-5,
                1.90e-5,
            ],
            [
                4.27e-7,
                1.59e-6,
                1.51e-6,
                1.37e-6,
                1.29e-6,
                1.15e-6,
                9.40e-7,
                6.71e-7,
                6.33e-7,
                9.52e-7,
                0.00e0,
            ],
        ]
    )


def rvp_to_dbz(x):
    """Calculates dBZ-values from DWD RVP6 values as given in DX-product
    files.

    Parameters
    ----------
    x : int
        a number or an array

    Examples
    --------
    >>> from wradlib.trafo import rvp_to_dbz
    >>> print(rvp_to_dbz(65.))
    0.0
    """
    return x * 0.5 - 32.5


def decibel(x):
    """Calculates the decibel representation of the input values

    :math:`dBZ=10 \\cdot \\log_{10} z`

    Parameters
    ----------
    x : a number or an array
        (must not be <= 0.)

    Examples
    --------
    >>> from wradlib.trafo import decibel
    >>> print(decibel(100.))
    20.0
    """
    return 10.0 * np.log10(x)


def idecibel(x):
    """Calculates the inverse of input decibel values

    :math:`z=10^{x \\over 10}`

    Parameters
    ----------
    x : a number or an array

    Examples
    --------
    >>> from wradlib.trafo import idecibel
    >>> print(idecibel(10.))
    10.0

    """
    return 10.0 ** (x / 10.0)


def r_to_depth(x, interval):
    """Computes rainfall depth (mm) from rainfall intensity (mm/h)

    Parameters
    ----------
    x : float,
        float or array of float
        rainfall intensity in mm/h
    interval : number
        time interval (s) the values of `x` represent

    Returns
    -------
    output : float
        float or array of float
        rainfall depth (mm)

    """
    return x * interval / 3600.0


def kdp_to_r(kdp, f, a=129.0, b=0.85):
    """Estimating rainfall intensity directly from specific differential phase.

    The general power law expression has been suggested by :cite:`Ryzhkov2005`.

    The default parameters have been set according to :cite:`Bringi2001`.

    Note
    ----
    **Please note that this way, rainfall intensities can become negative.**
    This is an intended behaviour in order to account for noisy :math:`K_{DP}`
    values.

    Parameters
    ----------
    kdp : float
        :math:`K_{DP}` as array of floats
    f : float
        radar frequency [GHz]

        - Standard frequencies in X-band range between 8.0 and 12.0 GHz,
        - Standard frequencies in C-band range between 4.0 and 8.0 GHz,
        - Standard frequencies in S-band range between 2.0 and 4.0 GHz.

    a : float
        linear coefficient of the power law
    b : float
        exponent of the power law

    Returns
    -------
    output : array
        array of rainfall intensity
    """
    return np.sign(kdp) * a * (np.abs(kdp) / f) ** b


def si_to_kmh(vals):
    """Conversion from SI wind speed units to km/hr.

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals : float
        float or array of floats
        Speed in SI units (m/s)

    Returns
    -------
    output: float
        float or array of floats
        Speed in km/hr

    Examples
    --------
    >>> from wradlib.trafo import si_to_kmh
    >>> print(si_to_kmh(1.))
    3.6
    """
    return vals * 3600.0 / 1000.0


def si_to_mph(vals):
    """Conversion from SI wind speed units to miles/hr

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals : float
        float or array of floats
        Speed in SI units (m/s)

    Returns
    -------
    output: float
        float or array of floats
        Speed in miles per hour

    Examples
    --------
    >>> from wradlib.trafo import si_to_mph
    >>> print(np.round(si_to_mph(1.), 3))
    2.237

    """
    return vals * 3600.0 / meters_per_mile


def si_2_kts(vals):
    """Conversion from SI wind speed units to knots

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals : float
        float or array of floats
        Speed in SI units (m/s)

    Returns
    -------
    output: float
        float or array of floats
        Speed in knots

    Examples
    --------
    >>> from wradlib.trafo import si_2_kts
    >>> print(np.round(si_2_kts(1.), 3))
    1.944

    """
    return vals * 3600.0 / meters_per_nautical_mile


def kmh_to_si(vals):
    """Conversion from km/hr to SI wind speed units

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals: float
        float or array of floats
        Wind speed in km/hr

    Returns
    -------
    output: float
        float or array of floats
        Wind speed in SI units (m/s)

    Examples
    --------
    >>> from wradlib.trafo import kmh_to_si
    >>> print(np.round(kmh_to_si(10.), 3))
    2.778

    """
    return vals * 1000.0 / 3600.0


def mph_to_si(vals):
    """Conversion from miles/hr to SI wind speed units

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals: float
        float or array of floats
        Wind speed in miles per hour

    Returns
    -------
    output: float
        float or array of floats
        Wind speed in SI units (m/s)

    Examples
    --------
    >>> from wradlib.trafo import mph_to_si
    >>> print(np.round(mph_to_si(10.), 2))
    4.47

    """
    return vals * meters_per_mile / 3600.0


def kts_to_si(vals):
    """Conversion from knots to SI wind speed units

    Note
    ----
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals: float
        float or array of floats
        Wind speed in knots

    Returns
    -------
    output: float
        float or array of floats
        Wind speed in SI units (m/s)

    Examples
    --------
    >>> from wradlib.trafo import kts_to_si
    >>> print(np.round(kts_to_si(1.), 3))
    0.514

    """
    return vals * meters_per_nautical_mile / 3600.0


if __name__ == "__main__":
    print("wradlib: Calling module <trafo> as main...")
