#-------------------------------------------------------------------------------
# Name:        trafo
# Purpose:
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     26.10.2011
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Data Transformation
^^^^^^^^^^^^^^^^^^^

Module <trafo> transforms data e.g. from RVP-units
to dBZ-values to Z-values and vice versa.

.. currentmodule:: wradlib.trafo

.. autosummary::
   :nosignatures:
   :toctree: generated/

   rvp2dBZ
   decibel
   idecibel
   r2depth
   kdp2r
   si2kmh
   si2mph
   si2kts
   kmh2si
   mph2si
   kts2si

"""
import numpy as np


# CONSTANTS
meters_per_mile = 1609.344
meters_per_nautical_mile  = 1852.


def rvp2dBZ(x):
    """Calculates dBZ-values from DWD RVP6 values as given in DX-product
    files.

    Parameters
    ----------
    x : a number or an array

    """
    return x*0.5-32.5


def decibel(x):
    """Calculates the decibel representation of the input values
    dBZ = 10*log10(z)

    Parameters
    ----------
    x : a number or an array (must not be <= 0.)

    """
    return 10.*np.log10(x)


def idecibel(x):
    """Calculates the inverse of input decibel values
    10.**(x/10.)

    Parameters
    ----------
    x : a number or an array

    """
    return 10.**(x/10.)


def r2depth(x, interval):
    """
    Computes rainfall depth (mm) from rainfall intensity (mm/h)

    Parameters
    ----------
    x : float or array of float
        rainfall intensity in mm/h
    interval : number
        time interval (s) the values of `x` represent

    Returns
    -------
    output : float or array of float
        rainfall depth (mm)

    """
    return x * interval / 3600.


def kdp2r(kdp, f, a=129., b=0.85):
    """Estimating rainfall intensity directly from specific differential phase.

    The general power law expression has been suggested by Ryzhkov et al. :cite:`Ryzhkov2005`.

    The default parameters have been set according to Bringi and Chandrasekar :cite:`Bringi2001`.

    **Please note that this way, rainfall intensities can become negative.** This is
    an intended behaviour in order to account for noisy Kdp values.

    Parameters
    ----------
    kdp : Kdp as array of floats

    f : radar frequency [GHz]

       Standard frequencies in X-band range between 8.0 and 12.0 GHz,

       Standard frequencies in C-band range between 4.0 and 8.0 GHz,

       Standard frequencies in S-band range between 2.0 and 4.0 GHz.

    a : linear coefficient of the power law

    b : exponent of the power law

    Returns
    -------
    output : array of rainfall intensity

    """
    return np.sign(kdp) * a * (np.abs(kdp) / f)**b


def si2kmh(vals):
    """Conversion from SI wind speed units to km/hr.
    
    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.
    
    Parameters
    ----------
    vals : float or array of floats
        Speed in SI units (m/s)
 
    Returns
    -------
    output: float or array of floats
        Speed in km/hr
    
    Examples
    --------
    >>> from wradlib.trafo import si2kmh
    >>> print si2kmh(1.)
    3.6
    """
    return vals * 3600. / 1000.


def si2mph(vals):
    """Conversion from SI wind speed units to miles/hr

    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals : float or array of floats
        Speed in SI units (m/s)
 
    Returns
    -------
    output: float
        Speed in miles per hour
    
    Examples
    --------
    >>> from wradlib.trafo import si2mph
    >>> print np.round( si2mph(1.), 3 )
    2.237

    """
    return vals * 3600. / meters_per_mile
    

def si2kts(vals):
    """Conversion from SI wind speed units to knots

    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.

    Parameters
    ----------
    vals : float or array of floats
        Speed in SI units (m/s)
 
    Returns
    -------
    output: float
        Speed in knots
    
    Examples
    --------
    >>> from wradlib.trafo import si2kts
    >>> print np.round( si2kts(1.), 3 )
    1.944

    """
    return vals * 3600. / meters_per_nautical_mile
    

def kmh2si(vals):
    """Conversion from km/hr to SI wind speed units

    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.
    
    Parameters
    ----------
    vals: float or array of floats
        Wind speed in km/hr
 
    Returns
    -------
    output: float or array of floats
        Wind speed in SI units (m/s)
    
    Examples
    --------
    >>> from wradlib.trafo import kmh2si
    >>> print np.round( kmh2si(10.), 3 )
    2.778

    """
    return vals * 1000. / 3600.


def mph2si(vals):
    """Conversion from miles/hr to SI wind speed units

    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.
    
    Parameters
    ----------
    vals: float or array of floats
        Wind speed in miles per hour
 
    Returns
    -------
    output: float or array of floats
        Wind speed in SI units (m/s)
    
    Examples
    --------
    >>> from wradlib.trafo import mph2si
    >>> print np.round( mph2si(10.), 2 )
    4.47

    """
    return vals * meters_per_mile / 3600.


def kts2si(vals):
    """Conversion from knots to SI wind speed units

    Notes
    -----
    ..versionadded:: 0.6.0
    
    Code was migrated from https://github.com/nguy/PyRadarMet.
    
    Parameters
    ----------
    vals: float or array of floats
        Wind speed in knots
 
    Returns
    -------
    output: float or array of floats
        Wind speed in SI units (m/s)
    
    Examples
    --------
    >>> from wradlib.trafo import kts2si
    >>> print np.round( kts2si(1.), 3 )
    0.514

    """
    return vals * meters_per_nautical_mile / 3600.


if __name__ == '__main__':
    print 'wradlib: Calling module <trafo> as main...'
