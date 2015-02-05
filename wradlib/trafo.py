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

"""
import numpy as np


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



if __name__ == '__main__':
    print 'wradlib: Calling module <trafo> as main...'
