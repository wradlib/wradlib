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
   rvp2r
   decibel
   idecibel

"""

def rvp2dBZ(x):
    """Calculates dBZ-values from DWD RVP6 values as given in DX-product
    files.

    Parameters
    ----------
    x : a number or an array

    """
    return x*0.5-32.5

def rvp2r(x, **kwargs):
    """Calculates rain rates from RVP6 values directly using z2r.

    Parameters
    ----------
    x : a number or an array

    """
    return z2r(idecibel(rvp2dBZ(x)), **kwargs)

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


if __name__ == '__main__':
    print 'wradlib: Calling module <trafo> as main...'
