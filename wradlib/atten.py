# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:         atten
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Attenuation Correction
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

    correctAttenuationHB

"""
import numpy as np
import os, sys
import math
import logging

logger = logging.getLogger('attcorr')


class AttenuationOverflowError(Exception):
    pass


def correctAttenuationHB(gateset, coefficients=None, mode='', thrs=59.0):
    """Gate-by-Gate attenuation correction according to Hitschfeld & Bordan
    [Hitschfeld1954]_



    Parameters
    ----------
    gateset : array
        multidimensional array. The range gates (over which iteration has to
        be performed) are supposed to vary along
        the *last* dimension so, e.g., for a set of `l` radar images stored in
        polar form with `m` azimuths and `n` range-bins the input array's
        shape can be either (l,m,n) or (m,l,n)
    coefficients : dictionary
        correction coefficients
        a: proportionality factor of the k-Z relation ( :math:`k=a*Z^{b}` )
        b: exponent of the k-Z relation
        l: length of a range gate.
        if set to None the following default dictionary will be used
        {'a':5.0e-3, 'b':0.69, 'l':1.0}

    mode : string
        controls how the function reacts, if the sum of signal and attenuation
        exceeds the
        threshold ``thrs``
        Possible values:
        'warn' : emit a warning through the module's logger but continue
        execution
        'zero' : set offending gates to 0.0
        'nan' : set offending gates to nan
        Any other mode will raise an Exception

    thrs : float
        threshold, for the sum of attenuation and signal, which is deemed
        unplausible.

    Returns
    -------
    k : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation for each range gate

    Raises
    ------
    AttenuationOverflowError
        Exception, if attenuation exceeds ``thrs`` and no handling ``mode`` is
        set.

    References
    ----------

    .. [Hitschfeld1954] Hitschfeld, W. & Bordan, J., 1954.
        Errors Inherent in the Radar Measurement of Rainfall at Attenuating
        Wavelengths. Journal of the Atmospheric Sciences, 11(1), p.58-67.
        DOI: 10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

    .. comment
        _[1] Krämer2008 - Krämer, Stefan 2008: Quantitative Radardatenaufbereitung
        für die Niederschlagsvorhersage und die Siedlungsentwässerung,
        Mitteilungen Institut für Wasserwirtschaft, Hydrologie und
        Landwirtschaftlichen Wasserbau
        Gottfried Wilhelm Leibniz Universität Hannover, Heft 92, ISSN 0343-8090

    """
    if coefficients is None:
        _coefficients = {'a':5.0e-3, 'b':0.69, 'l':1.0}
    else:
        _coefficients = coefficients

    a = _coefficients['a']
    b = _coefficients['b']
    l = _coefficients['l']


    k = np.empty(gateset.shape)
    ksum = 0.

    # multidimensional version
    # assumes that iteration is only along the last dimension (i.e. range gates)
    # all other dimensions are calculated simultaneously to gain some speed
    for gate in range(gateset.shape[-1]):
        # calculate k in dB/km from k-Z relation
        # c.f. Krämer2008(p. 147)
        kn = a * (10.0**((gateset[...,gate] + ksum)/10.0))**b  * 2.0 * l
        #kn = 10**(log10(a)+0.1*bin*b)
        #dBkn = 10*math.log10(a) + (bin+ksum)*b + 10*math.log10(2*l)
        ksum += kn

        k[...,gate] = ksum
        # stop-criterion, if corrected reflectivity is larger than 59 dBZ
        overflow = (gateset[...,gate] + ksum) > thrs
        if np.any(overflow):
            if mode == 'warn':
                logger.warning('dB-sum over threshold (%3.1f)'%thrs)
            if mode == 'nan':
                k[gate,overflow] = np.nan
            if mode == 'zero':
                k[gate,overflow] = 0.0
            else:
                raise AttenuationOverflowError

    return k


if __name__ == '__main__':
    print 'wradlib: Calling module <atten> as main...'
