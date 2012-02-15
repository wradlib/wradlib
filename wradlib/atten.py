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
    correctAttenuationKraemer
    correctAttenuationHJ

"""
import numpy as np
import os, sys
import math
import logging

logger = logging.getLogger('attcorr')


class AttenuationOverflowError(Exception):
    pass


def correctAttenuationHB(gateset, a = 1.67e-4, b = 0.7, l = 1.0, mode='',
                         thrs=59.0):
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
        data havs to be provided in decibel representation of reflectivity (dBZ)

    a : float
        proportionality factor of the k-Z relation ( :math:`k=a*Z^{b}` ).
        Per default set to 1.67e-4.

    b : float
        exponent of the k-Z relation ( :math:`k=a*Z^{b}` ). Per default set to
        0.7.

    l : float
        length of a range gate. Per default set to 1.0.

    mode : string
        controls how the function reacts, if the sum of signal and attenuation
        exceeds the
        threshold ``thrs``
        Possible values:
        'warn' : emit a warning through the module's logger but continue
        execution
        'zero' : set offending gates to 0.0
        'nan' : set offending gates to nan
        Any other mode and default setting will raise an Exception.

    thrs : float
        threshold, for the sum of attenuation and signal, which is deemed
        unplausible.

    Returns
    -------
    k : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation for each range gate.

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
        Gottfried Wilhelm Leibniz Universität Hannover, Heft 92, ISSN 0343-8090.

    """
    if coefficients is None:
        _coefficients = {'a':1.67e-4, 'b':0.7, 'l':1.0}
    else:
        _coefficients = coefficients

    a = _coefficients['a']
    b = _coefficients['b']
    l = _coefficients['l']


    k = np.empty(gateset.shape)
    k[...,0] = 0.
    ksum = 0.

    # multidimensional version
    # assumes that iteration is only along the last dimension (i.e. range gates)
    # all other dimensions are calculated simultaneously to gain some speed
    for gate in range(gateset.shape[-1]-1):
        # calculate k in dB/km from k-Z relation
        # c.f. Krämer2008(p. 147)
        kn = a * (10.0**((gateset[...,gate] + ksum)/10.0))**b  * 2.0 * l
        #kn = 10**(log10(a)+0.1*bin*b)
        #dBkn = 10*math.log10(a) + (bin+ksum)*b + 10*math.log10(2*l)
        ksum += kn

        k[...,gate+1] = ksum
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


def correctAttenuationKraemer(gateset,  a_max = 1.67e-4, a_min = 2.33e-5,
                              b = 0.7, n = 30, l = 1.0, mode = 'zero',
                              thrs_dBZ = 59.0):
    """Gate-by-Gate attenuation correction according to Stefan Kraemer
    [Kraemer2008]_.



    Parameters
    ----------
    gateset : array
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the *last* dimension so,
        e.g., for a set of `l` radar images stored in polar form with `m`
        azimuths and `n` range-bins the input array's shape can be either
        (l,m,n) or (m,l,n).

        Data havs to be provided in decibel representation of reflectivity
        (dBZ).

    a_max : float
        initial value for linear coefficient of the k-Z relation
        ( :math:`k=a*Z^{b}` ). Per default set to 1.67e-4.

    a_min : float
        minimal allowed linear coefficient of the k-Z relation
        ( :math:`k=a*Z^{b}` ) in the downwards iteration of a in case of signal
        overflow (sum of signal and attenuation exceeds the threshold ``thrs``).
        Per default set to 2.33e-5.

    b : float
        exponent of the k-Z relation ( :math:`k=a*Z^{b}` ). Per default set to
        0.7.

    n : integer
        number of iterations from a_max to a_min. Per default set to 30.

    l : float
        length of a range gate. Per default set to 1.0.

    mode : string
        Controls how the function reacts in case of signal overflow (sum of
        signal and attenuation exceeds the threshold ``thrs``).
        Possible values:

        'warn' : emit a warning through the module's logger but continue
        execution

        'zero' : set offending gates to 0.0

        'nan' : set offending gates to nan

        Per default set to 'zero'. Any other mode will raise an Exception.

    thrs_dBZ : float
        Threshold, for the attenuation corrected signal in dBZ, which is deemed
        unplausible. Per default set to 59.0 dBZ.

    Returns
    -------
    k : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation for each range gate.

    Raises
    ------
    AttenuationOverflowError
        Exception, if attenuation exceeds ``thrs`` even with smallest possible
        linear coefficient (a_min) and no handling ``mode`` is set.

    References
    ----------

    .. [Kraemer2008] Krämer, Stefan 2008: Quantitative Radardatenaufbereitung
        für die Niederschlagsvorhersage und die Siedlungsentwässerung,
        Mitteilungen Institut für Wasserwirtschaft, Hydrologie und
        Landwirtschaftlichen Wasserbau
        Gottfried Wilhelm Leibniz Universität Hannover, Heft 92, ISSN 0343-8090.

    """

    if np.max(np.isnan(gateset)): raise Exception('There are not processable NaN in the gateset!')

    da = (a_max - a_min) / (n - 1)
    ai = a_max + da
    k = np.zeros(gateset.shape)
    k[...,0] = 0.0
    # indexing all rows of last dimension (radarbeams)
    beams2correct = np.where(np.max(k, axis = k.ndim - 1) > (-1.))
    # iterate over possible a-parameters
    for i in range(n):
        ai = ai - da
        # subset of beams that have to be corrected and corresponding attenuations
        sub_gateset = gateset[beams2correct]
        sub_k = k[beams2correct]
        for gate in range(gateset.shape[-1] - 1):
            kn = ai * (10.0**((sub_gateset[...,gate] + sub_k[...,gate])/10.0))**b  * 2.0 * l
            sub_k[...,gate + 1] = sub_k[...,gate] + kn
        # integration of the calculated attenuation subset to the whole attenuation matrix
        k[beams2correct] = sub_k
        # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the threshold
        beams2correct = np.where(np.max(gateset + k, axis = k.ndim - 1) > thrs_dBZ)
        # if there is no beam left for correction, the iteration can be interrupted prematurely
        if len(k[beams2correct]) == 0: break
    if len(k[beams2correct]) > 0:
        if mode == 'warn': logger.warning('dB-sum over threshold (%3.1f)'%thrs)
        elif mode == 'nan':  k[beams2correct] = np.nan
        elif mode == 'zero': k[beams2correct] = 0.0
        else: raise AttenuationOverflowError

    return k


def correctAttenuationHJ(gateset, a_max = 1.67e-4, a_min = 2.33e-5, b = 0.7,
                         n = 30, l = 1.0, mode = 'zero', thrs_dBZ = 59.0,
                         max_PIA = 20.0):
    """Gate-by-Gate attenuation correction based on Stefan Kraemer
    [Kraemer2008]_, expanded by Stephan Jacobi, Maik Heistermann and
    Thomas Pfaff [Jacobi2011]_.



    Parameters
    ----------
    gateset : array
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the *last* dimension so,
        e.g., for a set of `l` radar images stored in polar form with `m`
        azimuths and `n` range-bins the input array's shape can be either
        (l,m,n) or (m,l,n).

        Data havs to be provided in decibel representation of reflectivity
        (dBZ).

    a_max : float
        initial value for linear coefficient of the k-Z relation
        ( :math:`k=a*Z^{b}` ). Per default set to 1.67e-4.

    a_min : float
        minimal allowed linear coefficient of the k-Z relation
        ( :math:`k=a*Z^{b}` ) in the downwards iteration of a in case of signal
        overflow (sum of signal and attenuation exceeds the threshold ``thrs``).
        Per default set to 2.33e-5.

    b : float
        exponent of the k-Z relation ( :math:`k=a*Z^{b}` ). Per default set to
        0.7.

    n : integer
        number of iterations from a_max to a_min. Per default set to 30.

    l : float
        length of a range gate. Per default set to 1.0.

    mode : string
        Controls how the function reacts in case of signal overflow (sum of
        signal and attenuation exceeds the threshold ``thrs``).
        Possible values:

        'warn' : emit a warning through the module's logger but continue
        execution

        'zero' : set offending gates to 0.0

        'nan' : set offending gates to nan

        Per default set to 'zero'. Any other mode will raise an Exception.

    thrs_dBZ : float
        Threshold, for the attenuation corrected signal in dBZ, which is deemed
        unplausible. Per default set to 59.0 dBZ.

    max_PIA : float
        threshold, for the maximum path integrated attenuation which allows
        reasonable attenuation corrections. Per default set to 20.0 dB.

    Returns
    -------
    k : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation [dB] for each range gate. In case the input array (gateset)
        contains NaNs the corresponding beams of the output array (k) will be
        set as NaN, too.

    Raises
    ------
    AttenuationOverflowError
        Exception, if attenuation exceeds ``thrs`` even with smallest possible
        linear coefficient (a_min) and no handling ``mode`` is set.

    References
    ----------

    .. [Jacobi2011] Jacobi, S., Heistermann, M., Pfaff, T. 2011: Evaluation and
        improvement of C-band radar attenuation correction for operational flash
        flood forecasting.
        Proceedings of the Weather Radar and Hydrology symposium, Exeter, UK,
        April 2011, IAHS Publ. 3XX, 2011, in review.

    """

#    if np.any(np.isnan(gateset)):
#        raise ValueError, 'There are NaNs in the gateset! Cannot continue.'
#    k = np.zeros(gateset.shape)
    da = (a_max - a_min) / (n - 1)
    ai = a_max + da
##  initialize an attenuation array with the same shape as the gateset,
##  filled with zeros, except that NaNs occuring in the gateset will cause a
##  initialization with Nans for the ENTIRE corresponding attenuation beam
    k = np.where(np.isnan(gateset), np.nan, 0.)
    k[np.where(np.isnan(k))[0]] = np.nan
    # indexing all rows of last dimension (radarbeams) except rows including NaNs
    beams2correct = np.where(np.max(k, axis=-1) > (-1.))
    # iterate over possible a-parameters
    for i in range(n):
        ai = ai - da
        # subset of beams that have to be corrected and corresponding attenuations
        sub_gateset = gateset[beams2correct]
        sub_k = k[beams2correct]
        for gate in range(gateset.shape[-1] - 1):
            kn = ai * (10.0**((sub_gateset[...,gate] + sub_k[...,gate])/10.0))**b  * 2.0 * l
            sub_k[...,gate + 1] = sub_k[...,gate] + kn
        # integration of the calculated attenuation subset to the whole attenuation matrix
        k[beams2correct] = sub_k
        # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the thresholds
        # of corrected attenuation or PIA or NaNs are occuring
        beams2correct = np.where(np.logical_or(np.max(gateset + k, axis=-1) > thrs_dBZ,
                                               np.max(k, axis=-1) > max_PIA))
        # if there is no beam left for correction, the iteration can be interrupted prematurely
        if len(k[beams2correct]) == 0: break
    if len(k[beams2correct]) > 0:
        if mode == 'warn': logger.warning('threshold exceeded (corrected dBZ or PIA) even for lowest a')
        elif mode == 'nan':  k[beams2correct] = np.nan
        elif mode == 'zero': k[beams2correct] = 0.0
        else: raise AttenuationOverflowError

    return k

if __name__ == '__main__':
    print 'wradlib: Calling module <atten> as main...'
