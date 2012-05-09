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
    correctAttenuationConstrained
    constraint_dBZ
    constraint_PIA

"""
import numpy as np
import scipy.ndimage as ndi
import scipy.interpolate as spi
import os, sys
import math
import logging
from .trafo import idecibel

logger = logging.getLogger('attcorr')


class AttenuationOverflowError(Exception):
    pass


class AttenuationIterationError(Exception):
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
        data has to be provided in decibel representation of reflectivity [dBZ]

    a : float
        proportionality factor of the k-Z relation ( :math:`k=a*Z^{b}` ).
        Per default set to 1.67e-4.

    b : float
        exponent of the k-Z relation ( :math:`k=a*Z^{b}` ). Per default set to
        0.7.

    l : float
        length of a range gate [km]. Per default set to 1.0.

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
    pia : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation [dB] for each range gate.

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


    pia = np.empty(gateset.shape)
    pia[...,0] = 0.
    ksum = 0.

    # multidimensional version
    # assumes that iteration is only along the last dimension (i.e. range gates)
    # all other dimensions are calculated simultaneously to gain some speed
    for gate in range(gateset.shape[-1]-1):
        # calculate k in dB/km from k-Z relation
        # c.f. Krämer2008(p. 147)
        k = a * (10.0**((gateset[...,gate] + ksum)/10.0))**b  * 2.0 * l
        #k = 10**(log10(a)+0.1*bin*b)
        #dBkn = 10*math.log10(a) + (bin+ksum)*b + 10*math.log10(2*l)
        ksum += k

        pia[...,gate+1] = ksum
        # stop-criterion, if corrected reflectivity is larger than 59 dBZ
        overflow = (gateset[...,gate] + ksum) > thrs
        if np.any(overflow):
            if mode == 'warn':
                logger.warning('dB-sum over threshold (%3.1f)'%thrs)
            if mode == 'nan':
                pia[gate,overflow] = np.nan
            if mode == 'zero':
                pia[gate,overflow] = 0.0
            else:
                raise AttenuationOverflowError

    return pia


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

        Data has to be provided in decibel representation of reflectivity
        [dBZ].

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
        length of a range gate [km]. Per default set to 1.0.

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
        Threshold, for the attenuation corrected signal [dBZ], which is deemed
        unplausible. Per default set to 59.0 dBZ.

    Returns
    -------
    pia : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation [dB] for each range gate.

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
    pia = np.zeros(gateset.shape)
    pia[...,0] = 0.0
    # indexing all rows of last dimension (radarbeams)
    beams2correct = np.where(np.max(pia, axis = pia.ndim - 1) > (-1.))
    # iterate over possible a-parameters
    for i in range(n):
        ai = ai - da
        # subset of beams that have to be corrected and corresponding attenuations
        sub_gateset = gateset[beams2correct]
        sub_pia = pia[beams2correct]
        for gate in range(gateset.shape[-1] - 1):
            k = ai * (10.0**((sub_gateset[...,gate] + sub_pia[...,gate])/10.0))**b  * 2.0 * l
            sub_pia[...,gate + 1] = sub_pia[...,gate] + k
        # integration of the calculated attenuation subset to the whole attenuation matrix
        pia[beams2correct] = sub_pia
        # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the threshold
        beams2correct = np.where(np.max(gateset + pia, axis = pia.ndim - 1) > thrs_dBZ)
        # if there is no beam left for correction, the iteration can be interrupted prematurely
        if len(pia[beams2correct]) == 0: break
    if len(pia[beams2correct]) > 0:
        if mode == 'warn': logger.warning('dB-sum over threshold (%3.1f)'%thrs)
        elif mode == 'nan':  pia[beams2correct] = np.nan
        elif mode == 'zero': pia[beams2correct] = 0.0
        else: raise AttenuationOverflowError

    return pia


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

        Data has to be provided in decibel representation of reflectivity
        [dBZ].

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
        length of a range gate [km]. Per default set to 1.0.

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
        Threshold, for the attenuation corrected signal [dBZ], which is deemed
        unplausible. Per default set to 59.0 dBZ.

    max_PIA : float
        threshold, for the maximum path integrated attenuation [dB] which allows
        reasonable attenuation corrections. Per default set to 20.0 dB.

    Returns
    -------
    pia : array
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
    pia = np.where(np.isnan(gateset), np.nan, 0.)
    pia[np.where(np.isnan(pia))[0]] = np.nan
    # indexing all rows of last dimension (radarbeams) except rows including NaNs
    beams2correct = np.where(np.max(pia, axis=-1) > (-1.))
    # iterate over possible a-parameters
    for i in range(n):
        ai = ai - da
        # subset of beams that have to be corrected and corresponding attenuations
        sub_gateset = gateset[beams2correct]
        sub_pia = pia[beams2correct]
        for gate in range(gateset.shape[-1] - 1):
            k = ai * (10.0**((sub_gateset[...,gate] + sub_pia[...,gate])/10.0))**b  * 2.0 * l
            sub_pia[...,gate + 1] = sub_pia[...,gate] + k
        # integration of the calculated attenuation subset to the whole attenuation matrix
        pia[beams2correct] = sub_pia
        # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the thresholds
        # of corrected attenuation or PIA or NaNs are occuring
        beams2correct = np.where(np.logical_or(np.max(gateset + pia, axis=-1) > thrs_dBZ,
                                               np.max(pia, axis=-1) > max_PIA))
        # if there is no beam left for correction, the iteration can be interrupted prematurely
        if len(pia[beams2correct]) == 0: break
    if len(pia[beams2correct]) > 0:
        if mode == 'warn': logger.warning('threshold exceeded (corrected dBZ or PIA) even for lowest a')
        elif mode == 'nan':  pia[beams2correct] = np.nan
        elif mode == 'zero': pia[beams2correct] = 0.0
        else: raise AttenuationOverflowError

    return pia


def correctAttenuationConstrained(gateset, a_max=1.67e-4, a_min=2.33e-5,
                                b_max=0.7, b_min=0.2, na=30, nb=5, l=1.0,
                                mode='error',
                                constraints=None, constr_args=None,
                                diagnostics={}):
    """Gate-by-Gate attenuation correction based on the iterative approach of
    Stefan Kraemer [Kraemer2008]_ with a generalized and arbitrary number of
    constraints.

    Parameters
    ----------
    gateset : array
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the *last* dimension so,
        e.g., for a set of `l` radar images stored in polar form with `m`
        azimuths and `n` range-bins the input array's shape can be either
        (l,m,n) or (m,l,n).

        Data has to be provided in decibel representation of reflectivity
        [dBZ].

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
        length of a range gate [km]. Per default set to 1.0.

    mode : string
        Controls how the function reacts in case of signal overflow (sum of
        signal and attenuation exceeds the threshold ``thrs``).
        Possible values:

        'warn' : emit a warning through the module's logger but continue
        execution

        'zero' : set offending gates to 0.0

        'nan' : set offending gates to nan

        Per default set to 'zero'. Any other mode will raise an Exception.

    constraints : list
        list of constraint functions. The signature of these functions has to be
        constraint_function(`gateset`, `k`, *`constr_args`). Their return value
        must be a boolean array of shape gateset.shape[:-1] set to True for
        beams, which do not fulfill the constraint.

    constr_args : list
        list of lists, which are to be passed to the individual constraint
        functions using the *args mechanism
        (len(constr_args) == len(constraints))

    diagnostics : dictionary
        dictionary of variables, which are usually not returned by the function
        but may be of interest for research or during testing. Defaults to {},
        in which case no diagnostics are generated. If a dictionary with
        certain keys is passed to the function, the respective diagnostics are
        generated.
        Currently implemented diagnostics:

            - 'a' - returns the values of the a coefficient of the k-Z
               relation, which was used to calculate the attenuation for the
               respective beam as a np.array. The shape of the returned array
               will be gateset.shape[:-1].

    Returns
    -------
    k : array
        Array with the same shape as ``gateset`` containing the calculated
        attenuation [dB] for each range gate.

    Raises
    ------
    AttenuationOverflowError
        Exception, if not all constraints are satisfied even with the smallest
        possible linear coefficient (a_min) and no handling ``mode`` is set.

    References
    ----------

    .. [Kraemer2008] Krämer, Stefan 2008: Quantitative Radardatenaufbereitung
        für die Niederschlagsvorhersage und die Siedlungsentwässerung,
        Mitteilungen Institut für Wasserwirtschaft, Hydrologie und
        Landwirtschaftlichen Wasserbau
        Gottfried Wilhelm Leibniz Universität Hannover, Heft 92, ISSN 0343-8090.

    Examples
    --------
    Implementing the original Hitschfeld & Bordan (1954) algorithm with
    otherwise default parameters
    >>> k = correctAttenuationConstrained(gateset, n=1, mode='nan')

    Implementing the basic Kraemer algorithm
    >>> k = correctAttenuationConstrained(gateset,
    ...                                   mode='nan',
    ...                                   constraints=[constraint_dBZ],
    ...                                   constr_args[[59.0]])

    Implementing the PIA algorithm by Jacobi et al.
    >>> k = correctAttenuationConstrained(gateset,
    ...                                   mode='nan',
    ...                                   constraints=[constraint_dBZ,
    ...                                                constraint_PIA],
    ...                                   constr_args[[59.0],
    ...                                               [20.0]])

    """

    if np.max(np.isnan(gateset)): raise Exception('There are not processable NaN in the gateset!')

    if constraints is None:
        constraints = []
    if constr_args is None:
        constr_args = []


    a_used = np.empty(gateset.shape[:-1])
    b_used = np.empty(gateset.shape[:-1])

    da = (a_max - a_min) / (na - 1)
    ai = a_max + da
    k = np.zeros(gateset.shape)
    db = (b_max - b_min) / (nb - 1)

    # indexing all rows of last dimension (radarbeams)
    beams2correct = np.where(np.max(k, axis=-1) > (-1.))
    # iterate over possible a-parameters
    for j in range(nb):
        bi = b_max - db*j
        for i in range(na):
            ai = a_max - da*i
            # subset of beams that have to be corrected and corresponding attenuations
            sub_gateset = gateset[beams2correct]
            sub_k = k[beams2correct]
            for gate in range(gateset.shape[-1] - 1):
                kn = ai * (10.0**((sub_gateset[...,gate] + sub_k[...,gate])/10.0))**bi  * 2.0 * l
                sub_k[...,gate + 1] = sub_k[...,gate] + kn
            # integration of the calculated attenuation subset to the whole attenuation matrix
            k[beams2correct] = sub_k
            a_used[beams2correct] = ai
            b_used[beams2correct] = bi
            # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the threshold
            incorrectbeams = np.zeros(gateset.shape[:-1], dtype=np.bool)
            for constraint, constr_arg in zip(constraints, constr_args):
                incorrectbeams |= constraint(gateset, k, *constr_arg)
            beams2correct = np.where(incorrectbeams) #np.where(np.max(gateset + k, axis = k.ndim - 1) > thrs_dBZ)
            # if there is no beam left for correction, the iteration can be interrupted prematurely
            if len(k[beams2correct]) == 0: break
        if len(k[beams2correct]) == 0: break
    if len(k[beams2correct]) > 0:
        if mode == 'warn': logger.warning('correction did not fulfill constraints within given parameter range')
        elif mode == 'nan':  k[beams2correct] = np.nan
        elif mode == 'zero': k[beams2correct] = 0.0
        else: raise AttenuationOverflowError

    if diagnostics.has_key('a'):
            diagnostics['a'] = a_used
    if diagnostics.has_key('b'):
            diagnostics['b'] = b_used

    return k


def constraint_dBZ(gateset, k, thrs_dBZ):
    """Constraint callback function for correctAttenuationConstrained.
    Selects beams, in which at least one pixel exceeds `thrs_dBZ` [dBZ].
    """
    return np.max(gateset + k, axis=-1) > thrs_dBZ


def constraint_PIA(gateset, k, thrs_PIA):
    """Constraint callback function for correctAttenuationConstrained.
    Selects beams, in which the path integrated attenuation exceeds `thrs_PIA`.
    """
    return np.max(k, axis=-1) > thrs_PIA


#-------------------------------------------------------------------------------
# new implementation of Kraemer algorithm
#-------------------------------------------------------------------------------
def calc_attenuation_forward(gateset, a, b, l):
    """Gate-by-Gate forward correction as described in [Kraemer2008]_"""
    k = np.zeros(gateset.shape)
    for gate in range(gateset.shape[-1]-1):
        kn = a * idecibel(gateset[...,gate] + k[...,gate])**b  * 2.0 * l
        k[...,gate+1] = k[...,gate] + kn
    return k


def calc_attenuation_backward(gateset, a, b, l, a_ref, tdiff, maxiter):
    """Gate-by-Gate backward correction as described in [Kraemer2008]_"""
    k = np.zeros(gateset.shape)
    k[...,-1] = a_ref
    for gate in range(gateset.shape[-1]-2, 0, -1):
        kright = np.zeros(gateset.shape[:-1])+a_ref/gateset.shape[-1]
        toprocess = np.ones(gateset.shape[:-1], dtype=np.bool)
        for j in range(maxiter):
            kleft = a * (idecibel(gateset[...,gate][toprocess]
                                  + k[...,gate+1][toprocess]
                                  - kright[toprocess]))**b * 2.0 * l
            diff = np.abs(kleft-kright)
            kright[toprocess] = kleft
            toprocess[diff<tdiff] = False
            if ~np.any(toprocess):
                break

        if j == maxiter-1:
            raise AttenuationIterationError

        k[...,gate] = k[...,gate+1] - kright
    #k = np.cumsum(k, axis=-1)
    return k


def bisectReferenceAttenuation(gateset,
                               a_ref,
                               a_max=1e-3,
                               a_min=1e-7,
                               b=0.7,
                               l=1,
                               thrs=0.05,
                               maxiter=np.inf):
    """Find the optimal linear coefficient for a gateset to
    achieve a given reference attenuation using a forward correction algorithm
    """
    a_hi = a_max
    a_lo = a_min

    itercount = 0
    while not np.all(a_hi == a_lo):
        a_mid = (a_hi+a_lo)/2
        k = calcAttenuationForward(gateset, a_mid, b, l)
        overshoot = ((k[...,-1] - a_ref)/a_ref) > thrs
        undershoot = ((k[...,-1] - a_ref)/a_ref) < -thrs
        hit = (np.abs(k[...,-1] - a_ref)/a_ref) < thrs
        a_hi[overshoot] = a_mid[overshoot]
        a_lo[undershoot] = a_mid[undershoot]
        a_hi[hit] = a_mid[hit]
        a_lo[hit] = a_mid[hit]
        itercount+=1
        if itercount > maxiter:
            raise AttenuationIterationError

    return k, a_hi


def sector_filter(mask, min_sector_size, axis=-1):
    """"""
    kernel = np.ones((min_sector_size,))
    forward_origin = -(min_sector_size - (min_sector_size // 2)) + min_sector_size%2
    backward_origin = (min_sector_size - (min_sector_size // 2)) - 1
    forward_sum = ndi.correlate1d(mask, kernel, axis=-1, mode='wrap', origin=forward_origin)
    backward_sum = ndi.correlate1d(mask, kernel, axis=-1, mode='wrap', origin=backward_origin)
    forward_corners = (forward_sum == min_sector_size)
    backward_corners = (backward_sum == min_sector_size)
    forward_large_sectors = ndi.morphology.binary_dilation(forward_corners, kernel, origin=forward_origin).astype(int)
    backward_large_sectors = ndi.morphology.binary_dilation(backward_corners, kernel, origin=backward_origin).astype(int)

    return (forward_large_sectors | backward_large_sectors)


def nd_pad(data, pad, axis=-1, mode='wrap'):
    """"""
    axislen = data.shape[axis]
    new_shape = np.array(data.shape)
    new_shape[axis] += 2*pad
    new_data = np.empty(new_shape)

    dataslices = [slice(None, None) for i in new_shape]
    dataslices[axis] = slice(pad, axislen+pad)

    new_data[dataslices]=data

    if mode=='wrap':
        old_leftslice = [slice(None, None) for i in new_shape]
        old_leftslice[axis] = slice(0, pad)
        old_rightslice = [slice(None, None) for i in new_shape]
        old_rightslice[axis] = slice(axislen-pad, axislen)

        new_leftslice = [slice(None, None) for i in new_shape]
        new_leftslice[axis] = slice(0,pad)
        new_rightslice = [slice(None, None) for i in new_shape]
        new_rightslice[axis] = slice(axislen+pad, axislen+2*pad)

        new_data[new_leftslice] = data[old_rightslice]
        new_data[new_rightslice] = data[old_leftslice]

    return new_data


def interp_atten(k, invalidbeams):
    """"""
    x = np.arange(3*k.shape[1])

    for i in range(k.shape[0]):
        sub_i = invalidbeams[i,:]
        sub_k = k[i,:,-1]
        pad_i = np.concatenate([sub_i]*3)
        pad_k = np.concatenate([sub_k]*3)

        intp = scipy.interpolate.interp1d(x[~pad_i], pad_k[~pad_i], mode='linear')

        k[i,sub_i,-1] = intp(x[k.shape[1]:2*k.shape[1]+1][sub_i])


def correctAttenuationConstrained2(gateset, a_max, a_min, na, b_max, b_min, nb, l,
                                   mode='error', constraints=None,
                                   thr_sec=10,
                                   constr_args=None,
                                   diagnostics={}):
    """"""
    if np.any(np.isnan(gateset)):
        raise Exception('There are not processable NaN in the gateset!')

    if constraints is None:
        constraints = []
    if constr_args is None:
        constr_args = []
    naz = gateset.shape[-2]
    nrng = gateset.shape[-1]
    tmp_gateset = gateset.reshape((-1,naz,nrng))

    k = np.zeros_like(tmp_gateset)

    a_used = np.empty(tmp_gateset.shape[:-1])
    b_used = np.empty(tmp_gateset.shape[:-1])

    # forward attenuation calculation
    # indexing all rows of last dimension (radarbeams)
    beams2correct = np.where(np.ones(tmp_gateset.shape[:-1], dtype=np.bool))
    invalidbeams = np.zeros(tmp_gateset.shape[:-1], dtype=np.bool)

    da = (a_max - a_min)/na
    db = (b_max - b_min)/nb

    # iterate over possible a-parameters
    for j in range(nb):
        bi = b_max - db*j
        for i in range(na):
            ai = a_max - da*i
            # subset of beams that have to be corrected and corresponding attenuations
            sub_gateset = tmp_gateset[beams2correct]
            sub_k = calc_attenuation_forward(sub_gateset, ai, bi, l)
            # mark overflow
            k[beams2correct] = sub_k
            a_used[beams2correct] = ai
            b_used[beams2correct] = bi
            # indexing the rows of the last dimension (radarbeam), if any corrected values exceed the threshold
            incorrectbeams = np.zeros(tmp_gateset.shape[:-1], dtype=np.bool)
            for constraint, constr_arg in zip(constraints, constr_args):
                incorrectbeams |= constraint(tmp_gateset, k, *constr_arg)

            # determine sectors larger than thr_sec
            beams2correct = np.where(sector_filter(incorrectbeams, thr_sec))
            if len(k[beams2correct]) == 0: break
        if len(k[beams2correct]) == 0: break

    # interpolate the reference attenuations to invalid sectors

    # do reference bisections over invalid sectors

    return k.reshape(gateset.shape)



if __name__ == '__main__':
    print 'wradlib: Calling module <atten> as main...'
