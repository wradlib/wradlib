#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Hydrometeor Classification (HMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}
"""
__all__ = ["msf_index_indep", "trapezoid", "fuzzyfi", "probability", "classify"]
__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np

pr_types = {
    0: ("LR", "Light Rain"),
    1: ("MR", "Moderate Rain"),
    2: ("HR", "Heavy Rain"),
    3: ("LD", "Large Drops"),
    4: ("HL", "Hail"),
    5: ("RH", "Rain/Hail"),
    6: ("GH", "Graupel/Hail"),
    7: ("DS", "Dry Snow"),
    8: ("WS", "Wet Snow"),
    9: ("HC", "H Crystals"),
    10: ("VC", "V Crystals"),
    11: ("NP", "No Precip"),
}


def msf_index_indep(msf, idp, obs):
    """Retrieve membership function values based on independent observable

    Parameters
    ----------
    msf : :class:`numpy:numpy.ndarray`
        Array of size (hmc-classes, observables, indep-ranges, 4) containing
        the values of the trapezoidal msf values for every hmc-class and
        observable within the independent observable range.
    idp : :class:`numpy:numpy.ndarray`
        Array of length of the independent observable containing the ranges
        of the independent observable.
    obs : :class:`numpy:numpy.ndarray`
        Array of arbitrary shape containing the data of the independent
        observable (eg. (rays, bins) or (scan, rays, bins)).

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        Array of shape (hmc-classes, observables, obs.shape, 4) containing the
        membership function values for every radar-bin for every hmc-class and
        observable.
    """
    bins = np.append(idp, idp[-1] + (idp[-1] - idp[-2]))
    idx = np.digitize(obs, bins) - 1
    idx_mask = np.zeros_like(idx, dtype=np.bool_)
    idxm = np.ma.array(idx, mask=idx_mask)
    idxm = np.ma.masked_outside(idxm, 0, bins.shape[0] - 2)
    out = np.zeros((msf.shape[0], msf.shape[1], obs.size, msf.shape[-1]))
    out[:, :, ~idxm.mask.flatten(), :] = msf[:, :, idxm.compressed(), :]
    out = np.reshape(out, ((msf.shape[0], msf.shape[1]) + obs.shape + (msf.shape[-1],)))
    return out


def trapezoid(msf, obs):
    """Calculates membership of `obs` using trapezoidal
    membership functions

    Parameters
    ----------
    msf : :class:`numpy:numpy.ndarray`
        Array which is of size (obs.shape, 4), containing the trapezoidal
        membership function values for every `obs` point for one particular
        hydrometeor class.
    obs : :class:`numpy:numpy.ndarray`
        Array of arbitrary size and dimensions containing
        the data from which the membership shall be calculated.

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        Array which is of (obs.shape) containing calculated membership
        probabilities.
    """
    out = np.zeros_like(obs)

    ones = (obs >= msf[..., 1]) & (obs <= msf[..., 2])
    out[ones] = 1.0

    lower = (obs >= msf[..., 0]) & (obs < msf[..., 1])
    out[lower] = (obs[lower] - msf[..., 0][lower]) / (
        msf[..., 1][lower] - msf[..., 0][lower]
    )

    higher = (obs > msf[..., 2]) & (obs <= msf[..., 3])
    out[higher] = (obs[higher] - msf[..., 3][higher]) / (
        msf[..., 2][higher] - msf[..., 3][higher]
    )

    return out


def fuzzyfi(msf, obs):
    """Iterate over all hmc-classes and retrieve memberships

    Parameters
    ----------
    msf : :class:`numpy:numpy.ndarray`
        Array which is of size (hmc-class, obs.shape, 4), containing the
        trapezoidal membership function values for every `obs` point for
        every hydrometeor class.
    obs : :class:`numpy:numpy.ndarray`
        Array of arbitrary size and dimensions containing
        the data from which the memberships shall be calculated.

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        Array which is of (hmc-class, obs.shape) containing calculated
        membership probabilities.
    """
    out = np.zeros(msf.shape[0:-1])

    for i, m in enumerate(msf):
        out[i] = trapezoid(m, obs)

    return out


def probability(data, weights):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array which is of size (hmc-class, obs, data.shape), containing the
        membership probability values.
    weights : :class:`numpy:numpy.ndarray`
        Array of length (observables) containing the weights for
        each observable.

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        Array which is of (hmc-class, data.shape) containing weighted
        hmc-membership probabilities.
    """
    data = data.copy()
    weights = weights.copy()
    maxw = np.sum(weights)
    weights.shape = (1, len(weights)) + len(data.shape[2:]) * (1,)
    weights = np.broadcast_to(weights, data.shape)
    return np.sum(data * weights, axis=1) / maxw


def classify(data, threshold=0.0):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : :py:class:`numpy:numpy.ndarray`
        Array which is of size (hmc-class, data.shape), containing the
        weighted hmc-membership probability values.

    Keyword Arguments
    -----------------
    threshold : float
        Threshold value where probability is considered no precip,
        defaults to 0

    Returns
    -------
    idx : :py:class:`numpy:numpy.ndarray`
        Array which is of (data.shape) containing the (sorted) index into
        the hydrometeor-class.
        No precip is added on the top.
    vals : :py:class:`numpy:numpy.ndarray`
        Array which is of (data.shape) containing the (sorted) probability
        scores. No precip is added on the top.
    """
    data = data.copy()
    shape = data.shape[0]

    # handle no precipitation
    nop = np.sum(data, axis=0) / data.shape[0]
    mask = nop <= threshold
    # add no precip field (with zero probability)
    noprec = np.zeros_like(nop)
    data = np.vstack((data, noprec[np.newaxis, ...]))

    # sort idx and vals
    idx = np.argsort(data, axis=0)
    vals = np.sort(data, axis=0)
    # set no precip in every class
    idx[:, mask] = shape
    vals[:, mask] = 1.0

    return idx, vals
