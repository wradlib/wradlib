#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Clutter Identification and Hydrometeor Classification (HMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}
"""
__all__ = [
    "filter_gabella",
    "filter_gabella_a",
    "filter_gabella_b",
    "filter_cloudtype",
    "filter_window_distance",
    "histo_cut",
    "classify_echo_fuzzy",
    "msf_index_indep",
    "trapezoid",
    "fuzzyfi",
    "probability",
    "classify",
    "ClassifyMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

from functools import singledispatch

import numpy as np
import xarray as xr
from scipy import ndimage

from wradlib import dp, util

#: Precipitation Types Mapping for HMC
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


@singledispatch
def filter_gabella_a(obj, wsize, tr1, *, cartesian=False, radial=False):
    """First part of the Gabella filter looking for large reflectivity \
    gradients.

    This function checks for each pixel in ``img`` how many pixels surrounding
    it in a window of ``wsize`` are by ``tr1`` smaller than the central pixel.

    Parameters
    ----------
    obj : :py:class:`numpy:numpy.ndarray`
        radar image to which the filter is to be applied
    wsize : int
        Size of the window surrounding the central pixel
    tr1 : float
        Threshold value
    cartesian : bool
        Specify if the input grid is Cartesian or polar
    radial : bool
        Specify if only radial information should be used

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        an array with the same shape as ``img``, containing the
        filter's results.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella` - the complete filter

    :func:`~wradlib.classify.filter_gabella_b` - the second part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    nn = wsize // 2
    count = -np.ones(obj.shape, dtype=int)
    range_shift = range(-nn, nn + 1)
    azimuth_shift = range(-nn, nn + 1)
    if radial:
        azimuth_shift = [0]
    for sa in azimuth_shift:
        refa = np.roll(obj, sa, axis=0)
        for sr in range_shift:
            refr = np.roll(refa, sr, axis=1)
            count += (obj - refr) < tr1
    count[:, 0:nn] = wsize**2
    count[:, -nn:] = wsize**2
    if cartesian:
        count[0:nn, :] = wsize**2
        count[-nn:, :] = wsize**2
    return count


@filter_gabella_a.register(xr.DataArray)
def _filter_gabella_a_xarray(obj, **kwargs):
    """First part of the Gabella filter looking for large reflectivity gradients.

    This function checks for each pixel in ``img`` how many pixels surrounding
    it in a window of ``wsize`` are by ``tr1`` smaller than the central pixel.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray`
        radar image to which the filter is to be applied

    Keyword Arguments
    -----------------
    wsize : int
        Size of the window surrounding the central pixel
    tr1 : float
        Threshold value
    cartesian : bool
        Specify if the input grid is Cartesian or polar
    radial : bool
        Specify if only radial information should be used

    Returns
    -------
    out : :py:class:`xarray:xarray.DataArray`
        an array with the same shape as ``img``, containing the
        filter's results.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella` - the complete filter

    :func:`~wradlib.classify.filter_gabella_b` - the second part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    dim0 = obj.wrl.util.dim0()
    wsize = kwargs.pop("wsize", 5)
    tr1 = kwargs.pop("tr1", 6.0)
    out = xr.apply_ufunc(
        filter_gabella_a,
        obj,
        wsize,
        tr1,
        input_core_dims=[[dim0, "range"], [], []],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "filter_gabella_a"
    return out


@singledispatch
def filter_gabella_b(obj, *, thrs=0.0):
    """Second part of the Gabella filter comparing area to circumference of \
    contiguous echo regions.

    Parameters
    ----------
    obj : :py:class:`numpy:numpy.ndarray`
    thrs : float
        Threshold below which the field values will be considered as no rain

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        contains in each pixel the ratio between area and circumference of the
        meteorological echo it is assigned to or 0 for non precipitation
        pixels.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella` - the complete filter

    :func:`~wradlib.classify.filter_gabella_a` - the first part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    conn = np.ones((3, 3))
    # create binary image of the rainfall field
    binimg = obj > thrs
    # label objects (individual rain cells, so to say)
    labelimg, nlabels = ndimage.label(binimg, conn)
    # erode the image, thus removing the 'boundary pixels'
    binimg_erode = ndimage.binary_erosion(binimg, structure=conn)
    # determine the size of each object
    labelhist, edges = np.histogram(
        labelimg, bins=nlabels + 1, range=(-0.5, labelimg.max() + 0.5)
    )
    # determine the size of the eroded objects
    erodelabelhist, edges = np.histogram(
        np.where(binimg_erode, labelimg, 0),
        bins=nlabels + 1,
        range=(-0.5, labelimg.max() + 0.5),
    )
    # the boundary is the difference between these two
    boundarypixels = labelhist - erodelabelhist
    # now get the ratio between object size and boundary
    ratio = labelhist.astype(np.float32) / boundarypixels
    # assign it back to the objects
    # first get the indices
    indices = np.digitize(labelimg.ravel(), edges) - 1
    # then produce a new field with the ratios in the right place
    result = ratio[indices.ravel()].reshape(obj.shape)

    return result


@filter_gabella_b.register(xr.DataArray)
def _filter_gabella_b_xarray(obj, **kwargs):
    """Second part of the Gabella filter comparing area to circumference of \
    contiguous echo regions.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray`

    Keyword Arguments
    -----------------
    thrs : float
        Threshold below which the field values will be considered as no rain

    Returns
    -------
    out: :py:class:`xarray:xarray.DataArray`
        contains in each pixel the ratio between area and circumference of the
        meteorological echo it is assigned to or 0 for non precipitation
        pixels.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella` - the complete filter

    :func:`~wradlib.classify.filter_gabella_a` - the first part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    dim0 = obj.wrl.util.dim0()
    out = xr.apply_ufunc(
        filter_gabella_b,
        obj,
        input_core_dims=[[dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "filter_gabella_b"
    return out


@singledispatch
def filter_gabella(
    obj,
    *,
    wsize=5,
    **kwargs,
):
    """Clutter identification filter developed by :cite:`Gabella2002`.

    This is a two-part identification algorithm using echo continuity and
    minimum echo area to distinguish between meteorological (rain) and non-
    meteorological echos (ground clutter etc.)

    Parameters
    ----------
    obj : :py:class:`numpy:numpy.ndarray`
    wsize : int, optional
        Size of the window surrounding the central pixel, defaults to 5.

    Keyword Arguments
    -----------------
    thrsnorain : float
    tr1 : float
    n_p : int
    tr2 : float
    rm_nans : bool
        True replaces nans with Inf
        False takes nans into acount
    radial : bool
        True to use radial information only in
        :func:`~wradlib.classify.filter_gabella_a`.
    cartesian : bool
        True if cartesian data are used, polar assumed if False.

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        boolean array with pixels identified as clutter set to True.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella_a` - the first part of the filter

    :func:`~wradlib.classify.filter_gabella_b` - the second part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    thrsnorain = kwargs.get("thrsnorain", 0.0)
    tr1 = kwargs.get("tr1", 6.0)
    n_p = kwargs.get("n_p", 6)
    tr2 = kwargs.get("tr2", 1.3)
    rm_nans = kwargs.get("rm_nans", True)
    radial = kwargs.get("radial", False)
    cartesian = kwargs.get("cartesian", False)

    bad = np.isnan(obj)
    if rm_nans:
        obj = obj.copy()
        obj[bad] = np.Inf
    ntr1 = filter_gabella_a(
        obj, wsize=wsize, tr1=tr1, cartesian=cartesian, radial=radial
    )
    if not rm_nans:
        f_good = ndimage.uniform_filter((~bad).astype(float), size=wsize)
        f_good[f_good == 0] = 1e-10
        ntr1 = ntr1 / f_good
        ntr1[bad] = n_p
    clutter1 = ntr1 < n_p
    ratio = filter_gabella_b(obj, thrs=thrsnorain)
    clutter2 = np.abs(ratio) < tr2
    return clutter1 | clutter2


@filter_gabella.register(xr.DataArray)
def _filter_gabella_xarray(obj, **kwargs):
    """Clutter identification filter developed by :cite:`Gabella2002`.

    This is a two-part identification algorithm using echo continuity and
    minimum echo area to distinguish between meteorological (rain) and non-
    meteorological echos (ground clutter etc.)

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray`

    Keyword Arguments
    -----------------
    wsize : int
        Size of the window surrounding the central pixel
    thrsnorain : float
    tr1 : float
    n_p : int
    tr2 : float
    rm_nans : bool
        True replaces nans with Inf
        False takes nans into acount
    radial : bool
        True to use radial information only in
        :func:`~wradlib.classify.filter_gabella_a`.
    cartesian : bool
        True if cartesian data are used, polar assumed if False.

    Returns
    -------
    output : :py:class:`xarray:xarray.DataArray`
        boolean array with pixels identified as clutter set to True.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella_a` - the first part of the filter

    :func:`~wradlib.classify.filter_gabella_b` - the second part of the filter

    Examples
    --------

    See :ref:`/notebooks/classify/clutter_gabella.ipynb`.

    """
    dim0 = obj.wrl.util.dim0()
    out = xr.apply_ufunc(
        filter_gabella,
        obj,
        input_core_dims=[[dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "filter_gabella"
    return out


@singledispatch
def histo_cut(obj, *, upper_frequency=0.01, lower_frequency=0.01):
    """Histogram based clutter identification.

    This identification algorithm uses the histogram of temporal accumulated
    rainfall. It iteratively detects classes whose frequency falls below a
    specified percentage (1% by default) of the frequency of the class with the
    biggest frequency and remove the values from the dataset until the changes
    from iteration to iteration falls below a threshold. This algorithm is able
    to detect static clutter as well as shadings.

    The tresholds for the upper frequency (clutter) and the lower frequency (shading)
    can be parameterized by the respective kwargs, `upper_frequency`/`lower_frequency`.

    It is suggested to choose a representative time periode for the input precipitation
    accumulation. The recommended time period should cover one year.

    Parameters
    ----------
    obj : :py:class:`numpy:numpy.ndarray`
        spatial array containing rain accumulation
    upper_frequency : float, optional
        Upper frequency percentage for clutter detection, defaults to 0.01.
    lower_frequency : float, optional
        Lower frequency percentage for shading detection, defaults to 0.01.

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        uint8 array with pixels identified as clutter set to 1 and shadings set to 2.
        Remaining pixels set to 0. Users strictly relying on a boolean mask might have
        to explicitely cast to boolean (adding `.astype(np.bool)` on the return).

    Examples
    --------

    See :ref:`/notebooks/classify/histo_cut.ipynb`.
    """

    prec_accum = np.array(obj)

    # initialization of data bounds for clutter and shade definition
    lower_bound = 0
    upper_bound = prec_accum.max()

    # predefinitions for the first iteration
    lower_bound_before = -51
    upper_bound_before = -51

    # iterate as long as the difference between current and
    # last iteration doesn't fall below the stop criterion
    while (abs(lower_bound - lower_bound_before) > 1) or (
        abs(upper_bound - upper_bound_before) > 1
    ):
        # masks for bins with sums over/under the data bounds
        upper_mask = (prec_accum <= upper_bound).astype(int)
        lower_mask = (prec_accum >= lower_bound).astype(int)
        # NaNs in place of masked bins
        # Kopie der Datenmatrix mit Nans an Stellen,
        # wo der Threshold erreicht wird
        prec_accum_masked = np.where((upper_mask * lower_mask) == 0, np.nan, prec_accum)

        # generate a histogram of the valid bins with 50 classes
        (n, bins) = np.histogram(
            prec_accum_masked[np.isfinite(prec_accum_masked)].ravel(), bins=50
        )
        # get the class with biggest occurence
        index = np.where(n == n.max())
        index = index[0][0]
        # separated stop criterion check in case one of the bounds
        # is already robust
        if abs(lower_bound - lower_bound_before) > 1:
            # get the index of the class which underscores the occurence of
            # the biggest class by lower_frequency (1%, default), beginning from
            # the class with the biggest occurence to the first class
            for i in range(index, -1, -1):
                if n[i] < (n[index] * lower_frequency):
                    break
        if abs(upper_bound - upper_bound_before) > 1:
            # get the index of the class which underscores the occurence of
            # the biggest class by upper_frequency (1%, default), beginning from
            # the class with the biggest occurence to the last class
            for j in range(index, len(n)):
                if n[j] < (n[index] * upper_frequency):
                    break

        lower_bound_before = lower_bound
        upper_bound_before = upper_bound
        # update the new boundaries
        lower_bound = bins[i]
        upper_bound = bins[j + 1]

    # create zero array and set clutter as 1 and shading as 2
    mask = np.zeros(prec_accum.shape, dtype=np.uint8)
    mask[prec_accum > upper_bound] = 1
    mask[prec_accum < lower_bound] = 2

    return mask


@histo_cut.register(xr.DataArray)
def _histo_cut_xarray(obj, **kwargs):
    """Histogram based clutter identification.

    This identification algorithm uses the histogram of temporal accumulated
    rainfall. It iteratively detects classes whose frequency falls below a
    specified percentage (1% by default) of the frequency of the class with the
    biggest frequency and remove the values from the dataset until the changes
    from iteration to iteration falls below a threshold. This algorithm is able
    to detect static clutter as well as shadings.

    The tresholds for the upper frequency (clutter) and the lower frequency (shading)
    can be parameterized by the respective kwargs, `upper_frequency`/`lower_frequency`.

    It is suggested to choose a representative time periode for the input precipitation
    accumulation. The recommended time period should cover one year.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray`
        spatial array containing rain accumulation

    Keyword Arguments
    -----------------
    upper_frequency : float
        Upper frequency percentage for clutter detection, defaults to 0.01.
    lower_frequency : float
        Lower frequency percentage for shading detection, defaults to 0.01.

    Returns
    -------
    output : :py:class:`xarray:xarray.DataArray`
        uint8 array with pixels identified as clutter set to 1 and shadings set to 2.
        Remaining pixels set to 0. Users strictly relying on a boolean mask might have
        to explicitely cast to boolean (adding `.astype(np.bool)` on the return).

    Examples
    --------

    See :ref:`/notebooks/classify/histo_cut.ipynb`.
    """
    dim0 = obj.wrl.util.dim0()
    out = xr.apply_ufunc(
        histo_cut,
        obj,
        input_core_dims=[[dim0, "range"]],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "histo_cut"
    return out


@singledispatch
def classify_echo_fuzzy(dat, *, weights=None, trpz=None):
    """Fuzzy echo classification and clutter identification based on \
    polarimetric moments.

    The implementation is based on :cite:`Vulpiani2012`. At the
    moment, it only distinguishes between meteorological and non-meteorological
    echos.

    .. versionchanged:: 1.4.0
       The implementation was extended using depolarization ratio (dr)
       and clutter phase alignment (cpa).

    .. versionchanged:: 2.0.0
       Returns probability of meteorological echos instead of clutter mask

    For Clutter Phase Alignment (CPA) see :cite:`Hubbert2009a` and
    :cite:`Hubbert2009b`

    For each decision variable and radar bin, the algorithm uses trapezoidal
    functions in order to define the membership to the non-meteorological
    echo class.
    Based on pre-defined weights, a linear combination of the different degrees
    of membership is computed. The echo is assumed to be non-meteorological
    in case the linear combination exceeds a threshold.

    At the moment, the following decision variables are considered:

        - Texture of differential reflectivity (zdr) (mandatory)

        - Texture of correlation coefficient (rho) (mandatory)

        - Texture of differential propagation phase (phidp) (mandatory)

        - Doppler velocity (dop) (mandatory)

        - Static clutter map (map) (mandatory)

        - Correlation coefficient (rho2) (additional)

        - Depolarization Ratio (dr), computed from
          correlation coefficient & differential reflectivity (additional)

        - clutter phase alignment (cpa) (additional)

    Parameters
    ----------
    dat : dict
        dictionary of arrays.
        Contains the data of the decision variables. The shapes of the arrays
        should be (..., number of beams, number of gates) and the shapes need
        to be identical or be broadcastable.
    weights : dict
        dictionary of floats.
        Defines the weights of the decision variables. Default is:
        zdr: 0.4,
        rho: 0.4,
        phi: 0.1,
        dop: 0.1,
        map: 0.5,
        rho2: 0.4,
        dr: 0.4,
        cpa: 0.4.
    trpz : dict
        dictionary of lists of floats.
        Contains the arguments of the trapezoidal membership functions for each
        decision variable. Default is:
        zdr: [0.7, 1.0, 9999, 9999],
        rho: [0.1, 0.15, 9999, 9999],
        phi: [15, 20, 10000, 10000],
        dop: [-0.2, -0.1, 0.1, 0.2],
        map: [1, 1, 9999, 9999],
        rho2: [-9999, -9999, 0.95, 0.98],
        dr: [-20, -12, 9999, 9999],
        cpa: [0.6, 0.9, 9999, 9999].

    Returns
    -------
    prob : :py:class:`numpy:numpy.ndarray`
        Array indicates probability of meteorological echos based on the fuzzy
        classification.
    mask : :py:class:`numpy:numpy.ndarray`
        Boolean array indicating where all the polarimetric moments
        had missing values which could be used as an additional information
        criterion.

    Note
    ----
    The boolean clutter mask (versions prior 2.0) can be calculated with the following
    code: `np.where(prob < thresh, True, False)`.

    See Also
    --------
    :func:`~wradlib.dp.texture` - texture

    :func:`~wradlib.dp.depolarization` - depolarization ratio

    """
    # Check the inputs
    # mandatory data keys
    dkeys = ["zdr", "rho", "phi", "dop", "map"]
    # usable wkeys
    wkeys = ["zdr", "rho", "phi", "dop", "map", "rho2", "dr", "cpa"]
    # usable tkeys
    tkeys = ["zdr", "rho", "phi", "dop", "map", "rho2", "dr", "cpa"]

    # default weights
    weights_default = {
        "zdr": 0.4,
        "rho": 0.4,
        "phi": 0.1,
        "dop": 0.1,
        "map": 0.5,
        "rho2": 0.4,
        "dr": 0.4,
        "cpa": 0.4,
    }
    if weights is None:
        weights = weights_default
    else:
        weights = dict(list(weights_default.items()) + list(weights.items()))

    # default trapezoidal membership functions
    trpz_default = {
        "zdr": [0.7, 1.0, 9999, 9999],
        "rho": [0.1, 0.15, 9999, 9999],
        "phi": [15, 20, 10000, 10000],
        "dop": [-0.2, -0.1, 0.1, 0.2],
        "map": [1, 1, 9999, 9999],
        "rho2": [-9999, -9999, 0.95, 0.98],
        "dr": [-20, -12, 9999, 9999],
        "cpa": [0.6, 0.9, 9999, 9999],
    }
    if trpz is None:
        trpz = trpz_default
    else:
        trpz = dict(list(trpz_default.items()) + list(trpz.items()))

    # check data conformity
    if not np.all(np.in1d(dkeys, list(dat.keys()))):
        raise ValueError(
            "Argument `dat` must be a dictionary " f"with mandatory keywords {*dkeys,}."
        )
    if not np.all(np.in1d(wkeys, list(weights.keys()))):
        raise ValueError(
            "Argument `weights` must be a dictionary " f"with keywords {*wkeys,}."
        )
    if not np.all(np.in1d(tkeys, list(trpz.keys()))):
        raise ValueError(
            "Argument `trpz` must be a dictionary " f"with keywords {*tkeys,}."
        )

    # copy rho to rho2
    dat["rho2"] = dat["rho"].copy()

    shape = None
    for key in dkeys:
        if dat[key] is not None:
            if shape is None:
                shape = dat[key].shape
            else:
                if dat[key].shape[-2:] != shape[-2:]:
                    raise ValueError(
                        "Arrays of the decision variables have inconsistent "
                        f"shapes: {dat[key].shape} vs. {shape}"
                    )
        else:
            util.warn(f"Missing decision variable: {key}", UserWarning)

    # If all dual-pol moments are NaN, can we assume that and echo is
    # non-meteorological?
    # Successively identify those bins where all moments are NaN
    nmom = ["rho", "zdr", "phi", "dr", "cpa"]  # 'dop'
    nan_mask = np.isnan(dat["rho"])
    for mom in nmom[1:]:
        try:
            nan_mask &= np.isnan(dat[mom])
        except KeyError:
            pass

    # Replace missing data by NaN
    dummy = np.zeros(shape) * np.nan
    for key in dat.keys():
        if dat[key] is None:
            dat[key] = dummy

    # membership in meteorological class for each variable
    qres = {}
    for key in dat.keys():
        if key not in tkeys:
            continue
        if key in ["zdr", "rho", "phi"]:
            d = dp.texture(dat[key])
        else:
            d = dat[key]
        qres[key] = 1.0 - util.trapezoid(
            d, trpz[key][0], trpz[key][1], trpz[key][2], trpz[key][3]
        )

    # create weight arrays which are zero where the data is NaN
    # This way, each pixel "adapts" to the local data availability
    wres = {}
    for key in dat.keys():
        if key not in wkeys:
            continue
        wres[key] = _weight_array(qres[key], weights[key])

    # Membership in meteorological class after combining all variables
    qsum = []
    wsum = []
    for key in dat.keys():
        if key not in wkeys:
            continue
        # weighted sum, also removing NaN from data
        qsum.append(np.nan_to_num(qres[key]) * wres[key])
        wsum.append(wres[key])

    q = np.array(qsum).sum(axis=0) / np.array(wsum).sum(axis=0)

    return q, nan_mask


@classify_echo_fuzzy.register(xr.Dataset)
def _classify_echo_fuzzy_xarray(obj, dat, **kwargs):
    """Fuzzy echo classification and clutter identification based on polarimetric moments.

    The implementation is based on :cite:`Vulpiani2012`. At the
    moment, it only distinguishes between meteorological and non-meteorological
    echos.

    For Clutter Phase Alignment (CPA) see :cite:`Hubbert2009a` and
    :cite:`Hubbert2009b`

    For each decision variable and radar bin, the algorithm uses trapezoidal
    functions in order to define the membership to the non-meteorological
    echo class.
    Based on pre-defined weights, a linear combination of the different degrees
    of membership is computed. The echo is assumed to be non-meteorological
    in case the linear combination exceeds a threshold.

    At the moment, the following decision variables are considered:
        - Texture of differential reflectivity (zdr) (mandatory)
        - Texture of correlation coefficient (rho) (mandatory)
        - Texture of differential propagation phase (phidp) (mandatory)
        - Doppler velocity (dop) (mandatory)
        - Static clutter map (map) (mandatory)
        - Correlation coefficient (rho2) (additional)
        - Depolarization Ratio (dr), computed from
          correlation coefficient & differential reflectivity (additional)
        - clutter phase alignment (cpa) (additional)

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.Dataset`
    dat : dict
        Mapping of moment names.

    Keyword Arguments
    -----------------
    weights : dict
        dictionary of floats.
        Defines the weights of the decision variables. Default is:
        zdr: 0.4,
        rho: 0.4,
        phi: 0.1,
        dop: 0.1,
        map: 0.5,
        rho2: 0.4,
        dr: 0.4,
        cpa: 0.4.
    trpz : dict
        dictionary of lists of floats.
        Contains the arguments of the trapezoidal membership functions for each
        decision variable. Default is:
        zdr: [0.7, 1.0, 9999, 9999],
        rho: [0.1, 0.15, 9999, 9999],
        phi: [15, 20, 10000, 10000],
        dop: [-0.2, -0.1, 0.1, 0.2],
        map: [1, 1, 9999, 9999],
        rho2: [-9999, -9999, 0.95, 0.98],
        dr: [-20, -12, 9999, 9999],
        cpa: [0.6, 0.9, 9999, 9999].

    Returns
    -------
    prob : :py:class:`xarray:xarray.DataArray`
        DataArray indicating probability of meteorological echos based on the fuzzy classification.
    mask : :py:class:`xarray:xarray.DataArray`
        DataArray indicating where all the polarimetric moments had missing values which
        could be used as an additional information criterion.

    See Also
    --------
    :func:`~wradlib.dp.texture` - texture

    :func:`~wradlib.dp.depolarization` - depolarization ratio
    """

    def _classify_echo_fuzzy_wrapper(*args, **kwargs):
        mom = ["rho", "phi", "ref", "dop", "zdr", "map"][: len(args)]
        dat = {name: value for name, value in zip(mom, args)}
        prob, mask = classify_echo_fuzzy(dat, **kwargs)
        return prob, mask

    mom = ["rho", "phi", "ref", "dop", "zdr", "map", "rho2", "dpr", "cpa"]
    args = [obj[dat[m]] for m in mom if m in dat]
    dim0 = args[0].wrl.util.dim0()
    input_core_dims = [[dim0, "range"]] * len(dat)
    prob, mask = xr.apply_ufunc(
        _classify_echo_fuzzy_wrapper,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=[[dim0, "range"], [dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    prob.name = "probability_classify_echo_fuzzy"
    mask.name = "mask_classify_echo_fuzzy"
    return prob, mask


def _weight_array(data, weight):
    """
    Generates weight array where valid values have the weight value
    and NaNs have 0 weight value.
    """
    w_array = weight * np.ones(np.shape(data))
    w_array[np.isnan(data)] = 0.0
    return w_array


def filter_cloudtype(
    img,
    cloud,
    *,
    thrs=0,
    snow=False,
    low=False,
    cirrus=False,
    smoothing=None,
    grid="polar",
    scale=None,
):
    """Identification of non-meteorological echoes based on cloud type.

    Parameters
    ----------
    img : :py:class:`numpy:numpy.ndarray`
        radar image to which the filter is to be applied
    cloud : :py:class:`numpy:numpy.ndarray`
        image with collocated cloud value from MSG SAFNWC PGE02 product
    thrs : float
        Threshold above which to identify clutter
    snow : bool
        Switch to use PGE02 class "land/sea snow" for clutter identification
    low : bool
        Switch to use PGE02 class very low stratus, very low cumulus and
        low cumulus for clutter identification
    cirrus : bool
        Switch to use PGE02 class "very thin cirrus" and "fractional clouds"
        for clutter identification
    smoothing : float
        Size [m] of the smoothing window used to take into account various
        localisation errors (e.g. advection, parallax)
    grid : str
        "polar" or "cartesian"
    scale : float or tuple
        float or tuple of 2 floats
        range [m] scale for polar grid
        x[m] and y[m] scale for cartesian grid

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        a boolean array containing TRUE where clutter has been identified.

    """
    noprecip = (cloud == 1) | (cloud == 2)
    if snow:
        noprecip = noprecip | (cloud == 3) | (cloud == 4)
    if low:
        noprecip = noprecip | (cloud == 5) | (cloud == 6) | (cloud == 7)
    if cirrus:
        noprecip = noprecip | (cloud == 14) | (cloud == 18)
    if smoothing is not None:
        myfilter = getattr(util, f"filter_window_{grid}")
        noprecip = myfilter(noprecip, smoothing, "minimum", scale)
    clutter = noprecip & (img > thrs)
    return clutter


@singledispatch
def filter_window_distance(img, rscale, *, fsize=1500, tr1=7):
    """2d filter looking for large reflectivity gradients.

    This function counts for each bin in ``img`` the percentage of surrounding
    bins in a window of half size ``fsize`` which are not ``tr1`` smaller than
    the central bin. The window is defined using geometrical distance.

    Parameters
    ----------
    img : :py:class:`numpy:numpy.ndarray`
        2d polar data to which the filter is to be applied
    rscale : float
        range [m] scale of the polar grid
    fsize : int
        Half-size [m] of the square window surrounding the central pixel
    tr1 : float
        Threshold value

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        an array with the same shape as ``img``, containing the
        filter's results.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella_a` - Original version of the filter

    :func:`~wradlib.classify.filter_gabella_b` - filter using an echo area
    """
    ascale = 2 * np.pi / img.shape[0]
    count = np.ones(img.shape, dtype=int)
    similar = np.zeros(img.shape, dtype=float)
    good = np.ones(img.shape, dtype=float)
    valid = ~np.isnan(img)
    hole = np.sum(~valid) > 0
    nr = int(round(fsize / rscale))
    range_shift = range(-nr, nr + 1)
    r = np.arange(img.shape[1]) * rscale + rscale / 2
    adist = r * ascale
    na = np.around(fsize / adist).astype(int)
    max_na = img.shape[0] / 10
    sa = 0
    while sa < max_na:
        imax = np.where(na >= sa)[0][-1] + 1
        refa1 = util.roll2d_polar(img, sa, axis=0)
        refa2 = util.roll2d_polar(img, -sa, axis=0)
        for sr in range_shift:
            refr1 = util.roll2d_polar(refa1, sr, axis=1)
            similar[:, 0:imax] += img[:, 0:imax] - refr1[:, 0:imax] < tr1
            if sa > 0:
                refr2 = util.roll2d_polar(refa2, sr, axis=1)
                similar[:, 0:imax] += img[:, 0:imax] - refr2[:, 0:imax] < tr1
        count[:, 0:imax] = 2 * sa + 1
        sa += 1
    similar[~valid] = np.nan
    count[~valid] = -1
    count[:, nr:-nr] = count[:, nr:-nr] * (2 * nr + 1)
    for i in range(0, nr):
        count[:, i] = count[:, i] * (nr + 1 + i)
        count[:, -i - 1] = count[:, -i - 1] * (nr + 1 + i)
    if hole:
        good = util.filter_window_polar(valid.astype(float), fsize, "uniform", rscale)
        count = count * good
        count[count == 0] = 1
    similar -= 1
    count -= 1
    similar = similar / count
    return similar


@filter_window_distance.register(xr.Dataset)
@filter_window_distance.register(xr.DataArray)
def _filter_window_distance_xarray(obj, **kwargs):
    """2d filter looking for large reflectivity gradients.

    This function counts for each bin in ``img`` the percentage of surrounding
    bins in a window of half size ``fsize`` which are not ``tr1`` smaller than
    the central bin. The window is defined using geometrical distance.

    Parameters
    ----------
    obj : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        2d polar data to which the filter is to be applied

    Keyword Arguments
    -----------------
    fsize : int
        Half-size [m] of the square window surrounding the central pixel
    tr1 : float
        Threshold value

    Returns
    -------
    output : :py:class:`xarray:xarray.DataArray` | :py:class:`xarray:xarray.Dataset`
        an array with the same shape as ``img``, containing the
        filter's results.

    See Also
    --------
    :func:`~wradlib.classify.filter_gabella_a` - Original version of the filter

    :func:`~wradlib.classify.filter_gabella_b` - filter using an echo area
    """
    dim0 = obj.wrl.util.dim0()
    rscale = obj.range.diff("range").median()
    if isinstance(obj, xr.Dataset):
        obj, keep = util.get_apply_ufunc_variables(obj, dim0)
    out = xr.apply_ufunc(
        filter_window_distance,
        obj,
        rscale.values,
        input_core_dims=[[dim0, "range"], []],
        output_core_dims=[[dim0, "range"]],
        dask="parallelized",
        kwargs=kwargs,
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    if isinstance(obj, xr.Dataset):
        out = xr.merge([out, keep])
    else:
        out.name = "filter_window_distance"

    return out


@singledispatch
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
        observable (e.g. (rays, bins) or (scan, rays, bins)).

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


@msf_index_indep.register(xr.Dataset)
def _msf_index_indep_xarray(msf, obs):
    msf = msf.to_array(dim="obs").transpose("hmc", ...)
    dim0 = obs.wrl.util.dim0()
    out = xr.apply_ufunc(
        msf_index_indep,
        msf,
        msf.idp,
        obs,
        input_core_dims=[["hmc", "obs", "idp", "trapezoid"], ["idp"], [dim0, "range"]],
        output_core_dims=[["hmc", "obs", dim0, "range", "trapezoid"]],
        dask="parallelized",
        output_dtypes=["i4"],
    )
    out.name = "msf_index_indep"
    return out


@singledispatch
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
    shape = msf.shape[:-1]
    obs = np.broadcast_to(obs, shape)
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


@singledispatch
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


@fuzzyfi.register(xr.DataArray)
def _fuzzyfi_xarray(msf, hmc_ds, msf_obs_mapping):
    dim0 = hmc_ds.wrl.util.dim0()
    rev = {v: k for k, v in msf_obs_mapping.items()}
    obs = hmc_ds[list(msf_obs_mapping.values())].rename(rev)
    obs = obs.to_array("obs")
    out = xr.apply_ufunc(
        trapezoid,
        msf,
        obs,
        input_core_dims=[
            ["hmc", "obs", dim0, "range", "trapezoid"],
            ["obs", dim0, "range"],
        ],
        output_core_dims=[["hmc", "obs", dim0, "range"]],
        output_dtypes=float,
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    out.name = "fuzzyfi"
    return out


@singledispatch
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


@probability.register(xr.DataArray)
def _probability_xarray(data, weights):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the membership probability values.
    weights : :class:`numpy:numpy.ndarray`
        Array of length (observables) containing the weights for
        each observable.

    Returns
    -------
    out : xarray.DataArray
        Array containing weighted hmc-membership probabilities.
    """
    w = weights.to_array(dim="obs")
    out = (data * w).sum("obs") / w.sum("obs")
    return out  # .transpose("hmc", ...)


@singledispatch
def classify(data, *, threshold=0.0):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : :py:class:`numpy:numpy.ndarray`
        Array which is of size (hmc-class, data.shape), containing the
        weighted hmc-membership probability values.
    threshold : float, optional
        Threshold value where probability is considered no precip,
        defaults to 0.

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


@classify.register(xr.DataArray)
def _classify_xarray(data, threshold=0.0):
    """Calculate probability of hmc-class for every data bin.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        Array which is of size (hmc-class, data.shape), containing the
        weighted hmc-membership probability values.

    Keyword Arguments
    -----------------
    threshold : float
        Threshold value where probability is considered no precip,
        defaults to 0

    Returns
    -------
    out : xarray.DataArray
        DataArray containing probability scores.
        No precip is added on the top.
    """
    # handle no precipitation
    nop = xr.where(data.sum("hmc") / len(data.hmc) <= threshold, 1, 0)
    nop = nop.assign_coords({"hmc": "NP"}).expand_dims(dim="hmc", axis=-1)
    return xr.concat([data, nop], dim="hmc")


class ClassifyMethods(util.XarrayMethods):
    """wradlib xarray SubAccessor methods for Classify."""

    @util.docstring(_filter_gabella_xarray)
    def filter_gabella(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return filter_gabella(self, *args, **kwargs)
        else:
            return filter_gabella(self._obj, *args, **kwargs)

    @util.docstring(_filter_gabella_a_xarray)
    def filter_gabella_a(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return filter_gabella_a(self, *args, **kwargs)
        else:
            return filter_gabella_a(self._obj, *args, **kwargs)

    @util.docstring(_filter_gabella_b_xarray)
    def filter_gabella_b(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return filter_gabella_b(self, *args, **kwargs)
        else:
            return filter_gabella_b(self._obj, *args, **kwargs)

    @util.docstring(_histo_cut_xarray)
    def histo_cut(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return histo_cut(self, *args, **kwargs)
        else:
            return histo_cut(self._obj, *args, **kwargs)

    @util.docstring(_classify_echo_fuzzy_xarray)
    def classify_echo_fuzzy(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return classify_echo_fuzzy(self, *args, **kwargs)
        else:
            return classify_echo_fuzzy(self._obj, *args, **kwargs)

    @util.docstring(_filter_window_distance_xarray)
    def filter_window_distance(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return filter_window_distance(self, *args, **kwargs)
        else:
            return filter_window_distance(self._obj, *args, **kwargs)

    @util.docstring(_msf_index_indep_xarray)
    def msf_index_indep(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return msf_index_indep(self, *args, **kwargs)
        else:
            return msf_index_indep(self._obj, *args, **kwargs)

    @util.docstring(_fuzzyfi_xarray)
    def fuzzyfi(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return fuzzyfi(self, *args, **kwargs)
        else:
            return fuzzyfi(self._obj, *args, **kwargs)

    @util.docstring(_probability_xarray)
    def probability(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return probability(self, *args, **kwargs)
        else:
            return probability(self._obj, *args, **kwargs)

    @util.docstring(_classify_xarray)
    def classify(self, *args, **kwargs):
        if not isinstance(self, ClassifyMethods):
            return classify(self, *args, **kwargs)
        else:
            return classify(self._obj, *args, **kwargs)

    # @util.docstring(_trapezoid_xarray)
    # def trapezoid(self, *args, **kwargs):
    #     if not isinstance(self, ClassifyMethods):
    #         return trapezoid(self, *args, **kwargs)
    #     else:
    #         return trapezoid(self._obj, *args, **kwargs)
