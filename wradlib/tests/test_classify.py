#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import classify, georef, io, ipol, util

from . import requires_data, requires_gdal, requires_h5py, requires_netcdf


def test_filter_gabella_a():
    pass


def test_filter_window_distance():
    img = np.zeros((36, 10), dtype=np.float32)
    img[2, 2] = 10  # isolated pixel
    img[5, 6:8] = 10  # line
    img[20, :] = 5  # spike
    img[9:12, 4:7] = 11  # precip field
    img[15:17, 5:7] = np.nan  # nans
    cl = img.copy()
    cl[img > 0] = True
    cl[img == 11] = False
    cl[np.isnan(img)] = False
    np.set_printoptions(precision=2)
    rscale = 250
    similar = classify.filter_window_distance(img, rscale, fsize=300, tr1=4)
    result = similar < 0.3
    np.set_printoptions(precision=3)
    assert (result == cl).all()


@requires_data
def test_filter_gabella():
    filename = util.get_wradlib_data_file("misc/polar_dBZ_fbg.gz")
    data = np.loadtxt(filename)
    classify.filter_gabella(data, wsize=5, thrsnorain=0.0, tr1=6.0, n_p=8, tr2=1.3)


@requires_data
def test_histo_cut():
    filename = util.get_wradlib_data_file("misc/annual_rainfall_fbg.gz")
    yearsum = np.loadtxt(filename)
    classify.histo_cut(yearsum)


@pytest.fixture()
def fuzzy_data():
    rhofile = util.get_wradlib_data_file("netcdf/TAG-20120801" "-140046-02-R.nc")
    phifile = util.get_wradlib_data_file("netcdf/TAG-20120801" "-140046-02-P.nc")
    reffile = util.get_wradlib_data_file("netcdf/TAG-20120801" "-140046-02-Z.nc")
    dopfile = util.get_wradlib_data_file("netcdf/TAG-20120801" "-140046-02-V.nc")
    zdrfile = util.get_wradlib_data_file("netcdf/TAG-20120801" "-140046-02-D.nc")
    mapfile = util.get_wradlib_data_file("hdf5/TAG_cmap_sweeps" "_0204050607.hdf5")
    # We need to organize our data as a dictionary
    dat = {}
    dat["rho"], attrs_rho = io.read_edge_netcdf(rhofile)
    dat["phi"], attrs_phi = io.read_edge_netcdf(phifile)
    dat["ref"], attrs_ref = io.read_edge_netcdf(reffile)
    dat["dop"], attrs_dop = io.read_edge_netcdf(dopfile)
    dat["zdr"], attrs_zdr = io.read_edge_netcdf(zdrfile)
    dat["map"] = io.from_hdf5(mapfile)[0][0]
    yield dat


@requires_data
@requires_netcdf
@requires_h5py
def test_classify_echo_fuzzy(fuzzy_data):
    weights = {
        "zdr": 0.4,
        "rho": 0.4,
        "rho2": 0.4,
        "phi": 0.1,
        "dop": 0.1,
        "map": 0.5,
    }
    prob, mask = classify.classify_echo_fuzzy(fuzzy_data, weights=weights)
    np.testing.assert_array_equal(
        prob[0, :4],
        np.array(
            [
                0.052631578947368425,
                0.1803048097462205,
                0.052631578947368425,
                0.052631578947368425,
            ]
        ),
    )


@pytest.fixture()
def cloudtype_data():
    # read the radar volume scan
    filename = "hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf"
    filename = util.get_wradlib_data_file(filename)
    pvol = io.read_opera_hdf5(filename)
    nrays = int(pvol["dataset1/where"]["nrays"])
    nbins = int(pvol["dataset1/where"]["nbins"])
    val = pvol["dataset1/data1/data"]
    gain = float(pvol["dataset1/data1/what"]["gain"])
    offset = float(pvol["dataset1/data1/what"]["offset"])
    val = val * gain + offset
    rscale = int(pvol["dataset1/where"]["rscale"])
    elangle = pvol["dataset1/where"]["elangle"]
    coord = georef.sweep_centroids(nrays, rscale, nbins, elangle)
    site = (
        pvol["where"]["lon"],
        pvol["where"]["lat"],
        pvol["where"]["height"],
    )

    coord, proj_radar = georef.spherical_to_xyz(
        coord[..., 0],
        coord[..., 1],
        coord[..., 2],
        site,
        re=6370040.0,
        ke=4.0 / 3.0,
    )
    filename = "hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5"
    filename = util.get_wradlib_data_file(filename)
    sat_gdal = io.read_safnwc(filename)
    val_sat = georef.read_gdal_values(sat_gdal)
    coord_sat = georef.read_gdal_coordinates(sat_gdal)
    proj_sat = georef.read_gdal_projection(sat_gdal)
    coord_sat = georef.reproject(coord_sat, src_crs=proj_sat, trg_crs=proj_radar)
    coord_radar = coord
    interp = ipol.Nearest(
        coord_sat[..., 0:2].reshape(-1, 2), coord_radar[..., 0:2].reshape(-1, 2)
    )
    val_sat = interp(val_sat.ravel()).reshape(val.shape)
    timelag = 9 * 60
    wind = 10
    error = np.absolute(timelag) * wind
    dat = dict(val=val, val_sat=val_sat, rscale=rscale, error=error)
    yield dat


@requires_data
@requires_gdal
@requires_h5py
def test_filter_cloudtype(cloudtype_data):
    val = cloudtype_data["val"]
    val_sat = cloudtype_data["val_sat"]
    rscale = cloudtype_data["rscale"]
    error = cloudtype_data["error"]
    nonmet = classify.filter_cloudtype(val, val_sat, scale=rscale, smoothing=error)
    nclutter = np.sum(nonmet)
    assert nclutter == 8141
    nonmet = classify.filter_cloudtype(
        val, val_sat, scale=rscale, smoothing=error, low=True
    )
    nclutter = np.sum(nonmet)
    assert nclutter == 17856


@requires_data
@pytest.fixture
def class_data():
    @dataclass(init=False, repr=False, eq=False)
    class TestHydrometeorClassification:
        filename = util.get_wradlib_data_file("misc/msf_xband.gz")
        msf = io.get_membership_functions(filename)
        msf_idp = msf[0, 0, :, 0]
        msf_obs = msf[..., 1:]

        hmca = np.array(
            [
                [4.34960938, 15.68457031, 14.62988281],
                [7.78125, 5.49902344, 5.03808594],
                [0.49659729, 0.22286987, 0.86561584],
                [-9.11071777, -1.60217285, 11.15356445],
                [25.6, 25.6, 25.6],
            ]
        )

        msf_val = classify.msf_index_indep(msf_obs, msf_idp, hmca[0])
        fu = classify.fuzzyfi(msf_val, hmca)
        w = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
        prob = classify.probability(fu, w)

    yield TestHydrometeorClassification


def test_msf_index_indep(class_data):
    tst = np.array([-20, 10, 110])
    res = np.array(
        [[[[0.0, 0.0, 0.0, 0.0], [5.0, 10.0, 35.0, 40.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    msf_val = classify.msf_index_indep(
        class_data.msf_obs[0:1, 0:1], class_data.msf_idp, tst
    )
    np.testing.assert_array_equal(msf_val, res)


def test_fuzzify(class_data):
    res = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    fu = classify.fuzzyfi(class_data.msf_val, class_data.hmca)
    np.testing.assert_array_equal(fu[0], res)


def test_probability(class_data):
    res = np.array(
        [
            [0.16666667, 0.5, 0.66666667],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.21230469, 0.33333333],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.33333333, 0.33333333],
            [0.0, 0.33333333, 0.47532552],
            [0.28997396, 0.5, 0.5],
            [0.28997396, 0.5, 0.33333333],
        ]
    )
    prob = classify.probability(class_data.fu, class_data.w)
    np.testing.assert_array_almost_equal(prob, res, decimal=8)


@pytest.mark.xfail(strict=False)
def test_classify(class_data):
    res_idx = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 4, 4],
            [4, 5, 5],
            [5, 6, 6],
            [6, 11, 11],
            [7, 3, 3],
            [8, 7, 7],
            [11, 8, 10],
            [0, 0, 8],
            [9, 9, 9],
            [10, 10, 0],
        ]
    )
    res_vals = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.21230469, 0.33333333],
            [0.0, 0.33333333, 0.33333333],
            [0.0, 0.33333333, 0.33333333],
            [0.16666667, 0.5, 0.47532552],
            [0.28997396, 0.5, 0.5],
            [0.28997396, 0.5, 0.66666667],
        ]
    )

    hmc_idx, hmc_vals = classify.classify(class_data.prob, threshold=0.0)

    np.testing.assert_array_almost_equal(hmc_vals, res_vals)
    np.testing.assert_array_almost_equal(hmc_idx, res_idx)
