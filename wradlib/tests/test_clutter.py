#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import pytest

from wradlib import clutter, georef, io, ipol, util

from . import requires_data, requires_gdal, requires_h5py, requires_netcdf


# -------------------------------------------------------------------------------
# testing the filter helper function
# -------------------------------------------------------------------------------
class TestClutter:
    img = np.zeros((36, 10), dtype=np.float32)
    img[2, 2] = 10  # isolated pixel
    img[5, 6:8] = 10  # line
    img[20, :] = 5  # spike
    img[9:12, 4:7] = 11  # precip field

    def test_filter_gabella_a(self):
        pass

    def test_filter_window_distance(self):
        self.img[15:17, 5:7] = np.nan  # nans
        cl = self.img.copy()
        cl[self.img > 0] = True
        cl[self.img == 11] = False
        cl[np.isnan(self.img)] = False
        np.set_printoptions(precision=2)
        rscale = 250
        similar = clutter.filter_window_distance(self.img, rscale, fsize=300, tr1=4)
        result = similar < 0.3
        np.set_printoptions(precision=3)
        assert (result == cl).all()


class TestFilterGabella:
    @requires_data
    def test_filter_gabella(self):
        filename = util.get_wradlib_data_file("misc/polar_dBZ_fbg.gz")
        data = np.loadtxt(filename)
        clutter.filter_gabella(data, wsize=5, thrsnorain=0.0, tr1=6.0, n_p=8, tr2=1.3)


class TestHistoCut:
    @requires_data
    def test_histo_cut_test(self):
        filename = util.get_wradlib_data_file("misc/annual_rainfall_fbg.gz")
        yearsum = np.loadtxt(filename)
        clutter.histo_cut(yearsum)


@pytest.fixture(scope="class")
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


class TestClassifyEchoFuzzyTest:
    @requires_data
    @requires_netcdf
    @requires_h5py
    def test_classify_echo_fuzzy(self, fuzzy_data):
        weights = {
            "zdr": 0.4,
            "rho": 0.4,
            "rho2": 0.4,
            "phi": 0.1,
            "dop": 0.1,
            "map": 0.5,
        }
        clutter.classify_echo_fuzzy(fuzzy_data, weights=weights, thresh=0.5)


@pytest.fixture(scope="class")
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
    sitecoords = (
        pvol["where"]["lon"],
        pvol["where"]["lat"],
        pvol["where"]["height"],
    )

    coord, proj_radar = georef.spherical_to_xyz(
        coord[..., 0],
        coord[..., 1],
        coord[..., 2],
        sitecoords,
        re=6370040.0,
        ke=4.0 / 3.0,
    )
    filename = "hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5"
    filename = util.get_wradlib_data_file(filename)
    sat_gdal = io.read_safnwc(filename)
    val_sat = georef.read_gdal_values(sat_gdal)
    coord_sat = georef.read_gdal_coordinates(sat_gdal)
    proj_sat = georef.read_gdal_projection(sat_gdal)
    coord_sat = georef.reproject(
        coord_sat, projection_source=proj_sat, projection_target=proj_radar
    )
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


class TestFilterCloudtype:
    @requires_data
    @requires_gdal
    @requires_h5py
    def test_filter_cloudtype(self, cloudtype_data):
        val = cloudtype_data["val"]
        val_sat = cloudtype_data["val_sat"]
        rscale = cloudtype_data["rscale"]
        error = cloudtype_data["error"]
        nonmet = clutter.filter_cloudtype(val, val_sat, scale=rscale, smoothing=error)
        nclutter = np.sum(nonmet)
        assert nclutter == 8141
        nonmet = clutter.filter_cloudtype(
            val, val_sat, scale=rscale, smoothing=error, low=True
        )
        nclutter = np.sum(nonmet)
        assert nclutter == 17856
