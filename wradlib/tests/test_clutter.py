#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np
import wradlib.clutter as clutter
import unittest
from .. import io as io
from .. import util as util
from .. import georef as georef
from .. import ipol as ipol


# -------------------------------------------------------------------------------
# testing the filter helper function
# -------------------------------------------------------------------------------
class TestClutter(unittest.TestCase):
    # def test_filter_gabella_a_trdefault(self):
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data)
    #     self.assertTrue(result == 4)
    #
    # def test_filter_gabella_a_tr1(self):
    #     pass
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data, tr1=5)
    #     self.assertTrue(result == 3)
    # -------------------------------------------------------------------------------
    # testing the first part of the filter
    # -------------------------------------------------------------------------------
    def filter_setup(self):
        img = np.zeros((36, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[9:12, 4:7] = 11  # precip field
        self.img = img
        pass

    def test_filter_gabella_a(self):
        pass

    def test_filter_window_distance(self):
        self.filter_setup()
        self.img[15:17, 5:7] = np.nan  # nans
        cl = self.img.copy()
        cl[self.img > 0] = True
        cl[self.img == 11] = False
        cl[np.isnan(self.img)] = False
        np.set_printoptions(precision=2)
        rscale = 250
        similar = clutter.filter_window_distance(self.img, rscale, fsize=300,
                                                 tr1=4)
        result = similar < 0.3
        np.set_printoptions(precision=3)
        self.assertTrue((result == cl).all())


class FilterGabellaTest(unittest.TestCase):
    def test_filter_gabella(self):
        filename = util.get_wradlib_data_file('misc/polar_dBZ_fbg.gz')
        data = np.loadtxt(filename)
        clutter.filter_gabella(data, wsize=5, thrsnorain=0., tr1=6.,
                               n_p=8, tr2=1.3)


class HistoCutTest(unittest.TestCase):
    def test_histo_cut_test(self):
        filename = util.get_wradlib_data_file('misc/annual_rainfall_fbg.gz')
        yearsum = np.loadtxt(filename)
        clutter.histo_cut(yearsum)


class ClassifyEchoFuzzyTest(unittest.TestCase):
    def setUp(self):
        rhofile = util.get_wradlib_data_file('netcdf/TAG-20120801'
                                             '-140046-02-R.nc')
        phifile = util.get_wradlib_data_file('netcdf/TAG-20120801'
                                             '-140046-02-P.nc')
        reffile = util.get_wradlib_data_file('netcdf/TAG-20120801'
                                             '-140046-02-Z.nc')
        dopfile = util.get_wradlib_data_file('netcdf/TAG-20120801'
                                             '-140046-02-V.nc')
        zdrfile = util.get_wradlib_data_file('netcdf/TAG-20120801'
                                             '-140046-02-D.nc')
        mapfile = util.get_wradlib_data_file('hdf5/TAG_cmap_sweeps'
                                             '_0204050607.hdf5')
        # We need to organize our data as a dictionary
        dat = {}
        dat["rho"], attrs_rho = io.read_edge_netcdf(rhofile)
        dat["phi"], attrs_phi = io.read_edge_netcdf(phifile)
        dat["ref"], attrs_ref = io.read_edge_netcdf(reffile)
        dat["dop"], attrs_dop = io.read_edge_netcdf(dopfile)
        dat["zdr"], attrs_zdr = io.read_edge_netcdf(zdrfile)
        dat["map"] = io.from_hdf5(mapfile)[0][0]
        self.dat = dat

    def test_classify_echo_fuzzy(self):
        weights = {"zdr": 0.4,
                   "rho": 0.4,
                   "rho2": 0.4,
                   "phi": 0.1,
                   "dop": 0.1,
                   "map": 0.5}
        clutter.classify_echo_fuzzy(self.dat, weights=weights,
                                    thresh=0.5)


class FilterCloudtypeTest(unittest.TestCase):
    def setUp(self):
        # read the radar volume scan
        filename = 'hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf'
        filename = util.get_wradlib_data_file(filename)
        pvol = io.read_opera_hdf5(filename)
        nrays = int(pvol["dataset1/where"]["nrays"])
        nbins = int(pvol["dataset1/where"]["nbins"])
        val = pvol["dataset%d/data1/data" % (1)]
        gain = float(pvol["dataset1/data1/what"]["gain"])
        offset = float(pvol["dataset1/data1/what"]["offset"])
        self.val = val * gain + offset
        self.rscale = int(pvol["dataset1/where"]["rscale"])
        elangle = pvol["dataset%d/where" % (1)]["elangle"]
        coord = georef.sweep_centroids(nrays, self.rscale, nbins, elangle)
        sitecoords = (pvol["where"]["lon"], pvol["where"]["lat"],
                      pvol["where"]["height"])

        coord, proj_radar = georef.spherical_to_xyz(coord[..., 0],
                                                    np.degrees(coord[..., 1]),
                                                    coord[..., 2], sitecoords,
                                                    re=6370040.,
                                                    ke=4. / 3.)
        filename = 'hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5'
        filename = util.get_wradlib_data_file(filename)
        sat_gdal = io.read_safnwc(filename)
        val_sat = georef.read_gdal_values(sat_gdal)
        coord_sat = georef.read_gdal_coordinates(sat_gdal)
        proj_sat = georef.read_gdal_projection(sat_gdal)
        coord_sat = georef.reproject(coord_sat, projection_source=proj_sat,
                                     projection_target=proj_radar)
        coord_radar = coord
        interp = ipol.Nearest(coord_sat[..., 0:2].reshape(-1, 2),
                              coord_radar[..., 0:2].reshape(-1, 2))
        self.val_sat = interp(val_sat.ravel()).reshape(val.shape)
        timelag = 9 * 60
        wind = 10
        self.error = np.absolute(timelag) * wind

    def test_filter_cloudtype(self):
        clutter.filter_cloudtype(self.val, self.val_sat,
                                 scale=self.rscale, smoothing=self.error)


if __name__ == '__main__':
    unittest.main()
