#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest

import wradlib.vis as vis
import wradlib.georef as georef
import numpy as np
import matplotlib.pyplot as pl
pl.interactive(True)  # noqa
import datetime as dt  # noqa


class PolarPlotTest(unittest.TestCase):
    def setUp(self):
        img = np.zeros((360, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[60:120, 2:7] = 11  # precip field
        self.img = img

    def test_plot_ppi(self):
        pl.figure()
        proj = georef.create_osr("dwd-radolan")
        ax, pm = vis.plot_ppi(self.img, re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, autoext=False)
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               line=dict(color='white',
                                         linestyle='solid'))
        pl.figure()
        ax, pm = vis.plot_ppi(self.img, site=(10., 45.), autoext=False,
                              proj=proj)
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               proj=proj,
                               line=dict(color='white',
                                         linestyle='solid'))

    def test_plot_rhi(self):
        pl.figure()
        ax, pm = vis.plot_rhi(self.img[0:90, :])
        pl.figure()
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5)
        pl.figure()
        ax, pm = vis.plot_rhi(self.img[0:90, :], refrac=False)
        pl.figure()
        ax, pm = vis.plot_rhi(self.img[0:90, :], autoext=False)
        pl.figure()
        ax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                              th=np.arange(90))

    def test_plot_cg_ppi(self):
        cgax, caax, paax, pm = vis.plot_cg_ppi(self.img, elev=2.0)
        cgax, caax, paax, pm = vis.plot_cg_ppi(self.img, autoext=False)
        cgax, caax, paax, pm = vis.plot_cg_ppi(self.img, refrac=False)

    def test_plot_cg_rhi(self):
        cgax, caax, paax, pm = vis.plot_cg_rhi(self.img[0:90, :])
        cgax, caax, paax, pm = vis.plot_cg_rhi(self.img[0:90, :], th_res=0.5)
        cgax, caax, paax, pm = vis.plot_cg_rhi(self.img[0:90, :], refrac=False)
        cgax, caax, paax, pm = vis.plot_cg_rhi(self.img[0:90, :],
                                               autoext=False)
        cgax, caax, paax, pm = vis.plot_cg_rhi(self.img[0:90, :],
                                               r=np.arange(10),
                                               th=np.arange(90), autoext=True)

    def test_plot_scan_strategy(self):
        pl.figure()
        ranges = np.arange(0, 100000, 1000)
        elevs = np.arange(1, 30, 3)
        site = (7.0, 53.0)
        vis.plot_scan_strategy(ranges, elevs, site)
        pl.figure()
        vis.plot_scan_strategy(ranges, elevs, site, ax=pl.gca())


class MiscPlotTest(unittest.TestCase):
    def test_plot_plan_and_vert(self):
        pass

    def test_plot_max_plan_and_vert(self):
        pass

    def test_plot_tseries(self):
        base = dt.datetime.today()
        date_list = np.array(
            [base - dt.timedelta(hours=x) for x in range(0, 48)])
        data = np.arange(0, len(date_list))
        data = np.vstack((data, data[::-1]))
        vis.plot_tseries(date_list, data.T)

    def test_add_lines(self):
        pass

    def test_add_patches(self):
        pass


if __name__ == '__main__':
    unittest.main()
