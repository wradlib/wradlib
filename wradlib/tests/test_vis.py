#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import sys
import unittest

import wradlib.vis as vis
import wradlib.georef as georef
import numpy as np
import matplotlib.pyplot as pl
pl.interactive(True)  # noqa
from tempfile import NamedTemporaryFile


class PolarPlotTest(unittest.TestCase):
    def setUp(self):
        img = np.zeros((360, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[60:120, 2:7] = 11  # precip field
        self.r = np.arange(0, 100000, 10000)
        self.az = np.arange(0, 360)
        self.img = img
        self.proj = georef.create_osr("dwd-radolan")

    def test_plot_ppi(self):
        ax, pm = vis.plot_ppi(self.img, re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az,
                              re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, refrac=True,
                              re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, refrac=True,
                              re=6371000., ke=(4. / 3.), ax=ax)
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, refrac=True,
                              re=6371000., ke=(4. / 3.), ax=212)
        ax, pm = vis.plot_ppi(self.img, autoext=False)
        vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               line=dict(color='white',
                                         linestyle='solid'))
        ax, pm = vis.plot_ppi(self.img, site=(10., 45.), autoext=False,
                              proj=self.proj)
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               proj=self.proj,
                               line=dict(color='white',
                                         linestyle='solid'))
        ax, pm = vis.plot_ppi(self.img, func='contour')
        ax, pm = vis.plot_ppi(self.img, func='contourf')

    def test_plot_rhi(self):
        ax, pm = vis.plot_rhi(self.img[0:90, :])
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5)
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, ax=212)
        ax, pm = vis.plot_rhi(self.img[0:90, :], refrac=False)
        ax, pm = vis.plot_rhi(self.img[0:90, :], autoext=False)
        ax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                              th=np.arange(90))
        ax, pm = vis.plot_rhi(self.img[0:90, :], func='contour')
        ax, pm = vis.plot_rhi(self.img[0:90, :], func='contourf')

    def test_plot_cg_ppi(self):
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=cgax)
        fig, ax = pl.subplots(2, 2)
        self.assertRaises(TypeError,
                          lambda: vis.plot_ppi(self.img, elev=2.0,
                                               cg=True, ax=ax[0, 0]))
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=111)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=121)
        cgax, pm = vis.plot_ppi(self.img, autoext=False, cg=True)
        cgax, pm = vis.plot_ppi(self.img, refrac=False, cg=True)
        cgax, pm = vis.plot_ppi(self.img, func='contour', cg=True)
        cgax, pm = vis.plot_ppi(self.img, func='contourf', cg=True)

    @unittest.skipIf(sys.version_info < (3, 5),
                     "not supported in this python version")
    def test_plot_cg_ppi_py3k(self):
        with self.assertWarns(UserWarning):
            cgax, pm = vis.plot_ppi(self.img, func='contourf',
                                    proj=self.proj, cg=True)

    def test_plot_cg_rhi(self):
        cgax, pm = vis.plot_rhi(self.img[0:90, :], cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], cg=True, ax=cgax)
        fig, ax = pl.subplots(2, 2)
        self.assertRaises(TypeError,
                          lambda: vis.plot_rhi(self.img[0:90, :],
                                               cg=True, ax=ax[0, 0]))
        cgax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], refrac=False, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], autoext=False, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                                th=np.arange(90), autoext=True, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func='contour', cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func='contourf',
                                cg=True)

    def test_create_cg(self):
        cgax, caax, paax = vis.create_cg('PPI')
        cgax, caax, paax = vis.create_cg('PPI', subplot=121)


class MiscPlotTest(unittest.TestCase):
    def test_plot_scan_strategy(self):
        ranges = np.arange(0, 100000, 1000)
        elevs = np.arange(1, 30, 3)
        site = (7.0, 53.0)
        vis.plot_scan_strategy(ranges, elevs, site)
        vis.plot_scan_strategy(ranges, elevs, site, ax=pl.gca())

    def test_plot_plan_and_vert(self):
        x = np.arange(0, 10)
        y = np.arange(0, 10)
        z = np.arange(0, 5)
        dataxy = np.zeros((len(x), len(y)))
        datazx = np.zeros((len(z), len(x)))
        datazy = np.zeros((len(z), len(y)))
        vol = np.zeros((len(z), len(y), len(x)))
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy)
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy,
                               title='Test')
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy,
                               saveto=NamedTemporaryFile(mode='w+b').name)
        vis.plot_max_plan_and_vert(x, y, z, vol)

    def test_add_lines(self):
        fig, ax = pl.subplots()
        x = np.arange(0, 10)
        y = np.arange(0, 10)
        xy = np.dstack((x, y))
        vis.add_lines(ax, xy)
        vis.add_lines(ax, np.array([xy]))

    def test_add_patches(self):
        fig, ax = pl.subplots()
        x = np.arange(0, 10)
        y = np.arange(0, 10)
        xy = np.dstack((x, y))
        vis.add_patches(ax, xy)
        vis.add_patches(ax, np.array([xy]))


if __name__ == '__main__':
    unittest.main()
