#!/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import sys

import wradlib.vis as vis
import wradlib.georef as georef
import wradlib.io as io
import numpy as np
import matplotlib.pyplot as pl
from wradlib.util import import_optional
from tempfile import NamedTemporaryFile
cartopy = import_optional('cartopy')
pl.interactive(True)  # noqa


class PolarPlotTest(unittest.TestCase):
    def setUp(self):
        img = np.zeros((360, 10), dtype=np.float32)
        img[2, 2] = 10  # isolated pixel
        img[5, 6:8] = 10  # line
        img[20, :] = 5  # spike
        img[60:120, 2:7] = 11  # precip field
        self.r = np.arange(0, 100000, 10000)
        self.az = np.arange(0, 360)
        self.el = np.arange(0, 90)
        self.th = np.zeros_like(self.az)
        self.az1 = np.ones_like(self.el) * 225
        self.img = img
        self.proj = georef.create_osr("dwd-radolan")

        self.da_ppi = io.create_xarray_dataarray(img, self.r, self.az, self.th)
        self.da_rhi = io.create_xarray_dataarray(img[0:90], self.r, self.az1,
                                                 self.el)

    def test_plot_ppi(self):
        # DeprecationTests
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_ppi(self.img, autoext=True)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_ppi(self.img, autoext=False)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_ppi(self.img, refrac=True)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_ppi(self.img, refrac=False)

        ax, pm = vis.plot_ppi(self.img, re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az,
                              re=6371000., ke=(4. / 3.))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az,
                              re=6371000., ke=(4. / 3.), ax=ax)
        ax, pm = vis.plot_ppi(self.img, self.r, self.az,
                              re=6371000., ke=(4. / 3.), ax=212)
        ax, pm = vis.plot_ppi(self.img)
        vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               line=dict(color='white',
                                         linestyle='solid'))
        ax, pm = vis.plot_ppi(self.img, site=(10., 45., 0.),
                              proj=self.proj)
        vis.plot_ppi_crosshair(site=(0, 0),
                               ranges=[2, 4, 8],
                               angles=[0, 45, 90, 180, 270],
                               proj=self.proj,
                               line=dict(color='white',
                                         linestyle='solid'))
        ax, pm = vis.plot_ppi(self.img, func='contour')
        ax, pm = vis.plot_ppi(self.img, func='contourf')
        with self.assertRaises(TypeError):
            ax, pm = vis.plot_ppi(self.img, proj=self.proj)
        with self.assertWarns(UserWarning):
            ax, pm = vis.plot_ppi(self.img, proj=None,
                                  site=(0, 0, 0))
        with self.assertWarns(UserWarning):
            ax, pm = vis.plot_ppi(self.img, proj=None,
                                  site=(0, 0))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, proj=self.proj,
                              site=(0, 0, 0))

    def test_plot_ppi_xarray(self):
        self.da_ppi.wradlib.rays
        self.da_ppi.wradlib.plot()
        self.da_ppi.wradlib.plot_ppi()
        self.da_ppi.wradlib.contour()
        self.da_ppi.wradlib.contourf()
        self.da_ppi.wradlib.pcolormesh()
        self.da_ppi.wradlib.plot(proj='cg')
        self.da_ppi.wradlib.plot_ppi(proj='cg')
        self.da_ppi.wradlib.contour(proj='cg')
        self.da_ppi.wradlib.contourf(proj='cg')
        self.da_ppi.wradlib.pcolormesh(proj='cg')
        with self.assertRaises(TypeError):
            self.da_ppi.wradlib.pcolormesh(proj=self.proj)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        with self.assertRaises(TypeError):
            self.da_ppi.wradlib.pcolormesh(proj={'rot': 0, 'scale': 1},
                                           ax=ax)

    @unittest.skipIf('cartopy' not in sys.modules, "without Cartopy")
    def test_plot_ppi_cartopy(self):
        if cartopy:
            site = (7, 45, 0.)
            map_proj = cartopy.crs.Mercator(central_longitude=site[1])
            ax, pm = vis.plot_ppi(self.img, self.r, self.az, proj=map_proj)
            self.assertIsInstance(ax, cartopy.mpl.geoaxes.GeoAxes)
            fig = pl.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection=map_proj)
            self.da_ppi.wradlib.plot_ppi(ax=ax)
            ax.gridlines(draw_labels=True)

    def test_plot_rhi(self):
        # DeprecationTests
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_rhi(self.img, autoext=True)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_rhi(self.img, autoext=False)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_rhi(self.img, refrac=True)
        with self.assertWarns(DeprecationWarning):
            ax, pm = vis.plot_rhi(self.img, refrac=False)
        ax, pm = vis.plot_rhi(self.img[0:90, :])
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5)
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, ax=212)
        ax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                              th=np.arange(90))
        ax, pm = vis.plot_rhi(self.img[0:90, :], func='contour')
        ax, pm = vis.plot_rhi(self.img[0:90, :], func='contourf')
        ax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                              th=np.arange(90), proj=self.proj,
                              site=(0, 0, 0))

    def test_plot_rhi_xarray(self):
        self.assertEqual(repr(self.da_rhi.wradlib).split("\n", 1)[1],
                         repr(self.da_rhi).split("\n", 1)[1])
        self.da_rhi.wradlib.rays
        self.da_rhi.wradlib.plot()
        self.da_rhi.wradlib.plot_rhi()
        self.da_rhi.wradlib.contour()
        self.da_rhi.wradlib.contourf()
        self.da_rhi.wradlib.pcolormesh()
        self.da_rhi.wradlib.plot(proj='cg')
        self.da_rhi.wradlib.plot_rhi(proj='cg')
        self.da_rhi.wradlib.contour(proj='cg')
        self.da_rhi.wradlib.contourf(proj='cg')
        self.da_rhi.wradlib.pcolormesh(proj='cg')

    def test_plot_cg_ppi(self):
        # DeprecationTests
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_ppi(self.img, autoext=True, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_ppi(self.img, autoext=False, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_ppi(self.img, refrac=True, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_ppi(self.img, refrac=False, cg=True)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, proj=self.proj,
                                site=(0, 0, 0))
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj='cg')
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=cgax)
        fig, ax = pl.subplots(2, 2)
        self.assertRaises(TypeError,
                          lambda: vis.plot_ppi(self.img, elev=2.0,
                                               cg=True, ax=ax[0, 0]))
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=111)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, cg=True, ax=121)
        cgax, pm = vis.plot_ppi(self.img, cg=True)
        cgax, pm = vis.plot_ppi(self.img, func='contour', cg=True)
        cgax, pm = vis.plot_ppi(self.img, func='contourf', cg=True)
        cgax, pm = vis.plot_ppi(self.img, func='contourf', cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_ppi(self.img, func='contourf',
                                    proj=self.proj, site=(0, 0, 0),
                                    cg=True)

    def test_plot_cg_rhi(self):
        # DeprecationTests
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_rhi(self.img, autoext=True, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_rhi(self.img, autoext=False, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_rhi(self.img, refrac=True, cg=True)
        with self.assertWarns(DeprecationWarning):
            cgax, pm = vis.plot_rhi(self.img, refrac=False, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], proj='cg')
        cgax, pm = vis.plot_rhi(self.img[0:90, :], cg=True, ax=cgax)
        fig, ax = pl.subplots(2, 2)
        self.assertRaises(TypeError,
                          lambda: vis.plot_rhi(self.img[0:90, :],
                                               cg=True, ax=ax[0, 0]))
        cgax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10),
                                th=np.arange(90), cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func='contour', cg=True)
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func='contourf',
                                cg=True)
        with self.assertWarns(UserWarning):
            cgax, pm = vis.plot_rhi(self.img[0:90, :], func='contourf',
                                    proj=self.proj, site=(0, 0, 0),
                                    cg=True)

    def test_create_cg(self):
        with self.assertWarns(DeprecationWarning):
            cgax, caax, paax = vis.create_cg('PPI')
        with self.assertWarns(DeprecationWarning):
            cgax, caax, paax = vis.create_cg('PPI', subplot=121)
        with self.assertWarns(DeprecationWarning):
            cgax, caax, paax = vis.create_cg('RHI')
        with self.assertWarns(DeprecationWarning):
            cgax, caax, paax = vis.create_cg('RHI', subplot=121)
        cgax, caax, paax = vis.create_cg()
        cgax, caax, paax = vis.create_cg(subplot=121)


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
