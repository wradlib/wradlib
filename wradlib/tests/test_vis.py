#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import sys
import tempfile
from distutils.version import LooseVersion

import matplotlib as mpl  # isort:skip

mpl.use("Agg")  # noqa: E402
import matplotlib.pyplot as pl
import numpy as np
import pytest

from wradlib import georef, util, vis

from . import requires_data, requires_secrets

cartopy = util.import_optional("cartopy")


class TestPolarPlot:
    img = np.zeros((360, 10), dtype=np.float32)
    img[2, 2] = 10  # isolated pixel
    img[5, 6:8] = 10  # line
    img[20, :] = 5  # spike
    img[60:120, 2:7] = 11  # precip field
    r = np.arange(0, 100000, 10000)
    az = np.arange(0, 360)
    el = np.arange(0, 90)
    th = np.zeros_like(az)
    az1 = np.ones_like(el) * 225
    img = img
    proj = georef.create_osr("dwd-radolan")

    da_ppi = georef.create_xarray_dataarray(img, r, az, th)
    da_ppi = georef.georeference_dataset(da_ppi, proj=None)
    da_rhi = georef.create_xarray_dataarray(img[0:90], r, az1, el)
    da_rhi = georef.georeference_dataset(da_rhi, proj=None)

    def test_plot_ppi(self):
        ax, pm = vis.plot_ppi(self.img, re=6371000.0, ke=(4.0 / 3.0))
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, re=6371000.0, ke=(4.0 / 3.0))
        ax, pm = vis.plot_ppi(
            self.img, self.r, self.az, re=6371000.0, ke=(4.0 / 3.0), ax=ax
        )
        ax, pm = vis.plot_ppi(
            self.img, self.r, self.az, re=6371000.0, ke=(4.0 / 3.0), ax=212
        )
        ax, pm = vis.plot_ppi(self.img)
        vis.plot_ppi_crosshair(site=(0, 0, 0), ranges=[2, 4, 8])
        vis.plot_ppi_crosshair(
            site=(0, 0, 0),
            ranges=[2, 4, 8],
            angles=[0, 45, 90, 180, 270],
            line=dict(color="white", linestyle="solid"),
        )
        ax, pm = vis.plot_ppi(self.img, self.r, site=(10.0, 45.0, 0.0), proj=self.proj)
        vis.plot_ppi_crosshair(
            site=(10.0, 45.0, 0.0),
            ranges=[2, 4, 8],
            angles=[0, 45, 90, 180, 270],
            proj=self.proj,
            line=dict(color="white", linestyle="solid"),
        )
        ax, pm = vis.plot_ppi(self.img, func="contour")
        ax, pm = vis.plot_ppi(self.img, func="contourf")
        ax, pm = vis.plot_ppi(self.img, self.r, self.az, proj=self.proj, site=(0, 0, 0))
        with pytest.warns(UserWarning):
            ax, pm = vis.plot_ppi(self.img, site=(10.0, 45.0, 0.0), proj=self.proj)
        with pytest.warns(UserWarning):
            ax, pm = vis.plot_ppi(self.img, proj=None, site=(0, 0, 0))
        with pytest.raises(TypeError):
            ax, pm = vis.plot_ppi(self.img, proj=self.proj)
        with pytest.raises(ValueError):
            ax, pm = vis.plot_ppi(self.img, site=(0, 0), proj=self.proj)
        with pytest.raises(ValueError):
            vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])

    def test_plot_ppi_xarray(self):
        self.da_ppi.wradlib.rays
        self.da_ppi.wradlib.plot()
        self.da_ppi.wradlib.plot_ppi()
        self.da_ppi.wradlib.contour()
        self.da_ppi.wradlib.contourf()
        self.da_ppi.wradlib.pcolormesh()
        self.da_ppi.wradlib.plot(proj="cg")
        self.da_ppi.wradlib.plot_ppi(proj="cg")
        self.da_ppi.wradlib.contour(proj="cg")
        self.da_ppi.wradlib.contourf(proj="cg")
        self.da_ppi.wradlib.pcolormesh(proj="cg")
        with pytest.raises(TypeError):
            self.da_ppi.wradlib.pcolormesh(proj=self.proj)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        with pytest.raises(TypeError):
            self.da_ppi.wradlib.pcolormesh(proj={"rot": 0, "scale": 1}, ax=ax)

    @pytest.mark.skipif("cartopy" not in sys.modules, reason="without Cartopy")
    def test_plot_ppi_cartopy(self):
        if cartopy:
            if (LooseVersion(cartopy.__version__) < LooseVersion("0.18.0")) and (
                LooseVersion(mpl.__version__) >= LooseVersion("3.3.0")
            ):
                pytest.skip("fails for cartopy < 0.18.0 and matplotlib >= 3.3.0")
            site = (7, 45, 0.0)
            map_proj = cartopy.crs.Mercator(central_longitude=site[1])
            ax, pm = vis.plot_ppi(self.img, self.r, self.az, proj=map_proj)
            assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxes)
            fig = pl.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection=map_proj)
            self.da_ppi.wradlib.plot_ppi(ax=ax)
            ax.gridlines(draw_labels=True)

    def test_plot_rhi(self):
        ax, pm = vis.plot_rhi(self.img[0:90, :])
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5)
        ax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, ax=212)
        ax, pm = vis.plot_rhi(self.img[0:90, :], r=np.arange(10), th=np.arange(90))
        ax, pm = vis.plot_rhi(self.img[0:90, :], func="contour")
        ax, pm = vis.plot_rhi(self.img[0:90, :], func="contourf")
        ax, pm = vis.plot_rhi(
            self.img[0:90, :],
            r=np.arange(10),
            th=np.arange(90),
            proj=self.proj,
            site=(0, 0, 0),
        )

    def test_plot_rhi_xarray(self):
        assert (
            repr(self.da_rhi.wradlib).split("\n", 1)[1]
            == repr(self.da_rhi).split("\n", 1)[1]
        )
        self.da_rhi.wradlib.rays
        self.da_rhi.wradlib.plot()
        self.da_rhi.wradlib.plot_rhi()
        self.da_rhi.wradlib.contour()
        self.da_rhi.wradlib.contourf()
        self.da_rhi.wradlib.pcolormesh()
        self.da_rhi.wradlib.plot(proj="cg")
        self.da_rhi.wradlib.plot_rhi(proj="cg")
        self.da_rhi.wradlib.contour(proj="cg")
        self.da_rhi.wradlib.contourf(proj="cg")
        self.da_rhi.wradlib.pcolormesh(proj="cg")

    def test_plot_cg_ppi(self):
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj="cg")
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj="cg", site=(0, 0, 0))
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj="cg", ax=cgax)
        fig, ax = pl.subplots(2, 2)
        with pytest.raises(TypeError):
            vis.plot_ppi(self.img, elev=2.0, proj="cg", ax=ax[0, 0])
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj="cg", ax=111)
        cgax, pm = vis.plot_ppi(self.img, elev=2.0, proj="cg", ax=121)
        cgax, pm = vis.plot_ppi(self.img, proj="cg")
        cgax, pm = vis.plot_ppi(self.img, func="contour", proj="cg")
        cgax, pm = vis.plot_ppi(self.img, func="contourf", proj="cg")
        cgax, pm = vis.plot_ppi(self.img, func="contourf", proj="cg")

    def test_plot_cg_rhi(self):
        cgax, pm = vis.plot_rhi(self.img[0:90, :], proj="cg")
        cgax, pm = vis.plot_rhi(self.img[0:90, :], proj="cg", ax=cgax)
        fig, ax = pl.subplots(2, 2)
        with pytest.raises(TypeError):
            vis.plot_rhi(self.img[0:90, :], proj="cg", ax=ax[0, 0])
        cgax, pm = vis.plot_rhi(self.img[0:90, :], th_res=0.5, proj="cg")
        cgax, pm = vis.plot_rhi(self.img[0:90, :], proj="cg")
        cgax, pm = vis.plot_rhi(
            self.img[0:90, :], r=np.arange(10), th=np.arange(90), proj="cg"
        )
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func="contour", proj="cg")
        cgax, pm = vis.plot_rhi(self.img[0:90, :], func="contourf", proj="cg")

    def test_create_cg(self):
        cgax, caax, paax = vis.create_cg()
        cgax, caax, paax = vis.create_cg(subplot=121)


class TestMiscPlot:
    @requires_data
    def test_plot_scan_strategy(self):
        ranges = np.arange(0, 10000, 100)
        elevs = np.arange(1, 30, 3)
        site = (7.0, 53.0, 100.0)
        vis.plot_scan_strategy(ranges, elevs, site)
        vis.plot_scan_strategy(ranges, elevs, site, cg=True)

    @requires_data
    @requires_secrets
    def test_plot_scan_strategy_terrain(self):
        ranges = np.arange(0, 10000, 100)
        elevs = np.arange(1, 30, 3)
        site = (7.0, 53.0, 100.0)
        vis.plot_scan_strategy(ranges, elevs, site, terrain=True)
        vis.plot_scan_strategy(ranges, elevs, site, cg=True, terrain=True)

    def test_plot_plan_and_vert(self):
        x = np.arange(0, 10)
        y = np.arange(0, 10)
        z = np.arange(0, 5)
        dataxy = np.zeros((len(x), len(y)))
        datazx = np.zeros((len(z), len(x)))
        datazy = np.zeros((len(z), len(y)))
        vol = np.zeros((len(z), len(y), len(x)))
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy)
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, title="Test")
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, saveto=tmp)
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
