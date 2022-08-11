#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
from packaging.version import Version

from wradlib import georef, util, vis

from . import (
    cartopy,
    mpl,
    requires_cartopy,
    requires_data,
    requires_gdal,
    requires_matplotlib,
    requires_secrets,
)

if not isinstance(mpl, util.OptionalModuleStub):
    mpl.use("Agg")

pl = util.import_optional("matplotlib.pyplot")


@pytest.fixture
def pol_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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
        da_ppi = georef.create_xarray_dataarray(img, r, az, th)
        da_rhi = georef.create_xarray_dataarray(img[0:90], r, az1, el)

    yield Data


@pytest.fixture
def prj_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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

    yield Data


@requires_matplotlib
class TestPolarPlot:
    def test_plot_ppi(self, pol_data):
        ax, pm = vis.plot_ppi(pol_data.img, re=6371000.0, ke=(4.0 / 3.0))
        ax, pm = vis.plot_ppi(
            pol_data.img, pol_data.r, pol_data.az, re=6371000.0, ke=(4.0 / 3.0)
        )
        ax, pm = vis.plot_ppi(
            pol_data.img, pol_data.r, pol_data.az, re=6371000.0, ke=(4.0 / 3.0), ax=ax
        )
        ax, pm = vis.plot_ppi(
            pol_data.img, pol_data.r, pol_data.az, re=6371000.0, ke=(4.0 / 3.0), ax=212
        )
        ax, pm = vis.plot_ppi(pol_data.img)
        vis.plot_ppi_crosshair(site=(0, 0, 0), ranges=[2, 4, 8])
        vis.plot_ppi_crosshair(
            site=(0, 0, 0),
            ranges=[2, 4, 8],
            angles=[0, 45, 90, 180, 270],
            line=dict(color="white", linestyle="solid"),
        )
        ax, pm = vis.plot_ppi(pol_data.img, func="contour")
        ax, pm = vis.plot_ppi(pol_data.img, func="contourf")
        with pytest.warns(UserWarning):
            ax, pm = vis.plot_ppi(pol_data.img, proj=None, site=(0, 0, 0))
        with pytest.raises(ValueError):
            vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])

    @requires_gdal
    def test_plot_ppi_proj(self, prj_data):
        ax, pm = vis.plot_ppi(
            prj_data.img, prj_data.r, site=(10.0, 45.0, 0.0), proj=prj_data.proj
        )
        vis.plot_ppi_crosshair(
            site=(10.0, 45.0, 0.0),
            ranges=[2, 4, 8],
            angles=[0, 45, 90, 180, 270],
            proj=prj_data.proj,
            line=dict(color="white", linestyle="solid"),
        )

        ax, pm = vis.plot_ppi(
            prj_data.img, prj_data.r, prj_data.az, proj=prj_data.proj, site=(0, 0, 0)
        )
        with pytest.warns(UserWarning):
            ax, pm = vis.plot_ppi(
                prj_data.img, site=(10.0, 45.0, 0.0), proj=prj_data.proj
            )

        with pytest.raises(TypeError):
            ax, pm = vis.plot_ppi(prj_data.img, proj=prj_data.proj)
        with pytest.raises(ValueError):
            ax, pm = vis.plot_ppi(prj_data.img, site=(0, 0), proj=prj_data.proj)
        with pytest.raises(ValueError):
            vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])

    @requires_gdal
    def test_plot_ppi_xarray(self, prj_data):
        prj_data.da_ppi.wradlib.rays
        prj_data.da_ppi.wradlib.plot()
        prj_data.da_ppi.wradlib.plot_ppi()
        prj_data.da_ppi.wradlib.contour()
        prj_data.da_ppi.wradlib.contourf()
        prj_data.da_ppi.wradlib.pcolormesh()
        prj_data.da_ppi.wradlib.plot(proj="cg")
        prj_data.da_ppi.wradlib.plot_ppi(proj="cg")
        prj_data.da_ppi.wradlib.contour(proj="cg")
        prj_data.da_ppi.wradlib.contourf(proj="cg")
        prj_data.da_ppi.wradlib.pcolormesh(proj="cg")
        fig = pl.figure()
        ax = fig.add_subplot(111)
        with pytest.raises(TypeError):
            prj_data.da_ppi.wradlib.pcolormesh(proj={"rot": 0, "scale": 1}, ax=ax)

    @requires_gdal
    def test_plot_ppi_xarray_proj(self, prj_data):
        with pytest.raises(TypeError):
            prj_data.da_ppi.wradlib.pcolormesh(proj=prj_data.proj)

    @requires_cartopy
    @requires_gdal
    def test_plot_ppi_cartopy(self, prj_data):
        if (Version(cartopy.__version__) < Version("0.18.0")) and (
            Version(mpl.__version__) >= Version("3.3.0")
        ):
            pytest.skip("fails for cartopy < 0.18.0 and matplotlib >= 3.3.0")
        site = (7, 45, 0.0)
        map_proj = cartopy.crs.Mercator(central_longitude=site[1])
        ax, pm = vis.plot_ppi(prj_data.img, prj_data.r, prj_data.az, proj=map_proj)
        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxes)
        fig = pl.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=map_proj)
        prj_data.da_ppi.wradlib.plot_ppi(ax=ax)
        ax.gridlines(draw_labels=True)

    def test_plot_rhi(self, pol_data):
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :])
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :], th_res=0.5)
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :], th_res=0.5, ax=212)
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :], r=np.arange(10), th=np.arange(90))
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :], func="contour")
        ax, pm = vis.plot_rhi(pol_data.img[0:90, :], func="contourf")
        ax, pm = vis.plot_rhi(
            pol_data.img[0:90, :],
            r=np.arange(10),
            th=np.arange(90),
            proj=None,
            site=(0, 0, 0),
        )

    @requires_gdal
    def test_plot_rhi_xarray(self, prj_data):
        assert (
            repr(prj_data.da_rhi.wradlib).split("\n", 1)[1]
            == repr(prj_data.da_rhi).split("\n", 1)[1]
        )
        prj_data.da_rhi.wradlib.rays
        prj_data.da_rhi.wradlib.plot()
        prj_data.da_rhi.wradlib.plot_rhi()
        prj_data.da_rhi.wradlib.contour()
        prj_data.da_rhi.wradlib.contourf()
        prj_data.da_rhi.wradlib.pcolormesh()
        prj_data.da_rhi.wradlib.plot(proj="cg")
        prj_data.da_rhi.wradlib.plot_rhi(proj="cg")
        prj_data.da_rhi.wradlib.contour(proj="cg")
        prj_data.da_rhi.wradlib.contourf(proj="cg")
        prj_data.da_rhi.wradlib.pcolormesh(proj="cg")

    def test_plot_cg_ppi(self, pol_data):
        cgax, pm = vis.plot_ppi(pol_data.img, elev=2.0, proj="cg")
        cgax, pm = vis.plot_ppi(pol_data.img, elev=2.0, proj="cg", site=(0, 0, 0))
        cgax, pm = vis.plot_ppi(pol_data.img, elev=2.0, proj="cg", ax=cgax)
        fig, ax = pl.subplots(2, 2)
        with pytest.raises(TypeError):
            vis.plot_ppi(pol_data.img, elev=2.0, proj="cg", ax=ax[0, 0])
        cgax, pm = vis.plot_ppi(pol_data.img, elev=2.0, proj="cg", ax=111)
        cgax, pm = vis.plot_ppi(pol_data.img, elev=2.0, proj="cg", ax=121)
        cgax, pm = vis.plot_ppi(pol_data.img, proj="cg")
        cgax, pm = vis.plot_ppi(pol_data.img, func="contour", proj="cg")
        cgax, pm = vis.plot_ppi(pol_data.img, func="contourf", proj="cg")
        cgax, pm = vis.plot_ppi(pol_data.img, func="contourf", proj="cg")

    def test_plot_cg_rhi(self, pol_data):
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], proj="cg")
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], proj="cg", ax=cgax)
        fig, ax = pl.subplots(2, 2)
        with pytest.raises(TypeError):
            vis.plot_rhi(pol_data.img[0:90, :], proj="cg", ax=ax[0, 0])
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], th_res=0.5, proj="cg")
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], proj="cg")
        cgax, pm = vis.plot_rhi(
            pol_data.img[0:90, :], r=np.arange(10), th=np.arange(90), proj="cg"
        )
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], func="contour", proj="cg")
        cgax, pm = vis.plot_rhi(pol_data.img[0:90, :], func="contourf", proj="cg")

    def test_create_cg(self):
        cgax, caax, paax = vis.create_cg()
        cgax, caax, paax = vis.create_cg(subplot=121)


@requires_matplotlib
class TestMiscPlot:
    @requires_data
    @requires_gdal
    def test_plot_scan_strategy(self):
        ranges = np.arange(0, 10000, 100)
        elevs = np.arange(1, 30, 3)
        site = (7.0, 53.0, 100.0)
        vis.plot_scan_strategy(ranges, elevs, site)
        vis.plot_scan_strategy(ranges, elevs, site, cg=True)
        vis.plot_scan_strategy(ranges, [1.0], site)
        vis.plot_scan_strategy(ranges, [1.0], site, cg=True)

    @requires_data
    @requires_secrets
    @requires_gdal
    @pytest.mark.xfail(strict=False)
    def test_plot_scan_strategy_terrain(self):
        ranges = np.arange(0, 10000, 100)
        elevs = np.arange(1, 30, 3)
        site = (-28.5, 38.5, 100.0)
        vis.plot_scan_strategy(ranges, elevs, site, terrain=True)
        vis.plot_scan_strategy(ranges, elevs, site, cg=True, terrain=True)
        vis.plot_scan_strategy(ranges, [1.0], site, terrain=True)
        vis.plot_scan_strategy(ranges, [1.0], site, cg=True, terrain=True)

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
