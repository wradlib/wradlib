#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

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
        da_ppi = georef.create_xarray_dataarray(img, r=r, phi=az, theta=th)
        da_ppi = georef.georeference(da_ppi)
        da_rhi = georef.create_xarray_dataarray(
            img[0:90], r=r, phi=az1, theta=el, sweep_mode="rhi"
        )
        da_rhi = georef.georeference(da_rhi)

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
        crs = georef.create_osr("dwd-radolan")
        da_ppi = georef.create_xarray_dataarray(img, r=r, phi=az, theta=th)
        da_ppi = georef.georeference(da_ppi, crs=None)
        print(da_ppi)
        da_rhi = georef.create_xarray_dataarray(img[0:90], r=r, phi=az1, theta=el)
        da_rhi = georef.georeference(da_rhi, crs=None)

    yield Data


@requires_matplotlib
def test_plot_ppi(pol_data):
    da = pol_data.da_ppi
    vis.plot(da)
    vis.plot(da, ax=pl.gca())
    vis.plot(da, ax=212)
    vis.plot_ppi_crosshair(site=(0, 0, 0), ranges=[2, 4, 8])
    vis.plot_ppi_crosshair(
        site=(0, 0, 0),
        ranges=[2, 4, 8],
        angles=[0, 45, 90, 180, 270],
        line=dict(color="white", linestyle="solid"),
    )
    vis.plot(da, func="contour")
    vis.plot(da, func="contourf")
    with pytest.raises(ValueError):
        vis.plot_ppi_crosshair(site=(0, 0), ranges=[2, 4, 8])


@requires_matplotlib
@requires_gdal
def test_plot_ppi_proj(prj_data):
    da = prj_data.da_rhi
    vis.plot(da)
    vis.plot_ppi_crosshair(
        site=(10.0, 45.0, 0.0),
        ranges=[2, 4, 8],
        angles=[0, 45, 90, 180, 270],
        crs=prj_data.crs,
        line=dict(color="white", linestyle="solid"),
    )


@requires_matplotlib
@requires_gdal
def test_plot_ppi_xarray(prj_data):
    assert hasattr(prj_data.da_ppi.wrl, "rays")
    vis.plot(prj_data.da_ppi)
    vis.plot(prj_data.da_ppi, func="contour")
    vis.plot(prj_data.da_ppi, func="contourf")
    vis.plot(prj_data.da_ppi, func="pcolormesh")
    vis.plot(prj_data.da_ppi, crs="cg")
    vis.plot(prj_data.da_ppi, crs="cg", func="contour")
    vis.plot(prj_data.da_ppi, crs="cg", func="contourf")
    vis.plot(prj_data.da_ppi, crs="cg", func="pcolormesh")
    fig = pl.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(TypeError):
        vis.plot(prj_data.da_ppi, crs={"rot": 0, "scale": 1}, func="pcolormesh", ax=ax)


@requires_matplotlib
@requires_gdal
def test_plot_ppi_xarray_accessor(prj_data):
    assert hasattr(prj_data.da_ppi.wrl, "rays")
    prj_data.da_ppi.wrl.vis.plot()
    prj_data.da_ppi.wrl.vis.contour()
    prj_data.da_ppi.wrl.vis.contourf()
    prj_data.da_ppi.wrl.vis.pcolormesh()
    prj_data.da_ppi.wrl.vis.plot(crs="cg")
    prj_data.da_ppi.wrl.vis.contour(crs="cg")
    prj_data.da_ppi.wrl.vis.contourf(crs="cg")
    prj_data.da_ppi.wrl.vis.pcolormesh(crs="cg")
    fig = pl.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(TypeError):
        prj_data.da_ppi.wrl.vis.pcolormesh(crs={"rot": 0, "scale": 1}, ax=ax)


@requires_matplotlib
@requires_gdal
def test_plot_ppi_xarray_proj(prj_data):
    with pytest.raises(TypeError):
        prj_data.da_ppi.wrl.vis.pcolormesh(crs=prj_data.crs)


@requires_matplotlib
@requires_cartopy
@requires_gdal
def test_plot_ppi_cartopy(prj_data):
    if (Version(cartopy.__version__) < Version("0.18.0")) and (
        Version(mpl.__version__) >= Version("3.3.0")
    ):
        pytest.skip("fails for cartopy < 0.18.0 and matplotlib >= 3.3.0")
    site = (7, 45, 0.0)
    map_proj = cartopy.crs.Mercator(central_longitude=site[1])
    vis.plot(prj_data.da_ppi, crs=map_proj)
    assert isinstance(pl.gca(), cartopy.mpl.geoaxes.GeoAxes)
    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=map_proj)
    prj_data.da_ppi.wrl.vis.plot(ax=ax)
    ax.gridlines(draw_labels=True)


@requires_matplotlib
@requires_gdal
def test_plot_rhi_xarray(prj_data):
    assert (
        repr(prj_data.da_rhi.wrl).split("\n", 1)[1]
        == repr(prj_data.da_rhi).split("\n", 1)[1]
    )
    assert hasattr(prj_data.da_rhi.wrl, "rays")
    prj_data.da_rhi.wrl.vis.plot()
    prj_data.da_rhi.wrl.vis.contour()
    prj_data.da_rhi.wrl.vis.contourf()
    prj_data.da_rhi.wrl.vis.pcolormesh()
    prj_data.da_rhi.wrl.vis.plot(crs="cg")
    prj_data.da_rhi.wrl.vis.contour(crs="cg")
    prj_data.da_rhi.wrl.vis.contourf(crs="cg")
    prj_data.da_rhi.wrl.vis.pcolormesh(crs="cg")


@requires_matplotlib
def test_plot_cg_ppi(pol_data):
    vis.plot(pol_data.da_ppi, crs="cg")
    vis.plot(pol_data.da_ppi, crs="cg")
    cgax = pl.gca()
    vis.plot(pol_data.da_ppi, crs="cg", ax=cgax)
    fig, ax = pl.subplots(2, 2)
    with pytest.raises(TypeError):
        vis.plot(pol_data.da_ppi, crs="cg", ax=ax[0, 0])
    vis.plot(pol_data.da_ppi, crs="cg", ax=111)
    vis.plot(pol_data.da_ppi, crs="cg", ax=121)
    vis.plot(pol_data.da_ppi, crs="cg")
    vis.plot(pol_data.da_ppi, func="contour", crs="cg")
    vis.plot(pol_data.da_ppi, func="contourf", crs="cg")
    vis.plot(pol_data.da_ppi, func="contourf", crs="cg")


@requires_matplotlib
def test_plot_cg_rhi(pol_data):
    da = pol_data.da_rhi
    vis.plot(da, crs="cg")
    cgax = pl.gca()
    vis.plot(da, crs="cg", ax=cgax)
    fig, ax = pl.subplots(2, 2)
    with pytest.raises(TypeError):
        vis.plot(da, crs="cg", ax=ax[0, 0])
    vis.plot(da, crs="cg")
    vis.plot(da, func="contour", crs="cg")
    vis.plot(da, func="contourf", crs="cg")


@requires_matplotlib
def test_create_cg():
    cgax, caax, paax = vis.create_cg()
    cgax, caax, paax = vis.create_cg(subplot=121)


@requires_matplotlib
@requires_data
@requires_gdal
def test_plot_scan_strategy():
    ranges = np.arange(0, 10000, 100)
    elevs = np.arange(1, 30, 3)
    site = (7.0, 53.0, 100.0)
    vis.plot_scan_strategy(ranges, elevs, site)
    vis.plot_scan_strategy(ranges, elevs, site, cg=True)
    vis.plot_scan_strategy(ranges, [1.0], site)
    vis.plot_scan_strategy(ranges, [1.0], site, cg=True)


@requires_matplotlib
@requires_data
@requires_secrets
@requires_gdal
@pytest.mark.xfail(strict=False)
def test_plot_scan_strategy_terrain():
    ranges = np.arange(0, 10000, 100)
    elevs = np.arange(1, 30, 3)
    site = (-28.5, 38.5, 100.0)
    vis.plot_scan_strategy(ranges, elevs, site, terrain=True)
    vis.plot_scan_strategy(ranges, elevs, site, cg=True, terrain=True)
    vis.plot_scan_strategy(ranges, [1.0], site, terrain=True)
    vis.plot_scan_strategy(ranges, [1.0], site, cg=True, terrain=True)


@requires_matplotlib
def test_plot_plan_and_vert():
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    z = np.arange(0, 5)
    dataxy = np.zeros((len(x), len(y)))
    datazx = np.zeros((len(z), len(x)))
    datazy = np.zeros((len(z), len(y)))
    vol = np.zeros((len(z), len(y), len(x)))
    vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy)
    vis.plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, title="Test")
    vis.plot_max_plan_and_vert(x, y, z, vol)


@requires_matplotlib
def test_add_lines():
    fig, ax = pl.subplots()
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    xy = np.dstack((x, y))
    vis.add_lines(ax, xy)
    vis.add_lines(ax, np.array([xy]))


@requires_matplotlib
def test_add_patches():
    fig, ax = pl.subplots()
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    xy = np.dstack((x, y))
    vis.add_patches(ax, xy)
    vis.add_patches(ax, np.array([xy]))
