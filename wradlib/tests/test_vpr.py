#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import georef, vpr

from . import requires_gdal


@pytest.fixture
def help_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
        site = (7.0, 53.0, 100.0)
        proj = georef.epsg_to_osr(31467)
        az = np.arange(0.0, 360.0, 2.0)
        r = np.arange(0, 50000, 1000)
        el = 2.5

    yield Data


class TestVPRHelperFunctions:
    def test_out_of_range(self):
        pass

    def test_blindspots(self):
        pass

    @requires_gdal
    def test_volcoords_from_polar(self, help_data):
        coords = vpr.volcoords_from_polar(
            help_data.site, help_data.el, help_data.az, help_data.r, help_data.proj
        )
        assert coords.shape == (9000, 3)

    @requires_gdal
    def test_volcoords_from_polar_irregular(self, help_data):
        # oneazforall, onerange4all, one elev
        coords = vpr.volcoords_from_polar_irregular(
            help_data.site, [help_data.el], help_data.az, help_data.r, help_data.proj
        )
        assert coords.shape == (9000, 3)

        # oneazforall, onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(
            help_data.site,
            [help_data.el, 5.0],
            help_data.az,
            help_data.r,
            help_data.proj,
        )
        assert coords.shape == (18000, 3)

        # onerange4all, two elev
        coords = vpr.volcoords_from_polar_irregular(
            help_data.site,
            [help_data.el, 5.0],
            [help_data.az, help_data.az],
            help_data.r,
            help_data.proj,
        )
        assert coords.shape == (18000, 3)

        # oneazforall, two elev
        coords = vpr.volcoords_from_polar_irregular(
            help_data.site,
            [help_data.el, 5.0],
            help_data.az,
            [help_data.r, help_data.r],
            help_data.proj,
        )
        assert coords.shape == (18000, 3)

    @requires_gdal
    def test_synthetic_polar_volume(self, help_data):
        nbins = [320, 240, 340, 300]
        rscale = [1000, 1000, 500, 500]
        elev = [0.3, 0.4, 3.0, 4.5]

        xyz = np.array([]).reshape((-1, 3))
        for i, vals in enumerate(zip(nbins, rscale, elev)):
            az = np.arange(0.0, 360.0, 2.0)
            r = np.arange(0, vals[0] * vals[1], vals[1])
            xyz_ = vpr.volcoords_from_polar(
                help_data.site, vals[2], az, r, help_data.proj
            )
            xyz = np.vstack((xyz, xyz_))

        vol = vpr.synthetic_polar_volume(xyz)
        assert vol.shape == (216000,)

    def test_norm_vpr_stats(self):
        vol = np.arange(2 * 3 * 4).astype("f4").reshape((4, 3, 2)) ** 2
        prof = vpr.norm_vpr_stats(vol, 1)
        np.allclose(prof, np.array([0.09343848, 1.0, 3.0396144, 6.2122827]))

    @requires_gdal
    def test_make_3d_grid(self, help_data):
        maxrange = 50000.0
        maxalt = 5000.0
        horiz_res = 4000.0
        vert_res = 1000.0
        outxyz, outshape = vpr.make_3d_grid(
            help_data.site, help_data.proj, maxrange, maxalt, horiz_res, vert_res
        )
        assert outshape == (6, 26, 26)
        assert outxyz.shape == (4056, 3)


@pytest.fixture
def cart_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
        site = (7.0, 53.0, 100.0)
        proj = georef.epsg_to_osr(31467)
        az = np.arange(0.0, 360.0, 2.0) + 1.0
        r = np.arange(0.0, 50000.0, 1000.0)
        elev = np.array([1.0, 3.0, 5.0, 10.0])
        # cartesian grid settings
        maxrange = 50000.0
        minelev = 1.0
        maxelev = 10.0
        maxalt = 8000.0
        horiz_res = 4000.0
        vert_res = 1000.0
        xyz = vpr.volcoords_from_polar(site, elev, az, r, proj)
        data = vpr.synthetic_polar_volume(xyz)
        trgxyz, trgshape = vpr.make_3d_grid(
            site, proj, maxrange, maxalt, horiz_res, vert_res
        )

    yield Data


class TestCartesianVolume:
    @requires_gdal
    def test_CartesianVolume(self, cart_data):
        gridder = vpr.CartesianVolume(
            cart_data.xyz,
            cart_data.trgxyz,
            cart_data.trgshape,
            cart_data.maxrange,
            cart_data.minelev,
            cart_data.maxelev,
        )
        out = gridder(cart_data.data)
        assert out.shape == (6084,)
        assert len(np.where(np.isnan(out))[0]) == 0

    @requires_gdal
    def test_CAPPI(self, cart_data):
        gridder = vpr.CAPPI(
            cart_data.xyz,
            cart_data.trgxyz,
            cart_data.trgshape,
            cart_data.maxrange,
            cart_data.minelev,
            cart_data.maxelev,
        )
        out = gridder(cart_data.data)
        assert out.shape == (6084,)
        assert len(np.where(np.isnan(out))[0]) == 3528

    @requires_gdal
    def test_PseudoCAPPI(self, cart_data):
        # interpolate to Cartesian 3-D volume grid
        gridder = vpr.PseudoCAPPI(
            cart_data.xyz,
            cart_data.trgxyz,
            cart_data.trgshape,
            cart_data.maxrange,
            cart_data.minelev,
            cart_data.maxelev,
        )
        out = gridder(cart_data.data)
        assert out.shape == (6084,)
        assert len(np.where(np.isnan(out))[0]) == 1744
