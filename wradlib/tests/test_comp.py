#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import numpy as np

from wradlib import comp, georef, io, ipol, util

from . import has_data, requires_data


class TestCompose:
    if has_data:
        filename = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        dx_file = util.get_wradlib_data_file(filename)
        data, metadata = io.read_dx(dx_file)
        radar_location = (8.005, 47.8744, 1517)
        elevation = 0.5  # in degree
        azimuths = np.arange(0, 360)  # in degrees
        ranges = np.arange(0, 128000.0, 1000.0)  # in meters
        polargrid = np.meshgrid(ranges, azimuths)
        coords, rad = georef.spherical_to_xyz(
            polargrid[0], polargrid[1], elevation, radar_location
        )
        x = coords[..., 0]
        y = coords[..., 1]

    @requires_data
    def test_extract_circle(self):
        xgrid = np.linspace(self.x.min(), self.x.mean(), 100)
        ygrid = np.linspace(self.y.min(), self.y.mean(), 100)
        grid_xy = np.meshgrid(xgrid, ygrid)
        grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
        comp.extract_circle(np.array([self.x.mean(), self.y.mean()]), 128000.0, grid_xy)

    @requires_data
    def test_togrid(self):
        xgrid = np.linspace(self.x.min(), self.x.mean(), 100)
        ygrid = np.linspace(self.y.min(), self.y.mean(), 100)
        grid_xy = np.meshgrid(xgrid, ygrid)
        grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
        xy = np.concatenate([self.x.ravel()[:, None], self.y.ravel()[:, None]], axis=1)
        comp.togrid(
            xy,
            grid_xy,
            128000.0,
            np.array([self.x.mean(), self.y.mean()]),
            self.data.ravel(),
            ipol.Nearest,
        )

    def test_compose(self):
        g1 = np.array(
            [
                np.nan,
                np.nan,
                10.0,
                np.nan,
                np.nan,
                np.nan,
                10.0,
                10.0,
                10.0,
                np.nan,
                10.0,
                10.0,
                10.0,
                10.0,
                np.nan,
                np.nan,
                10.0,
                10.0,
                10.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )
        g2 = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                11.0,
                11.0,
                11.0,
                np.nan,
                np.nan,
                11.0,
                11.0,
                11.0,
                11.0,
                np.nan,
                11.0,
                11.0,
                11.0,
                np.nan,
                np.nan,
                np.nan,
                11.0,
                np.nan,
                np.nan,
            ]
        )
        q1 = np.array(
            [
                np.nan,
                np.nan,
                3.47408756e09,
                np.nan,
                np.nan,
                np.nan,
                8.75744493e08,
                8.75744493e08,
                1.55045236e09,
                np.nan,
                3.47408756e09,
                8.75744493e08,
                5.98145272e04,
                1.55045236e09,
                np.nan,
                np.nan,
                1.55045236e09,
                1.55045236e09,
                1.55045236e09,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )
        q2 = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.55045236e09,
                1.55045236e09,
                1.55045236e09,
                np.nan,
                np.nan,
                1.55045236e09,
                5.98145272e04,
                8.75744493e08,
                3.47408756e09,
                np.nan,
                1.55045236e09,
                8.75744493e08,
                8.75744493e08,
                np.nan,
                np.nan,
                np.nan,
                3.47408756e09,
                np.nan,
                np.nan,
            ]
        )

        composite = comp.compose_weighted(
            [g1, g2], [1.0 / (q1 + 0.001), 1.0 / (q2 + 0.001)]
        )
        composite1 = comp.compose_ko([g1, g2], [1.0 / (q1 + 0.001), 1.0 / (q2 + 0.001)])
        res = np.array(
            [
                np.nan,
                np.nan,
                10.0,
                np.nan,
                np.nan,
                np.nan,
                10.3609536,
                10.3609536,
                10.5,
                np.nan,
                10.0,
                10.3609536,
                10.5,
                10.6390464,
                11.0,
                np.nan,
                10.5,
                10.6390464,
                10.6390464,
                np.nan,
                np.nan,
                np.nan,
                11.0,
                np.nan,
                np.nan,
            ]
        )
        res1 = np.array(
            [
                np.nan,
                np.nan,
                10.0,
                np.nan,
                np.nan,
                np.nan,
                10.0,
                10.0,
                10.0,
                np.nan,
                10.0,
                10.0,
                10.0,
                11.0,
                11.0,
                np.nan,
                10.0,
                11.0,
                11.0,
                np.nan,
                np.nan,
                np.nan,
                11.0,
                np.nan,
                np.nan,
            ]
        )
        np.testing.assert_allclose(composite, res)
        np.testing.assert_allclose(composite1, res1)
