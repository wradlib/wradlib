# !/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import sys

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal, ogr, osr

import wradlib
from wradlib import georef, util

from . import has_data, requires_data

np.set_printoptions(
    edgeitems=3,
    infstr="inf",
    linewidth=75,
    nanstr="nan",
    precision=8,
    suppress=False,
    threshold=1000,
    formatter=None,
)


@pytest.fixture(params=[10, 13, 15])
def r(request):
    return request.param


class TestCoordinateTransform:
    r = np.array([0.0, 0.0, 111.0, 111.0, 111.0, 111.0, 111.0]) * 1000
    az = np.array([0.0, 180.0, 0.0, 90.0, 180.0, 270.0, 360.0])
    th = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    csite = (9.0, 48.0, 0)
    result_xyz = tuple(
        (
            np.array([0.0, 0.0, 0.0, 110993.6738, 0.0, -110976.7856, 0.0]),
            np.array([0.0, -0.0, 110993.6738, 0.0, -110993.6738, -0.0, 110976.7856]),
            np.array(
                [
                    0.0,
                    0.0,
                    725.7159843,
                    725.7159843,
                    725.7159843,
                    1694.22337134,
                    1694.22337134,
                ]
            ),
        )
    )
    result = tuple(
        (
            np.array([9.0, 9.0, 9.0, 10.49189531, 9.0, 7.50810469, 9.0]),
            np.array(
                [
                    48.0,
                    48.0,
                    48.99839742,
                    47.99034027,
                    47.00160258,
                    47.99034027,
                    47.99034027,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    967.03198482,
                    967.03198482,
                    967.03198482,
                    1935.45679527,
                    1935.45679527,
                ]
            ),
        )
    )
    result_n = tuple(
        (
            np.array([9.0, 9.0, 9.0, 10.48716091, 9.0, 7.51306531, 9.0]),
            np.array(
                [
                    48.0,
                    48.0,
                    48.99814438,
                    47.99037251,
                    47.00168131,
                    47.99037544,
                    48.997993,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    725.7159843,
                    725.7159843,
                    725.7159843,
                    1694.22337134,
                    1694.22337134,
                ]
            ),
        )
    )

    def test_spherical_to_xyz(self):
        def check(rc, azc, elc, outc, squeeze=False, strict_dims=False):
            assert (
                georef.spherical_to_xyz(
                    rc, azc, elc, self.csite, squeeze=squeeze, strict_dims=strict_dims,
                )[0].shape
                == outc
            )

        check(np.arange(10), np.arange(36), 10.0, (1, 36, 10, 3))
        check(np.arange(10), np.arange(36), np.arange(36), (1, 36, 10, 3,))
        check(np.arange(10), np.arange(36), np.arange(36), (36, 10, 3,), squeeze=True)
        check(
            np.arange(10),
            np.arange(36),
            np.arange(36),
            (36, 36, 10, 3),
            squeeze=None,
            strict_dims=True,
        )
        check(np.arange(10), np.arange(36), np.arange(18), (18, 36, 10, 3))
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        check(r, phi, 10, (1, 36, 10, 3))
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        check(r, phi, np.arange(36), (1, 36, 10, 3))
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        check(r, phi, np.arange(18), (18, 36, 10, 3))
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        check(r, phi, np.arange(36), (36, 36, 10, 3,), strict_dims=True)
        check(10, 36, 10.0, (1, 1, 1, 3))
        check(np.arange(10), 36, 10.0, (1, 1, 10, 3))
        check(10, np.arange(36), 10.0, (1, 36, 1, 3))
        check(10, 36.0, np.arange(10), (10, 1, 1, 3))
        check(10, np.arange(36), np.arange(10), (10, 36, 1, 3))
        coords, rad = georef.spherical_to_xyz(
            self.r, self.az, self.th, self.csite, squeeze=True
        )
        np.testing.assert_allclose(
            coords[..., 0], self.result_xyz[0], rtol=2e-10, atol=3e-5
        )
        np.testing.assert_allclose(
            coords[..., 1], self.result_xyz[1], rtol=2e-10, atol=3e-5
        )
        np.testing.assert_allclose(
            coords[..., 2], self.result_xyz[2], rtol=2e-10, atol=3e-5
        )
        re = georef.get_earth_radius(self.csite[1])
        coords, rad = georef.spherical_to_xyz(
            self.r, self.az, self.th, self.csite, re=re, squeeze=True,
        )
        np.testing.assert_allclose(
            coords[..., 0], self.result_xyz[0], rtol=2e-10, atol=3e-5
        )
        np.testing.assert_allclose(
            coords[..., 1], self.result_xyz[1], rtol=2e-10, atol=3e-5
        )
        np.testing.assert_allclose(
            coords[..., 2], self.result_xyz[2], rtol=2e-10, atol=3e-5
        )

    def test_bin_altitude(self):
        altitude = georef.bin_altitude(
            np.arange(10.0, 101.0, 10.0) * 1000.0, 2.0, 0, 6370040.0
        )
        altref = np.array(
            [
                354.87448647,
                721.50702113,
                1099.8960815,
                1490.04009656,
                1891.93744678,
                2305.58646416,
                2730.98543223,
                3168.13258613,
                3617.02611263,
                4077.66415017,
            ]
        )
        np.testing.assert_allclose(altref, altitude)

    def test_bin_distance(self):
        distance = georef.bin_distance(
            np.arange(10.0, 101.0, 10.0) * 1000.0, 2.0, 0, 6370040.0
        )
        distref = np.array(
            [
                9993.49302358,
                19986.13717891,
                29977.90491409,
                39968.76869178,
                49958.70098959,
                59947.6743006,
                69935.66113377,
                79922.63401441,
                89908.5654846,
                99893.4281037,
            ]
        )
        np.testing.assert_allclose(distref, distance)

    def test_site_distance(self):
        altitude = georef.bin_altitude(
            np.arange(10.0, 101.0, 10.0) * 1000.0, 2.0, 0, 6370040.0
        )
        distance = georef.site_distance(
            np.arange(10.0, 101.0, 10.0) * 1000.0, 2.0, altitude, 6370040.0
        )
        distref = np.array(
            [
                9993.49302358,
                19986.13717891,
                29977.90491409,
                39968.76869178,
                49958.70098959,
                59947.6743006,
                69935.66113377,
                79922.63401441,
                89908.5654846,
                99893.4281037,
            ]
        )
        np.testing.assert_allclose(distref, distance)

    def test_spherical_to_proj(self):
        coords = georef.spherical_to_proj(self.r, self.az, self.th, self.csite)
        np.testing.assert_allclose(coords[..., 0], self.result_n[0])
        np.testing.assert_allclose(coords[..., 1], self.result_n[1])
        np.testing.assert_allclose(coords[..., 2], self.result_n[2])

    @requires_data
    def test_maximum_intensity_projection(self):
        angle = 0.0
        elev = 0.0

        filename = util.get_wradlib_data_file("misc/polar_dBZ_tur.gz")
        data = np.loadtxt(filename)
        # we need to have meter here for the georef function inside mip
        d1 = np.arange(data.shape[1], dtype=np.float) * 1000
        d2 = np.arange(data.shape[0], dtype=np.float)
        data = np.roll(data, (d2 >= angle).nonzero()[0][0], axis=0)

        # calculate max intensity proj
        georef.maximum_intensity_projection(data, r=d1, az=d2, angle=angle, elev=elev)
        georef.maximum_intensity_projection(data, autoext=False)


class TestCoordinateHelper:
    def test_centroid_to_polyvert(self):
        np.testing.assert_array_equal(
            georef.centroid_to_polyvert(np.array([0.0, 1.0]), [0.5, 1.5]),
            np.array(
                [[-0.5, -0.5], [-0.5, 2.5], [0.5, 2.5], [0.5, -0.5], [-0.5, -0.5]]
            ),
        )
        np.testing.assert_array_equal(
            georef.centroid_to_polyvert(np.arange(4).reshape((2, 2)), 0.5),
            np.array(
                [
                    [[-0.5, 0.5], [-0.5, 1.5], [0.5, 1.5], [0.5, 0.5], [-0.5, 0.5]],
                    [[1.5, 2.5], [1.5, 3.5], [2.5, 3.5], [2.5, 2.5], [1.5, 2.5]],
                ]
            ),
        )
        with pytest.raises(ValueError):
            georef.centroid_to_polyvert([[0.0], [1.0]], [0.5, 1.5])

    def test_spherical_to_polyvert(self):
        sph = georef.get_default_projection()
        polyvert = georef.spherical_to_polyvert(
            np.array([10000.0, 10100.0]),
            np.array([45.0, 90.0]),
            0,
            (9.0, 48.0),
            proj=sph,
        )
        arr = np.asarray(
            [
                [
                    [9.05084865, 48.08224715, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.12264494, 48.03400725, 6.0],
                    [9.05084865, 48.08224715, 6.0],
                ],
                [
                    [9.05136309, 48.0830778, 6.0],
                    [9.05187756, 48.08390846, 6.0],
                    [9.12512428, 48.03469291, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                ],
                [
                    [9.12264494, 48.03400725, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.05084865, 48.08224715, 6.0],
                    [9.12264494, 48.03400725, 6.0],
                ],
                [
                    [9.1238846, 48.03435008, 6.0],
                    [9.12512428, 48.03469291, 6.0],
                    [9.05187756, 48.08390846, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                ],
            ]
        )
        np.testing.assert_array_almost_equal(polyvert, arr, decimal=3)
        polyvert, pr = georef.spherical_to_polyvert(
            np.array([10000.0, 10100.0]), np.array([45.0, 90.0]), 0, (9.0, 48.0)
        )
        arr = np.asarray(
            [
                [
                    [3.7885640e03, 9.1464023e03, 6.0],
                    [3.8268320e03, 9.2387900e03, 6.0],
                    [9.2387900e03, 3.8268323e03, 6.0],
                    [9.1464023e03, 3.7885645e03, 6.0],
                    [3.7885640e03, 9.1464023e03, 6.0],
                ],
                [
                    [3.8268320e03, 9.2387900e03, 6.0],
                    [3.8651003e03, 9.3311777e03, 6.0],
                    [9.3311777e03, 3.8651006e03, 6.0],
                    [9.2387900e03, 3.8268323e03, 6.0],
                    [3.8268320e03, 9.2387900e03, 6.0],
                ],
                [
                    [9.1464023e03, 3.7885645e03, 6.0],
                    [9.2387900e03, 3.8268323e03, 6.0],
                    [3.8268320e03, 9.2387900e03, 6.0],
                    [3.7885640e03, 9.1464023e03, 6.0],
                    [9.1464023e03, 3.7885645e03, 6.0],
                ],
                [
                    [9.2387900e03, 3.8268323e03, 6.0],
                    [9.3311777e03, 3.8651006e03, 6.0],
                    [3.8651003e03, 9.3311777e03, 6.0],
                    [3.8268320e03, 9.2387900e03, 6.0],
                    [9.2387900e03, 3.8268323e03, 6.0],
                ],
            ]
        )
        np.testing.assert_array_almost_equal(polyvert, arr, decimal=3)

    def test_spherical_to_centroids(self):
        r = np.array([10000.0, 10100.0])
        az = np.array([45.0, 90.0])
        sitecoords = (9.0, 48.0, 0.0)
        sph = georef.get_default_projection()
        centroids = georef.spherical_to_centroids(r, az, 0, sitecoords, proj=sph)
        arr = np.asarray(
            [
                [[9.09439583, 48.06323717, 6.0], [9.09534571, 48.06387232, 6.0]],
                [[9.1333325, 47.99992262, 6.0], [9.13467253, 47.99992106, 6.0]],
            ]
        )
        np.testing.assert_array_almost_equal(centroids, arr, decimal=3)

        centroids, pr = georef.spherical_to_centroids(r, az, 0, sitecoords)
        arr = np.asarray(
            [
                [[7.0357090e03, 7.0357090e03, 6.0], [7.1064194e03, 7.1064194e03, 6.0]],
                [[9.9499951e03, 0.0, 6.0], [1.0049995e04, 0.0, 6.0]],
            ]
        )
        np.testing.assert_array_almost_equal(centroids, arr, decimal=3)

    def test_sweep_centroids(self):
        np.testing.assert_array_equal(
            georef.sweep_centroids(1, 100.0, 1, 2.0), np.array([[[50.0, 180.0, 2.0]]])
        )

    def test__check_polar_coords(self):
        r = np.array([50.0, 100.0, 150.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50.0, 100.0, 150.0, 200.0])
        az = np.array([10.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        with pytest.warns(UserWarning):
            georef.polar._check_polar_coords(r, az)

        r = np.array([0, 50.0, 100.0, 150.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50.0, 100.0, 100.0, 150.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([100.0, 50.0, 150.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50.0, 100.0, 125.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50.0, 100.0, 150.0, 200.0])
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 361.0])
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50.0, 100.0, 150.0, 200.0])
        az = np.array([225.0, 270.0, 315.0, 0.0, 45.0, 90.0, 135.0, 180.0])[::-1]
        with pytest.raises(ValueError):
            georef.polar._check_polar_coords(r, az)

    def test__get_range_resolution(self):
        r = np.array([50.0])
        with pytest.raises(ValueError):
            georef.polar._get_range_resolution(r)
        r = np.array([50.0, 100.0, 150.0, 190.0, 250.0])
        with pytest.raises(ValueError):
            georef.polar._get_range_resolution(r)

    def test__get_azimuth_resolution(self):
        az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 224.0, 270.0, 315.0])
        with pytest.raises(ValueError):
            georef.polar._get_azimuth_resolution(az)


class TestProjections:
    wgs84_gdal2 = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,'
        'AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],'
        'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]]'
    )
    wgs84_gdal3 = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,'
        'AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],'
        'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]],'
        'AXIS["Latitude",NORTH],'
        'AXIS["Longitude",EAST],'
        'AUTHORITY["EPSG","4326"]]'
    )

    if gdal.VersionInfo()[0] >= "3":
        wgs84 = wgs84_gdal3
    else:
        wgs84 = wgs84_gdal2

    def test_create_osr(self):
        self.maxDiff = None
        radolan_wkt = (
            'PROJCS["Radolan projection",'
            'GEOGCS["Radolan Coordinate System",'
            'DATUM["Radolan Kugel",'
            'SPHEROID["Erdkugel",6370040.0,0.0]],'
            'PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.017453292519943295],'
            'AXIS["Longitude",EAST],'
            'AXIS["Latitude",NORTH]],'
            'PROJECTION["polar_stereographic"],'
            'PARAMETER["central_meridian",10.0],'
            'PARAMETER["latitude_of_origin",90.0],'
            'PARAMETER["scale_factor",{0:8.10f}],'
            'PARAMETER["false_easting",0.0],'
            'PARAMETER["false_northing",0.0],'
            'UNIT["m*1000.0",1000.0],'
            'AXIS["X",EAST],'
            'AXIS["Y",NORTH]]'
        )

        radolan_wkt3 = (
            'PROJCS["Radolan Projection",'
            'GEOGCS["Radolan Coordinate System",'
            'DATUM["Radolan_Kugel",'
            'SPHEROID["Erdkugel",6370040,0]],'
            'PRIMEM["Greenwich",0,'
            'AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.0174532925199433,'
            'AUTHORITY["EPSG","9122"]]],'
            'PROJECTION["Polar_Stereographic"],'
            'PARAMETER["latitude_of_origin",90],'
            'PARAMETER["central_meridian",10],'
            'PARAMETER["scale_factor",{0:8.12f}],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["kilometre",1000,'
            'AUTHORITY["EPSG","9036"]],'
            'AXIS["Easting",SOUTH],'
            'AXIS["Northing",SOUTH]]'
        )

        scale = (1.0 + np.sin(np.radians(60.0))) / (1.0 + np.sin(np.radians(90.0)))

        aeqd_wkt = (
            'PROJCS["unnamed",'
            'GEOGCS["WGS 84",'
            'DATUM["unknown",'
            'SPHEROID["WGS84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Azimuthal_Equidistant"],'
            'PARAMETER["latitude_of_center",{0:-f}],'
            'PARAMETER["longitude_of_center",{1:-f}],'
            'PARAMETER["false_easting",{2:-f}],'
            'PARAMETER["false_northing",{3:-f}],'
            'UNIT["Meter",1]]'
        )

        aeqd_wkt3 = (
            'PROJCS["unnamed",'
            'GEOGCS["WGS 84",'
            'DATUM["unknown",'
            'SPHEROID["WGS84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Azimuthal_Equidistant"],'
            'PARAMETER["latitude_of_center",{0:d}],'
            'PARAMETER["longitude_of_center",{1:d}],'
            'PARAMETER["false_easting",{2:d}],'
            'PARAMETER["false_northing",{3:d}],'
            'UNIT["Meter",1],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )

        if gdal.VersionInfo()[0] >= "3":
            radolan_wkt = radolan_wkt3.format(scale)
            aeqd_wkt = aeqd_wkt3.format(49, 5, 0, 0)
        else:
            radolan_wkt = radolan_wkt.format(scale)
            aeqd_wkt = aeqd_wkt.format(49, 5, 0, 0)

        assert georef.create_osr("dwd-radolan").ExportToWkt() == radolan_wkt

        assert georef.create_osr("aeqd", lon_0=5, lat_0=49).ExportToWkt() == aeqd_wkt
        assert (
            georef.create_osr("aeqd", lon_0=5, lat_0=49, x_0=0, y_0=0).ExportToWkt()
            == aeqd_wkt
        )
        with pytest.raises(ValueError):
            georef.create_osr("lambert")

    def test_proj4_to_osr(self):
        projstr = (
            "+proj=tmerc +lat_0=0 +lon_0=9 +k=1 "
            "+x_0=3500000 +y_0=0 +ellps=bessel "
            "+towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 "
            "+units=m +no_defs"
        )

        srs = georef.proj4_to_osr(projstr)
        p4 = srs.ExportToProj4()
        srs2 = osr.SpatialReference()
        srs2.ImportFromProj4(p4)
        assert srs.IsSame(srs2)
        with pytest.raises(ValueError):
            georef.proj4_to_osr("+proj=lcc1")

    def test_get_earth_radius(self):
        assert georef.get_earth_radius(50.0), 6365631.51753728

    def test_reproject(self):
        proj_gk = osr.SpatialReference()
        proj_gk.ImportFromEPSG(31466)
        proj_wgs84 = osr.SpatialReference()
        proj_wgs84.ImportFromEPSG(4326)
        x, y, z = georef.reproject(
            7.0, 53.0, 0.0, projection_source=proj_wgs84, projection_target=proj_gk
        )
        lon, lat = georef.reproject(
            x, y, projection_source=proj_gk, projection_target=proj_wgs84
        )
        assert pytest.approx(lon) == 7.0
        assert pytest.approx(lat) == 53.0

        lonlat = georef.reproject(
            np.stack((x, y), axis=-1),
            projection_source=proj_gk,
            projection_target=proj_wgs84,
        )
        assert pytest.approx(lonlat[0]) == 7.0
        assert pytest.approx(lonlat[1]) == 53.0

        with pytest.raises(TypeError):
            georef.reproject(np.stack((x, y, x, y), axis=-1))

        lon, lat, alt = georef.reproject(
            x, y, z, projection_source=proj_gk, projection_target=proj_wgs84
        )
        assert pytest.approx(lon, abs=1e-5) == 7.0
        assert pytest.approx(lat, abs=1e-3) == 53.0
        assert pytest.approx(alt, abs=1e-3) == 0.0

        with pytest.raises(TypeError):
            georef.reproject(x, y, x, y)
        with pytest.raises(TypeError):
            georef.reproject([np.arange(10)], [np.arange(11)])
        with pytest.raises(TypeError):
            georef.reproject([np.arange(10)], [np.arange(10)], [np.arange(11)])

    def test_get_default_projection(self):
        assert georef.get_default_projection().ExportToWkt() == self.wgs84

    def test_epsg_to_osr(self):
        assert georef.epsg_to_osr(4326).ExportToWkt() == self.wgs84
        assert georef.epsg_to_osr().ExportToWkt() == self.wgs84

    def test_wkt_to_osr(self):
        assert georef.wkt_to_osr(self.wgs84).IsSame(georef.get_default_projection())
        assert georef.wkt_to_osr().IsSame(georef.get_default_projection())

    def test_get_radar_projection(self):
        sitecoords = [5, 52, 90]
        p0 = georef.get_radar_projection(sitecoords)
        if gdal.VersionInfo()[0] >= "3":
            assert p0.GetName() == "Unknown Azimuthal Equidistant"
        assert p0.IsProjected()
        assert p0.IsSameGeogCS(georef.get_default_projection())
        assert p0.GetNormProjParm("latitude_of_center") == sitecoords[1]
        assert p0.GetNormProjParm("longitude_of_center") == sitecoords[0]

    def test_get_earth_projection(self):
        georef.get_earth_projection("ellipsoid")
        georef.get_earth_projection("geoid")
        georef.get_earth_projection("sphere")

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="known break on windows")
    def test_geoid_to_ellipsoid(self):
        coords = np.array([[5.0, 50.0, 300.0], [2, 54, 300], [50, 5, 300]])
        geoid = georef.get_earth_projection("geoid")
        ellipsoid = georef.get_earth_projection("ellipsoid")
        newcoords = georef.reproject(
            coords, projection_source=geoid, projection_target=ellipsoid
        )
        assert np.any(np.not_equal(coords[..., 2], newcoords[..., 2]))

        newcoords = georef.reproject(
            newcoords, projection_source=ellipsoid, projection_target=geoid
        )
        np.testing.assert_allclose(coords, newcoords)

    def test_get_extent(self):
        arr = np.asarray(
            [
                [
                    [9.05084865, 48.08224715, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.12264494, 48.03400725, 6.0],
                    [9.05084865, 48.08224715, 6.0],
                ],
                [
                    [9.05136309, 48.0830778, 6.0],
                    [9.05187756, 48.08390846, 6.0],
                    [9.12512428, 48.03469291, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                ],
                [
                    [9.12264494, 48.03400725, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.05084865, 48.08224715, 6.0],
                    [9.12264494, 48.03400725, 6.0],
                ],
                [
                    [9.1238846, 48.03435008, 6.0],
                    [9.12512428, 48.03469291, 6.0],
                    [9.05187756, 48.08390846, 6.0],
                    [9.05136309, 48.0830778, 6.0],
                    [9.1238846, 48.03435008, 6.0],
                ],
            ]
        )

        ref = [9.05084865, 9.12512428, 48.03400725, 48.08390846]
        extent = georef.get_extent(arr)
        np.testing.assert_array_almost_equal(ref, extent)


class TestPixMap:
    def test_pixel_coordinates(self):
        pass

    def test_pixel_to_map(self):
        pass


@requires_data
class TestGdal:
    if has_data:
        filename = "geo/bonn_new.tif"
        geofile = util.get_wradlib_data_file(filename)
        ds = wradlib.io.open_raster(geofile)
        (data, coords, proj) = georef.extract_raster_dataset(ds)

        filename = "hdf5/belgium.comp.hdf"
        geofile = util.get_wradlib_data_file(filename)
        ds2 = wradlib.io.open_raster(geofile)
        (data2, coords2, proj2) = georef.extract_raster_dataset(ds, mode="edge")

    corner_gdalinfo = np.array([[3e5, 1e6], [3e5, 3e5], [1e6, 3e5], [1e6, 1e6]])

    corner_geo_gdalinfo = np.array(
        [
            [-0.925465, 53.6928559],
            [-0.266697, 47.4167912],
            [9.0028805, 47.4160381],
            [9.6641599, 53.6919969],
        ]
    )

    def test_read_gdal_coordinates(self):
        center_coords = georef.read_gdal_coordinates(self.ds)
        assert center_coords.shape[-1] == 2
        edge_coords = georef.read_gdal_coordinates(self.ds, mode="edge")
        ul_center = (edge_coords[0, 0] + edge_coords[1, 1]) / 2
        np.testing.assert_array_almost_equal(center_coords[0, 0], ul_center)

    requires_data

    def test_read_gdal_projection(self):
        georef.read_gdal_projection(self.ds)

    def test_read_gdal_values(self):
        georef.read_gdal_values(self.ds)
        georef.read_gdal_values(self.ds, nodata=9999.0)

    def test_reproject_raster_dataset(self):
        georef.reproject_raster_dataset(
            self.ds, spacing=0.005, resample=gdal.GRA_Bilinear, align=True
        )
        georef.reproject_raster_dataset(
            self.ds, size=(1000, 1000), resample=gdal.GRA_Bilinear, align=True
        )
        with pytest.raises(NameError):
            georef.reproject_raster_dataset(self.ds)
        dst = georef.epsg_to_osr(25832)
        georef.reproject_raster_dataset(
            self.ds,
            spacing=100.0,
            resample=gdal.GRA_Bilinear,
            align=True,
            projection_target=dst,
        )

    def test_create_raster_dataset(self):
        data, coords = georef.set_raster_origin(
            self.data.copy(), self.coords.copy(), "upper"
        )
        ds = georef.create_raster_dataset(
            data, coords, projection=self.proj, nodata=-32768
        )

        data, coords, proj = georef.extract_raster_dataset(ds)
        np.testing.assert_array_equal(data, self.data)
        np.testing.assert_array_almost_equal(coords, self.coords)
        assert proj.ExportToWkt() == self.proj.ExportToWkt()

        data, coords = georef.set_raster_origin(
            self.data2.copy(), self.coords2.copy(), "upper"
        )
        ds = georef.create_raster_dataset(
            data, coords, projection=self.proj, nodata=-32768
        )

        data, coords, proj = georef.extract_raster_dataset(ds, mode="edge")
        np.testing.assert_array_equal(data, self.data2)
        np.testing.assert_array_almost_equal(coords, self.coords2)
        assert proj.ExportToWkt() == self.proj.ExportToWkt()

    def test_set_raster_origin(self):
        testfunc = georef.set_raster_origin
        data, coords = testfunc(self.data.copy(), self.coords.copy(), "upper")
        np.testing.assert_array_equal(data, self.data)
        np.testing.assert_array_equal(coords, self.coords)
        data, coords = testfunc(self.data.copy(), self.coords.copy(), "lower")
        np.testing.assert_array_equal(data, np.flip(self.data, axis=-2))
        np.testing.assert_array_equal(coords, np.flip(self.coords, axis=-3))

        data, coords = testfunc(
            self.data.copy()[:, :3600], self.coords.copy()[:3600, :3600], "lower"
        )
        np.testing.assert_array_equal(data, np.flip(self.data[:, :3600], axis=-2))

        np.testing.assert_array_equal(
            coords, np.flip(self.coords[:3600, :3600], axis=-3)
        )

    def test_set_raster_indexing(self):
        data, coords = georef.set_raster_origin(
            self.data.copy(), self.coords.copy(), "lower"
        )
        data, coords = georef.set_raster_indexing(data, coords, "ij")
        np.testing.assert_array_equal(
            data, np.swapaxes(np.flip(self.data, axis=-2), 0, 1)
        )
        np.testing.assert_array_equal(
            coords, np.swapaxes(np.flip(self.coords, axis=-3), 0, 1)
        )
        data, coords = georef.set_raster_indexing(data, coords, "xy")
        np.testing.assert_array_equal(data, np.flip(self.data, axis=-2))
        np.testing.assert_array_equal(coords, np.flip(self.coords, axis=-3))

    def test_set_coordinate_indexing(self):
        coords = georef.set_coordinate_indexing(self.coords.copy(), "ij")
        np.testing.assert_array_equal(coords, np.swapaxes(self.coords, 0, 1))
        coords = georef.set_coordinate_indexing(self.coords.copy(), "xy")
        np.testing.assert_array_equal(coords, self.coords)

    def test_extract_raster_dataset(self):
        ds = self.ds
        data, coords, proj = georef.extract_raster_dataset(ds)
        assert coords.shape[-1] == 2
        data, coords, proj = georef.extract_raster_dataset(ds, mode="edge")
        assert coords.shape[-1] == 2

    def test_get_raster_elevation(self):
        georef.get_raster_elevation(self.ds, download={"region": "Eurasia"})

    def test_get_raster_extent(self):

        extent = georef.get_raster_extent(self.ds2)
        window_map = np.array([3e5, 1e6, 3e5, 1e6])
        np.testing.assert_array_almost_equal(extent, window_map, decimal=3)

        extent = georef.get_raster_extent(self.ds2, geo=True)
        window_geo = np.array([-0.925465, 9.6641599, 47.4160381, 53.6928559])
        np.testing.assert_array_almost_equal(extent, window_geo)

        extent = georef.get_raster_extent(self.ds2, window=False)
        np.testing.assert_array_almost_equal(extent, self.corner_gdalinfo, decimal=3)

        extent = georef.get_raster_extent(self.ds2, geo=True, window=False)
        np.testing.assert_array_almost_equal(extent, self.corner_geo_gdalinfo)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="known break on windows")
    def test_merge_raster_datasets(self):
        download = {"region": "Eurasia"}
        datasets = wradlib.io.dem.get_srtm(
            [3, 4, 47, 48], merge=False, download=download
        )
        georef.merge_raster_datasets(datasets)

    def test_raster_to_polyvert(self):
        ds = self.ds
        polyvert = georef.raster_to_polyvert(ds)
        nx = ds.RasterXSize
        ny = ds.RasterYSize
        assert polyvert.shape == (ny, nx, 5, 2)


class TestGetGrids:
    # calculate xy and lonlat grids with georef function
    radolan_grid_xy = georef.get_radolan_grid(900, 900, trig=True)
    radolan_grid_ll = georef.get_radolan_grid(900, 900, trig=True, wgs84=True)

    def test_get_radolan_grid_equality(self):
        # create radolan projection osr object
        scale = (1.0 + np.sin(np.radians(60.0))) / (1.0 + np.sin(np.radians(90.0)))
        dwd_string = (
            "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 "
            "+k={0:10.8f} +x_0=0 +y_0=0 +a=6370040 +b=6370040 "
            "+to_meter=1000 +no_defs".format(scale)
        )
        proj_stereo = georef.proj4_to_osr(dwd_string)

        # create wgs84 projection osr object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        # transform radolan polar stereographic projection to wgs84 and wgs84
        # to polar stereographic
        # using osr transformation routines
        radolan_grid_ll = georef.reproject(
            self.radolan_grid_xy,
            projection_source=proj_stereo,
            projection_target=proj_wgs,
        )
        radolan_grid_xy = georef.reproject(
            self.radolan_grid_ll,
            projection_source=proj_wgs,
            projection_target=proj_stereo,
        )

        # check source and target arrays for equality
        np.testing.assert_allclose(radolan_grid_ll, self.radolan_grid_ll)
        np.testing.assert_allclose(radolan_grid_xy, self.radolan_grid_xy)

        radolan_grid_xy = georef.get_radolan_grid(900, 900)
        radolan_grid_ll = georef.get_radolan_grid(900, 900, wgs84=True)

        # check source and target arrays for equality
        np.testing.assert_allclose(radolan_grid_ll, self.radolan_grid_ll)
        np.testing.assert_allclose(radolan_grid_xy, self.radolan_grid_xy)

    def test_get_radolan_grid_raises(self):
        with pytest.raises(TypeError):
            georef.get_radolan_grid("900", "900")
        with pytest.raises(ValueError):
            georef.get_radolan_grid(2000, 2000)

    def test_get_radolan_grid_shape(self):
        radolan_grid_xy = georef.get_radolan_grid()
        assert radolan_grid_xy.shape == (900, 900, 2)

    def test_radolan_coords(self):
        x, y = georef.get_radolan_coords(7.0, 53.0)
        assert pytest.approx(x) == -208.15159184860158
        assert pytest.approx(y) == -3971.7689758313813
        # Also test with trigonometric approach
        x, y = georef.get_radolan_coords(7.0, 53.0, trig=True)
        assert x == -208.15159184860175
        assert y == -3971.7689758313832

    def test_xyz_to_spherical(self):
        xyz = np.array([[1000, 1000, 1000]])
        r, phi, theta = georef.xyz_to_spherical(xyz)
        assert pytest.approx(r[0]) == 1732.11878135
        assert pytest.approx(phi[0]) == 45.0
        assert pytest.approx(theta[0]) == 35.25802956


class TestSatellite:
    if has_data:
        f = "gpm/2A-CS-151E24S154E30S.GPM.Ku.V7-20170308.20141206-S095002-E095137.004383.V05A.HDF5"  # noqa
        gpm_file = util.get_wradlib_data_file(f)
        pr_data = wradlib.io.read_generic_hdf5(gpm_file)
        pr_lon = pr_data["NS/Longitude"]["data"]
        pr_lat = pr_data["NS/Latitude"]["data"]
        zenith = pr_data["NS/PRE/localZenithAngle"]["data"]
        wgs84 = georef.get_default_projection()
        a = wgs84.GetSemiMajor()
        b = wgs84.GetSemiMinor()
        rad = georef.proj4_to_osr(
            (
                "+proj=aeqd +lon_0={lon:f} " + "+lat_0={lat:f} +a={a:f} +b={b:f}" + ""
            ).format(lon=pr_lon[68, 0], lat=pr_lat[68, 0], a=a, b=b)
        )
        pr_x, pr_y = georef.reproject(
            pr_lon, pr_lat, projection_source=wgs84, projection_target=rad
        )
        re = georef.get_earth_radius(pr_lat[68, 0], wgs84) * 4.0 / 3.0
        pr_xy = np.dstack((pr_x, pr_y))
        alpha = zenith
        zt = 407000.0
        dr = 125.0
        bw_pr = 0.71
        nbin = 176
        nray = pr_lon.shape[1]

        pr_out = np.array(
            [
                [
                    [
                        [-58533.78453556, 124660.60390174],
                        [-58501.33048429, 124677.58873852],
                    ],
                    [
                        [-53702.13393133, 127251.83656509],
                        [-53670.98686161, 127268.11882882],
                    ],
                ],
                [
                    [
                        [-56444.00788528, 120205.5374491],
                        [-56411.55421163, 120222.52300741],
                    ],
                    [
                        [-51612.2360682, 122796.78620764],
                        [-51581.08938314, 122813.06920719],
                    ],
                ],
            ]
        )
        r_out = np.array(
            [0.0, 125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0, 1000.0, 1125.0]
        )
        z_out = np.array(
            [
                0.0,
                119.51255112,
                239.02510224,
                358.53765337,
                478.05020449,
                597.56275561,
                717.07530673,
                836.58785786,
                956.10040898,
                1075.6129601,
            ]
        )

    @requires_data
    def test_correct_parallax(self):
        xy, r, z = georef.correct_parallax(self.pr_xy, self.nbin, self.dr, self.alpha)
        pr_out = np.array(
            [
                [
                    [
                        [-16582.50734831, 35678.47219358],
                        [-16547.94607589, 35696.40777009],
                    ],
                    [
                        [-11742.02016667, 38252.32622057],
                        [-11708.84553319, 38269.52268457],
                    ],
                ],
                [
                    [
                        [-14508.62005182, 31215.98689653],
                        [-14474.05905935, 31233.92329553],
                    ],
                    [
                        [-9667.99183645, 33789.86576047],
                        [-9634.81750708, 33807.06305397],
                    ],
                ],
            ]
        )
        r_out = np.array(
            [0.0, 125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0, 1000.0, 1125.0]
        )
        z_out = np.array(
            [
                0.0,
                118.78164113,
                237.56328225,
                356.34492338,
                475.1265645,
                593.90820563,
                712.68984675,
                831.47148788,
                950.25312901,
                1069.03477013,
            ]
        )

        np.testing.assert_allclose(xy[60:62, 0:2, 0:2, :], pr_out, rtol=1e-12)
        np.testing.assert_allclose(r[0:10], r_out, rtol=1e-12)
        np.testing.assert_allclose(z[0, 0, 0:10], z_out, rtol=1e-10)

    @requires_data
    def test_dist_from_orbit(self):
        beta = abs(-17.04 + np.arange(self.nray) * self.bw_pr)
        xy, r, z = georef.correct_parallax(self.pr_xy, self.nbin, self.dr, self.alpha)
        dists = georef.dist_from_orbit(self.zt, self.alpha, beta, r, re=self.re)
        bd = np.array(
            [
                426553.58667772,
                426553.50342119,
                426553.49658156,
                426553.51025979,
                426553.43461609,
                426553.42515894,
                426553.46559985,
                426553.37020786,
                426553.44407286,
                426553.42173696,
            ]
        )
        sd = np.array(
            [
                426553.58667772,
                424895.63462839,
                423322.25176564,
                421825.47714885,
                420405.9414294,
                419062.44208923,
                417796.86827302,
                416606.91482435,
                415490.82582636,
                414444.11587979,
            ]
        )
        np.testing.assert_allclose(dists[0:10, 0, 0], bd, rtol=1e-12)
        np.testing.assert_allclose(dists[0, 0:10, 0], sd, rtol=1e-12)


class TestVector:
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(31466)
    wgs84 = georef.get_default_projection()

    npobj = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5630100.0],
            [2600100.0, 5630100.0],
            [2600100.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )
    lonlat = np.array(
        [
            [7.41779154, 50.79679579],
            [7.41781875, 50.79769443],
            [7.4192367, 50.79767718],
            [7.41920947, 50.79677854],
            [7.41779154, 50.79679579],
        ]
    )

    ogrobj = georef.numpy_to_ogr(npobj, "Polygon")
    ogrobj.AssignSpatialReference(None)
    projobj = georef.numpy_to_ogr(npobj, "Polygon")
    projobj.AssignSpatialReference(proj)

    # filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
    # ds, layer = wradlib.io.open_vector(filename)

    def test_ogr_create_layer(self):
        ds = wradlib.io.gdal_create_dataset("Memory", "test", gdal_type=gdal.OF_VECTOR)
        with pytest.raises(TypeError):
            georef.ogr_create_layer(ds, "test")
        lyr = georef.ogr_create_layer(
            ds, "test", geom_type=ogr.wkbPoint, fields=[("test", ogr.OFTReal)]
        )
        assert isinstance(lyr, ogr.Layer)

    def test_ogr_to_numpy(self):
        np.testing.assert_allclose(georef.ogr_to_numpy(self.ogrobj), self.npobj)

    def test_get_vector_points(self):
        # this also tests equality with `ogr_to_numpy`
        x = np.array(list(georef.get_vector_points(self.ogrobj))[0])
        y = georef.ogr_to_numpy(self.ogrobj)
        np.testing.assert_allclose(x, y)

    def test_get_vector_points_warning(self):
        point_wkt = "POINT (1198054.34 648493.09)"
        point = ogr.CreateGeometryFromWkt(point_wkt)
        with pytest.warns(UserWarning):
            list(georef.get_vector_points(point))

    @requires_data
    def test_get_vector_coordinates(self):
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        ds, layer = wradlib.io.open_vector(filename)

        # this also tests equality with `ogr_to_numpy`
        x, attrs = georef.get_vector_coordinates(layer, key="FID")
        assert attrs == list(range(13))

        x, attrs = georef.get_vector_coordinates(layer)
        y = []
        layer.ResetReading()
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetNextFeature()
            if feature:
                geom = feature.GetGeometryRef()
                y.append(georef.ogr_to_numpy(geom))
        y = np.array(y)
        for x1, y1 in zip(x, y):
            np.testing.assert_allclose(x1, y1)

        layer.ResetReading()
        x, attrs = georef.get_vector_coordinates(
            layer, source_srs=self.proj, dest_srs=self.wgs84
        )

        layer.ResetReading()
        x, attrs = georef.get_vector_coordinates(layer, dest_srs=self.wgs84)

    def test_transform_geometry(self):
        geom = georef.transform_geometry(self.projobj, dest_srs=self.wgs84)
        x = list(georef.get_vector_points(geom))[0]
        np.testing.assert_allclose(x, self.lonlat, rtol=1e-05)

    def test_transform_geometry_warning(self):
        with pytest.warns(UserWarning):
            georef.transform_geometry(self.ogrobj, dest_srs=self.wgs84)

    @requires_data
    def test_ogr_copy_layer(self):
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        src_ds, layer = wradlib.io.open_vector(filename)
        ds = wradlib.io.gdal_create_dataset("Memory", "test", gdal_type=gdal.OF_VECTOR)
        georef.ogr_copy_layer(src_ds, 0, ds)
        assert isinstance(ds.GetLayer(), ogr.Layer)

    @requires_data
    def test_ogr_copy_layer_by_name(self):
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        src_ds, layer = wradlib.io.open_vector(filename)
        ds = wradlib.io.gdal_create_dataset("Memory", "test", gdal_type=gdal.OF_VECTOR)
        georef.ogr_copy_layer_by_name(src_ds, "agger_merge", ds)
        assert isinstance(ds.GetLayer(), ogr.Layer)
        with pytest.raises(ValueError):
            georef.ogr_copy_layer_by_name(src_ds, "agger_merge1", ds)

    def test_ogr_add_feature(self):
        ds = wradlib.io.gdal_create_dataset("Memory", "test", gdal_type=gdal.OF_VECTOR)
        georef.ogr_create_layer(
            ds, "test", geom_type=ogr.wkbPoint, fields=[("index", ogr.OFTReal)]
        )

        point = np.array([1198054.34, 648493.09])
        parr = np.array([point, point, point])
        georef.ogr_add_feature(ds, parr)
        georef.ogr_add_feature(ds, parr, name="test")

    def test_ogr_add_geometry(self):
        ds = wradlib.io.gdal_create_dataset("Memory", "test", gdal_type=gdal.OF_VECTOR)
        lyr = georef.ogr_create_layer(
            ds, "test", geom_type=ogr.wkbPoint, fields=[("test", ogr.OFTReal)]
        )
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(1198054.34, 648493.09)
        georef.ogr_add_geometry(lyr, point, [42.42])

    def test_geocol_to_numpy(self):
        # Create a geometry collection
        geomcol = ogr.Geometry(ogr.wkbGeometryCollection)

        # Create polygon
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(1179091.1646903288, 712782.8838459781)
        ring.AddPoint(1161053.0218226474, 667456.2684348812)
        ring.AddPoint(1214704.933941905, 641092.8288590391)
        ring.AddPoint(1228580.428455506, 682719.3123998424)
        ring.AddPoint(1218405.0658121984, 721108.1805541387)
        ring.AddPoint(1179091.1646903288, 712782.8838459781)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        geomcol.AddGeometry(poly)

        # Add a point
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(-122.23, 47.09)
        geomcol.AddGeometry(point)

        # Add a line
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(-122.60, 47.14)
        line.AddPoint(-122.48, 47.23)
        geomcol.AddGeometry(line)

        arr = georef.ogr_geocol_to_numpy(geomcol)[..., 0:2]

        res = np.array(
            [
                [1179091.1646903288, 712782.8838459781],
                [1161053.0218226474, 667456.2684348812],
                [1214704.933941905, 641092.8288590391],
                [1228580.428455506, 682719.3123998424],
                [1218405.0658121984, 721108.1805541387],
                [1179091.1646903288, 712782.8838459781],
            ]
        )

        np.testing.assert_allclose(arr, res)

    def test_get_centroid(self):
        cent1 = georef.get_centroid(self.npobj)
        cent2 = georef.get_centroid(self.ogrobj)

        assert cent1 == (2600050.0, 5630050.0)
        assert cent2 == (2600050.0, 5630050.0)


class TestXarray:
    func = georef.create_xarray_dataarray
    da = func(
        np.random.rand(360, 1000),
        r=np.arange(0.0, 100000.0, 100.0),
        phi=np.arange(0.0, 360.0),
        theta=np.ones(360) * 1.0,
        site=(9.0, 48.0, 100.0),
        proj=True,
        sweep_mode="azimuth_surveillance",
    )
    da = georef.georeference_dataset(da)

    def test_create_xarray_dataarray(self):
        img = np.zeros((360, 10), dtype=np.float32)
        r = np.arange(0, 100000, 10000)
        az = np.arange(0, 360)
        th = np.zeros_like(az)
        proj = georef.epsg_to_osr(4326)
        with pytest.raises(TypeError):
            georef.create_xarray_dataarray(img)
        georef.create_xarray_dataarray(img, r, az, th, proj=proj)

    def test_georeference_dataset(self):
        src_da = self.da.copy()
        src_da.drop(["x", "y", "z", "gr", "rays", "bins"])
        da = georef.georeference_dataset(src_da)
        xr.testing.assert_equal(self.da, da)
