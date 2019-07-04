# !/usr/bin/env python
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import wradlib.georef as georef
import wradlib.util as util
from wradlib.io import (read_generic_hdf5, open_raster, gdal_create_dataset,
                        open_vector)
import numpy as np
from osgeo import gdal, osr, ogr

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                    precision=8, suppress=False, threshold=1000,
                    formatter=None)


class CoordinateTransformTest(unittest.TestCase):
    def setUp(self):
        self.r = np.array([0., 0., 111., 111., 111., 111.]) * 1000
        self.az = np.array([0., 180., 0., 90., 180., 270.])
        self.th = np.array([0., 0., 0., 0., 0., 0.5])
        self.csite = (9.0, 48.0, 0)
        self.result_xyz = tuple(
            (np.array([0., 0., 0., 110993.6738, 0., -110976.7856]),
             np.array([0., -0., 110993.6738, 0., -110976.7856, -0.]),
             np.array([0., 0., 725.7159843, 725.7159843, 725.7159843,
                       1694.22337134])))
        self.result = tuple(
            (np.array([9., 9., 9., 10.49189531, 9., 7.50810469]),
             np.array([48., 48., 48.99839742, 47.99034027, 47.00160258,
                       47.99034027]),
             np.array([0., 0., 967.03198482, 967.03198482, 967.03198482,
                       1935.45679527])))
        self.result_n = tuple(
            (np.array([9., 9., 9., 10.48716091, 9., 7.51306531]),
             np.array([48., 48., 48.99814438, 47.99037251, 47.00168131,
                       47.99037544]),
             np.array([0., 0., 725.7159843, 725.7159843, 725.7159843,
                       1694.22337134])))

    def test_spherical_to_xyz(self):
        self.assertTrue((1, 36, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10),
                                                np.arange(36),
                                                10., self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((1, 36, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10), np.arange(36),
                                                np.arange(36), self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((36, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10), np.arange(36),
                                                np.arange(36), self.csite,
                                                squeeze=True,
                                                strict_dims=False)[0].shape)
        self.assertTrue((36, 36, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10), np.arange(36),
                                                np.arange(36), self.csite,
                                                strict_dims=True)[0].shape)
        self.assertTrue((18, 36, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10), np.arange(36),
                                                np.arange(18), self.csite,
                                                strict_dims=False)[0].shape)
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        self.assertTrue((1, 36, 10, 3) ==
                        georef.spherical_to_xyz(r, phi, 10, self.csite,
                                                squeeze=False,
                                                strict_dims=False)[0].shape)
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        self.assertTrue((1, 36, 10, 3) ==
                        georef.spherical_to_xyz(r, phi, np.arange(36),
                                                self.csite, squeeze=False,
                                                strict_dims=False)[0].shape)
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        self.assertTrue((18, 36, 10, 3) ==
                        georef.spherical_to_xyz(r, phi, np.arange(18),
                                                self.csite, squeeze=False,
                                                strict_dims=False)[0].shape)
        r, phi = np.meshgrid(np.arange(10), np.arange(36))
        self.assertTrue((36, 36, 10, 3) ==
                        georef.spherical_to_xyz(r, phi, np.arange(36),
                                                self.csite, squeeze=False,
                                                strict_dims=True)[0].shape)
        self.assertTrue((1, 1, 1, 3) ==
                        georef.spherical_to_xyz(10, 36, 10., self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((1, 1, 10, 3) ==
                        georef.spherical_to_xyz(np.arange(10), 36, 10.,
                                                self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((1, 36, 1, 3) ==
                        georef.spherical_to_xyz(10, np.arange(36), 10.,
                                                self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((10, 1, 1, 3) ==
                        georef.spherical_to_xyz(10, 36., np.arange(10),
                                                self.csite,
                                                squeeze=False)[0].shape)
        self.assertTrue((10, 36, 1, 3) ==
                        georef.spherical_to_xyz(10, np.arange(36),
                                                np.arange(10),
                                                self.csite,
                                                squeeze=False)[0].shape)

        coords, rad = georef.spherical_to_xyz(self.r, self.az,
                                              self.th, self.csite,
                                              squeeze=True, strict_dims=False)
        self.assertTrue(np.allclose(coords[..., 0], self.result_xyz[0],
                        rtol=1e-03))
        self.assertTrue(np.allclose(coords[..., 1], self.result_xyz[1],
                        rtol=1e-03))
        self.assertTrue(np.allclose(coords[..., 2], self.result_xyz[2],
                        rtol=1e-03))
        re = georef.get_earth_radius(self.csite[1])
        coords, rad = georef.spherical_to_xyz(self.r, self.az,
                                              self.th, self.csite,
                                              re=re, squeeze=True)
        self.assertTrue(np.allclose(coords[..., 0], self.result_xyz[0],
                                    rtol=1e-03))
        self.assertTrue(np.allclose(coords[..., 1], self.result_xyz[1],
                                    rtol=1e-03))
        self.assertTrue(np.allclose(coords[..., 2], self.result_xyz[2],
                                    rtol=1e-03))

    def test_bin_altitude(self):
        altitude = georef.bin_altitude(np.arange(10., 101., 10.)
                                       * 1000., 2., 0, 6370040.)
        altref = np.array([354.87448647, 721.50702113, 1099.8960815,
                           1490.04009656, 1891.93744678, 2305.58646416,
                           2730.98543223, 3168.13258613, 3617.02611263,
                           4077.66415017])
        np.testing.assert_allclose(altref, altitude)

    def test_bin_distance(self):
        distance = georef.bin_distance(np.arange(10., 101., 10.) * 1000., 2.,
                                       0, 6370040.)
        distref = np.array([9993.49302358, 19986.13717891, 29977.90491409,
                            39968.76869178, 49958.70098959, 59947.6743006,
                            69935.66113377, 79922.63401441, 89908.5654846,
                            99893.4281037])
        np.testing.assert_allclose(distref, distance)

    def test_site_distance(self):
        altitude = georef.bin_altitude(np.arange(10., 101., 10.) * 1000., 2.,
                                       0, 6370040.)
        distance = georef.site_distance(np.arange(10., 101., 10.) * 1000., 2.,
                                        altitude, 6370040.)
        distref = np.array([9993.49302358, 19986.13717891, 29977.90491409,
                            39968.76869178, 49958.70098959, 59947.6743006,
                            69935.66113377, 79922.63401441, 89908.5654846,
                            99893.4281037])
        np.testing.assert_allclose(distref, distance)

    def test_spherical_to_proj(self):
        coords = georef.spherical_to_proj(self.r, self.az,
                                          self.th, self.csite)
        self.assertTrue(np.allclose(coords[..., 0], self.result_n[0]))
        self.assertTrue(np.allclose(coords[..., 1], self.result_n[1]))
        self.assertTrue(np.allclose(coords[..., 2], self.result_n[2]))


class CoordinateHelperTest(unittest.TestCase):
    def test_centroid_to_polyvert(self):
        np.testing.assert_array_equal(
            georef.centroid_to_polyvert(np.array([0., 1.]),
                                        [0.5, 1.5]),
            np.array([[-0.5, -0.5],
                      [-0.5, 2.5],
                      [0.5, 2.5],
                      [0.5, -0.5],
                      [-0.5, -0.5]]))
        np.testing.assert_array_equal(
            georef.centroid_to_polyvert(np.arange(4).reshape((2, 2)), 0.5),
            np.array([[[-0.5, 0.5],
                       [-0.5, 1.5],
                       [0.5, 1.5],
                       [0.5, 0.5],
                       [-0.5, 0.5]],
                      [[1.5, 2.5],
                       [1.5, 3.5],
                       [2.5, 3.5],
                       [2.5, 2.5],
                       [1.5, 2.5]]]))
        with self.assertRaises(ValueError):
            georef.centroid_to_polyvert([[0.], [1.]], [0.5, 1.5])

    def test_spherical_to_polyvert(self):
        sph = georef.get_default_projection()
        polyvert = georef.spherical_to_polyvert(np.array([10000., 10100.]),
                                                np.array([45., 90.]), 0,
                                                (9., 48.), proj=sph)
        arr = np.asarray([[[9.05084865, 48.08224715, 6.],
                           [9.05136309, 48.0830778, 6.],
                           [9.1238846, 48.03435008, 6.],
                           [9.12264494, 48.03400725, 6.],
                           [9.05084865, 48.08224715, 6.]],
                          [[9.05136309, 48.0830778, 6.],
                           [9.05187756, 48.08390846, 6.],
                           [9.12512428, 48.03469291, 6.],
                           [9.1238846, 48.03435008, 6.],
                           [9.05136309, 48.0830778, 6.]],
                          [[9.12264494, 48.03400725, 6.],
                           [9.1238846, 48.03435008, 6.],
                           [9.05136309, 48.0830778, 6.],
                           [9.05084865, 48.08224715, 6.],
                           [9.12264494, 48.03400725, 6.]],
                          [[9.1238846, 48.03435008, 6.],
                           [9.12512428, 48.03469291, 6.],
                           [9.05187756, 48.08390846, 6.],
                           [9.05136309, 48.0830778, 6.],
                           [9.1238846, 48.03435008, 6.]]])
        np.testing.assert_array_almost_equal(polyvert, arr, decimal=3)
        polyvert, pr = georef.spherical_to_polyvert(np.array([10000., 10100.]),
                                                    np.array([45., 90.]), 0,
                                                    (9., 48.))
        arr = np.asarray([[[3.7885640e+03, 9.1464023e+03, 6.],
                           [3.8268320e+03, 9.2387900e+03, 6.],
                           [9.2387900e+03, 3.8268323e+03, 6.],
                           [9.1464023e+03, 3.7885645e+03, 6.],
                           [3.7885640e+03, 9.1464023e+03, 6.]],
                          [[3.8268320e+03, 9.2387900e+03, 6.],
                           [3.8651003e+03, 9.3311777e+03, 6.],
                           [9.3311777e+03, 3.8651006e+03, 6.],
                           [9.2387900e+03, 3.8268323e+03, 6.],
                           [3.8268320e+03, 9.2387900e+03, 6.]],
                          [[9.1464023e+03, 3.7885645e+03, 6.],
                           [9.2387900e+03, 3.8268323e+03, 6.],
                           [3.8268320e+03, 9.2387900e+03, 6.],
                           [3.7885640e+03, 9.1464023e+03, 6.],
                           [9.1464023e+03, 3.7885645e+03, 6.]],
                          [[9.2387900e+03, 3.8268323e+03, 6.],
                           [9.3311777e+03, 3.8651006e+03, 6.],
                           [3.8651003e+03, 9.3311777e+03, 6.],
                           [3.8268320e+03, 9.2387900e+03, 6.],
                           [9.2387900e+03, 3.8268323e+03, 6.]]])
        np.testing.assert_array_almost_equal(polyvert, arr, decimal=3)

    def test_spherical_to_centroids(self):
        r = np.array([10000., 10100.])
        az = np.array([45., 90.])
        sitecoords = (9., 48., 0.)
        sph = georef.get_default_projection()
        centroids = georef.spherical_to_centroids(r, az, 0, sitecoords,
                                                  proj=sph)
        arr = np.asarray([[[9.09439583, 48.06323717, 6.],
                           [9.09534571, 48.06387232, 6.]],
                          [[9.1333325, 47.99992262, 6.],
                           [9.13467253, 47.99992106, 6.]]])
        np.testing.assert_array_almost_equal(centroids, arr, decimal=3)

        centroids, pr = georef.spherical_to_centroids(r, az, 0, sitecoords)
        arr = np.asarray([[[7.0357090e+03, 7.0357090e+03, 6.],
                           [7.1064194e+03, 7.1064194e+03, 6.]],
                          [[9.9499951e+03, 0., 6.],
                           [1.0049995e+04, 0., 6.]]])
        np.testing.assert_array_almost_equal(centroids, arr, decimal=3)

    def test_sweep_centroids(self):
        self.assertTrue(np.allclose(georef.sweep_centroids(1, 100., 1, 2.0),
                                    np.array([[[50., 3.14159265, 2.]]])))

    def test__check_polar_coords(self):
        r = np.array([50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50., 100., 150., 200.])
        az = np.array([10., 45., 90., 135., 180., 225., 270., 315.])
        with self.assertWarns(UserWarning):
            georef.polar._check_polar_coords(r, az)

        r = np.array([0, 50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50., 100., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([100., 50., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50., 100., 125., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 361.])
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

        r = np.array([50., 100., 150., 200.])
        az = np.array([225., 270., 315., 0., 45., 90., 135., 180.])[::-1]
        with self.assertRaises(ValueError):
            georef.polar._check_polar_coords(r, az)

    def test__get_range_resolution(self):
        r = np.array([50.])
        with self.assertRaises(ValueError):
            georef.polar._get_range_resolution(r)
        r = np.array([50., 100., 150., 190., 250.])
        with self.assertRaises(ValueError):
            georef.polar._get_range_resolution(r)

    def test__get_azimuth_resolution(self):
        az = np.array([0., 45., 90., 135., 180., 224., 270., 315.])
        with self.assertRaises(ValueError):
            georef.polar._get_azimuth_resolution(az)


class ProjectionsTest(unittest.TestCase):
    def test_create_osr(self):
        self.maxDiff = None
        radolan_wkt = ('PROJCS["Radolan projection",'
                       'GEOGCS["Radolan Coordinate System",'
                       'DATUM["Radolan Kugel",'
                       'SPHEROID["Erdkugel",6370040.0,0.0]],'
                       'PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],'
                       'UNIT["degree",0.017453292519943295],'
                       'AXIS["Longitude",EAST],'
                       'AXIS["Latitude",NORTH]],'
                       'PROJECTION["polar_stereographic"],'
                       'PARAMETER["central_meridian",10.0],'
                       'PARAMETER["latitude_of_origin",60.0],'
                       'PARAMETER["scale_factor",{0:8.10f}],'
                       'PARAMETER["false_easting",0.0],'
                       'PARAMETER["false_northing",0.0],'
                       'UNIT["m*1000.0",1000.0],'
                       'AXIS["X",EAST],'
                       'AXIS["Y",NORTH]]'.
                       format((1. + np.sin(np.radians(60.))) /
                              (1. + np.sin(np.radians(90.)))))
        self.assertEqual(georef.create_osr('dwd-radolan').ExportToWkt(),
                         radolan_wkt)
        aeqd_wkt = ('PROJCS["unnamed",'
                    'GEOGCS["WGS 84",'
                    'DATUM["unknown",'
                    'SPHEROID["WGS84",6378137,298.257223563]],'
                    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                    'PROJECTION["Azimuthal_Equidistant"],'
                    'PARAMETER["latitude_of_center",{0:-f}],'
                    'PARAMETER["longitude_of_center",{1:-f}],'
                    'PARAMETER["false_easting",{2:-f}],'
                    'PARAMETER["false_northing",{3:-f}]]'.
                    format(49., 5., 0, 0))

        self.assertEqual(georef.create_osr('aeqd',
                                           lon_0=5.,
                                           lat_0=49).ExportToWkt(),
                         aeqd_wkt)
        self.assertEqual(georef.create_osr('aeqd',
                                           lon_0=5.,
                                           lat_0=49,
                                           x_0=0,
                                           y_0=0).ExportToWkt(),
                         aeqd_wkt)
        with self.assertRaises(ValueError):
            georef.create_osr('lambert')

    def test_proj4_to_osr(self):
        srs = georef.proj4_to_osr('+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=0 '
                                  '+k_0=0.99987742 +x_0=600000 +y_0=2200000 '
                                  '+a=6378249.2 +b=6356515 '
                                  '+towgs84=-168,-60,320,0,0,0,0 '
                                  '+pm=paris +units=m +no_defs')
        p4 = srs.ExportToProj4()
        srs2 = osr.SpatialReference()
        srs2.ImportFromProj4(p4)
        self.assertTrue(srs.IsSame(srs2))
        with self.assertRaises(ValueError):
            georef.proj4_to_osr("+proj=lcc1")

    def test_get_earth_radius(self):
        self.assertEqual(georef.get_earth_radius(50.), 6365631.51753728)

    def test_reproject(self):
        proj_gk = osr.SpatialReference()
        proj_gk.ImportFromEPSG(31466)
        proj_wgs84 = osr.SpatialReference()
        proj_wgs84.ImportFromEPSG(4326)
        x, y, z = georef.reproject(7., 53., 0., projection_source=proj_wgs84,
                                   projection_target=proj_gk)
        lon, lat = georef.reproject(x, y, projection_source=proj_gk,
                                    projection_target=proj_wgs84)
        self.assertAlmostEqual(lon, 7.0)
        self.assertAlmostEqual(lat, 53.0)

        lonlat = georef.reproject(np.stack((x, y), axis=-1),
                                  projection_source=proj_gk,
                                  projection_target=proj_wgs84)
        self.assertAlmostEqual(lonlat[0], 7.0)
        self.assertAlmostEqual(lonlat[1], 53.0)

        with self.assertRaises(TypeError):
            georef.reproject(np.stack((x, y, x, y), axis=-1))

        lon, lat, alt = georef.reproject(x, y, z, projection_source=proj_gk,
                                         projection_target=proj_wgs84)
        self.assertAlmostEqual(lon, 7., places=5)
        self.assertAlmostEqual(lat, 53., places=3)
        self.assertAlmostEqual(alt, 0., places=3)

        with self.assertRaises(TypeError):
            georef.reproject(x, y, x, y)
        with self.assertRaises(TypeError):
            georef.reproject([np.arange(10)], [np.arange(11)])
        with self.assertRaises(TypeError):
            georef.reproject([np.arange(10)], [np.arange(10)],
                             [np.arange(11)])

    def test_get_default_projection(self):
        self.assertEqual(georef.get_default_projection().ExportToWkt(),
                         ('GEOGCS["WGS 84",DATUM["WGS_1984",'
                          'SPHEROID["WGS 84",6378137,298.257223563,'
                          'AUTHORITY["EPSG","7030"]],'
                          'AUTHORITY["EPSG","6326"]],'
                          'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                          'UNIT["degree",0.0174532925199433,'
                          'AUTHORITY["EPSG","9122"]],'
                          'AUTHORITY["EPSG","4326"]]'))

    def test_epsg_to_osr(self):
        self.assertEqual(georef.epsg_to_osr(4326).ExportToWkt(),
                         ('GEOGCS["WGS 84",DATUM["WGS_1984",'
                          'SPHEROID["WGS 84",6378137,298.257223563,'
                          'AUTHORITY["EPSG","7030"]],'
                          'AUTHORITY["EPSG","6326"]],'
                          'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                          'UNIT["degree",0.0174532925199433,'
                          'AUTHORITY["EPSG","9122"]],'
                          'AUTHORITY["EPSG","4326"]]'))

        self.assertEqual(georef.epsg_to_osr().ExportToWkt(),
                         ('GEOGCS["WGS 84",DATUM["WGS_1984",'
                          'SPHEROID["WGS 84",6378137,298.257223563,'
                          'AUTHORITY["EPSG","7030"]],'
                          'AUTHORITY["EPSG","6326"]],'
                          'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                          'UNIT["degree",0.0174532925199433,'
                          'AUTHORITY["EPSG","9122"]],'
                          'AUTHORITY["EPSG","4326"]]'))

    def test_wkt_to_osr(self):
        self.assertTrue(georef.wkt_to_osr('GEOGCS["WGS 84",DATUM["WGS_1984",'
                                          'SPHEROID["WGS 84",6378137,'
                                          '298.257223563,'
                                          'AUTHORITY["EPSG","7030"]],'
                                          'AUTHORITY["EPSG","6326"]],'
                                          'PRIMEM["Greenwich",0,'
                                          'AUTHORITY["EPSG","8901"]],'
                                          'UNIT["degree",0.0174532925199433,'
                                          'AUTHORITY["EPSG","9122"]],'
                                          'AUTHORITY["EPSG","4326"]]').IsSame(
            georef.get_default_projection()))

        self.assertTrue(
            georef.wkt_to_osr().IsSame(georef.get_default_projection()))


class PixMapTest(unittest.TestCase):
    def test_pixel_coordinates(self):
        pass

    def test_pixel_to_map(self):
        pass

    def test_pixel_to_map3d(self):
        pass


class GdalTests(unittest.TestCase):
    def setUp(self):
        filename = 'geo/bonn_new.tif'
        geofile = util.get_wradlib_data_file(filename)
        self.ds = open_raster(geofile)
        (self.data,
         self.coords,
         self.proj) = georef.extract_raster_dataset(self.ds)
        self.x_sp, self.y_sp = self.coords[1, 1] - self.coords[0, 0]

    def test_read_gdal_coordinates(self):
        coords = georef.read_gdal_coordinates(self.ds)
        self.assertEqual(coords.shape[-1], 3)
        coords = georef.read_gdal_coordinates(self.ds, z=False)
        self.assertEqual(coords.shape[-1], 2)

    def test_read_gdal_projection(self):
        georef.read_gdal_projection(self.ds)

    def test_read_gdal_values(self):
        georef.read_gdal_values(self.ds)
        georef.read_gdal_values(self.ds, nodata=9999.)

    def test_reproject_raster_dataset(self):
        georef.reproject_raster_dataset(self.ds, spacing=0.005,
                                        resample=gdal.GRA_Bilinear,
                                        align=True)
        georef.reproject_raster_dataset(self.ds, size=(1000, 1000),
                                        resample=gdal.GRA_Bilinear,
                                        align=True)
        with self.assertRaises(NameError):
            georef.reproject_raster_dataset(self.ds)
        dst = georef.epsg_to_osr(31466)
        georef.reproject_raster_dataset(self.ds, spacing=100.,
                                        resample=gdal.GRA_Bilinear,
                                        align=True, projection_target=dst)

    def test_create_raster_dataset(self):
        data, coords = georef.set_raster_origin(self.data.copy(),
                                                self.coords.copy(),
                                                'upper')
        ds = georef.create_raster_dataset(data,
                                          coords,
                                          projection=self.proj,
                                          nodata=-32768)

        data, coords, proj = georef.extract_raster_dataset(ds)
        np.testing.assert_array_equal(data, self.data)
        np.testing.assert_array_almost_equal(coords, self.coords)
        self.assertEqual(proj.ExportToWkt(), self.proj.ExportToWkt())

    def test_set_raster_origin(self):
        testfunc = georef.set_raster_origin
        data, coords = testfunc(self.data.copy(),
                                self.coords.copy(), 'upper')
        np.testing.assert_array_equal(data, self.data)
        np.testing.assert_array_equal(coords, self.coords)
        data, coords = testfunc(self.data.copy(),
                                self.coords.copy(), 'lower')
        np.testing.assert_array_equal(data, np.flip(self.data, axis=-2))
        np.testing.assert_array_equal(coords, np.flip(self.coords, axis=-3))

        data, coords = testfunc(self.data.copy()[:, :3600],
                                self.coords.copy()[:3600, :3600],
                                'lower')
        np.testing.assert_array_equal(data, np.flip(self.data[:, :3600],
                                                    axis=-2))

        np.testing.assert_array_equal(coords,
                                      np.flip(self.coords[:3600, :3600] +
                                              [0, self.y_sp], axis=-3))

    def test_extract_raster_dataset(self):
        data, coords, proj = georef.extract_raster_dataset(self.ds)


class GetGridsTest(unittest.TestCase):
    def setUp(self):
        # calculate xy and lonlat grids with georef function
        self.radolan_grid_xy = georef.get_radolan_grid(900, 900, trig=True)
        self.radolan_grid_ll = georef.get_radolan_grid(900, 900, trig=True,
                                                       wgs84=True)

    def test_get_radolan_grid_equality(self):
        # create radolan projection osr object
        scale = (1. + np.sin(np.radians(60.))) / (1. + np.sin(np.radians(90.)))
        dwd_string = ('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 '
                      '+k={0:10.8f} +x_0=0 +y_0=0 +a=6370040 +b=6370040 '
                      '+to_meter=1000 +no_defs'.format(scale))
        proj_stereo = georef.proj4_to_osr(dwd_string)

        # create wgs84 projection osr object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        # transform radolan polar stereographic projection to wgs84 and wgs84
        # to polar stereographic
        # using osr transformation routines
        radolan_grid_ll = georef.reproject(self.radolan_grid_xy,
                                           projection_source=proj_stereo,
                                           projection_target=proj_wgs)
        radolan_grid_xy = georef.reproject(self.radolan_grid_ll,
                                           projection_source=proj_wgs,
                                           projection_target=proj_stereo)

        # check source and target arrays for equality
        self.assertTrue(np.allclose(radolan_grid_ll, self.radolan_grid_ll))
        self.assertTrue(np.allclose(radolan_grid_xy, self.radolan_grid_xy))

        radolan_grid_xy = georef.get_radolan_grid(900, 900)
        radolan_grid_ll = georef.get_radolan_grid(900, 900, wgs84=True)

        # check source and target arrays for equality
        self.assertTrue(np.allclose(radolan_grid_ll, self.radolan_grid_ll))
        self.assertTrue(np.allclose(radolan_grid_xy, self.radolan_grid_xy))

    def test_get_radolan_grid_raises(self):
        with self.assertRaises(TypeError):
            georef.get_radolan_grid('900', '900')
        with self.assertRaises(ValueError):
            georef.get_radolan_grid(2000, 2000)

    def test_get_radolan_grid_shape(self):
        radolan_grid_xy = georef.get_radolan_grid()
        self.assertEqual((900, 900, 2), radolan_grid_xy.shape)

    def test_radolan_coords(self):
        x, y = georef.get_radolan_coords(7.0, 53.0)
        self.assertAlmostEqual(x, -208.15159184860158)
        self.assertAlmostEqual(y, -3971.7689758313813)
        # Also test with trigonometric approach
        x, y = georef.get_radolan_coords(7.0, 53.0, trig=True)
        self.assertEqual(x, -208.15159184860175)
        self.assertEqual(y, -3971.7689758313832)

    def test_xyz_to_spherical(self):
        xyz = np.array([[1000, 1000, 1000]])
        r, phi, theta = georef.xyz_to_spherical(xyz)
        self.assertAlmostEqual(r[0], 1732.11878135)
        self.assertAlmostEqual(phi[0], 45.)
        self.assertAlmostEqual(theta[0], 35.25802956)


class SatelliteTest(unittest.TestCase):
    def setUp(self):
        f = 'gpm/2A-CS-151E24S154E30S.GPM.Ku.V7-20170308.20141206-S095002-E095137.004383.V05A.HDF5'  # noqa
        gpm_file = util.get_wradlib_data_file(f)
        pr_data = read_generic_hdf5(gpm_file)
        pr_lon = pr_data['NS/Longitude']['data']
        pr_lat = pr_data['NS/Latitude']['data']
        zenith = pr_data['NS/PRE/localZenithAngle']['data']
        wgs84 = georef.get_default_projection()
        a = wgs84.GetSemiMajor()
        b = wgs84.GetSemiMinor()
        rad = georef.proj4_to_osr(('+proj=aeqd +lon_0={lon:f} ' +
                                   '+lat_0={lat:f} +a={a:f} +b={b:f}' +
                                   '').format(lon=pr_lon[68, 0],
                                              lat=pr_lat[68, 0],
                                              a=a, b=b))
        pr_x, pr_y = georef.reproject(pr_lon, pr_lat,
                                      projection_source=wgs84,
                                      projection_target=rad)
        self.re = georef.get_earth_radius(pr_lat[68, 0], wgs84) * 4. / 3.
        self.pr_xy = np.dstack((pr_x, pr_y))
        self.alpha = zenith
        self.zt = 407000.
        self.dr = 125.
        self.bw_pr = 0.71
        self.nbin = 176
        self.nray = pr_lon.shape[1]

        self.pr_out = np.array([[[[-58533.78453556, 124660.60390174],
                                  [-58501.33048429, 124677.58873852]],
                                 [[-53702.13393133, 127251.83656509],
                                  [-53670.98686161, 127268.11882882]]],
                                [[[-56444.00788528, 120205.5374491],
                                  [-56411.55421163, 120222.52300741]],
                                 [[-51612.2360682, 122796.78620764],
                                  [-51581.08938314, 122813.06920719]]]])
        self.r_out = np.array([0., 125., 250., 375., 500., 625., 750., 875.,
                               1000., 1125.])
        self.z_out = np.array([0., 119.51255112, 239.02510224, 358.53765337,
                               478.05020449, 597.56275561, 717.07530673,
                               836.58785786, 956.10040898, 1075.6129601])

    def test_correct_parallax(self):
        xy, r, z = georef.correct_parallax(self.pr_xy, self.nbin,
                                           self.dr, self.alpha)
        self.xyz = np.concatenate((xy, z[..., np.newaxis]), axis=-1)
        pr_out = np.array([[[[-16582.50734831, 35678.47219358],
                             [-16547.94607589, 35696.40777009]],
                            [[-11742.02016667, 38252.32622057],
                             [-11708.84553319, 38269.52268457]]],
                           [[[-14508.62005182, 31215.98689653],
                             [-14474.05905935, 31233.92329553]],
                            [[-9667.99183645, 33789.86576047],
                             [-9634.81750708, 33807.06305397]]]])
        r_out = np.array([0., 125., 250., 375., 500., 625., 750., 875.,
                          1000., 1125.])
        z_out = np.array([0., 118.78164113, 237.56328225, 356.34492338,
                          475.1265645, 593.90820563, 712.68984675,
                          831.47148788, 950.25312901, 1069.03477013])

        np.testing.assert_allclose(xy[60:62, 0:2, 0:2, :], pr_out, rtol=1e-12)
        np.testing.assert_allclose(r[0:10], r_out, rtol=1e-12)
        np.testing.assert_allclose(z[0, 0, 0:10], z_out, rtol=1e-10)

    def test_dist_from_orbit(self):
        beta = abs(-17.04 + np.arange(self.nray) * self.bw_pr)
        xy, r, z = georef.correct_parallax(self.pr_xy, self.nbin,
                                           self.dr, self.alpha)
        dists = georef.dist_from_orbit(self.zt, self.alpha, beta, r,
                                       re=self.re)
        bd = np.array([426553.58667772, 426553.50342119, 426553.49658156,
                       426553.51025979, 426553.43461609, 426553.42515894,
                       426553.46559985, 426553.37020786, 426553.44407286,
                       426553.42173696])
        sd = np.array([426553.58667772, 424895.63462839, 423322.25176564,
                       421825.47714885, 420405.9414294,  419062.44208923,
                       417796.86827302, 416606.91482435, 415490.82582636,
                       414444.11587979])
        np.testing.assert_allclose(dists[0:10, 0, 0], bd, rtol=1e-12)
        np.testing.assert_allclose(dists[0, 0:10, 0], sd, rtol=1e-12)


class VectorTest(unittest.TestCase):
    def setUp(self):
        self.proj = osr.SpatialReference()
        self.proj.ImportFromEPSG(31466)
        self.wgs84 = georef.get_default_projection()

        self.npobj = np.array([[2600000., 5630000.], [2600000., 5630100.],
                               [2600100., 5630100.], [2600100., 5630000.],
                               [2600000., 5630000.]])
        self.lonlat = np.array([[7.41779154, 50.79679579],
                                [7.41781875, 50.79769443],
                                [7.4192367, 50.79767718],
                                [7.41920947, 50.79677854],
                                [7.41779154, 50.79679579]])

        self.ogrobj = georef.numpy_to_ogr(self.npobj, 'Polygon')
        self.ogrobj.AssignSpatialReference(None)
        self.projobj = georef.numpy_to_ogr(self.npobj, 'Polygon')
        self.projobj.AssignSpatialReference(self.proj)

        filename = util.get_wradlib_data_file('shapefiles/agger/'
                                              'agger_merge.shp')
        self.ds, self.layer = open_vector(filename)

    def test_ogr_create_layer(self):
        ds = gdal_create_dataset('Memory', 'test',
                                 gdal_type=gdal.OF_VECTOR)
        with self.assertRaises(TypeError):
            georef.ogr_create_layer(ds, 'test')
        lyr = georef.ogr_create_layer(ds, 'test', geom_type=ogr.wkbPoint,
                                          fields=[('test', ogr.OFTReal)])
        self.assertTrue(isinstance(lyr, ogr.Layer))

    def test_ogr_to_numpy(self):
        self.assertTrue(
            np.allclose(georef.ogr_to_numpy(self.ogrobj), self.npobj))

    def test_get_vector_points(self):
        # this also tests equality with `ogr_to_numpy`
        x = np.array(list(georef.get_vector_points(self.ogrobj))[0])
        y = georef.ogr_to_numpy(self.ogrobj)
        np.testing.assert_allclose(x, y)

    def test_get_vector_points_warning(self):
        point_wkt = "POINT (1198054.34 648493.09)"
        point = ogr.CreateGeometryFromWkt(point_wkt)
        with self.assertWarns(UserWarning):
            list(georef.get_vector_points(point))

    def test_get_vector_coordinates(self):
        # this also tests equality with `ogr_to_numpy`

        x, attrs = georef.get_vector_coordinates(self.layer, key='FID')
        self.assertEqual(attrs, list(range(13)))

        x, attrs = georef.get_vector_coordinates(self.layer)
        y = []
        self.layer.ResetReading()
        for i in range(self.layer.GetFeatureCount()):
            feature = self.layer.GetNextFeature()
            if feature:
                geom = feature.GetGeometryRef()
                y.append(georef.ogr_to_numpy(geom))
        y = np.array(y)
        for x1, y1 in zip(x, y):
            np.testing.assert_allclose(x1, y1)

        self.layer.ResetReading()
        x, attrs = georef.get_vector_coordinates(self.layer,
                                                 source_srs=self.proj,
                                                 dest_srs=self.wgs84)

        self.layer.ResetReading()
        x, attrs = georef.get_vector_coordinates(self.layer,
                                                 dest_srs=self.wgs84)

    def test_transform_geometry(self):
        geom = georef.transform_geometry(self.projobj, dest_srs=self.wgs84)
        x = list(georef.get_vector_points(geom))[0]
        np.testing.assert_allclose(x, self.lonlat)

    def test_transform_geometry_warning(self):
        with self.assertWarns(UserWarning):
            georef.transform_geometry(self.ogrobj, dest_srs=self.wgs84)

    def test_ogr_copy_layer(self):
        ds = gdal_create_dataset('Memory', 'test',
                                 gdal_type=gdal.OF_VECTOR)
        georef.ogr_copy_layer(self.ds, 0, ds)
        self.assertTrue(isinstance(ds.GetLayer(), ogr.Layer))

    def test_ogr_copy_layer_by_name(self):
        ds = gdal_create_dataset('Memory', 'test',
                                 gdal_type=gdal.OF_VECTOR)
        georef.ogr_copy_layer_by_name(self.ds, 'agger_merge', ds)
        self.assertTrue(isinstance(ds.GetLayer(), ogr.Layer))
        with self.assertRaises(ValueError):
            georef.ogr_copy_layer_by_name(self.ds, 'agger_merge1', ds)

    def test_ogr_add_feature(self):
        ds = gdal_create_dataset('Memory', 'test',
                                 gdal_type=gdal.OF_VECTOR)
        georef.ogr_create_layer(ds, 'test', geom_type=ogr.wkbPoint,
                                fields=[('index', ogr.OFTReal)])

        point = np.array([1198054.34, 648493.09])
        parr = np.array([point, point, point])
        georef.ogr_add_feature(ds, parr)
        georef.ogr_add_feature(ds, parr, name='test')

    def test_ogr_add_geometry(self):
        ds = gdal_create_dataset('Memory', 'test',
                                 gdal_type=gdal.OF_VECTOR)
        lyr = georef.ogr_create_layer(ds, 'test', geom_type=ogr.wkbPoint,
                                      fields=[('test', ogr.OFTReal)])
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

        res = np.array([[1179091.1646903288, 712782.8838459781],
                        [1161053.0218226474, 667456.2684348812],
                        [1214704.933941905, 641092.8288590391],
                        [1228580.428455506, 682719.3123998424],
                        [1218405.0658121984, 721108.1805541387],
                        [1179091.1646903288, 712782.8838459781]])

        np.testing.assert_allclose(arr, res)

    def test_get_centroid(self):
        cent1 = georef.get_centroid(self.npobj)
        cent2 = georef.get_centroid(self.ogrobj)

        self.assertEqual(cent1, (2600050.0, 5630050.0))
        self.assertEqual(cent2, (2600050.0, 5630050.0))


if __name__ == '__main__':
    unittest.main()
