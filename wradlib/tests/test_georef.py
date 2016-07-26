# !/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import wradlib.georef as georef
import numpy as np
from osgeo import osr


class CoordinateTransformTest(unittest.TestCase):
    def setUp(self):
        self.r = np.array([0., 0., 111., 111., 111., 111.]) * 1000
        self.az = np.array([0., 180., 0., 90., 180., 270.])
        self.th = np.array([0., 0., 0., 0., 0., 0.5])
        self.csite = (9.0, 48.0)
        self.result = tuple(
            (np.array([9., 9., 9., 10.49189531, 9., 7.50810469]),
             np.array([48., 48., 48.99839742, 47.99034027, 47.00160258,
                       47.99034027]),
             np.array([0., 0., 967.03198482, 967.03198482, 967.03198482,
                       1935.45679527])))
        self.result_n = tuple(
            (np.array([9., 9., 9., 10.49189531, 9., 7.50810469]),
             np.array([48., 48., 48.99839742, 47.99034027, 47.00160258,
                       47.99034027]),
             np.array([0., 0., 725.7159843, 725.7159843, 725.7159843,
                       1694.22337134])))

    def test_hor2aeq(self):
        self.assertTrue(np.allclose(georef.hor2aeq(0.25, 0.5, 0.75),
                                    (-0.29983281824238966,
                                     0.22925926995789672)))

    def test_aeq2hor(self):
        self.assertTrue(np.allclose(
            georef.aeq2hor(0.22925926995789672, -0.29983281824238966, 0.75),
            (0.25, 0.5)))

    def test_polar2lonlat(self):
        self.assertTrue(
            np.allclose(georef.polar2lonlat(self.r, self.az, self.csite),
                        self.result[:2]))

    def test_polar2lonlatalt(self):
        self.assertTrue(np.allclose(
            georef.polar2lonlatalt(self.r, self.az, self.th, self.csite),
            self.result, rtol=1e-03))

    def test_polar2lonlatalt_n(self):
        self.assertTrue(np.allclose(
            georef.polar2lonlatalt_n(self.r, self.az, self.th, self.csite),
            self.result_n, rtol=1e-04))

    def test__latscale(self):
        self.assertEqual(georef._latscale(), 111178.17148373958)

    def test__lonscale(self):
        self.assertTrue(
            np.allclose(georef._lonscale(np.arange(-90., 90., 10.)),
                        np.array(
                            [6.80769959e-12, 1.93058869e+04, 3.80251741e+04,
                             5.55890857e+04,
                             7.14639511e+04, 8.51674205e+04, 9.62831209e+04,
                             1.04473307e+05,
                             1.09489125e+05, 1.11178171e+05, 1.09489125e+05,
                             1.04473307e+05,
                             9.62831209e+04, 8.51674205e+04, 7.14639511e+04,
                             5.55890857e+04,
                             3.80251741e+04, 1.93058869e+04])))

    def test_beam_height_n(self):
        self.assertTrue(np.allclose(
            georef.beam_height_n(np.arange(10., 101., 10.) * 1000., 2.),
            np.array([354.87448647, 721.50702113, 1099.8960815,
                      1490.04009656, 1891.93744678, 2305.58646416,
                      2730.98543223, 3168.13258613, 3617.02611263,
                      4077.66415017])))

    def test_arc_distance_n(self):
        self.assertTrue(np.allclose(
            georef.arc_distance_n(np.arange(10., 101., 10.) * 1000., 2.),
            np.array(
                [9993.49302358, 19986.13717891, 29977.90491409, 39968.76869178,
                 49958.70098959, 59947.6743006, 69935.66113377, 79922.63401441,
                 89908.5654846, 99893.4281037])))


class CoordinateHelperTest(unittest.TestCase):
    def test_centroid2polyvert(self):
        self.assertTrue(
            np.allclose(georef.centroid2polyvert([0., 1.], [0.5, 1.5]),
                        np.array([[-0.5, -0.5],
                                  [-0.5, 2.5],
                                  [0.5, 2.5],
                                  [0.5, -0.5],
                                  [-0.5, -0.5]])))

        self.assertTrue(np.allclose(
            georef.centroid2polyvert(np.arange(4).reshape((2, 2)), 0.5),
            np.array([[[-0.5, 0.5],
                       [-0.5, 1.5],
                       [0.5, 1.5],
                       [0.5, 0.5],
                       [-0.5, 0.5]],
                      [[1.5, 2.5],
                       [1.5, 3.5],
                       [2.5, 3.5],
                       [2.5, 2.5],
                       [1.5, 2.5]]])))

    def test_polar2polyvert(self):
        self.assertTrue(np.allclose(
            georef.polar2polyvert(np.array([10000., 10100.]),
                                  np.array([45., 90.]), (9., 48.)),
            np.array([[[9.05100794, 48.08225674],
                       [9.051524, 48.0830875],
                       [9.12427234, 48.03435375],
                       [9.12302879, 48.03401088],
                       [9.05100794, 48.08225674]],
                      [[9.051524, 48.0830875],
                       [9.05204008, 48.08391826],
                       [9.12551589, 48.03469661],
                       [9.12427234, 48.03435375],
                       [9.051524, 48.0830875]],
                      [[9.12302879, 48.03401088],
                       [9.12427234, 48.03435375],
                       [9.051524, 48.0830875],
                       [9.05100794, 48.08225674],
                       [9.12302879, 48.03401088]],
                      [[9.12427234, 48.03435375],
                       [9.12551589, 48.03469661],
                       [9.05204008, 48.08391826],
                       [9.051524, 48.0830875],
                       [9.12427234, 48.03435375]]])))

    def test_polar2centroids(self):
        r = np.array([10000., 10100.])
        az = np.array([45., 90.])
        sitecoords = (9., 48.)
        self.assertTrue(np.allclose(georef.polar2centroids(r, az, sitecoords),
                                    tuple((np.array([[9.09469143, 9.09564428],
                                                     [9.13374952,
                                                      9.13509373]]),
                                           np.array(
                                               [[48.06324434, 48.06387957],
                                                [47.99992237, 47.9999208]])))))

    def test_sweep_centroids(self):
        self.assertTrue(np.allclose(georef.sweep_centroids(1, 100., 1, 2.0),
                                    np.array([[[50., 3.14159265, 2.]]])))

    def test__check_polar_coords(self):
        r = np.array([50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        r = np.array([0, 50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        r = np.array([100., 50., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        r = np.array([50., 100., 125., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        r = np.array([50., 100., 150., 200.])
        az = np.array([0., 45., 90., 135., 180., 225., 270., 315., 361.])
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        r = np.array([50., 100., 150., 200.])
        az = np.array([225., 270., 315., 0., 45., 90., 135., 180.])[::-1]
        self.assertRaises(ValueError,
                          lambda: georef._check_polar_coords(r, az))

        # only py3k
        # r = np.array([50., 100., 150., 200.])
        # az = np.array([10., 45., 90., 135., 180., 225., 270., 315.])
        # self.assertWarns(UserWarning,
        #                  lambda: georef._check_polar_coords(r, az))


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

    def test_get_earth_radius(self):
        self.assertEqual(georef.get_earth_radius(50.), 6365631.51753728)

    def test_reproject(self):
        proj_gk = osr.SpatialReference()
        proj_gk.ImportFromEPSG(31466)
        proj_wgs84 = osr.SpatialReference()
        proj_wgs84.ImportFromEPSG(4326)
        x, y = georef.reproject(7., 53., projection_source=proj_wgs84,
                                projection_target=proj_gk)
        lon, lat = georef.reproject(x, y, projection_source=proj_gk,
                                    projection_target=proj_wgs84)
        self.assertAlmostEqual(lon, 7.0)
        self.assertAlmostEqual(lat, 53.0)

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
    def test_read_gdal_coordinates(self):
        pass

    def test_read_gdal_projection(self):
        pass

    def test_read_gdal_values(self):
        pass

    def test_resample_raster_dataset(self):
        pass

    def test_get_shape_points(self):
        pass

    def test_transform_geometry(self):
        pass

    def test_get_shape_coordinates(self):
        pass


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

    def test_get_radolan_grid_raises(self):
        self.assertRaises(TypeError,
                          lambda: georef.get_radolan_grid('900', '900'))
        self.assertRaises(ValueError,
                          lambda: georef.get_radolan_grid(2000, 2000))

    def test_get_radolan_grid_shape(self):
        radolan_grid_xy = georef.get_radolan_grid()
        self.assertEqual((900, 900, 2), radolan_grid_xy.shape)

    def test_radolan_coords(self):
        x, y = georef.get_radolan_coords(7.0, 53.0)
        self.assertEqual(x, -208.15159184860158)
        self.assertEqual(y, -3971.7689758313813)


if __name__ == '__main__':
    unittest.main()
