#!/usr/bin/env python
# Copyright (c) 2011-2018-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import tempfile

import wradlib.georef as georef
import wradlib.zonalstats as zonalstats
import wradlib.util as util
import numpy as np
from osgeo import osr

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                    precision=8, suppress=False, threshold=1000,
                    formatter=None)


class DataSourceTest(unittest.TestCase):
    def setUp(self):
        # create synthetic box
        self.box0 = np.array([[2600000., 5630000.], [2600000., 5640000.],
                              [2610000., 5640000.], [2610000., 5630000.],
                              [2600000., 5630000.]])

        self.box1 = np.array([[2700000., 5630000.], [2700000., 5640000.],
                              [2710000., 5640000.], [2710000., 5630000.],
                              [2700000., 5630000.]])

        self.data = np.array([self.box0, self.box1])

        self.ds = zonalstats.DataSource(self.data)

        self.values1 = np.array([47.11, 47.11])
        self.values2 = np.array([47.11, 15.08])

    def test__check_src(self):
        filename = util.get_wradlib_data_file('shapefiles/agger/'
                                              'agger_merge.shp')
        self.assertEqual(len(zonalstats.DataSource(filename).data), 13)
        self.assertRaises(RuntimeError,
                          lambda: zonalstats.DataSource('test_zonalstats.py'))

    def test_data(self):
        self.assertTrue(np.allclose(self.ds.data, self.data))

    def test__get_data(self):
        self.assertTrue(np.allclose(self.ds._get_data(), self.data))

    def test_get_data_by_idx(self):
        self.assertTrue(np.allclose(self.ds.get_data_by_idx([0]), self.box0))
        self.assertTrue(np.allclose(self.ds.get_data_by_idx([1]), self.box1))
        self.assertTrue(
            np.allclose(self.ds.get_data_by_idx([0, 1]), self.data))

    def test_get_data_by_att(self):
        self.assertTrue(
            np.allclose(self.ds.get_data_by_att('index', 0), self.box0))
        self.assertTrue(
            np.allclose(self.ds.get_data_by_att('index', 1), self.box1))

    def test_get_data_by_geom(self):
        lyr = self.ds.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        for i, feature in enumerate(lyr):
            geom = feature.GetGeometryRef()
            self.assertTrue(
                np.allclose(self.ds.get_data_by_geom(geom), self.data[i]))

    def test_set_attribute(self):
        self.ds.set_attribute('test', self.values1)
        self.assertTrue(
            np.allclose(self.ds.get_attributes(['test']), self.values1))
        self.ds.set_attribute('test', self.values2)
        self.assertTrue(
            np.allclose(self.ds.get_attributes(['test']), self.values2))

    def test_get_attributes(self):
        self.ds.set_attribute('test', self.values2)
        self.assertEqual(self.ds.get_attributes(['test'], filt=('index', 0)),
                         self.values2[0])
        self.assertEqual(self.ds.get_attributes(['test'], filt=('index', 1)),
                         self.values2[1])

    def test_get_geom_properties(self):
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(31466)
        filename = util.get_wradlib_data_file('shapefiles/agger/'
                                              'agger_merge.shp')
        test = zonalstats.DataSource(filename, proj)
        np.testing.assert_array_equal(
            [[76722499.98474795]],
            test.get_geom_properties(['Area'],
                                     filt=('FID', 1)))

    def test_dump_vector(self):
        self.ds.dump_vector(tempfile.NamedTemporaryFile(mode='w+b').name)

    def test_dump_raster(self):
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(31466)
        filename = util.get_wradlib_data_file('shapefiles/agger/'
                                              'agger_merge.shp')
        test = zonalstats.DataSource(filename, proj)
        test.dump_raster(tempfile.NamedTemporaryFile(mode='w+b').name,
                         driver='netCDF', pixel_size=100.)
        test.dump_raster(tempfile.NamedTemporaryFile(mode='w+b').name,
                         driver='netCDF', pixel_size=100., attr='FID')


@unittest.skipIf(not util.has_geos(), "GDAL without GEOS")
class ZonalDataBaseTest(unittest.TestCase):
    def setUp(self):

        # GK3-Projection
        self.proj = osr.SpatialReference()
        self.proj.ImportFromEPSG(31466)

        # create synthetic box
        self.box0 = np.array([[2600000., 5630000.], [2600000., 5640000.],
                              [2610000., 5640000.], [2610000., 5630000.],
                              [2600000., 5630000.]])

        self.box1 = np.array([[2610000., 5630000.], [2610000., 5640000.],
                              [2620000., 5640000.], [2620000., 5630000.],
                              [2610000., 5630000.]])

        self.box3 = np.array([[2595000., 5625000.], [2595000., 5635000.],
                              [2605000., 5635000.], [2605000., 5625000.],
                              [2595000., 5625000.]])

        self.box4 = np.array([[2615000., 5635000.], [2615000., 5645000.],
                              [2625000., 5645000.], [2625000., 5635000.],
                              [2615000., 5635000.]])

        self.box5 = np.array([[2600000., 5635000.], [2605000., 5635000.],
                              [2605000., 5630000.], [2600000., 5630000.],
                              [2600000., 5635000.]])

        self.box6 = np.array([[2615000., 5635000.], [2615000., 5640000.],
                              [2620000., 5640000.], [2620000., 5635000.],
                              [2615000., 5635000.]])

        self.box7 = np.array([[2715000., 5635000.], [2715000., 5640000.],
                              [2720000., 5640000.], [2720000., 5635000.],
                              [2715000., 5635000.]])

        self.src = np.array([self.box0, self.box1])
        self.trg = np.array([self.box3, self.box4])
        self.dst = np.array([[self.box5], [self.box6]])
        self.zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.f = tempfile.NamedTemporaryFile(mode='w+b').name
        self.zdb.dump_vector(self.f)

    def test___init__(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.assertIsInstance(zdb.src, zonalstats.DataSource)
        self.assertIsInstance(zdb.trg, zonalstats.DataSource)
        self.assertIsInstance(zdb.dst, zonalstats.DataSource)
        self.assertEqual(zdb._count_intersections, 2)

    def test_coun_intersections(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.assertEqual(zdb.count_intersections, 2)

    def test_srs(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.assertEqual(zdb.srs, self.proj)

    def test_isecs(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        np.testing.assert_equal(zdb.isecs, self.dst)

    def test_get_isec(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        np.testing.assert_equal(zdb.get_isec(0), [self.box5])
        np.testing.assert_equal(zdb.get_isec(1), [self.box6])

    def test_get_source_index(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.assertEqual(zdb.get_source_index(0), 0)
        self.assertEqual(zdb.get_source_index(1), 1)

    def test_dump_vector(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        f = tempfile.NamedTemporaryFile(mode='w+b').name
        zdb.dump_vector(f)

    def test_load_vector(self):
        zonalstats.ZonalDataBase(self.f)

    def test__get_intersection(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.assertRaises(TypeError, lambda: zdb._get_intersection())
        np.testing.assert_equal(zdb._get_intersection(trg=self.box3),
                                [self.box5])
        np.testing.assert_equal(zdb._get_intersection(idx=0),
                                [self.box5])
        self.assertRaises(TypeError,
                          lambda: zdb._get_intersection(idx=2))
        zdb = zonalstats.ZonalDataBase(self.src, [self.box7], srs=self.proj)
        zdb.trg = None
        self.assertRaises(TypeError,
                          lambda: zdb._get_intersection(idx=0))


@unittest.skipIf(not util.has_geos(), "GDAL without GEOS")
class ZonalDataPolyTest(unittest.TestCase):
    def setUp(self):

        # GK3-Projection
        self.proj = osr.SpatialReference()
        self.proj.ImportFromEPSG(31466)

        # create synthetic box
        self.box0 = np.array([[2600000., 5630000.], [2600000., 5640000.],
                              [2610000., 5640000.], [2610000., 5630000.],
                              [2600000., 5630000.]])

        self.box1 = np.array([[2610000., 5630000.], [2610000., 5640000.],
                              [2620000., 5640000.], [2620000., 5630000.],
                              [2610000., 5630000.]])

        self.box3 = np.array([[2595000., 5625000.], [2595000., 5635000.],
                              [2605000., 5635000.], [2605000., 5625000.],
                              [2595000., 5625000.]])

        self.box4 = np.array([[2615000., 5635000.], [2615000., 5645000.],
                              [2625000., 5645000.], [2625000., 5635000.],
                              [2615000., 5635000.]])

        self.box5 = np.array([[2600000., 5635000.], [2605000., 5635000.],
                              [2605000., 5630000.], [2600000., 5630000.],
                              [2600000., 5635000.]])

        self.box6 = np.array([[2615000., 5635000.], [2615000., 5640000.],
                              [2620000., 5640000.], [2620000., 5635000.],
                              [2615000., 5635000.]])

        self.box7 = np.array([[2715000., 5635000.], [2715000., 5640000.],
                              [2720000., 5640000.], [2720000., 5635000.],
                              [2715000., 5635000.]])

        self.src = np.array([self.box0, self.box1])
        self.trg = np.array([self.box3, self.box4])
        self.dst = np.array([[self.box5], [self.box6]])
        self.zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.f = tempfile.NamedTemporaryFile(mode='w+b').name
        self.zdb.dump_vector(self.f)

    def test__get_idx_weights(self):
        zdp = zonalstats.ZonalDataPoly(self.src, self.trg, srs=self.proj)
        self.assertEqual(zdp._get_idx_weights(),
                         ([np.array([0]), np.array([1])],
                          [np.array([25000000.]), np.array([25000000.])]))


@unittest.skipIf(not util.has_geos(), "GDAL without GEOS")
class ZonalDataPointTest(unittest.TestCase):
    def setUp(self):

        # GK3-Projection
        self.proj = osr.SpatialReference()
        self.proj.ImportFromEPSG(31466)

        # create synthetic box
        self.point0 = np.array([2600000., 5630000.])

        self.point1 = np.array([2620000., 5640000.])

        self.box3 = np.array([[2595000., 5625000.], [2595000., 5635000.],
                              [2605000., 5635000.], [2605000., 5625000.],
                              [2595000., 5625000.]])

        self.box4 = np.array([[2615000., 5635000.], [2615000., 5645000.],
                              [2625000., 5645000.], [2625000., 5635000.],
                              [2615000., 5635000.]])

        self.box5 = np.array([[2600000., 5635000.], [2605000., 5635000.],
                              [2605000., 5630000.], [2600000., 5630000.],
                              [2600000., 5635000.]])

        self.box6 = np.array([[2615000., 5635000.], [2615000., 5640000.],
                              [2620000., 5640000.], [2620000., 5635000.],
                              [2615000., 5635000.]])

        self.box7 = np.array([[2715000., 5635000.], [2715000., 5640000.],
                              [2720000., 5640000.], [2720000., 5635000.],
                              [2715000., 5635000.]])

        self.src = np.array([self.point0, self.point1])
        self.trg = np.array([self.box3, self.box4])
        self.dst = np.array([[self.point0], [self.point1]])
        self.zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.f = tempfile.NamedTemporaryFile(mode='w+b').name
        self.zdb.dump_vector(self.f)

    def test__get_idx_weights(self):
        zdp = zonalstats.ZonalDataPoint(self.src, self.trg, srs=self.proj)
        print(zdp._get_idx_weights())
        self.assertEqual(zdp._get_idx_weights(),
                         ([np.array([0]), np.array([1])],
                          [np.array([1.]), np.array([1.])]))


@unittest.skipIf(not util.has_geos(), "GDAL without GEOS")
class ZonalStatsBaseTest(unittest.TestCase):
    def setUp(self):

        # GK3-Projection
        self.proj = osr.SpatialReference()
        self.proj.ImportFromEPSG(31466)

        # create synthetic box
        self.box0 = np.array([[2600000., 5630000.], [2600000., 5640000.],
                              [2610000., 5640000.], [2610000., 5630000.],
                              [2600000., 5630000.]])

        self.box1 = np.array([[2610000., 5630000.], [2610000., 5640000.],
                              [2620000., 5640000.], [2620000., 5630000.],
                              [2610000., 5630000.]])

        self.box3 = np.array([[2595000., 5625000.], [2595000., 5635000.],
                              [2605000., 5635000.], [2605000., 5625000.],
                              [2595000., 5625000.]])

        self.box4 = np.array([[2615000., 5635000.], [2615000., 5645000.],
                              [2625000., 5645000.], [2625000., 5635000.],
                              [2615000., 5635000.]])

        self.box5 = np.array([[2600000., 5635000.], [2605000., 5635000.],
                              [2605000., 5630000.], [2600000., 5630000.],
                              [2600000., 5635000.]])

        self.box6 = np.array([[2615000., 5635000.], [2615000., 5640000.],
                              [2620000., 5640000.], [2620000., 5635000.],
                              [2615000., 5635000.]])

        self.box7 = np.array([[2715000., 5635000.], [2715000., 5640000.],
                              [2720000., 5640000.], [2720000., 5635000.],
                              [2715000., 5635000.]])

        self.src = np.array([self.box0, self.box1])
        self.trg = np.array([self.box3, self.box4])
        self.dst = np.array([[self.box5], [self.box6]])
        self.zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        self.zdp = zonalstats.ZonalDataPoly(self.src, self.trg, srs=self.proj)

    def test__init__(self):
        self.assertRaises(NotImplementedError,
                          lambda: zonalstats.ZonalStatsBase(self.zdb))
        zonalstats.ZonalStatsBase(self.zdp)
        self.assertRaises(TypeError, lambda: zonalstats.ZonalStatsBase('test'))
        self.assertRaises(TypeError, lambda: zonalstats.ZonalStatsBase())
        self.assertRaises(TypeError,
                          lambda: zonalstats.ZonalStatsBase(ix=np.arange(10),
                                                            w=np.arange(11)))

    def test_w(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.w, np.array([[25000000.], [25000000.]]))
        np.testing.assert_equal(zdp.ix, np.array([[0], [1]]))

    def test__check_vals(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        self.assertRaises(AssertionError,
                          lambda: zdp._check_vals(np.arange(3)))

    def test_mean(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.mean(np.arange(10, 21, 10)),
                                np.array([10, 20]))

    def test_var(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.var(np.arange(10, 21, 10)),
                                np.array([0, 0]))


@unittest.skipIf(not util.has_geos(), "GDAL without GEOS")
class ZonalDataTest(unittest.TestCase):
    def setUp(self):
        global skip
        # setup test grid and catchment
        lon = 7.071664
        lat = 50.730521
        r = np.array(range(50, 100 * 1000 + 50, 100))
        a = np.array(range(0, 360, 1))
        rays = a.shape[0]
        bins = r.shape[0]

        # setup OSR objects
        self.proj_gk = osr.SpatialReference()
        self.proj_gk.ImportFromEPSG(31466)
        self.proj_ll = osr.SpatialReference()
        self.proj_ll.ImportFromEPSG(4326)

        # create polar grid polygon vertices in lat,lon
        radar_ll = georef.spherical_to_polyvert(r, a, 0, (lon, lat),
                                                proj=self.proj_ll)[..., 0:2]

        # create polar grid centroids in lat,lon
        coords = georef.spherical_to_centroids(r, a, 0, (lon, lat),
                                               proj=self.proj_ll)

        radar_llc = coords[..., 0:2]

        # project ll grids to GK2
        self.radar_gk = georef.reproject(radar_ll,
                                         projection_source=self.proj_ll,
                                         projection_target=self.proj_gk)
        self.radar_gkc = georef.reproject(radar_llc,
                                          projection_source=self.proj_ll,
                                          projection_target=self.proj_gk)

        # reshape
        self.radar_gk.shape = (rays, bins, 5, 2)
        self.radar_gkc.shape = (rays, bins, 2)

        self.box0 = np.array([[2600000., 5630000.], [2600000., 5630100.],
                              [2600100., 5630100.], [2600100., 5630000.],
                              [2600000., 5630000.]])

        self.box1 = np.array([[2600100., 5630000.], [2600100., 5630100.],
                              [2600200., 5630100.], [2600200., 5630000.],
                              [2600100., 5630000.]])

        self.data = np.array([self.box0, self.box1])

        # create catchment bounding box
        buffer = 5000.
        bbox = zonalstats.get_bbox(self.data[..., 0], self.data[..., 1])
        bbox = dict(left=bbox['left'] - buffer, right=bbox['right'] + buffer,
                    bottom=bbox['bottom'] - buffer, top=bbox['top'] + buffer)

        mask, shape = zonalstats.mask_from_bbox(self.radar_gkc[..., 0],
                                                self.radar_gkc[..., 1],
                                                bbox,
                                                polar=True)

        self.radar_gkc = self.radar_gkc[mask, :]
        self.radar_gk = self.radar_gk[mask]

        self.zdpoly = zonalstats.ZonalDataPoly(self.radar_gk, self.data,
                                               srs=self.proj_gk)
        # self.zdpoly.dump_vector('test_zdpoly')
        self.zdpoint = zonalstats.ZonalDataPoint(self.radar_gkc, self.data,
                                                 srs=self.proj_gk)
        # self.zdpoint.dump_vector('test_zdpoint')

        isec_poly0 = np.array([np.array([[2600000., 5630000.],
                                         [2600000., 5630100.],
                                         [2600009.61157242, 5630100.],
                                         [2600041.77844048, 5630000.],
                                         [2600000., 5630000.]]),
                               np.array([[2600009.61157242, 5630100.],
                                         [2600100., 5630100.],
                                         [2600100., 5630000.],
                                         [2600041.77844048, 5630000.],
                                         [2600009.61157242, 5630100.]]),
                               np.array([[2600091.80406488, 5630100.],
                                         [2600100., 5630100.],
                                         [2600100., 5630074.58501104],
                                         [2600091.80406488, 5630100.]])])
        isec_poly1 = np.array([np.array([[2600100., 5630000.],
                                         [2600100., 5630100.],
                                         [2600114.66582085, 5630100.],
                                         [2600146.83254704, 5630000.],
                                         [2600100., 5630000.]]),
                               np.array([[2600114.66582085, 5630100.],
                                         [2600200., 5630100.],
                                         [2600200., 5630000.],
                                         [2600146.83254704, 5630000.],
                                         [2600114.66582085, 5630100.]]),
                               np.array([[2600197.20644071, 5630100.],
                                         [2600200., 5630100.],
                                         [2600200., 5630091.33737992],
                                         [2600197.20644071, 5630100.]])])

        isec_point0 = np.array([[2600077.2899581, 5630056.0874306]])
        isec_point1 = np.array([[2600172.498418, 5630086.7127034]])

        self.isec_poly = np.array([isec_poly0, isec_poly1])
        self.isec_point = np.array([isec_point0, isec_point1])

    def test_srs(self):
        self.assertEqual(self.zdpoly.srs, self.proj_gk)
        self.assertEqual(self.zdpoint.srs, self.proj_gk)

    def test_isecs(self):
        # need to iterate over nested array for correct testing
        for i in range(len(self.zdpoly.isecs)):
            for k in range(len(self.zdpoly.isecs[i])):
                np.testing.assert_array_almost_equal(self.zdpoly.isecs[i, k],
                                                     self.isec_poly[i, k],
                                                     decimal=7)

        np.testing.assert_array_almost_equal(self.zdpoint.isecs,
                                             self.isec_point, decimal=7)

    def test_get_isec(self):
        for i in [0, 1]:
            for k, arr in enumerate(self.zdpoly.get_isec(i)):
                np.testing.assert_array_almost_equal(arr,
                                                     self.isec_poly[i, k],
                                                     decimal=7)
            np.testing.assert_array_almost_equal(self.zdpoint.get_isec(i),
                                                 self.isec_point[i], decimal=7)

    def test_get_source_index(self):
        np.testing.assert_array_equal(self.zdpoly.get_source_index(0),
                                      np.array([2254, 2255]))
        np.testing.assert_array_equal(self.zdpoly.get_source_index(1),
                                      np.array([2255, 2256]))
        np.testing.assert_array_equal(self.zdpoint.get_source_index(0),
                                      np.array([2255]))
        np.testing.assert_array_equal(self.zdpoint.get_source_index(1),
                                      np.array([2256]))


class ZonalStatsUtilTest(unittest.TestCase):
    def setUp(self):
        self.npobj = np.array([[2600000., 5630000.], [2600000., 5630100.],
                               [2600100., 5630100.], [2600100., 5630000.],
                               [2600000., 5630000.]])

        self.ogrobj = georef.numpy_to_ogr(self.npobj, 'Polygon')

    def test_angle_between(self):
        self.assertAlmostEqual(zonalstats.angle_between(355., 5.), 10.)
        self.assertAlmostEqual(zonalstats.angle_between(5., 355.), -10.)

    def test_get_clip_mask(self):
        proj_gk = osr.SpatialReference()
        proj_gk.ImportFromEPSG(31466)
        coords = np.array([[2600020., 5630020.], [2600030., 5630030.],
                           [2600040., 5630040.], [2700100., 5630030.],
                           [2600040., 5640000.]])
        mask = zonalstats.get_clip_mask(coords, self.npobj, proj_gk)
        out = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, out)


if __name__ == '__main__':
    unittest.main()
