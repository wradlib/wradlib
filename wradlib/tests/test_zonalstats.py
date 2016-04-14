#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import tempfile

import wradlib.georef as georef
import wradlib.zonalstats as zonalstats
import wradlib.util as util
import numpy as np
from osgeo import osr, ogr, gdal


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
        self.assertRaises(IOError,
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

    def test_dump_vector(self):
        self.ds.dump_vector(tempfile.NamedTemporaryFile(mode='w+b').name)

    def test_dump_raster(self):
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(31466)
        filename = util.get_wradlib_data_file('shapefiles/agger/'
                                              'agger_merge.shp')
        test = zonalstats.DataSource(filename, proj)
        self.assertRaises(AttributeError, test.dump_raster(
            tempfile.NamedTemporaryFile(mode='w+b').name, 'netCDF',
            pixel_size=100.))


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
        # create polar grid polygon vertices in lat,lon
        radar_ll = georef.polar2polyvert(r, a, (lon, lat))

        # create polar grid centroids in lat,lon
        rlon, rlat = georef.polar2centroids(r, a, (lon, lat))
        radar_llc = np.dstack((rlon, rlat))

        # setup OSR objects
        self.proj_gk = osr.SpatialReference()
        self.proj_gk.ImportFromEPSG(31466)
        self.proj_ll = osr.SpatialReference()
        self.proj_ll.ImportFromEPSG(4326)

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
                                         [2600000., 5630057.83273596],
                                         [2600018.65014816, 5630000.],
                                         [2600000., 5630000.]]),
                               np.array([[2600000., 5630057.83273596],
                                         [2600000., 5630100.],
                                         [2600091.80406488, 5630100.],
                                         [2600100., 5630074.58501104],
                                         [2600100., 5630000.],
                                         [2600018.65014816, 5630000.],
                                         [2600000., 5630057.83273596]]),
                               np.array([[2600091.80406488, 5630100.],
                                         [2600100., 5630100.],
                                         [2600100., 5630074.58501104],
                                         [2600091.80406488, 5630100.]])])
        isec_poly1 = np.array([np.array([[2600100., 5630000.],
                                         [2600100., 5630074.58501104],
                                         [2600124.05249566, 5630000.],
                                         [2600100., 5630000.]]),
                               np.array([[2600100., 5630074.58501104],
                                         [2600100., 5630100.],
                                         [2600197.20644071, 5630100.],
                                         [2600200., 5630091.33737992],
                                         [2600200., 5630000.],
                                         [2600124.05249566, 5630000.],
                                         [2600100., 5630074.58501104]]),
                               np.array([[2600197.20644071, 5630100.],
                                         [2600200., 5630100.],
                                         [2600200., 5630091.33737992],
                                         [2600197.20644071, 5630100.]])])

        isec_point0 = np.array([[2600062.31245173, 5630031.20266055]])
        isec_point1 = np.array([[2600157.8352244, 5630061.85098382]])

        self.isec_poly = np.array([isec_poly0, isec_poly1])
        self.isec_point = np.array([isec_point0, isec_point1])

    def test_srs(self):
        self.assertEqual(self.zdpoly.srs, self.proj_gk)
        self.assertEqual(self.zdpoint.srs, self.proj_gk)

    def test_isecs(self):
        self.assertEqual(self.zdpoly.isecs.__str__(), self.isec_poly.__str__())
        self.assertEqual(self.zdpoint.isecs.__str__(),
                         self.isec_point.__str__())

    def test_get_isec(self):
        self.assertEqual(self.zdpoly.get_isec(0).__str__(),
                         self.isec_poly[0].__str__())
        self.assertEqual(self.zdpoly.get_isec(1).__str__(),
                         self.isec_poly[1].__str__())
        self.assertEqual(self.zdpoint.get_isec(0).__str__(),
                         self.isec_point[0].__str__())
        self.assertEqual(self.zdpoint.get_isec(1).__str__(),
                         self.isec_point[1].__str__())

    def test_get_source_index(self):
        self.assertTrue(np.allclose(self.zdpoly.get_source_index(0),
                                    np.array([2254, 2255, 2256])))
        self.assertTrue(np.allclose(self.zdpoly.get_source_index(1),
                                    np.array([2255, 2256, 2257])))
        self.assertTrue(
            np.allclose(self.zdpoint.get_source_index(0), np.array([2255])))
        self.assertTrue(
            np.allclose(self.zdpoint.get_source_index(1), np.array([2256])))


class ZonalStatsTest(unittest.TestCase):
    # TODO: create tests for ZonalStatsBase class and descendants
    pass


class ZonalStatsUtilTest(unittest.TestCase):
    def setUp(self):
        self.npobj = np.array([[2600000., 5630000.], [2600000., 5630100.],
                               [2600100., 5630100.], [2600100., 5630000.],
                               [2600000., 5630000.]])

        self.ogrobj = zonalstats.numpy_to_ogr(self.npobj, 'Polygon')

    def test_gdal_create_dataset(self):
        ds = zonalstats.gdal_create_dataset('GTiff', 'test.tif', 100, 100,
                                            gdal.GDT_Float32)
        del ds
        ds = zonalstats.gdal_create_dataset('GTiff', 'test.tif', 100, 100,
                                            gdal.GDT_Float32, remove=True)
        self.assertTrue(isinstance(ds, gdal.Dataset))
        self.assertRaises(IOError,
                          lambda: zonalstats
                          .gdal_create_dataset('GXF',
                                               'test.gxf',
                                               100, 100,
                                               gdal.GDT_Float32))

    def test_ogr_create_datasource(self):
        ds = zonalstats.ogr_create_datasource('Memory', 'test')
        self.assertTrue(isinstance(ds, ogr.DataSource))

    def test_ogr_create_layer(self):
        ds = zonalstats.ogr_create_datasource('Memory', 'test')
        self.assertRaises(TypeError,
                          lambda: zonalstats.ogr_create_layer(ds, 'test'))
        lyr = zonalstats.ogr_create_layer(ds, 'test', geom_type=ogr.wkbPoint,
                                          fields=[('test', ogr.OFTReal)])
        self.assertTrue(isinstance(lyr, ogr.Layer))

    def test_ogr_to_numpy(self):
        self.assertTrue(
            np.allclose(zonalstats.ogr_to_numpy(self.ogrobj), self.npobj))

    def test_angle_between(self):
        self.assertAlmostEqual(zonalstats.angle_between(355., 5.), 10.)
        self.assertAlmostEqual(zonalstats.angle_between(5., 355.), -10.)


if __name__ == '__main__':
    unittest.main()
