#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import tempfile

import numpy as np
import pytest
from osgeo import osr

from wradlib import georef, util, zonalstats

from . import requires_data

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


class TestDataSource:
    # create synthetic box
    box0 = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5640000.0],
            [2610000.0, 5640000.0],
            [2610000.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    box1 = np.array(
        [
            [2700000.0, 5630000.0],
            [2700000.0, 5640000.0],
            [2710000.0, 5640000.0],
            [2710000.0, 5630000.0],
            [2700000.0, 5630000.0],
        ]
    )

    data = np.array([box0, box1])

    ds = zonalstats.DataSource(data)

    values1 = np.array([47.11, 47.11])
    values2 = np.array([47.11, 15.08])

    @requires_data
    def test__check_src(self):
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        assert len(zonalstats.DataSource(filename).data) == 13
        with pytest.raises(RuntimeError):
            zonalstats.DataSource("test_zonalstats.py")

    def test_data(self):
        assert np.allclose(self.ds.data, self.data)

    def test__get_data(self):
        ds = zonalstats.DataSource(self.data)
        assert np.allclose(ds._get_data(), self.data)

    def test_get_data_by_idx(self):
        ds = zonalstats.DataSource(self.data)
        assert np.allclose(ds.get_data_by_idx([0]), self.box0)
        assert np.allclose(ds.get_data_by_idx([1]), self.box1)
        assert np.allclose(ds.get_data_by_idx([0, 1]), self.data)

    def test_get_data_by_att(self):
        ds = zonalstats.DataSource(self.data)
        assert np.allclose(ds.get_data_by_att("index", 0), self.box0)
        assert np.allclose(ds.get_data_by_att("index", 1), self.box1)

    def test_get_data_by_geom(self):
        ds = zonalstats.DataSource(self.data)
        lyr = ds.ds.GetLayer()
        lyr.ResetReading()
        lyr.SetSpatialFilter(None)
        lyr.SetAttributeFilter(None)
        for i, feature in enumerate(lyr):
            geom = feature.GetGeometryRef()
            assert np.allclose(ds.get_data_by_geom(geom), self.data[i])

    def test_set_attribute(self):
        ds = zonalstats.DataSource(self.data)
        ds.set_attribute("test", self.values1)
        assert np.allclose(ds.get_attributes(["test"]), self.values1)
        ds.set_attribute("test", self.values2)
        assert np.allclose(ds.get_attributes(["test"]), self.values2)

    def test_get_attributes(self):
        ds = zonalstats.DataSource(self.data)
        ds.set_attribute("test", self.values2)
        assert ds.get_attributes(["test"], filt=("index", 0)) == self.values2[0]
        assert ds.get_attributes(["test"], filt=("index", 1)) == self.values2[1]

    @requires_data
    def test_get_geom_properties(self):
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(31466)
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        test = zonalstats.DataSource(filename, proj)
        np.testing.assert_array_equal(
            [[76722499.98474795]], test.get_geom_properties(["Area"], filt=("FID", 1))
        )

    def test_dump_vector(self):
        ds = zonalstats.DataSource(self.data)
        ds.dump_vector(tempfile.NamedTemporaryFile(mode="w+b").name)

    @requires_data
    def test_dump_raster(self):
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(31466)
        filename = util.get_wradlib_data_file("shapefiles/agger/" "agger_merge.shp")
        test = zonalstats.DataSource(filename, srs=proj)
        test.dump_raster(
            tempfile.NamedTemporaryFile(mode="w+b").name,
            driver="netCDF",
            pixel_size=100.0,
        )
        test.dump_raster(
            tempfile.NamedTemporaryFile(mode="w+b").name,
            driver="netCDF",
            pixel_size=100.0,
            attr="FID",
        )


@pytest.mark.skipif(not util.has_geos(), reason="GDAL without GEOS")
class TestZonalDataBase:
    # GK3-Projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(31466)

    # create synthetic box
    box0 = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5640000.0],
            [2610000.0, 5640000.0],
            [2610000.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    box1 = np.array(
        [
            [2610000.0, 5630000.0],
            [2610000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5630000.0],
            [2610000.0, 5630000.0],
        ]
    )

    box3 = np.array(
        [
            [2595000.0, 5625000.0],
            [2595000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5625000.0],
            [2595000.0, 5625000.0],
        ]
    )

    box4 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5645000.0],
            [2625000.0, 5645000.0],
            [2625000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box5 = np.array(
        [
            [2600000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5630000.0],
            [2600000.0, 5630000.0],
            [2600000.0, 5635000.0],
        ]
    )

    box6 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box7 = np.array(
        [
            [2715000.0, 5635000.0],
            [2715000.0, 5640000.0],
            [2720000.0, 5640000.0],
            [2720000.0, 5635000.0],
            [2715000.0, 5635000.0],
        ]
    )

    src = np.array([box0, box1])
    trg = np.array([box3, box4])
    dst = np.array([[box5], [box6]])
    zdb = zonalstats.ZonalDataBase(src, trg, srs=proj)
    f = tempfile.NamedTemporaryFile(mode="w+b").name
    zdb.dump_vector(f)

    def test___init__(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        assert isinstance(zdb.src, zonalstats.DataSource)
        assert isinstance(zdb.trg, zonalstats.DataSource)
        assert isinstance(zdb.dst, zonalstats.DataSource)
        assert zdb._count_intersections == 2
        zd = zonalstats.DataSource(self.src, name="src", srs=self.proj)
        zdb = zonalstats.ZonalDataBase(zd, self.trg, srs=self.proj)
        assert isinstance(zdb.src, zonalstats.DataSource)
        assert isinstance(zdb.trg, zonalstats.DataSource)
        assert isinstance(zdb.dst, zonalstats.DataSource)
        assert zdb._count_intersections == 2
        zd1 = zonalstats.DataSource(self.src, name="src", srs=self.proj)
        zd2 = zonalstats.DataSource(self.trg, name="trg", srs=self.proj)
        zdb = zonalstats.ZonalDataBase(zd1, zd2, srs=self.proj)
        assert isinstance(zdb.src, zonalstats.DataSource)
        assert isinstance(zdb.trg, zonalstats.DataSource)
        assert isinstance(zdb.dst, zonalstats.DataSource)
        assert zdb._count_intersections == 2

    def test_count_intersections(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        assert zdb.count_intersections == 2

    def test_srs(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        assert zdb.srs == self.proj

    def test_isecs(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        np.testing.assert_equal(zdb.isecs, self.dst)

    def test_get_isec(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        np.testing.assert_equal(zdb.get_isec(0), [self.box5])
        np.testing.assert_equal(zdb.get_isec(1), [self.box6])

    def test_get_source_index(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        assert zdb.get_source_index(0) == 0
        assert zdb.get_source_index(1) == 1

    def test_dump_vector(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        f = tempfile.NamedTemporaryFile(mode="w+b").name
        zdb.dump_vector(f)

    def test_load_vector(self):
        zonalstats.ZonalDataBase(self.f)

    def test__get_intersection(self):
        zdb = zonalstats.ZonalDataBase(self.src, self.trg, srs=self.proj)
        with pytest.raises(TypeError):
            zdb._get_intersection()
        np.testing.assert_equal(zdb._get_intersection(trg=self.box3), [self.box5])
        np.testing.assert_equal(zdb._get_intersection(idx=0), [self.box5])
        with pytest.raises(TypeError):
            zdb._get_intersection(idx=2)
        zdb = zonalstats.ZonalDataBase(self.src, [self.box7], srs=self.proj)
        zdb.trg = None
        with pytest.raises(TypeError):
            zdb._get_intersection(idx=0)


@pytest.mark.skipif(not util.has_geos(), reason="GDAL without GEOS")
class TestZonalDataPoly:
    # GK3-Projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(31466)

    # create synthetic box
    box0 = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5640000.0],
            [2610000.0, 5640000.0],
            [2610000.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    box1 = np.array(
        [
            [2610000.0, 5630000.0],
            [2610000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5630000.0],
            [2610000.0, 5630000.0],
        ]
    )

    box3 = np.array(
        [
            [2595000.0, 5625000.0],
            [2595000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5625000.0],
            [2595000.0, 5625000.0],
        ]
    )

    box4 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5645000.0],
            [2625000.0, 5645000.0],
            [2625000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box5 = np.array(
        [
            [2600000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5630000.0],
            [2600000.0, 5630000.0],
            [2600000.0, 5635000.0],
        ]
    )

    box6 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box7 = np.array(
        [
            [2715000.0, 5635000.0],
            [2715000.0, 5640000.0],
            [2720000.0, 5640000.0],
            [2720000.0, 5635000.0],
            [2715000.0, 5635000.0],
        ]
    )

    src = np.array([box0, box1])
    trg = np.array([box3, box4])
    dst = np.array([[box5], [box6]])
    zdb = zonalstats.ZonalDataBase(src, trg, srs=proj)
    f = tempfile.NamedTemporaryFile(mode="w+b").name
    zdb.dump_vector(f)

    def test__get_idx_weights(self):
        zdp = zonalstats.ZonalDataPoly(self.src, self.trg, srs=self.proj)
        assert zdp._get_idx_weights() == (
            [np.array([0]), np.array([1])],
            [np.array([25000000.0]), np.array([25000000.0])],
        )


@pytest.mark.skipif(not util.has_geos(), reason="GDAL without GEOS")
class TestZonalDataPoint:
    # GK3-Projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(31466)

    # create synthetic box
    point0 = np.array([2600000.0, 5630000.0])

    point1 = np.array([2620000.0, 5640000.0])

    box3 = np.array(
        [
            [2595000.0, 5625000.0],
            [2595000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5625000.0],
            [2595000.0, 5625000.0],
        ]
    )

    box4 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5645000.0],
            [2625000.0, 5645000.0],
            [2625000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box5 = np.array(
        [
            [2600000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5630000.0],
            [2600000.0, 5630000.0],
            [2600000.0, 5635000.0],
        ]
    )

    box6 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box7 = np.array(
        [
            [2715000.0, 5635000.0],
            [2715000.0, 5640000.0],
            [2720000.0, 5640000.0],
            [2720000.0, 5635000.0],
            [2715000.0, 5635000.0],
        ]
    )

    src = np.array([point0, point1])
    trg = np.array([box3, box4])
    dst = np.array([[point0], [point1]])
    zdb = zonalstats.ZonalDataBase(src, trg, srs=proj)
    f = tempfile.NamedTemporaryFile(mode="w+b").name
    zdb.dump_vector(f)

    def test__get_idx_weights(self):
        zdp = zonalstats.ZonalDataPoint(self.src, self.trg, srs=self.proj)
        assert zdp._get_idx_weights() == (
            [np.array([0]), np.array([1])],
            [np.array([1.0]), np.array([1.0])],
        )


@pytest.mark.skipif(not util.has_geos(), reason="GDAL without GEOS")
class TestZonalStatsBase:
    # GK3-Projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(31466)

    # create synthetic box
    box0 = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5640000.0],
            [2610000.0, 5640000.0],
            [2610000.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    box1 = np.array(
        [
            [2610000.0, 5630000.0],
            [2610000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5630000.0],
            [2610000.0, 5630000.0],
        ]
    )

    box3 = np.array(
        [
            [2595000.0, 5625000.0],
            [2595000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5625000.0],
            [2595000.0, 5625000.0],
        ]
    )

    box4 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5645000.0],
            [2625000.0, 5645000.0],
            [2625000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box5 = np.array(
        [
            [2600000.0, 5635000.0],
            [2605000.0, 5635000.0],
            [2605000.0, 5630000.0],
            [2600000.0, 5630000.0],
            [2600000.0, 5635000.0],
        ]
    )

    box6 = np.array(
        [
            [2615000.0, 5635000.0],
            [2615000.0, 5640000.0],
            [2620000.0, 5640000.0],
            [2620000.0, 5635000.0],
            [2615000.0, 5635000.0],
        ]
    )

    box7 = np.array(
        [
            [2715000.0, 5635000.0],
            [2715000.0, 5640000.0],
            [2720000.0, 5640000.0],
            [2720000.0, 5635000.0],
            [2715000.0, 5635000.0],
        ]
    )

    src = np.array([box0, box1])
    trg = np.array([box3, box4])
    dst = np.array([[box5], [box6]])
    zdb = zonalstats.ZonalDataBase(src, trg, srs=proj)
    zdp = zonalstats.ZonalDataPoly(src, trg, srs=proj)

    def test__init__(self):
        with pytest.raises(NotImplementedError):
            zonalstats.ZonalStatsBase(self.zdb)
        zonalstats.ZonalStatsBase(self.zdp)
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase("test")
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase()
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase(ix=np.arange(10), w=np.arange(11))

    def test_w(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.w, np.array([[25000000.0], [25000000.0]]))
        np.testing.assert_equal(zdp.ix, np.array([[0], [1]]))

    def test__check_vals(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        with pytest.raises(AssertionError):
            zdp._check_vals(np.arange(3))

    def test_mean(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.mean(np.arange(10, 21, 10)), np.array([10, 20]))

    def test_var(self):
        zdp = zonalstats.ZonalStatsBase(self.zdp)
        np.testing.assert_equal(zdp.var(np.arange(10, 21, 10)), np.array([0, 0]))


@pytest.mark.skipif(not util.has_geos(), reason="GDAL without GEOS")
class TestZonalData:
    global skip
    # setup test grid and catchment
    lon = 7.071664
    lat = 50.730521
    r = np.array(range(50, 100 * 1000 + 50, 100))
    a = np.array(range(0, 360, 1))
    rays = a.shape[0]
    bins = r.shape[0]

    # setup OSR objects
    proj_gk = osr.SpatialReference()
    proj_gk.ImportFromEPSG(31466)
    proj_ll = osr.SpatialReference()
    proj_ll.ImportFromEPSG(4326)

    # create polar grid polygon vertices in lat,lon
    radar_ll = georef.spherical_to_polyvert(r, a, 0, (lon, lat), proj=proj_ll)[..., 0:2]

    # create polar grid centroids in lat,lon
    coords = georef.spherical_to_centroids(r, a, 0, (lon, lat), proj=proj_ll)

    radar_llc = coords[..., 0:2]

    # project ll grids to GK2
    radar_gk = georef.reproject(
        radar_ll, projection_source=proj_ll, projection_target=proj_gk
    )
    radar_gkc = georef.reproject(
        radar_llc, projection_source=proj_ll, projection_target=proj_gk
    )

    # reshape
    radar_gk.shape = (rays, bins, 5, 2)
    radar_gkc.shape = (rays, bins, 2)

    box0 = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5630100.0],
            [2600100.0, 5630100.0],
            [2600100.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    box1 = np.array(
        [
            [2600100.0, 5630000.0],
            [2600100.0, 5630100.0],
            [2600200.0, 5630100.0],
            [2600200.0, 5630000.0],
            [2600100.0, 5630000.0],
        ]
    )

    data = np.array([box0, box1])

    # create catchment bounding box
    buffer = 5000.0
    bbox = zonalstats.get_bbox(data[..., 0], data[..., 1])
    bbox = dict(
        left=bbox["left"] - buffer,
        right=bbox["right"] + buffer,
        bottom=bbox["bottom"] - buffer,
        top=bbox["top"] + buffer,
    )

    mask, shape = zonalstats.mask_from_bbox(
        radar_gkc[..., 0], radar_gkc[..., 1], bbox, polar=True
    )

    radar_gkc = radar_gkc[mask, :]
    radar_gk = radar_gk[mask]

    zdpoly = zonalstats.ZonalDataPoly(radar_gk, data, srs=proj_gk)
    # zdpoly.dump_vector('test_zdpoly')
    zdpoint = zonalstats.ZonalDataPoint(radar_gkc, data, srs=proj_gk)
    # zdpoint.dump_vector('test_zdpoint')

    isec_poly0 = np.array(
        [
            np.array(
                [
                    [2600000.0, 5630000.0],
                    [2600000.0, 5630100.0],
                    [2600009.61157242, 5630100.0],
                    [2600041.77844048, 5630000.0],
                    [2600000.0, 5630000.0],
                ]
            ),
            np.array(
                [
                    [2600009.61157242, 5630100.0],
                    [2600100.0, 5630100.0],
                    [2600100.0, 5630000.0],
                    [2600041.77844048, 5630000.0],
                    [2600009.61157242, 5630100.0],
                ]
            ),
            np.array(
                [
                    [2600091.80406488, 5630100.0],
                    [2600100.0, 5630100.0],
                    [2600100.0, 5630074.58501104],
                    [2600091.80406488, 5630100.0],
                ]
            ),
        ]
    )
    isec_poly1 = np.array(
        [
            np.array(
                [
                    [2600100.0, 5630000.0],
                    [2600100.0, 5630100.0],
                    [2600114.66582085, 5630100.0],
                    [2600146.83254704, 5630000.0],
                    [2600100.0, 5630000.0],
                ]
            ),
            np.array(
                [
                    [2600114.66582085, 5630100.0],
                    [2600200.0, 5630100.0],
                    [2600200.0, 5630000.0],
                    [2600146.83254704, 5630000.0],
                    [2600114.66582085, 5630100.0],
                ]
            ),
            np.array(
                [
                    [2600197.20644071, 5630100.0],
                    [2600200.0, 5630100.0],
                    [2600200.0, 5630091.33737992],
                    [2600197.20644071, 5630100.0],
                ]
            ),
        ]
    )

    isec_point0 = np.array([[2600077.2899581, 5630056.0874306]])
    isec_point1 = np.array([[2600172.498418, 5630086.7127034]])

    isec_poly = np.array([isec_poly0, isec_poly1])
    isec_point = np.array([isec_point0, isec_point1])

    def test_srs(self):
        assert self.zdpoly.srs == self.proj_gk
        assert self.zdpoint.srs == self.proj_gk

    def test_isecs(self):
        # need to iterate over nested array for correct testing
        for i in range(len(self.zdpoly.isecs)):
            for k in range(len(self.zdpoly.isecs[i])):
                np.testing.assert_allclose(
                    self.zdpoly.isecs[i, k], self.isec_poly[i, k], rtol=1e-6
                )

        np.testing.assert_allclose(self.zdpoint.isecs, self.isec_point, rtol=1e-6)

    def test_get_isec(self):
        for i in [0, 1]:
            for k, arr in enumerate(self.zdpoly.get_isec(i)):
                np.testing.assert_allclose(arr, self.isec_poly[i, k], rtol=1e-6)
            np.testing.assert_allclose(
                self.zdpoint.get_isec(i), self.isec_point[i], rtol=1e-6
            )

    def test_get_source_index(self):
        np.testing.assert_array_equal(
            self.zdpoly.get_source_index(0), np.array([2254, 2255])
        )
        np.testing.assert_array_equal(
            self.zdpoly.get_source_index(1), np.array([2255, 2256])
        )
        np.testing.assert_array_equal(
            self.zdpoint.get_source_index(0), np.array([2255])
        )
        np.testing.assert_array_equal(
            self.zdpoint.get_source_index(1), np.array([2256])
        )


class TestZonalStatsUtil:
    npobj = np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5630100.0],
            [2600100.0, 5630100.0],
            [2600100.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )

    ogrobj = georef.numpy_to_ogr(npobj, "Polygon")

    def test_angle_between(self):
        assert zonalstats.angle_between(355.0, 5.0) == pytest.approx(10.0)
        assert zonalstats.angle_between(5.0, 355.0) == pytest.approx(-10.0)

    def test_get_clip_mask(self):
        proj_gk = osr.SpatialReference()
        proj_gk.ImportFromEPSG(31466)
        coords = np.array(
            [
                [2600020.0, 5630020.0],
                [2600030.0, 5630030.0],
                [2600040.0, 5630040.0],
                [2700100.0, 5630030.0],
                [2600040.0, 5640000.0],
            ]
        )
        mask = zonalstats.get_clip_mask(coords, self.npobj, proj_gk)
        out = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, out)
