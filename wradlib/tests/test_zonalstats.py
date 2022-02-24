#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

from wradlib import georef, io, zonalstats

from . import osr, requires_gdal, requires_geos

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


@pytest.fixture
def data_base():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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
                # [2600000.0, 5635000.0],
                [2605000.0, 5635000.0],
                [2605000.0, 5630000.0],
                [2600000.0, 5630000.0],
                [2600000.0, 5635000.0],
                [2605000.0, 5635000.0],
            ]
        )

        box6 = np.array(
            [
                # [2615000.0, 5635000.0],
                [2615000.0, 5640000.0],
                [2620000.0, 5640000.0],
                [2620000.0, 5635000.0],
                [2615000.0, 5635000.0],
                [2615000.0, 5640000.0],
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

    yield Data


@requires_geos
class TestZonalDataBase:
    @requires_gdal
    def test___init__(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        assert isinstance(zdb.src, io.VectorSource)
        assert isinstance(zdb.trg, io.VectorSource)
        assert isinstance(zdb.dst, io.VectorSource)
        assert zdb._count_intersections == 2
        zd = io.VectorSource(data_base.src, name="src", srs=data_base.proj)
        zdb = zonalstats.ZonalDataBase(zd, data_base.trg, srs=data_base.proj)
        assert isinstance(zdb.src, io.VectorSource)
        assert isinstance(zdb.trg, io.VectorSource)
        assert isinstance(zdb.dst, io.VectorSource)
        assert zdb._count_intersections == 2
        zd1 = io.VectorSource(data_base.src, name="src", srs=data_base.proj)
        zd2 = io.VectorSource(data_base.trg, name="trg", srs=data_base.proj)
        zdb = zonalstats.ZonalDataBase(zd1, zd2, srs=data_base.proj)
        assert isinstance(zdb.src, io.VectorSource)
        assert isinstance(zdb.trg, io.VectorSource)
        assert isinstance(zdb.dst, io.VectorSource)
        assert zdb._count_intersections == 2

    @requires_gdal
    def test_count_intersections(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        assert zdb.count_intersections == 2

    @requires_gdal
    def test_srs(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        assert zdb.srs == data_base.proj

    @requires_gdal
    def test_isecs(self, data_base):
        # todo: Normalize Polygons before comparison.
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        np.testing.assert_equal(zdb.isecs, data_base.dst)

    @requires_gdal
    def test_get_isec(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        np.testing.assert_equal(zdb.get_isec(0), [data_base.box5])
        np.testing.assert_equal(zdb.get_isec(1), [data_base.box6])

    @requires_gdal
    def test_get_source_index(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        assert zdb.get_source_index(0) == 0
        assert zdb.get_source_index(1) == 1

    @requires_gdal
    def test_dump_vector(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        f = tempfile.NamedTemporaryFile(mode="w+b").name
        zdb.dump_vector(f)

    @requires_gdal
    def test_load_vector(self, data_base):
        zonalstats.ZonalDataBase(data_base.f)

    @requires_gdal
    def test__get_intersection(self, data_base):
        zdb = zonalstats.ZonalDataBase(data_base.src, data_base.trg, srs=data_base.proj)
        with pytest.raises(TypeError):
            zdb._get_intersection()
        np.testing.assert_equal(
            zdb._get_intersection(trg=data_base.box3), [data_base.box5]
        )
        np.testing.assert_equal(zdb._get_intersection(idx=0), [data_base.box5])
        with pytest.raises(TypeError):
            zdb._get_intersection(idx=2)
        zdb = zonalstats.ZonalDataBase(
            data_base.src, [data_base.box7], srs=data_base.proj
        )
        zdb.trg = None
        with pytest.raises(TypeError):
            zdb._get_intersection(idx=0)


@pytest.fixture
def data_poly():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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

    yield Data


@requires_geos
class TestZonalDataPoly:
    @requires_gdal
    def test__get_idx_weights(self, data_poly):
        zdp = zonalstats.ZonalDataPoly(data_poly.src, data_poly.trg, srs=data_poly.proj)
        assert zdp._get_idx_weights() == (
            [np.array([0]), np.array([1])],
            [np.array([25000000.0]), np.array([25000000.0])],
        )


@pytest.fixture
def data_point():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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

    yield Data


@requires_geos
class TestZonalDataPoint:
    @requires_gdal
    def test__get_idx_weights(self, data_point):
        zdp = zonalstats.ZonalDataPoint(
            data_point.src, data_point.trg, srs=data_point.proj
        )
        assert zdp._get_idx_weights() == (
            [np.array([0]), np.array([1])],
            [np.array([1.0]), np.array([1.0])],
        )


@pytest.fixture
def stats_base():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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

    yield Data


@requires_geos
class TestZonalStatsBase:
    @requires_gdal
    def test__init__(self, stats_base):
        with pytest.raises(NotImplementedError):
            zonalstats.ZonalStatsBase(stats_base.zdb)
        zonalstats.ZonalStatsBase(stats_base.zdp)
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase("test")
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase()
        with pytest.raises(TypeError):
            zonalstats.ZonalStatsBase(ix=np.arange(10), w=np.arange(11))

    @requires_gdal
    def test_w(self, stats_base):
        zdp = zonalstats.ZonalStatsBase(stats_base.zdp)
        np.testing.assert_equal(zdp.w, np.array([[25000000.0], [25000000.0]]))
        np.testing.assert_equal(zdp.ix, np.array([[0], [1]]))

    @requires_gdal
    def test__check_vals(self, stats_base):
        zdp = zonalstats.ZonalStatsBase(stats_base.zdp)
        with pytest.raises(AssertionError):
            zdp._check_vals(np.arange(3))

    @requires_gdal
    def test_mean(self, stats_base):
        zdp = zonalstats.ZonalStatsBase(stats_base.zdp)
        np.testing.assert_equal(zdp.mean(np.arange(10, 21, 10)), np.array([10, 20]))

    @requires_gdal
    def test_var(self, stats_base):
        zdp = zonalstats.ZonalStatsBase(stats_base.zdp)
        np.testing.assert_equal(zdp.var(np.arange(10, 21, 10)), np.array([0, 0]))


@pytest.fixture
def zonal_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
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
        coords = georef.spherical_to_polyvert(r, a, 0, (lon, lat), proj=proj_ll)
        radar_ll = coords[..., 0:2]

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
                        # [2600000.0, 5630000.0],
                        [2600000.0, 5630100.0],
                        [2600009.61157242, 5630100.0],
                        [2600041.77844048, 5630000.0],
                        [2600000.0, 5630000.0],
                        [2600000.0, 5630100.0],
                    ]
                ),
                np.array(
                    [
                        # [2600009.61157242, 5630100.0],
                        [2600100.0, 5630100.0],
                        [2600100.0, 5630000.0],
                        [2600041.77844048, 5630000.0],
                        [2600009.61157242, 5630100.0],
                        [2600100.0, 5630100.0],
                    ]
                ),
                np.array(
                    [
                        # [2600091.80406488, 5630100.0],
                        [2600100.0, 5630100.0],
                        [2600100.0, 5630074.58501104],
                        [2600091.80406488, 5630100.0],
                        [2600100.0, 5630100.0],
                    ]
                ),
            ],
            dtype=object,
        )
        isec_poly1 = np.array(
            [
                np.array(
                    [
                        # [2600100.0, 5630000.0],
                        [2600100.0, 5630100.0],
                        [2600114.66582085, 5630100.0],
                        [2600146.83254704, 5630000.0],
                        [2600100.0, 5630000.0],
                        [2600100.0, 5630100.0],
                    ]
                ),
                np.array(
                    [
                        # [2600114.66582085, 5630100.0],
                        [2600200.0, 5630100.0],
                        [2600200.0, 5630000.0],
                        [2600146.83254704, 5630000.0],
                        [2600114.66582085, 5630100.0],
                        [2600200.0, 5630100.0],
                    ]
                ),
                np.array(
                    [
                        # [2600197.20644071, 5630100.0],
                        [2600200.0, 5630100.0],
                        [2600200.0, 5630091.33737992],
                        [2600197.20644071, 5630100.0],
                        [2600200.0, 5630100.0],
                    ]
                ),
            ],
            dtype=object,
        )

        isec_point0 = np.array([[2600077.2899581, 5630056.0874306]])
        isec_point1 = np.array([[2600172.498418, 5630086.7127034]])

        isec_poly = np.array([isec_poly0, isec_poly1])
        isec_point = np.array([isec_point0, isec_point1])

    yield Data


@requires_geos
class TestZonalData:
    @requires_gdal
    def test_srs(self, zonal_data):
        assert zonal_data.zdpoly.srs == zonal_data.proj_gk
        assert zonal_data.zdpoint.srs == zonal_data.proj_gk

    @requires_gdal
    def test_isecs(self, zonal_data):
        # need to iterate over nested array for correct testing
        for i, ival in enumerate(zonal_data.zdpoly.isecs):
            for k, kval in enumerate(ival):
                np.testing.assert_allclose(
                    kval.astype(float), zonal_data.isec_poly[i, k], rtol=1e-6
                )

        np.testing.assert_allclose(
            zonal_data.zdpoint.isecs.astype(float), zonal_data.isec_point, rtol=1e-6
        )

    @requires_gdal
    def test_get_isec(self, zonal_data):
        for i in [0, 1]:
            for k, arr in enumerate(zonal_data.zdpoly.get_isec(i)):
                np.testing.assert_allclose(
                    arr.astype(float), zonal_data.isec_poly[i, k], rtol=1e-6
                )
            np.testing.assert_allclose(
                zonal_data.zdpoint.get_isec(i).astype(float),
                zonal_data.isec_point[i],
                rtol=1e-6,
            )

    @requires_gdal
    def test_get_source_index(self, zonal_data):
        np.testing.assert_array_equal(
            zonal_data.zdpoly.get_source_index(0), np.array([2254, 2255])
        )
        np.testing.assert_array_equal(
            zonal_data.zdpoly.get_source_index(1), np.array([2255, 2256])
        )
        np.testing.assert_array_equal(
            zonal_data.zdpoint.get_source_index(0), np.array([2255])
        )
        np.testing.assert_array_equal(
            zonal_data.zdpoint.get_source_index(1), np.array([2256])
        )


@pytest.fixture
def npobj():
    yield np.array(
        [
            [2600000.0, 5630000.0],
            [2600000.0, 5630100.0],
            [2600100.0, 5630100.0],
            [2600100.0, 5630000.0],
            [2600000.0, 5630000.0],
        ]
    )


@pytest.fixture
def ogrobj(npobj):
    yield georef.numpy_to_ogr(npobj, "Polygon")


class TestZonalStatsUtil:
    def test_angle_between(self):
        assert zonalstats.angle_between(355.0, 5.0) == pytest.approx(10.0)
        assert zonalstats.angle_between(5.0, 355.0) == pytest.approx(-10.0)

    @requires_gdal
    def test_get_clip_mask(self, npobj):
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
        mask = zonalstats.get_clip_mask(coords, npobj, proj_gk)
        out = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, out)
