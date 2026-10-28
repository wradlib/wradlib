#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
import xradar as xd
from open_radar_data import DATASETS

from wradlib import comp, georef, io, ipol

from . import (
    gdal,
    get_wradlib_data_file,
    osr,
    requires_dask,
    requires_gdal,
    rioxarray,
)


@pytest.fixture
def comp_data():
    @dataclass(init=False, repr=False, eq=False)
    class Data:
        filename = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        dx_file = get_wradlib_data_file(filename)
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

    yield Data


def test_extract_circle(comp_data):
    x = comp_data.x
    y = comp_data.y
    xgrid = np.linspace(x.min(), x.mean(), 100)
    ygrid = np.linspace(y.min(), y.mean(), 100)
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
    comp.extract_circle(np.array([x.mean(), y.mean()]), 128000.0, grid_xy)


def test_togrid(comp_data):
    x = comp_data.x
    y = comp_data.y
    data = comp_data.data
    xgrid = np.linspace(x.min(), x.mean(), 100)
    ygrid = np.linspace(y.min(), y.mean(), 100)
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
    xy = np.concatenate([x.ravel()[:, None], y.ravel()[:, None]], axis=1)
    comp.togrid(
        xy,
        grid_xy,
        128000.0,
        np.array([x.mean(), y.mean()]),
        data.ravel(),
        ipol.Nearest,
    )


@pytest.mark.parametrize("quality_dtype", [np.int32, np.float64])
@pytest.mark.parametrize("shape", [(7,), (1, 7)])
def test_compose_ko_missing_radar_data(quality_dtype, shape):
    radargrids = [
        np.array([np.nan, 0.0, np.nan, 4.0, 6.0, np.nan, 1.0]).reshape(shape),
        np.array([2.0, 2.0, np.nan, np.nan, 7.0, 3.0, 8.0]).reshape(shape),
    ]
    qualitygrids = [
        np.array([2, 2, 3, 1, 1, 1, 1], dtype=quality_dtype).reshape(shape),
        np.array([1, 1, 4, 2, 2, 2, 1], dtype=quality_dtype).reshape(shape),
    ]
    original_radar = [grid.copy() for grid in radargrids]
    original_quality = [grid.copy() for grid in qualitygrids]

    composite = comp.compose_ko(radargrids, qualitygrids)

    np.testing.assert_equal(
        composite, np.array([2.0, 0.0, np.nan, 4.0, 7.0, 3.0, 1.0]).reshape(shape)
    )
    assert len(radargrids) == len(qualitygrids) == 2
    np.testing.assert_equal(radargrids, original_radar)
    np.testing.assert_equal(qualitygrids, original_quality)


def test_compose():
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


@pytest.mark.parametrize("lazy", [False, pytest.param(True, marks=requires_dask)])
@pytest.mark.parametrize("radar_last", [False, True])
def test_compose_weighted_xarray_valid_data(lazy, radar_last):
    radargrids = xr.DataArray(
        [
            [0.0, np.nan, 0.0, -1.0, np.nan, 2.0, 2.0],
            [2.0, 2.0, 0.0, 1.0, np.nan, np.nan, 6.0],
        ],
        dims=("radar", "x"),
        coords={"radar": ["a", "b"], "x": np.arange(7)},
        name="DBZH",
        attrs={"units": "dBZ"},
    )
    qualitygrids = xr.ones_like(radargrids)
    qualitygrids[1, -1] = 3.0
    if lazy:
        radargrids = radargrids.chunk({"x": 2})
        qualitygrids = qualitygrids.chunk({"x": 2})
    if radar_last:
        radargrids = radargrids.transpose("x", "radar")
        qualitygrids = qualitygrids.transpose("x", "radar")
    original_radar = radargrids.copy(deep=True)
    original_quality = qualitygrids.copy(deep=True)

    composite = radargrids.wrl.comp.compose_weighted(qualitygrids).compute()

    expected = xr.DataArray(
        [1.0, 2.0, 0.0, 0.0, np.nan, 2.0, 5.0],
        dims="x",
        coords={"x": np.arange(7)},
        name="DBZH",
        attrs={"units": "dBZ"},
    )
    xr.testing.assert_identical(composite, expected)
    xr.testing.assert_identical(radargrids, original_radar)
    xr.testing.assert_identical(qualitygrids, original_quality)


@requires_gdal
def test_sweep_to_raster_geographic():

    filename = "hdf5/IDR66_20141206_094829.vol.h5"
    filename = get_wradlib_data_file(filename)
    sweep = xr.open_datatree(filename, engine="odim")
    sweep = sweep["sweep_0"].to_dataset(inherit="all_coords")
    sweep = xd.georeference.get_x_y_z(sweep)

    lon = float(sweep.longitude.values)
    lat = float(sweep.latitude.values)
    lon_min = round(lon) - 2
    lat_min = round(lat) - 2
    lon_max = round(lon) + 2
    lat_max = round(lat) + 2
    bounds = [lon_min, lat_min, lon_max, lat_max]
    resolution = 30
    raster = georef.create_raster_geographic(
        bounds=bounds,
        resolution=resolution,
    )
    transform = comp.transform_binned(sweep, raster)

    # check normal operation
    composite = comp.sweep_to_raster(sweep, raster, transform=transform)
    # check accessor-based operation on Dataset
    composite1 = sweep.wrl.comp.sweep_to_raster(raster, transform=transform)
    # check accessor-based operation on DataArray
    composite2 = sweep.DBZH.wrl.comp.sweep_to_raster(raster, transform=transform)

    # intercomparison
    xr.testing.assert_equal(composite, composite1)
    xr.testing.assert_equal(composite.DBZH, composite2)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        composite.to_netcdf(tmp.name)

    ds = gdal.Open(tmp.name)
    gt = ds.GetGeoTransform()
    np.testing.assert_equal(
        gt, [lon_min * 3600, resolution, 0.0, lat_max * 3600, 0.0, -resolution]
    )

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    assert srs.GetAngularUnitsName() == "arc-second"
    ds.Close()

    ds = rioxarray.open_rasterio(tmp.name)
    ds2 = ds.rio.reproject("EPSG:4326")
    np.testing.assert_equal(
        ds2.rio.bounds(),
        bounds,
    )


def test_sweep_to_raster():

    filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
    datatree = xr.open_datatree(filename, engine="odim")
    sweep = datatree["sweep_0"].to_dataset(inherit="all_coords")

    location = (sweep.longitude.values, sweep.latitude.values)
    crs = georef.get_radar_projection(location)
    bounds = [-10e3, -10e3, 10e3, 10e3]
    resolution = 500
    raster = georef.create_raster_xarray(crs=crs, bounds=bounds, resolution=resolution)

    with pytest.raises(
        ValueError,
        match="Sweep has no x and y coordinates. Please georeference first.",
    ):
        comp.transform_binned(sweep, raster)

    sweep = xd.georeference.get_x_y_z(sweep)

    # call by function
    composite = comp.sweep_to_raster(sweep, raster)

    # call by accessor
    composite = sweep.wrl.comp.sweep_to_raster(raster)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        composite.to_netcdf(tmp.name)
