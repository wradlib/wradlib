# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc
import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from wradlib import io

from . import (
    get_wradlib_data_file,
    has_data,
    requires_data,
    requires_h5netcdf,
    requires_h5py,
    requires_netcdf,
    requires_xarray_backend_api,
    requires_xmltodict,
)


@requires_data
@requires_xarray_backend_api
def test_radolan_backend(file_or_filelike):
    filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
    test_attrs = {
        "radarlocations": [
            "boo",
            "ros",
            "emd",
            "hnr",
            "pro",
            "ess",
            "asd",
            "neu",
            "nhb",
            "oft",
            "tur",
            "isn",
            "fbg",
            "mem",
        ],
        "radolanversion": "2.13.1",
        "formatversion": 3,
        "radarid": "10000",
    }
    with get_wradlib_data_file(filename, file_or_filelike) as rwfile:
        data, meta = io.read_radolan_composite(rwfile)
        data[data == -9999.0] = np.nan
        with xr.open_dataset(rwfile, engine="radolan") as ds:
            assert ds["RW"].encoding["dtype"] == np.uint16
            if file_or_filelike == "file":
                assert ds["RW"].encoding["source"] == os.path.abspath(rwfile)
            else:
                assert ds["RW"].encoding["source"] == "None"
            assert ds.attrs == test_attrs
            assert ds["RW"].shape == (900, 900)
            np.testing.assert_almost_equal(
                ds["RW"].values,
                (data * 3600.0 / meta["intervalseconds"]),
                decimal=5,
            )


@contextlib.contextmanager
def get_measured_volume(file, format, fileobj, **kwargs):
    with get_wradlib_data_file(file, fileobj) as radfile:
        engine = format.lower()
        if engine == "odim":
            pytest.importorskip("h5netcdf")
            open_ = io.open_odim_dataset
        if engine == "gamic":
            pytest.importorskip("h5netcdf")
            open_ = io.open_gamic_dataset
        if engine == "cfradial1":
            pytest.importorskip("netCDF4")
            open_ = io.open_cfradial1_dataset
        if engine == "cfradial2":
            pytest.importorskip("netCDF4")
            open_ = io.open_cfradial2_dataset
        if engine == "iris":
            open_ = io.open_iris_dataset
        if engine == "rainbow":
            pytest.importorskip("xmltodict")
            open_ = io.open_rainbow_dataset
        if engine == "furuno":
            open_ = io.open_furuno_dataset
        yield open_(radfile, **kwargs)


def create_volume_repr(swp, ele, cls):
    ele = [f"{e:.1f}" for e in ele]
    repr = "".join(
        [
            f"<wradlib.{cls}>\n",
            f"Dimension(s): (sweep: {swp})\n",
            f"Elevation(s): ({', '.join(ele)})",
        ]
    )
    return repr


class DataVolume:
    def test_volumes(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            assert isinstance(vol, io.xarray.RadarVolume)
            repr = create_volume_repr(self.sweeps, self.elevations, type(vol).__name__)
            assert vol.__repr__() == repr
        del vol
        gc.collect()

    def test_sweeps(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            for i, ds in enumerate(vol):
                assert isinstance(ds, xr.Dataset)
                assert self.azimuths[i] == ds.dims["azimuth"]
                assert self.ranges[i] == ds.dims["range"]

    @requires_h5py
    def test_odim_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="odim").name
            if self.format.lower() == "cfradial2":
                for i in range(len(vol)):
                    ds = io.xarray._rewrite_time_reference_units(vol[i])
                    vol[i] = xr.decode_cf(ds)
            vol.to_odim(tmp_local, source="WMO:12345")
        del vol
        gc.collect()

    @requires_netcdf
    def test_cfradial2_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_cfradial2(tmp_local)
        del vol
        gc.collect()

    @requires_netcdf
    def test_netcdf_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_netcdf(tmp_local, timestep=slice(0, None))
        del vol
        gc.collect()


class MeasuredDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, fileobj, **kwargs):
        if "cfradial" in self.format.lower() and fileobj == "filelike":
            pytest.skip(f"{self.format.lower()} doesn't work with filelike")
        if "rainbow" in self.format.lower() and fileobj == "filelike":
            pytest.skip(f"{self.format.lower()} doesn't work with filelike")
        with get_measured_volume(
            self.name,
            self.format,
            fileobj,
            backend_kwargs=self.backend_kwargs,
            **kwargs,
        ) as vol:
            yield vol


@requires_data
@requires_h5netcdf
@requires_xarray_backend_api
class TestKNMIVolume(MeasuredDataVolume):
    if has_data:
        name = "hdf5/knmi_polar_volume.h5"
        format = "ODIM"
        volumes = 1
        sweeps = 14
        moments = ["DBZH"]
        elevations = [
            0.3,
            0.4,
            0.8,
            1.1,
            2.0,
            3.0,
            4.5,
            6.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
        ]
        azimuths = [360] * sweeps
        ranges = [320, 240, 240, 240, 240, 340, 340, 300, 300, 240, 240, 240, 240, 240]

        dsdesc = "dataset{}"
        mdesc = "data{}"

        backend_kwargs = dict(keep_azimuth=True)


@requires_data
@requires_h5netcdf
@requires_xarray_backend_api
class TestGamicVolume(MeasuredDataVolume):
    name = "hdf5/DWD-Vol-2_99999_20180601054047_00.h5"
    format = "GAMIC"
    volumes = 1
    sweeps = 10
    moments = [
        "DBZH",
        "DBZV",
        "DBTH",
        "DBTV",
        "ZDR",
        "VRADH",
        "VRADV",
        "WRADH",
        "WRADV",
        "PHIDP",
        "KDP",
        "RHOHV",
    ]
    elevations = [28.0, 18.0, 14.0, 11.0, 8.2, 6.0, 4.5, 3.1, 1.7, 0.6]
    azimuths = [360, 360, 360, 360, 360, 360, 360, 360, 360, 360]
    ranges = [360, 500, 620, 800, 1050, 1400, 1000, 1000, 1000, 1000]

    dsdesc = "scan{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(keep_azimuth=True)


@requires_data
@requires_netcdf
@requires_xarray_backend_api
class TestCfRadial1Volume(MeasuredDataVolume):
    name = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
    format = "CfRadial1"
    volumes = 1
    sweeps = 9
    moments = [
        "DBZH",
        "DBZV",
        "DBTH",
        "DBTV",
        "ZDR",
        "VRADH",
        "VRADV",
        "WRADH",
        "WRADV",
        "PHIDP",
        "KDP",
        "RHOHV",
    ]
    elevations = [0.5, 1.1, 1.8, 2.6, 3.6, 4.7, 6.5, 9.1, 12.8]
    azimuths = [483, 483, 482, 483, 481, 482, 482, 484, 483]
    ranges = [996, 996, 996, 996, 996, 996, 996, 996, 996]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = {}


@requires_data
@requires_netcdf
@requires_xarray_backend_api
class TestCfRadial2Volume(MeasuredDataVolume):
    name = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR_cfradial2.nc"
    format = "CfRadial2"
    volumes = 1
    sweeps = 9
    moments = [
        "DBZH",
        "DBZV",
        "DBTH",
        "DBTV",
        "ZDR",
        "VRADH",
        "VRADV",
        "WRADH",
        "WRADV",
        "PHIDP",
        "KDP",
        "RHOHV",
    ]
    elevations = [0.5, 1.1, 1.8, 2.6, 3.6, 4.7, 6.5, 9.1, 12.8]
    azimuths = [480, 480, 480, 480, 480, 480, 480, 480, 480]
    ranges = [996, 996, 996, 996, 996, 996, 996, 996, 996]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = {}


@requires_data
@requires_xarray_backend_api
class TestIrisVolume01(MeasuredDataVolume):
    name = "sigmet/cor-main131125105503.RAW2049"
    format = "Iris"
    volumes = 1
    sweeps = 10
    moments = [
        "DBZH",
        "DBZV",
        "DBTH",
        "DBTV",
        "ZDR",
        "VRADH",
        "VRADV",
        "WRADH",
        "WRADV",
        "PHIDP",
        "KDP",
        "RHOHV",
    ]
    elevations = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    azimuths = [360, 360, 360, 360, 360, 360, 360, 360, 360, 360]
    ranges = [664, 664, 664, 664, 664, 664, 664, 664, 664, 664]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(reindex_angle=False)


@requires_data
@requires_xarray_backend_api
class TestIrisVolume02(MeasuredDataVolume):
    name = "sigmet/SUR210819000227.RAWKPJV"
    format = "Iris"
    volumes = 1
    sweeps = 1
    moments = [
        "DBTH",
        "DBZH",
        "VRADH",
        "WRADH",
        "ZDR",
        "KDP",
        "RHOHV",
        "SQIH",
        "PHIDP",
        "DB_HCLASS2",
        "SNRH",
    ]
    elevations = [0.5]
    azimuths = [360]
    ranges = [833]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(reindex_angle=1.0)


@requires_data
@requires_xmltodict
@requires_xarray_backend_api
class TestRainbowVolume01(MeasuredDataVolume):
    name = "rainbow/2013051000000600dBZ.vol"
    format = "Rainbow"
    volumes = 1
    sweeps = 14
    moments = [
        "DBZH",
    ]
    elevations = [
        0.6,
        1.4,
        2.4,
        3.5,
        4.8,
        6.3,
        8.0,
        9.9,
        12.2,
        14.8,
        17.9,
        21.3,
        25.4,
        30.0,
    ]
    azimuths = [360] * sweeps
    ranges = [400] * sweeps

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(reindex_angle=1.0)


@requires_data
@requires_xarray_backend_api
class TestFurunoVolume01(MeasuredDataVolume):
    name = "furuno/2006_20220324_000000_000.scnx.gz"
    format = "furuno"
    volumes = 1
    sweeps = 1
    moments = [
        "RR",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "SQIH",
        "WRADH",
        "QUAL",
    ]
    elevations = [0.5]
    azimuths = [720]
    ranges = [936]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(
        reindex_angle=dict(
            tolerance=1.0, start_angle=0, stop_angle=360, angle_res=0.5, direction=1
        ),
        obsmode=1,
    )


@requires_data
@requires_xarray_backend_api
class TestFurunoVolume02(MeasuredDataVolume):
    name = "furuno/0080_20210730_160000_01_02.scn.gz"
    format = "furuno"
    volumes = 1
    sweeps = 1
    moments = [
        "RR",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "SQIH",
        "WRADH",
        "QUAL",
    ]
    elevations = [7.8]
    azimuths = [1385]
    ranges = [602]

    dsdesc = "sweep{}"
    mdesc = "moment_{}"

    backend_kwargs = dict(
        reindex_angle=dict(
            tolerance=1.0, start_angle=0, stop_angle=360, angle_res=0.26, direction=1
        ),
        obsmode=1,
    )
