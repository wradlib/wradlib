# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc
import os
import tempfile

import h5py
import numpy as np
import pytest
import xarray as xr

from wradlib import io, util

from . import (
    get_wradlib_data_file,
    has_data,
    requires_data,
    requires_xarray_backend_api,
)
from .test_io_odim import (  # noqa: F401
    base_gamic_data,
    base_odim_data_00,
    base_odim_data_01,
    base_odim_data_02,
    base_odim_data_03,
    write_group,
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
    # h5file = util.get_wradlib_data_file(file)
    with get_wradlib_data_file(file, fileobj) as h5file:
        engine = format.lower()
        if engine == "odim":
            open_ = io.open_odim_dataset
        if engine == "gamic":
            open_ = io.open_gamic_dataset
        if engine == "cfradial1":
            open_ = io.open_cfradial1_dataset
        if engine == "cfradial2":
            open_ = io.open_cfradial2_dataset
        yield open_(h5file, **kwargs)


@contextlib.contextmanager
def get_synthetic_volume(name, file_or_filelike, **kwargs):
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=name).name
    if "gamic" in name:
        format = "GAMIC"
    else:
        format = "ODIM"
    with h5py.File(str(tmp_local), "w") as f:
        data = globals()[name]()
        write_group(f, data)
    with get_wradlib_data_file(tmp_local, file_or_filelike) as h5file:
        engine = format.lower()
        if engine == "odim":
            open_ = io.open_odim_dataset
        if engine == "gamic":
            open_ = io.open_gamic_dataset
        yield open_(h5file, **kwargs)


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
            print("repr", repr)
            print(vol.__repr__())
            assert vol.__repr__() == repr
            print(vol[0])
        del vol
        gc.collect()

    def test_sweeps(self, file_or_filelike):
        if self.format.lower() in ["gamic", "odim"]:
            backend_kwargs = dict(keep_azimuth=True)
        else:
            backend_kwargs = {}
        with self.get_volume_data(
            file_or_filelike, backend_kwargs=backend_kwargs
        ) as vol:
            for i, ds in enumerate(vol):
                assert isinstance(ds, xr.Dataset)
                assert self.azimuths[i] == ds.dims["azimuth"]
                assert self.ranges[i] == ds.dims["range"]

    def test_odim_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="odim").name
            if self.format.lower() == "cfradial2":
                for i in range(len(vol)):
                    ds = io.xarray._rewrite_time_reference_units(vol[i])
                    vol[i] = xr.decode_cf(ds)
            vol.to_odim(tmp_local)
        del vol
        gc.collect()

    def test_cfradial2_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_cfradial2(tmp_local)
        del vol
        gc.collect()

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
            pytest.skip("cfradial doesn't work with filelike")
        with get_measured_volume(self.name, self.format, fileobj, **kwargs) as vol:
            yield vol


class SyntheticDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, fileobj, **kwargs):
        with get_synthetic_volume(self.name, fileobj, **kwargs) as vol:
            yield vol


@requires_data
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

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "dataset{}"
        mdesc = "data{}"


@requires_data
@requires_xarray_backend_api
class TestGamicVolume(MeasuredDataVolume):
    if has_data:
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

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "scan{}"
        mdesc = "moment_{}"


@requires_data
@requires_xarray_backend_api
class TestCfRadial1Volume(MeasuredDataVolume):
    if has_data:
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

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "sweep{}"
        mdesc = "moment_{}"


@requires_data
@requires_xarray_backend_api
class TestCfRadial2Volume(MeasuredDataVolume):
    if has_data:
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

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "sweep{}"
        mdesc = "moment_{}"


@requires_xarray_backend_api
class TestSyntheticOdimVolume01(SyntheticDataVolume):
    name = "base_odim_data_00"
    format = "ODIM"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [360, 360]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "dataset{}"
    mdesc = "data{}"


@requires_xarray_backend_api
class TestSyntheticOdimVolume02(SyntheticDataVolume):
    name = "base_odim_data_01"
    format = "ODIM"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [360, 360]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "dataset{}"
    mdesc = "data{}"


@requires_xarray_backend_api
class TestSyntheticOdimVolume03(SyntheticDataVolume):
    name = "base_odim_data_02"
    format = "ODIM"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [360, 360]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "dataset{}"
    mdesc = "data{}"


@requires_xarray_backend_api
class TestSyntheticOdimVolume04(SyntheticDataVolume):
    name = "base_odim_data_03"
    format = "ODIM"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [360, 360]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "dataset{}"
    mdesc = "data{}"


@requires_xarray_backend_api
class TestSyntheticGamicVolume01(SyntheticDataVolume):
    name = "base_gamic_data"
    format = "GAMIC"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [360, 360]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "scan{}"
    mdesc = "moment_{}"
