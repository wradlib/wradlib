# Copyright (c) 2011-2021, wradlib developers.
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
    h5py,
    has_data,
    requires_data,
    requires_h5netcdf,
    requires_h5py,
    requires_netcdf,
    requires_xarray_backend_api,
    requires_xmltodict,
)


def create_a1gate(i):
    return i + 20


def create_time():
    return xr.DataArray(1307700610.0, attrs=io.xarray.time_attrs)


def create_startazT(i, nrays=361):
    start = 1307700610.0
    arr = np.linspace(start, start + 360, 360, endpoint=False, dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate(i))
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_stopazT(i, nrays=360):
    start = 1307700611.0
    arr = np.linspace(start, start + 360, 360, endpoint=False, dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate(i))
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_startazA(nrays=360):
    arr = np.linspace(0, 360, 360, endpoint=False, dtype=np.float32)
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_stopazA(nrays=360):
    arr = np.linspace(1, 361, 360, endpoint=False, dtype=np.float32)
    # arr = np.arange(1, 361, 1, dtype=np.float32)
    arr[arr >= 360] -= 360
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_startelA(i, nrays=360):
    arr = np.ones(360, dtype=np.float32) * (i + 0.5)
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_stopelA(i, nrays=360):
    arr = np.ones(360, dtype=np.float32) * (i + 0.5)
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_ray_time(i, decode=False, nrays=360):
    time_data = (create_startazT(i, nrays=nrays) + create_stopazT(i, nrays=nrays)) / 2.0
    da = xr.DataArray(time_data, dims=["azimuth"], attrs=io.xarray.time_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({"arr": da})).arr
    return da


def create_azimuth(decode=False, nrays=360):
    startaz = create_startazA(nrays=nrays)
    stopaz = create_stopazA(nrays=nrays)
    zero_index = np.where(stopaz < startaz)
    stopaz[zero_index[0]] += 360
    azimuth_data = (startaz + stopaz) / 2.0
    da = xr.DataArray(
        azimuth_data, dims=["azimuth"], attrs=io.xarray.az_attrs_template.copy()
    )
    if decode:
        da = xr.decode_cf(xr.Dataset({"arr": da})).arr
    return da


def create_elevation(i, decode=False, nrays=360):
    startel = create_startelA(i, nrays=nrays)
    stopel = create_stopelA(i, nrays=nrays)
    elevation_data = (startel + stopel) / 2.0
    da = xr.DataArray(
        elevation_data, dims=["azimuth"], attrs=io.xarray.el_attrs_template.copy()
    )
    if decode:
        da = xr.decode_cf(xr.Dataset({"arr": da})).arr
    return da


def create_range(i, decode=False):
    where = create_dset_where(i)
    ngates = where["nbins"]
    range_start = where["rstart"] * 1000.0
    bin_range = where["rscale"]
    cent_first = range_start + bin_range / 2.0
    range_data = np.arange(
        cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
    )
    range_attrs = io.xarray.range_attrs
    range_attrs["meters_to_center_of_first_gate"] = cent_first[0]
    range_attrs["meters_between_gates"] = bin_range[0]
    da = xr.DataArray(range_data, dims=["range"], attrs=range_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({"arr": da})).arr
    return da


def create_root_where():
    return {"height": 99.5, "lon": 7.071624, "lat": 50.730599}


def create_root_what():
    return {"version": "9"}


def get_group_attrs(data, dsdesc, grp=None):
    if grp is not None:
        try:
            grp = data[dsdesc].get(grp, None)
        except KeyError:
            grp = data.get(dsdesc + "/" + grp, None)
    else:
        try:
            grp = data[dsdesc]
        except KeyError:
            pass
    if grp == {}:
        grp = None
    if grp:
        try:
            grp = grp["attrs"]
        except KeyError:
            pass
        for k, v in grp.items():
            try:
                v = v.item()
            except (ValueError, AttributeError):
                pass
            try:
                v = v.decode()
            except (UnicodeDecodeError, AttributeError):
                pass
            grp[k] = v
    return grp


def create_site(data):
    for k, v in data.items():
        try:
            data[k] = v.item()
        except AttributeError:
            pass
    site = xr.Dataset(coords=data)
    site = site.rename({"height": "altitude", "lon": "longitude", "lat": "latitude"})
    return site


def create_dset_how(i, nrays=360):
    return {
        "startazA": create_startazA(nrays=nrays),
        "stopazA": create_stopazA(nrays=nrays),
        "startelA": create_startelA(i, nrays=nrays),
        "stopelA": create_stopelA(i, nrays=nrays),
        "startazT": create_startazT(i, nrays=nrays),
        "stopazT": create_stopazT(i, nrays=nrays),
    }


def create_dset_where(i, nrays=360):
    return {
        "a1gate": np.array([create_a1gate(i)], dtype=np.int_),
        "elangle": np.array([i + 0.5], dtype=np.float32),
        "nrays": np.array([nrays], dtype=np.int_),
        "nbins": np.array([100], dtype=np.int_),
        "rstart": np.array([0], dtype=np.float32),
        "rscale": np.array([1000], dtype=np.float32),
    }


def create_dset_what():
    return {
        "startdate": np.array([b"20110610"], dtype="|S9"),
        "starttime": np.array([b"101010"], dtype="|S7"),
        "enddate": np.array([b"20110610"], dtype="|S9"),
        "endtime": np.array([b"101610"], dtype="|S7"),
    }


def create_dbz_what():
    return {
        "gain": np.array([0.5], dtype=np.float32),
        "nodata": np.array([255.0], dtype=np.float32),
        "offset": np.array([-31.5], dtype=np.float32),
        "quantity": np.array([b"DBZH"], dtype="|S5"),
        "undetect": np.array([0.0], dtype=np.float32),
    }


def create_data(nrays=360):
    np.random.seed(42)
    data = np.random.randint(0, 255, (360, 100), dtype=np.uint8)
    if nrays == 361:
        data = np.insert(data, 10, data[-1], axis=0)
    return data


def create_dataset(i, type=None, nrays=360):
    what = create_dbz_what()
    attrs = {}
    attrs["scale_factor"] = what["gain"]
    attrs["add_offset"] = what["offset"]
    attrs["_FillValue"] = what["nodata"]
    attrs["_Undetect"] = what["undetect"]

    if type == "GAMIC":
        attrs["add_offset"] -= 0.5 + 127.5 / 254
        attrs["scale_factor"] = 127.5 / 254
        attrs["_FillValue"] = what["undetect"]
        attrs["_Undetect"] = what["undetect"]

    attrs["coordinates"] = b"elevation azimuth range"
    ds = xr.Dataset({"DBZH": (["azimuth", "range"], create_data(nrays=nrays), attrs)})
    return ds


def create_coords(i, nrays=360):
    ds = xr.Dataset(
        coords={
            "time": create_time(),
            "rtime": create_ray_time(i, nrays=nrays),
            "azimuth": create_azimuth(nrays=nrays),
            "elevation": create_elevation(i, nrays=nrays),
            "range": create_range(i),
        }
    )
    return ds


def base_odim_data_00(nrays=360):
    data = {}
    root_attrs = [("Conventions", np.array([b"ODIM_H5/V2_0"], dtype="|S13"))]
    data["attrs"] = root_attrs
    foo_data = create_data(nrays=nrays)

    dataset = ["dataset1", "dataset2"]
    datas = ["data1"]

    data["where"] = {}
    data["where"]["attrs"] = create_root_where()
    data["what"] = {}
    data["what"]["attrs"] = create_root_what()
    for i, grp in enumerate(dataset):
        sub = {}
        sub["how"] = {}
        sub["where"] = {}
        sub["where"]["attrs"] = create_dset_where(i, nrays=nrays)
        sub["what"] = {}
        sub["what"]["attrs"] = create_dset_what()
        for j, mom in enumerate(datas):
            sub2 = {}
            sub2["data"] = foo_data
            sub2["what"] = {}
            sub2["what"]["attrs"] = create_dbz_what()
            sub[mom] = sub2
        data[grp] = sub
    return data


def base_odim_data_01():
    data = base_odim_data_00()
    dataset = ["dataset1", "dataset2"]
    for i, grp in enumerate(dataset):
        sub = data[grp]
        sub["how"] = {}
        sub["how"]["attrs"] = create_dset_how(i)
    return data


def base_odim_data_02():
    data = base_odim_data_00(nrays=361)
    dataset = ["dataset1", "dataset2"]
    for i, grp in enumerate(dataset):
        sub = data[grp]
        sub["how"] = {}
        sub["how"]["attrs"] = create_dset_how(i, nrays=361)
    return data


def base_odim_data_03():
    data = base_odim_data_00()
    dataset = ["dataset1", "dataset2"]
    for i, grp in enumerate(dataset):
        sub = data[grp]
        sub["how"] = {}
        sub["how"]["attrs"] = create_dset_how(i)
        sub["how"]["attrs"]["startelA"][0] = 10.0
    return data


def base_gamic_data():
    data = {}
    foo_data = create_data()
    dataset = ["scan0", "scan1"]
    datas = ["moment_0"]

    dt_type = np.dtype(
        {
            "names": [
                "azimuth_start",
                "azimuth_stop",
                "elevation_start",
                "elevation_stop",
                "timestamp",
            ],
            "formats": ["<f8", "<f8", "<f8", "<f8", "<i8"],
            "offsets": [0, 8, 16, 24, 32],
            "itemsize": 40,
        }
    )

    data["where"] = {}
    data["where"]["attrs"] = create_root_where()
    data["what"] = {}
    data["what"]["attrs"] = create_root_what()

    for i, grp in enumerate(dataset):
        sub = {}
        sub["how"] = {}
        sub["how"]["attrs"] = {
            "range_samples": 1.0,
            "range_step": 1000.0,
            "ray_count": 360,
            "bin_count": 100,
            "timestamp": b"2011-06-10T10:10:10.000Z",
            "elevation": i + 0.5,
        }
        for j, mom in enumerate(datas):
            sub2 = {}
            sub2["data"] = np.roll(foo_data, shift=-create_a1gate(i), axis=0)
            sub2["attrs"] = {
                "dyn_range_min": -32.0,
                "dyn_range_max": 95.5,
                "format": b"UV8",
                "moment": b"Zh",
                "unit": b"dBZ",
            }

            rh = np.zeros((360,), dtype=dt_type)
            rh["azimuth_start"] = np.roll(
                create_startazA(), shift=(360 - create_a1gate(i))
            )
            rh["azimuth_stop"] = np.roll(
                create_stopazA(), shift=(360 - create_a1gate(i))
            )
            rh["elevation_start"] = create_startelA(i)
            rh["elevation_stop"] = create_stopelA(i)
            rh["timestamp"] = np.roll(
                create_ray_time(i).values * 1e6, shift=-create_a1gate(i)
            )
            sub[mom] = sub2
            sub["ray_header"] = rh

        data[grp] = sub
    return data


def write_odim_dataset(grp, data):
    grp.create_dataset("data", data=data)


def write_gamic_dataset(grp, name, data):
    da = grp.create_dataset(name, data=data["data"])
    da.attrs.update(data["attrs"])


def write_gamic_ray_header(grp, data):
    dt_type = np.dtype(
        {
            "names": [
                "azimuth_start",
                "azimuth_stop",
                "elevation_start",
                "elevation_stop",
                "timestamp",
            ],
            "formats": ["<f8", "<f8", "<f8", "<f8", "<i8"],
            "offsets": [0, 8, 16, 24, 32],
            "itemsize": 40,
        }
    )
    rh = grp.create_dataset("ray_header", (360,), dtype=dt_type)
    rh[...] = data


def write_group(grp, data):
    for k, v in data.items():
        if k == "attrs":
            grp.attrs.update(v)
        elif k == "data":
            write_odim_dataset(grp, v)
        elif "moment" in k:
            write_gamic_dataset(grp, k, v)
        elif "ray_header" in k:
            write_gamic_ray_header(grp, v)
        else:
            if v:
                subgrp = grp.create_group(k)
                write_group(subgrp, v)


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


@contextlib.contextmanager
def get_synthetic_volume(name, file_or_filelike, **kwargs):
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=name).name
    if "gamic" in name:
        format = "GAMIC"
    else:
        format = "ODIM"
    pytest.importorskip("h5py")
    with h5py.File(str(tmp_local), "w") as f:
        data = globals()[name]()
        write_group(f, data)
    with get_wradlib_data_file(tmp_local, file_or_filelike) as h5file:
        engine = format.lower()
        if engine == "odim":
            pytest.importorskip("h5netcdf")
            open_ = io.open_odim_dataset
        if engine == "gamic":
            pytest.importorskip("h5netcdf")
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
            vol.to_odim(tmp_local)
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


class SyntheticDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, fileobj, **kwargs):
        with get_synthetic_volume(
            self.name, fileobj, backend_kwargs=self.backend_kwargs, **kwargs
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

    backend_kwargs = dict(reindex_angle=1.0, obsmode=1)


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

    backend_kwargs = dict(reindex_angle=1.0, obsmode=1)


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

    backend_kwargs = dict(keep_azimuth=True)


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

    backend_kwargs = dict(keep_azimuth=True)


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

    backend_kwargs = dict(keep_azimuth=True)


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

    backend_kwargs = dict(keep_azimuth=True)


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

    backend_kwargs = dict(keep_azimuth=True)
