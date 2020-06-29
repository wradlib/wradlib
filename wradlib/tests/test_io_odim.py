# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc

import h5py
import numpy as np
import pytest
import xarray as xr

from wradlib import io, util

from . import has_data, requires_data


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
    da = xr.DataArray(azimuth_data, dims=["azimuth"], attrs=io.xarray.az_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({"arr": da})).arr
    return da


def create_elevation(i, decode=False, nrays=360):
    startel = create_startelA(i, nrays=nrays)
    stopel = create_stopelA(i, nrays=nrays)
    elevation_data = (startel + stopel) / 2.0
    da = xr.DataArray(elevation_data, dims=["azimuth"], attrs=io.xarray.el_attrs)
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
        "a1gate": np.array([create_a1gate(i)], dtype=np.int),
        "elangle": np.array([i + 0.5], dtype=np.float32),
        "nrays": np.array([nrays], dtype=np.int),
        "nbins": np.array([100], dtype=np.int),
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
    if type == "GAMIC":
        attrs["add_offset"] -= 0.5
    attrs["_FillValue"] = what["nodata"]
    attrs["coordinates"] = b"elevation azimuth range"
    attrs["_Undetect"] = what["undetect"]
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


@contextlib.contextmanager
def get_measured_volume(file, get_loader, format):
    h5file = util.get_wradlib_data_file(file)
    yield io.xarray.open_odim(h5file, loader=get_loader, flavour=format)


@contextlib.contextmanager
def get_synthetic_volume(name, get_loader, **kwargs):
    import tempfile

    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=name).name
    if "gamic" in name:
        format = "GAMIC"
    else:
        format = "ODIM"
    with h5py.File(str(tmp_local), "w") as f:
        data = globals()[name]()
        write_group(f, data)
    yield io.xarray.open_odim(tmp_local, loader=get_loader, flavour=format, **kwargs)


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


@pytest.fixture(params=["h5py", "h5netcdf", "netcdf4"])
def get_loader(request):
    return request.param


@pytest.fixture(params=[360, 361])
def get_nrays(request):
    return request.param


def create_volume_repr(swp, ele):
    repr = "".join(
        [
            "<wradlib.XRadVolume>\n",
            f"Dimension(s): (sweep: {swp})\n",
            f"Elevation(s): {tuple(ele)}",
        ]
    )
    return repr


def create_timeseries_repr(time, azi, range, ele):
    repr = "".join(
        [
            "<wradlib.XRadTimeSeries>\n",
            f"Dimension(s): (time: {time}, azimuth: {azi}, ",
            f"range: {range})\n",
            f"Elevation(s): ({ele})",
        ]
    )
    return repr


def create_sweep_repr(format, azi, range, ele, mom):
    format = format.lower().capitalize()
    repr = "".join(
        [
            f"<wradlib.XRadSweep{format}>\n",
            f"Dimension(s): (azimuth: {azi}, range: {range})\n",
            f"Elevation(s): ({ele})\n",
            f'Moment(s): ({", ".join(mom)})',
        ]
    )
    return repr


def create_moment_repr(azi, range, ele, mom):
    repr = "".join(
        [
            "<wradlib.XRadMoment>\n",
            f"Dimension(s): (azimuth: {azi}, range: {range})\n",
            f"Elevation(s): ({ele})\n",
            f"Moment: ({mom})",
        ]
    )
    return repr


class DataMoment:
    def test_moments(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
            num = 1 if self.format == "ODIM" else 0
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    for k, mom in enumerate(swp):
                        repr = create_moment_repr(
                            self.azimuths[i],
                            self.ranges[i],
                            self.elevations[i],
                            self.moments[k],
                        )
                    assert isinstance(mom, io.xarray.XRadMoment)
                    assert mom.__repr__() == repr
                    assert mom.engine == engine
                    assert self.name.split("/")[-1] in mom.filename
                    assert mom.quantity == self.moments[k]
                    assert mom.parent == vol[i][j]
                    ncpath = "/".join([self.dsdesc, self.mdesc]).format(
                        i + num, k + num
                    )
                    assert mom.ncpath == ncpath
                    assert mom.ncid == mom.ncfile[mom.ncpath]
        del mom
        del ts
        del swp
        del vol
        gc.collect()

    def test_moment_data(self, get_loader):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(
            get_loader,
            decode_coords=False,
            mask_and_scale=False,
            decode_times=False,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                if "02" in self.name:
                    ds = create_dataset(i, nrays=361)["DBZH"]
                else:
                    ds = create_dataset(i)["DBZH"]
                for j, swp in enumerate(ts):
                    for k, mom in enumerate(swp):
                        xr.testing.assert_equal(mom.data, ds)
        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=False,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    for k, mom in enumerate(swp):
                        data = create_dataset(i)
                        data = data.assign_coords(create_coords(i).coords)
                        data = data.assign_coords(
                            create_site(self.data["where"]["attrs"]).coords
                        )
                        data = data.assign_coords(
                            {"sweep_mode": "azimuth_surveillance"}
                        )
                        data = xr.decode_cf(data, mask_and_scale=False)
                        xr.testing.assert_equal(mom.data, data["DBZH"])

        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=True,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    for k, mom in enumerate(swp):
                        data = create_dataset(i, type=self.format)
                        data = data.assign_coords(create_coords(i).coords)
                        data = data.assign_coords(
                            create_site(self.data["where"]["attrs"]).coords
                        )
                        data = data.assign_coords(
                            {"sweep_mode": "azimuth_surveillance"}
                        )
                        data = xr.decode_cf(data)
                        xr.testing.assert_equal(mom.data, data["DBZH"])
        del mom
        del swp
        del ts
        del vol
        gc.collect()


class DataSweep(DataMoment):
    def test_sweeps(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            num = 1 if self.format == "ODIM" else 0
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    repr = create_sweep_repr(
                        self.format,
                        self.azimuths[i],
                        self.ranges[i],
                        self.elevations[i],
                        self.moments,
                    )
                    assert isinstance(swp, io.xarray.XRadSweep)
                    assert self.name.split("/")[-1] in swp.filename
                    assert swp.__repr__() == repr

                    # mixins
                    attrs = get_group_attrs(
                        self.data, self.dsdesc.format(i + num), "how"
                    )
                    np.testing.assert_equal(swp.how, attrs)
                    attrs = get_group_attrs(
                        self.data, self.dsdesc.format(i + num), "what"
                    )
                    assert swp.what == attrs
                    attrs = get_group_attrs(
                        self.data, self.dsdesc.format(i + num), "where"
                    )
                    assert swp.where == attrs

                    # methods
                    if self.name == "base_odim_data_00":
                        with pytest.raises(TypeError):
                            swp._get_azimuth_how()
        del swp
        del ts
        del vol
        gc.collect()

    def test_sweep_data(self, get_loader):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(
            get_loader,
            decode_coords=False,
            mask_and_scale=False,
            decode_times=False,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                if "02" in self.name:
                    ds = create_dataset(i, nrays=361)
                else:
                    ds = create_dataset(i)
                for j, swp in enumerate(ts):
                    xr.testing.assert_equal(swp.data, ds)
        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=False,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    data = create_dataset(i)
                    data = data.assign_coords(create_coords(i).coords)
                    data = data.assign_coords(
                        create_site(self.data["where"]["attrs"]).coords
                    )
                    data = data.assign_coords({"sweep_mode": "azimuth_surveillance"})
                    data = xr.decode_cf(data, mask_and_scale=False)
                    xr.testing.assert_equal(swp.data, data)
        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=True,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                for j, swp in enumerate(ts):
                    data = create_dataset(i, type=self.format)
                    data = data.assign_coords(create_coords(i).coords)
                    data = data.assign_coords(
                        create_site(self.data["where"]["attrs"]).coords
                    )
                    data = data.assign_coords({"sweep_mode": "azimuth_surveillance"})
                    data = xr.decode_cf(data)
                    xr.testing.assert_equal(swp.data, data)
        del swp
        del ts
        del vol
        gc.collect()

    def test_sweep_coords_data(self, get_loader):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(
            get_loader,
            decode_coords=False,
            mask_and_scale=False,
            decode_times=False,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                if "02" in self.name:
                    ds = create_coords(i, nrays=361)
                else:
                    ds = create_coords(i)
                for j, swp in enumerate(ts):
                    xr.testing.assert_equal(swp.coords, ds)
        del swp
        del ts
        del vol
        gc.collect()

    def test_sweep_errors(self, get_loader):
        if not (get_loader == "netcdf4" and self.format == "GAMIC"):
            pytest.skip("only test gamic using netcdf4")
        with pytest.raises(ValueError):
            with self.get_volume_data(
                get_loader,
                decode_coords=False,
                mask_and_scale=False,
                decode_times=False,
                chunks=None,
                parallel=False,
            ) as vol:
                print(vol)


class DataTimeSeries(DataSweep):
    def test_timeseries(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
            num = 1 if self.format == "ODIM" else 0
            for i, ts in enumerate(vol):
                repr = create_timeseries_repr(
                    self.volumes, self.azimuths[i], self.ranges[i], self.elevations[i]
                )
                assert isinstance(ts, io.xarray.XRadTimeSeries)
                assert ts.__repr__() == repr
                assert ts.engine == engine
                assert ts.ncid == ts.ncfile[ts.ncpath]
                assert ts.ncpath == self.dsdesc.format(i + num)
                assert ts.parent == vol
                assert self.name.split("/")[-1] in ts.filename
        del ts
        del vol
        gc.collect()

    def test_timeseries_data(self, get_loader):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(
            get_loader,
            decode_coords=False,
            mask_and_scale=False,
            decode_times=False,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                if "02" in self.name:
                    ds = create_dataset(i, type=self.format, nrays=361)
                else:
                    ds = create_dataset(i, type=self.format)
                xr.testing.assert_equal(ts.data, ds.expand_dims("time"))

        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=False,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                data = create_dataset(i, type=self.format)
                data = data.assign_coords(create_coords(i).coords)
                data = data.assign_coords(
                    create_site(self.data["where"]["attrs"]).coords
                )
                data = data.assign_coords({"sweep_mode": "azimuth_surveillance"})
                data = xr.decode_cf(data, mask_and_scale=False)
                xr.testing.assert_equal(ts.data, data.expand_dims("time"))

        with self.get_volume_data(
            get_loader,
            decode_coords=True,
            mask_and_scale=True,
            decode_times=True,
            chunks=None,
            parallel=False,
        ) as vol:
            for i, ts in enumerate(vol):
                data = create_dataset(i, type=self.format)
                data = data.assign_coords(create_coords(i).coords)
                data = data.assign_coords(
                    create_site(self.data["where"]["attrs"]).coords
                )
                data = data.assign_coords({"sweep_mode": "azimuth_surveillance"})
                data = xr.decode_cf(data)
                xr.testing.assert_equal(ts.data, data.expand_dims("time"))

        del ts
        del vol
        gc.collect()


class DataVolume(DataTimeSeries):
    def test_volumes(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
            assert isinstance(vol, io.xarray.XRadVolume)
            repr = create_volume_repr(self.sweeps, self.elevations)
            assert vol.__repr__() == repr
            assert vol.engine == engine
            # assert vol.filename == odim_data[0]
            assert self.name.split("/")[-1] in vol.filename
            assert vol.ncid == vol.ncfile
            assert vol.ncpath == "/"
            assert vol.parent is None
            xr.testing.assert_equal(vol.site, create_site(self.data["where"]["attrs"]))

            # mixins
            assert vol.how == get_group_attrs(self.data, "how")
            assert vol.what == get_group_attrs(self.data, "what")
            assert vol.where == get_group_attrs(self.data, "where")

        del vol
        gc.collect()

    def test_odim_output(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="odim").name
            vol.to_odim(tmp_local)
        del vol
        gc.collect()

    def test_cfradial2_output(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_cfradial2(tmp_local)
        del vol
        gc.collect()

    def test_netcdf_output(self, get_loader):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        with self.get_volume_data(get_loader) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_netcdf(tmp_local, timestep=slice(0, None))
        del vol
        gc.collect()


class MeasuredDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, get_loader, **kwargs):
        with get_measured_volume(self.name, get_loader, self.format) as vol:
            yield vol


class SyntheticDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, get_loader, **kwargs):
        with get_synthetic_volume(self.name, get_loader, **kwargs) as vol:
            yield vol


@requires_data
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
        azimuths = [361, 361, 361, 360, 361, 360, 360, 361, 360, 360]
        ranges = [360, 500, 620, 800, 1050, 1400, 1000, 1000, 1000, 1000]

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "scan{}"
        mdesc = "moment_{}"


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


class TestSyntheticOdimVolume03(SyntheticDataVolume):
    name = "base_odim_data_02"
    format = "ODIM"
    volumes = 1
    sweeps = 2
    moments = ["DBZH"]
    elevations = [0.5, 1.5]
    azimuths = [361, 361]
    ranges = [100, 100]

    data = globals()[name]()

    dsdesc = "dataset{}"
    mdesc = "data{}"


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
