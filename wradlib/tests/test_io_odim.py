# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc

import numpy as np
import pytest
import xarray as xr

from wradlib import io, util

from . import (
    get_wradlib_data_file,
    h5py,
    has_data,
    requires_data,
    requires_h5netcdf,
    requires_h5py,
    requires_netcdf,
)
from .test_io_backends import (  # noqa: F401
    base_gamic_data,
    base_odim_data_00,
    base_odim_data_01,
    base_odim_data_02,
    base_odim_data_03,
    create_azimuth,
    create_coords,
    create_data,
    create_dataset,
    create_dbz_what,
    create_elevation,
    create_range,
    create_ray_time,
    create_site,
    create_time,
    get_group_attrs,
    write_group,
)


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@contextlib.contextmanager
def get_measured_volume(file, loader, format, fileobj):
    if loader == "noloader":
        pass
    elif loader == "netcdf4":
        pytest.importorskip("netCDF4")
    else:
        pytest.importorskip(loader)
        pytest.importorskip("h5netcdf")
    with get_wradlib_data_file(file, fileobj) as h5file:
        yield io.xarray_depr.open_odim(h5file, loader=loader, flavour=format)


@contextlib.contextmanager
def get_synthetic_volume(name, get_loader, file_or_filelike, **kwargs):
    import tempfile

    if get_loader == "noloader":
        pass
    elif get_loader == "netcdf4":
        pytest.importorskip("netCDF4")
    else:
        pytest.importorskip(get_loader)
        pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")

    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=name).name
    if "gamic" in name:
        format = "GAMIC"
    else:
        format = "ODIM"
    with h5py.File(str(tmp_local), "w") as f:
        data = globals()[name]()
        write_group(f, data)
    with get_wradlib_data_file(tmp_local, file_or_filelike) as h5file:
        yield io.xarray_depr.open_odim(
            h5file, loader=get_loader, flavour=format, **kwargs
        )


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
    def test_moments(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
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
                    assert isinstance(mom, io.xarray_depr.XRadMoment)
                    assert mom.__repr__() == repr
                    assert mom.engine == engine
                    if file_or_filelike == "file":
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

    def test_moment_data(self, get_loader, file_or_filelike):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(
            get_loader,
            file_or_filelike,
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
            file_or_filelike,
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
            file_or_filelike,
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
    def test_sweeps(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
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
                    assert isinstance(swp, io.xarray_depr.XRadSweep)
                    if file_or_filelike == "file":
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

    def test_sweep_data(self, get_loader, file_or_filelike):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(
            get_loader,
            file_or_filelike,
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
            file_or_filelike,
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
            file_or_filelike,
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

    def test_sweep_coords_data(self, get_loader, file_or_filelike):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(
            get_loader,
            file_or_filelike,
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

    def test_sweep_errors(self, get_loader, file_or_filelike):
        if not (get_loader == "netcdf4" and self.format == "GAMIC"):
            pytest.skip("only test gamic using netcdf4")
        with pytest.raises(ValueError):
            with self.get_volume_data(
                get_loader,
                file_or_filelike,
                decode_coords=False,
                mask_and_scale=False,
                decode_times=False,
                chunks=None,
                parallel=False,
            ) as vol:
                print(vol)


class DataTimeSeries(DataSweep):
    def test_timeseries(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
            engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
            num = 1 if self.format == "ODIM" else 0
            for i, ts in enumerate(vol):
                repr = create_timeseries_repr(
                    self.volumes, self.azimuths[i], self.ranges[i], self.elevations[i]
                )
                assert isinstance(ts, io.xarray_depr.XRadTimeSeries)
                assert ts.__repr__() == repr
                assert ts.engine == engine
                assert ts.ncid == ts.ncfile[ts.ncpath]
                assert ts.ncpath == self.dsdesc.format(i + num)
                assert ts.parent == vol
                if file_or_filelike == "file":
                    assert self.name.split("/")[-1] in ts.filename
        del ts
        del vol
        gc.collect()

    def test_timeseries_data(self, get_loader, file_or_filelike):
        if isinstance(self, MeasuredDataVolume):
            pytest.skip("requires synthetic data")
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(
            get_loader,
            file_or_filelike,
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
            file_or_filelike,
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
            file_or_filelike,
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
    def test_unknown_loader_error(self):
        with pytest.raises(ValueError) as err:
            with self.get_volume_data("noloader", "file") as vol:
                print(vol)
        assert "Unknown loader" in str(err.value)

    def test_gamic_netcdf4_error(self):
        if self.format != "GAMIC":
            pytest.skip("need GAMIC file")
        with pytest.raises(ValueError) as err:
            with self.get_volume_data("netcdf4", "file") as vol:
                print(vol)
        assert "GAMIC files can't be read using netcdf4" in str(err.value)

    def test_file_like_h5py_error(self):
        with pytest.raises(ValueError) as err:
            with self.get_volume_data("h5py", "filelike") as vol:
                print(vol)
        assert "file-like objects can't be read using h5py" in str(err.value)

    def test_volumes(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
            engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
            assert isinstance(vol, io.xarray_depr.XRadVolume)
            repr = create_volume_repr(self.sweeps, self.elevations)
            assert vol.__repr__() == repr
            assert vol.engine == engine
            # assert vol.filename == odim_data[0]
            if file_or_filelike == "file":
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

    @requires_h5py
    def test_odim_output(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="odim").name
            vol.to_odim(tmp_local)
        del vol
        gc.collect()

    @requires_netcdf
    def test_cfradial2_output(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_cfradial2(tmp_local)
        del vol
        gc.collect()

    @requires_netcdf
    def test_netcdf_output(self, get_loader, file_or_filelike):
        if get_loader == "netcdf4" and self.format == "GAMIC":
            pytest.skip("gamic needs hdf-based loader")
        if get_loader == "h5py" and file_or_filelike == "filelike":
            pytest.skip("file_like needs 'h5netcdf' or 'netcdf4'")
        with self.get_volume_data(get_loader, file_or_filelike) as vol:
            import tempfile

            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_netcdf(tmp_local, timestep=slice(0, None))
        del vol
        gc.collect()


class MeasuredDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, loader, fileobj, **kwargs):
        with get_measured_volume(self.name, loader, self.format, fileobj) as vol:
            yield vol

    @property
    def data(self):
        return self._data


class SyntheticDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, loader, fileobj, **kwargs):
        with get_synthetic_volume(self.name, loader, fileobj, **kwargs) as vol:
            yield vol


@requires_data
@requires_h5netcdf
@requires_netcdf
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

        _data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

    # @property
    # def data(self):
    #     return io.read_generic_hdf5(util.get_wradlib_data_file(self.name))


@requires_data
@requires_h5netcdf
@requires_netcdf
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

        dsdesc = "scan{}"
        mdesc = "moment_{}"

        _data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

    # @property
    # def data(self):
    #     return io.read_generic_hdf5(util.get_wradlib_data_file(self.name))


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
