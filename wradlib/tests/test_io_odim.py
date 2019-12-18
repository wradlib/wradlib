# Copyright (c) 2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import gc
import contextlib
import pytest
import numpy as np
import xarray as xr
import h5py

from wradlib import io, util


def create_a1gate():
    return 20


def create_time():
    return xr.DataArray(1307700610.0, attrs=io.xarray.time_attrs)


def create_startazT():
    arr = np.arange(1307700610.0, 1307700970.0, 1.0,
                    dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate())
    return arr


def create_stopazT():
    arr = np.arange(1307700611.0, 1307700971.0, 1.0, dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate())
    return arr


def create_startazA():
    arr = np.arange(0, 360, 1, dtype=np.float32)
    return arr


def create_stopazA():
    arr = np.arange(1, 361, 1, dtype=np.float32)
    arr[arr >= 360] -= 360
    return arr


def create_startelA():
    arr = np.ones(360, dtype=np.float32) * 0.5
    return arr


def create_stopelA():
    arr = np.ones(360, dtype=np.float32) * 0.5
    return arr


def create_ray_time(decode=False):
    time_data = (create_startazT() + create_stopazT()) / 2.
    da = xr.DataArray(time_data, dims=['azimuth'],
                      attrs=io.xarray.time_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({'arr': da})).arr
    return da


def create_azimuth(decode=False):
    startaz = create_startazA()
    stopaz = create_stopazA()
    zero_index = np.where(stopaz < startaz)
    stopaz[zero_index[0]] += 360
    azimuth_data = (startaz + stopaz) / 2.
    da = xr.DataArray(azimuth_data, dims=['azimuth'],
                      attrs=io.xarray.az_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({'arr': da})).arr
    return da


def create_elevation(decode=False):
    startel = create_startelA()
    stopel = create_stopelA()
    elevation_data = (startel + stopel) / 2.
    da = xr.DataArray(elevation_data, dims=['azimuth'],
                      attrs=io.xarray.el_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({'arr': da})).arr
    return da


def create_range(i, decode=False):
    where = create_dset_where(i)
    ngates = where['nbins']
    range_start = where['rstart'] * 1000.
    bin_range = where['rscale']
    cent_first = range_start + bin_range / 2.
    range_data = np.arange(cent_first,
                           range_start + bin_range * ngates,
                           bin_range,
                           dtype='float32')
    range_attrs = io.xarray.range_attrs
    range_attrs[
        'meters_to_center_of_first_gate'] = cent_first[0]
    range_attrs[
        'meters_between_gates'] = bin_range[0]
    da = xr.DataArray(range_data, dims=['range'], attrs=range_attrs)
    if decode:
        da = xr.decode_cf(xr.Dataset({'arr': da})).arr
    return da


def create_root_where():
    return {'height': 99.5,
            'lon': 7.071624,
            'lat': 50.730599}


def create_site():
    site = xr.Dataset(coords=create_root_where())
    site = site.rename({'height': 'altitude',
                        'lon': 'longitude',
                        'lat': 'latitude'})
    return site


def create_dset_how():
    return {'startazA': create_startazA(),
            'stopazA': create_stopazA(),
            'startelA': np.ones(360) * 0.5,
            'stopelA':  np.ones(360) * 0.5,
            'startazT': create_startazT(),
            'stopazT': create_stopazT()}


def create_dset_where(i):
    return {'a1gate': np.array([i + create_a1gate()], dtype=np.int),
            'elangle': np.array([i + 0.5], dtype=np.float32),
            'nrays': np.array([360], dtype=np.int),
            'nbins': np.array([100], dtype=np.int),
            'rstart': np.array([0], dtype=np.float32),
            'rscale': np.array([1000], dtype=np.float32)}


def create_dset_what():
    return {'startdate': np.array([b'20110610'], dtype='|S9'),
            'starttime': np.array([b'101010'], dtype='|S7'),
            'enddate': np.array([b'20110610'], dtype='|S9'),
            'endtime': np.array([b'101610'], dtype='|S7')}


def create_dbz_what():
    return {'gain': np.array([0.5], dtype=np.float32),
            'nodata': np.array([255.], dtype=np.float32),
            'offset': np.array([-31.5], dtype=np.float32),
            'quantity': np.array([b'DBZH'], dtype='|S5'),
            'undetect': np.array([0.], dtype=np.float32)}


def create_data():
    np.random.seed(42)
    return np.random.randint(0, 255, (360, 100), dtype=np.uint8)


def create_dataset(type=None):
    what = create_dbz_what()
    attrs = {}
    attrs['scale_factor'] = what['gain']
    attrs['add_offset'] = what['offset']
    attrs['_FillValue'] = what['nodata']
    attrs['coordinates'] = 'elevation azimuth range'
    attrs['_Undetect'] = what['undetect']
    ds = xr.Dataset({'DBZH': (['azimuth', 'range'], create_data(), attrs)})
    if type is None:
        return ds
    return ds


def create_coords(i):
    ds = xr.Dataset(coords={'time':  create_time(),
                            'rtime': create_ray_time(),
                            'azimuth': create_azimuth(),
                            'elevation': create_elevation(),
                            'range': create_range(i)})
    return ds


@contextlib.contextmanager
def get_odim_volume():
    filename = 'hdf5/knmi_polar_volume.h5'
    h5file = util.get_wradlib_data_file(filename)
    yield h5file


def odim_data_00(f):
    foo_data = create_data()
    dataset = ['dataset1', 'dataset2']
    f.attrs.update([('Conventions', np.array([b'ODIM_H5/V2_0'],
                                             dtype='|S13'))])
    root_where = f.create_group('where')
    root_where.attrs.update(create_root_where())
    for i, grp in enumerate(dataset):
        dset = f.create_group(grp)
        where = dset.create_group('where')
        where.attrs.update(create_dset_where(i))
        what = dset.create_group('what')
        what.attrs.update(create_dset_what())
        da = dset.create_group('data1')
        da.create_dataset('data', data=foo_data)
        ds_what = da.create_group('what')
        ds_what.attrs.update(create_dbz_what())


def odim_data_01(f):
    odim_data_00(f)


def odim_data_02(f):
    odim_data_00(f)
    dataset = ['dataset1', 'dataset2']
    for i, grp in enumerate(dataset):
        dset = f[grp]
        how = dset.create_group('how')
        how.attrs.update(create_dset_how())


@pytest.fixture(params=['data1', 'data2'])
def get_odim_data(request):
    return int(request.param[4:])


@pytest.fixture
def odim_data(tmpdir_factory, get_odim_data):
    fname = f"test_odim_{get_odim_data:02d}.h5"
    tmp_local = tmpdir_factory.mktemp("data").join(fname)
    datasets = {1: odim_data_01,
                2: odim_data_02}
    with h5py.File(str(tmp_local), 'w') as f:
        datasets[get_odim_data](f)
    return tmp_local


class XRadVolume:

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        yield io.xarray.open_odim(path, loader=self.loader, **kwargs)

    def test_open_sweep(self):
        with get_odim_volume() as vol_file:
            with self.open(vol_file) as vol:
                assert isinstance(vol, io.xarray.XRadVolume)
                repr = ''.join(['<wradlib.XRadVolume>\n',
                                'Dimensions: (sweep: 14)\n',
                                'Elevations: (0.3, 0.4, 0.8, 1.1, 2.0,',
                                ' 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, ',
                                '15.0, 20.0, 25.0)'])
                assert vol.__repr__() == repr
                assert isinstance(vol[0], io.xarray.XRadTimeSeries)
                repr = ''.join(['<wradlib.XRadTimeSeries>\n',
                                'Dimensions: (time: 1, azimuth: 360, ',
                                'range: 320)\n',
                                'Elevation: (0.3)'])
                assert vol[0].__repr__() == repr
                assert isinstance(vol[0][0], io.xarray.XRadSweep)
                repr = ''.join(['<wradlib.XRadSweepOdim>\n',
                                'Dimensions: (azimuth: 360, range: 320)\n',
                                'Elevation: (0.3)\n',
                                'Moment(s): (DBZH)'])
                assert vol[0][0].__repr__() == repr
                assert isinstance(vol[0][0][0], io.xarray.XRadMoment)
                repr = ''.join(['<wradlib.XRadMoment>\n',
                                'Dimensions: (azimuth: 360, range: 320)\n',
                                'Elevation: (0.3)\n',
                                'Moment: (DBZH)'])
                assert vol[0][0][0].__repr__() == repr

        del vol
        gc.collect()

    def test_open_test_data(self, odim_data):
        with self.open(odim_data) as vol:
            assert isinstance(vol, io.xarray.XRadVolume)
            repr = ''.join(['<wradlib.XRadVolume>\n',
                            'Dimensions: (sweep: 2)\n',
                            'Elevations: (0.5, 1.5)'])
            assert vol.__repr__() == repr
            assert isinstance(vol[0], io.xarray.XRadTimeSeries)
            repr = ''.join(['<wradlib.XRadTimeSeries>\n',
                            'Dimensions: (time: 1, azimuth: 360, ',
                            'range: 100)\n',
                            'Elevation: (0.5)'])
            assert vol[0].__repr__() == repr
            assert isinstance(vol[0][0], io.xarray.XRadSweep)
            repr = ''.join(['<wradlib.XRadSweepOdim>\n',
                            'Dimensions: (azimuth: 360, range: 100)\n',
                            'Elevation: (0.5)\n',
                            'Moment(s): (DBZH)'])
            assert vol[0][0].__repr__() == repr
            assert isinstance(vol[0][0][0], io.xarray.XRadMoment)
            repr = ''.join(['<wradlib.XRadMoment>\n',
                            'Dimensions: (azimuth: 360, range: 100)\n',
                            'Elevation: (0.5)\n',
                            'Moment: (DBZH)'])
            assert vol[0][0][0].__repr__() == repr
        del vol
        gc.collect()

    def test_moment(self, odim_data):
        with self.open(odim_data) as vol:
            mom = vol[0][0][0]
            assert mom.engine == self.engine
            assert mom.filename == odim_data
            assert mom.quantity == 'DBZH'
            assert mom.parent == vol[0][0]
            assert mom.time.values == np.datetime64('2011-06-10T10:10:10')
            assert mom.ncpath == 'dataset1/data1'
            assert mom.ncid == mom.ncfile[mom.ncpath]
        del mom
        del vol
        gc.collect()

    def test_sweep(self, odim_data):
        with self.open(odim_data) as vol:
            sweep = vol[0][0]

            assert sweep.a1gate == 20
            np.testing.assert_array_equal(sweep.azimuth,
                                          create_azimuth())
            np.testing.assert_array_equal(sweep.elevation,
                                          np.ones((360)) * 0.5)
            assert sweep.engine == self.engine
            assert sweep.filename == odim_data
            assert sweep.fixed_angle == 0.5
            assert sweep.nbins == 100
            assert sweep.ncid == sweep.ncfile[sweep.ncpath]
            assert sweep.ncpath == 'dataset1'
            assert sweep.nrays == 360
            assert sweep.parent == vol[0]
            xr.testing.assert_equal(sweep.ray_times,
                                    create_ray_time(decode=True))
            assert sweep.time.values == np.datetime64(
                '2011-06-10T10:10:10.000000000')
        del sweep
        del vol
        gc.collect()

    def test_timeseries(self, odim_data):
        with self.open(odim_data) as vol:
            ts = vol[0]
            assert ts.engine == self.engine
            assert ts.filename == odim_data
            assert ts.ncid == ts.ncfile[ts.ncpath]
            assert ts.ncpath == 'dataset1'
            assert ts.parent == vol

        del ts
        del vol
        gc.collect()

    def test_volume(self, odim_data):
        with self.open(odim_data) as vol:
            assert vol.engine == self.engine
            assert vol.filename == odim_data
            assert vol.ncid == vol.ncfile
            assert vol.ncpath == '/'
            assert vol.parent is None
            xr.testing.assert_equal(vol.site, create_site())

        del vol
        gc.collect()

    def test_odimh5_group_mixin(self, odim_data, get_odim_data):
        with self.open(odim_data) as vol:
            # volume
            assert vol.how is None
            assert vol.what is None
            assert vol.where == {'height': 99.5,
                                 'lat': 50.730599,
                                 'lon': 7.071624}
            assert vol.attrs['Conventions'] == 'ODIM_H5/V2_0'
            assert vol.engine == self.engine
            assert vol.filename == odim_data
            assert vol.groups == ['dataset1', 'dataset2', 'where']
            assert vol.ncid == vol.ncfile
            assert vol.ncpath == '/'
            assert vol.parent is None

            # timeseries
            ts = vol[0]
            if get_odim_data == 1:
                assert ts.how is None
            else:
                np.testing.assert_equal(ts.how, create_dset_how())
            assert ts.what == {'enddate': '20110610', 'endtime': '101610',
                               'startdate': '20110610', 'starttime': '101010'}
            assert ts.where == {'a1gate': 20, 'elangle': 0.5, 'nbins': 100,
                                'nrays': 360, 'rscale': 1000.0, 'rstart': 0.0}
            assert ts.attrs == {}
            assert ts.engine == self.engine
            assert ts.filename == odim_data
            assert set(ts.groups) & set(['data1', 'what', 'where'])
            assert ts.ncid == ts.ncfile[ts.ncpath]
            assert ts.ncpath == 'dataset1'
            assert ts.parent == vol

            # sweep
            sweep = ts[0]
            if get_odim_data == 1:
                assert sweep.how is None
            else:
                np.testing.assert_equal(sweep.how, create_dset_how())
            assert sweep.what == {'enddate': '20110610', 'endtime': '101610',
                                  'startdate': '20110610',
                                  'starttime': '101010'}
            assert sweep.where == {'a1gate': 20, 'elangle': 0.5, 'nbins': 100,
                                   'nrays': 360, 'rscale': 1000.0,
                                   'rstart': 0.0}
            assert sweep.attrs == {}
            assert sweep.engine == self.engine
            assert sweep.filename == odim_data
            assert set(sweep.groups) & set(['data1', 'what', 'where'])
            assert sweep.ncid == sweep.ncfile[sweep.ncpath]
            assert sweep.ncpath == 'dataset1'
            assert sweep.parent == ts

            # moment
            mom = sweep[0]
            assert mom.how is None
            assert mom.what == {'gain': 0.5, 'nodata': 255.0, 'offset': -31.5,
                                'quantity': 'DBZH', 'undetect': 0.0}
            assert mom.where is None
            assert mom.attrs == {}
            assert mom.engine == self.engine
            assert mom.filename == odim_data
            assert mom.groups == self.mom_groups
            assert mom.ncid == mom.ncfile[mom.ncpath]
            assert mom.ncpath == 'dataset1/data1'
            assert mom.parent == sweep
        del mom
        del sweep
        del ts
        del vol
        gc.collect()

    def test_sweep_methods(self, odim_data, get_odim_data):
        with self.open(odim_data) as vol:
            sweep = vol[0][0]

            assert sweep._get_a1gate() == 20
            assert sweep._get_fixed_angle() == 0.5
            if get_odim_data == 1:
                with pytest.raises(TypeError):
                    sweep._get_azimuth_how()
            np.testing.assert_equal(sweep._get_azimuth_where(),
                                    np.arange(0.5, 360, 1))
            np.testing.assert_equal(sweep._get_azimuth().values,
                                    np.arange(0.5, 360, 1))
            if get_odim_data == 1:
                with pytest.raises(TypeError):
                    sweep._get_elevation_how()
            np.testing.assert_equal(sweep._get_elevation_where(),
                                    np.ones((360)) * 0.5)
            np.testing.assert_equal(sweep._get_elevation().values,
                                    np.ones((360)) * 0.5)
            if get_odim_data == 1:
                with pytest.raises(TypeError):
                    sweep._get_time_how()
            time1 = sweep._get_time_what()
            assert time1[0] == 1307700950.5
            assert time1[-1] == 1307700949.5
            time2 = sweep._get_ray_times()
            assert time2.values[0] == 1307700950.5
            assert time2.values[-1] == 1307700949.5
            np.testing.assert_equal(sweep._get_range().values,
                                    np.arange(500, 100100, 1000))
            assert sweep._get_nrays() == 360
            assert sweep._get_nbins() == 100
            assert sweep._get_time().values == 1307700610.0
            assert sweep._get_time_fast() == 1307700610.0
        del sweep
        del vol
        gc.collect()

    def test_sweep_data(self, odim_data, get_odim_data):
        if self.engine == 'h5netcdf':
            pytest.skip("requires enhancements in xarray and h5netcdf")
        with self.open(odim_data, engine=self.engine,
                       decode_coords=False,
                       mask_and_scale=False, decode_times=False,
                       chunks=None, parallel=False) as vol:
            sweep = vol[0][0]
            xr.testing.assert_equal(sweep.data, create_dataset())
        with self.open(odim_data, engine=self.engine,
                       decode_coords=True,
                       mask_and_scale=False, decode_times=True,
                       chunks=None, parallel=False) as vol:
            swp = 0
            sweep = vol[0][swp]
            data = create_dataset()
            data = data.assign_coords(create_coords(swp).coords)
            data = data.assign_coords(create_site().coords)
            data = data.assign_coords({'sweep_mode': 'azimuth_surveillance'})
            data = xr.decode_cf(data, mask_and_scale=False)
            xr.testing.assert_equal(sweep.data, data)
        with self.open(odim_data, engine=self.engine,
                       decode_coords=True,
                       mask_and_scale=True, decode_times=True,
                       chunks=None, parallel=False) as vol:
            swp = 0
            sweep = vol[0][swp]
            data = create_dataset()
            data = data.assign_coords(create_coords(swp).coords)
            data = data.assign_coords(create_site().coords)
            data = data.assign_coords({'sweep_mode': 'azimuth_surveillance'})
            data = xr.decode_cf(data)
            xr.testing.assert_equal(sweep.data, data)
        del sweep
        del vol
        gc.collect()

    def test_sweep_coords_data(self, odim_data, get_odim_data):
        with self.open(odim_data, engine=self.engine,
                       decode_coords=False,
                       mask_and_scale=False, decode_times=False,
                       chunks=None, parallel=False) as vol:
            swp = 0
            sweep = vol[0][swp]
            xr.testing.assert_equal(sweep.coords, create_coords(swp))
        del sweep
        del vol
        gc.collect()

    def test_timeseries_data(self, odim_data, get_odim_data):
        if self.engine == 'h5netcdf':
            pytest.skip("requires enhancements in xarray and h5netcdf")
        with self.open(odim_data, engine=self.engine,
                       decode_coords=False,
                       mask_and_scale=False, decode_times=False,
                       chunks=None, parallel=False) as vol:
            ts = vol[0]
            print(ts.data)
            xr.testing.assert_equal(ts.data,
                                    create_dataset().expand_dims('time'))
        with self.open(odim_data, engine=self.engine,
                       decode_coords=True,
                       mask_and_scale=False, decode_times=True,
                       chunks=None, parallel=False) as vol:
            ts = vol[0]
            data = create_dataset()
            data = data.assign_coords(create_coords(0).coords)
            data = data.assign_coords(create_site().coords)
            data = data.assign_coords({'sweep_mode': 'azimuth_surveillance'})
            data = xr.decode_cf(data, mask_and_scale=False)
            xr.testing.assert_equal(ts.data, data.expand_dims('time'))
        with self.open(odim_data, engine=self.engine,
                       decode_coords=True,
                       mask_and_scale=True, decode_times=True,
                       chunks=None, parallel=False) as vol:
            ts = vol[0]
            data = create_dataset()
            data = data.assign_coords(create_coords(0).coords)
            data = data.assign_coords(create_site().coords)
            data = data.assign_coords({'sweep_mode': 'azimuth_surveillance'})
            data = xr.decode_cf(data)
            xr.testing.assert_equal(ts.data, data.expand_dims('time'))
        del ts
        del vol
        gc.collect()


class TestH5NetCDF(XRadVolume):
    loader = 'h5netcdf'
    engine = 'h5netcdf'
    mom_groups = ['what', 'data']


class TestNetCDF4(XRadVolume):
    loader = 'netcdf4'
    engine = 'netcdf4'
    mom_groups = ['what']


class TestH5Py(XRadVolume):
    loader = 'h5py'
    engine = 'h5netcdf'
    mom_groups = ['data', 'what']
