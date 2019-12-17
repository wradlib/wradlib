# Copyright (c) 2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import gc
import contextlib
import pytest
import numpy as np
import xarray as xr
import h5py

from wradlib import io, util


@contextlib.contextmanager
def get_odim_volume():
    filename = 'hdf5/knmi_polar_volume.h5'
    h5file = util.get_wradlib_data_file(filename)
    yield h5file


def create_dbz_what():
    what = [('gain', np.array([0.5], dtype=np.float32)),
            ('nodata', np.array([255.], dtype=np.float32)),
            ('offset', np.array([-31.5], dtype=np.float32)),
            ('quantity', np.array([b'DBZH'], dtype='|S5')),
            ('undetect', np.array([0.], dtype=np.float32))]
    return what


@pytest.fixture(scope="session")
def odim_data(tmpdir_factory):
    tmp_local = tmpdir_factory.mktemp("data").join("test_odim.h5")
    with h5py.File(str(tmp_local), 'w') as f:
        foo_data = np.random.randint(0, 255, (360, 100), dtype=np.uint8)
        dataset = ['dataset1', 'dataset2']
        f.attrs.update([('Conventions',  np.array([b'ODIM_H5/V2_0'],
                                                  dtype='|S13'))])
        root_where = f.create_group('where')
        root_where.attrs.update([('height', 99.5),
                                 ('lon', 7.071624),
                                 ('lat', 50.730599)])
        for i, grp in enumerate(dataset):
            dset = f.create_group(grp)
            where = dset.create_group('where')
            where.attrs.update(
                [('a1gate', np.array([i + 20], dtype=np.int)),
                 ('elangle', np.array([i + 0.5], dtype=np.float32)),
                 ('nrays', np.array([360], dtype=np.int)),
                 ('nbins', np.array([100], dtype=np.int)),
                 ('rstart', np.array([0], dtype=np.float32)),
                 ('rscale', np.array([1000], dtype=np.float32))])
            what = dset.create_group('what')
            what.attrs.update(
                [('startdate', np.array([b'20110610'], dtype='|S9')),
                 ('starttime', np.array([b'101010'], dtype='|S7')),
                 ('enddate', np.array([b'20110610'], dtype='|S9')),
                 ('endtime', np.array([b'101610'], dtype='|S7'))
                 ])
            da = dset.create_group('data1')
            da.create_dataset('data', data=foo_data)
            ds_what = da.create_group('what')
            ds_what.attrs.update(create_dbz_what())
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
                                          np.arange(0.5, 360, 1))
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
            assert sweep.ray_times.values[0] == np.datetime64(
                '2011-06-10T10:15:50.500000000')
            assert sweep.ray_times.values[-1] == np.datetime64(
                '2011-06-10T10:15:49.500000000')
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
            xr.testing.assert_equal(vol.site,
                                    xr.Dataset(coords={'altitude': 99.5,
                                                       'latitude': 50.730599,
                                                       'longitude': 7.071624})
                                    )

        del vol
        gc.collect()

    def test_odimh5_group_mixin(self, odim_data):
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
            assert ts.how is None
            assert ts.what == {'enddate': '20110610', 'endtime': '101610',
                               'startdate': '20110610', 'starttime': '101010'}
            assert ts.where == {'a1gate': 20, 'elangle': 0.5, 'nbins': 100,
                                'nrays': 360, 'rscale': 1000.0, 'rstart': 0.0}
            assert ts.attrs == {}
            assert ts.engine == self.engine
            assert ts.filename == odim_data
            assert ts.groups == ['data1', 'what', 'where']
            assert ts.ncid == ts.ncfile[ts.ncpath]
            assert ts.ncpath == 'dataset1'
            assert ts.parent == vol

            # sweep
            sweep = ts[0]
            assert sweep.how is None
            assert sweep.what == {'enddate': '20110610', 'endtime': '101610',
                                  'startdate': '20110610',
                                  'starttime': '101010'}
            assert sweep.where == {'a1gate': 20, 'elangle': 0.5, 'nbins': 100,
                                   'nrays': 360, 'rscale': 1000.0,
                                   'rstart': 0.0}
            assert sweep.attrs == {}
            assert sweep.engine == self.engine
            assert sweep.filename == odim_data
            assert sweep.groups == ['data1', 'what', 'where']
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

        del vol
        gc.collect()

    def test_sweep_methods(self, odim_data):
        with self.open(odim_data) as vol:
            sweep = vol[0][0]

            assert sweep._get_a1gate() == 20
            assert sweep._get_fixed_angle() == 0.5
            with pytest.raises(TypeError):
                sweep._get_azimuth_how()
            np.testing.assert_equal(sweep._get_azimuth_where(),
                                    np.arange(0.5, 360, 1))
            np.testing.assert_equal(sweep._get_azimuth().values,
                                    np.arange(0.5, 360, 1))
            with pytest.raises(TypeError):
                sweep._get_elevation_how()
            np.testing.assert_equal(sweep._get_elevation_where(),
                                    np.ones((360)) * 0.5)
            np.testing.assert_equal(sweep._get_elevation().values,
                                    np.ones((360)) * 0.5)
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
