#!/usr/bin/env python
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Xarray powered Data I/O
^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   CfRadial
   OdimH5
"""

import collections
import numpy as np
import datetime as dt
import netCDF4 as nc
import h5py

from .. import util as util
from .. import georef as georef

xr = util.import_optional('xarray')

# Cf/Radial 2.0 - ODIM_H5 mapping
moments_mapping = {
    'DBZH': {'standard_name': 'radar_equivalent_reflectivity_factor_h',
             'long_name': 'Equivalent reflectivity factor H',
             'short_name': 'DBZH',
             'units': 'dBZ',
             'gamic': 'zh'},
    'DBZV': {'standard_name': 'radar_equivalent_reflectivity_factor_v',
             'long_name': 'Equivalent reflectivity factor V',
             'short_name': 'DBZV',
             'units': 'dBZ',
             'gamic': 'zv'},
    'ZH': {'standard_name': 'radar_linear_equivalent_reflectivity_factor_h',
           'long_name': 'Linear equivalent reflectivity factor H',
           'short_name': 'ZH',
           'units': 'unitless',
           'gamic': None},
    'ZV': {'standard_name': 'radar_equivalent_reflectivity_factor_v',
           'long_name': 'Linear equivalent reflectivity factor V',
           'short_name': 'ZV',
           'units': 'unitless',
           'gamic': None},
    'DBTH': {'standard_name': 'radar_equivalent_reflectivity_factor_h',
             'long_name': 'Total power H (uncorrected reflectivity)',
             'short_name': 'DBTH',
             'units': 'dBZ',
             'gamic': 'uzh'},
    'DBTV': {'standard_name': 'radar_equivalent_reflectivity_factor_v',
             'long_name': 'Total power V (uncorrected reflectivity)',
             'short_name': 'DBTV',
             'units': 'dBZ',
             'gamic': 'uzv',
             },
    'TH': {'standard_name': 'radar_linear_equivalent_reflectivity_factor_h',
           'long_name': 'Linear total power H (uncorrected reflectivity)',
           'short_name': 'TH',
           'units': 'unitless',
           'gamic': None},
    'TV': {'standard_name': 'radar_linear_equivalent_reflectivity_factor_v',
           'long_name': 'Linear total power V (uncorrected reflectivity)',
           'short_name': 'TV',
           'units': 'unitless',
           'gamic': None},
    'VRADH': {
        'standard_name': 'radial_velocity_of_scatterers_away_'
                         'from_instrument_h',
        'long_name': 'Radial velocity of scatterers away from instrument H',
        'short_name': 'VRADH',
        'units': 'meters per seconds',
        'gamic': 'vh'},
    'VRADV': {
        'standard_name': 'radial_velocity_of_scatterers_'
                         'away_from_instrument_v',
        'long_name': 'Radial velocity of scatterers away from instrument V',
        'short_name': 'VRADV',
        'units': 'meters per second',
        'gamic': 'vv',
    },
    'WRADH': {'standard_name': 'radar_doppler_spectrum_width_h',
              'long_name': 'Doppler spectrum width H',
              'short_name': 'WRADH',
              'units': 'meters per seconds',
              'gamic': 'wh'},
    'WRADV': {'standard_name': 'radar_doppler_spectrum_width_v',
              'long_name': 'Doppler spectrum width V',
              'short_name': 'WRADV',
              'units': 'meters per second',
              'gamic': 'wv'},
    'ZDR': {'standard_name': 'radar_differential_reflectivity_hv',
            'long_name': 'Log differential reflectivity H/V',
            'short_name': 'ZDR',
            'units': 'dB',
            'gamic': 'zdr'},
    'LDR': {'standard_name': 'radar_linear_depolarization_ratio',
            'long_name': 'Log-linear depolarization ratio HV',
            'short_name': 'LDR',
            'units': 'dB',
            'gamic': 'ldr'},
    'PHIDP': {'standard_name': 'radar_differential_phase_hv',
              'long_name': 'Differential phase HV',
              'short_name': 'PHIDP',
              'units': 'degrees',
              'gamic': 'phidp'},
    'UPHIDP': {'standard_name': 'radar_differential_phase_hv',
               'long_name': 'Differential phase HV',
               'short_name': 'UPHIDP',
               'units': 'degrees',
               'gamic': 'phidp'},
    'KDP': {'standard_name': 'radar_specific_differential_phase_hv',
            'long_name': 'Specific differential phase HV',
            'short_name': 'KDP',
            'units': 'degrees per kilometer',
            'gamic': 'kdp'},
    'RHOHV': {'standard_name': 'radar_correlation_coefficient_hv',
              'long_name': 'Correlation coefficient HV',
              'short_name': 'RHOHV',
              'units': 'unitless',
              'gamic': 'rhohv'},
    'SQIH': {'standard_name': 'signal_quality_index_h',
             'long_name': 'Signal Quality H',
             'short_name': 'SQIH',
             'units': 'unitless',
             'gamic': None},
    'SQIV': {'standard_name': 'signal_quality_index_v',
             'long_name': 'Signal QualityV',
             'short_name': 'SQIV',
             'units': 'unitless',
             'gamic': None},
}

ODIM_NAMES = {value['short_name']: key for (key, value) in
              moments_mapping.items()}
GAMIC_NAMES = {value['gamic']: key for (key, value) in moments_mapping.items()}

range_attrs = {'units': 'meters',
               'standard_name': 'projection_range_coordinate',
               'long_name': 'range_to_measurement_volume',
               'spacing_is_constant': 'true',
               'axis': 'radial_range_coordinate',
               'meters_to_center_of_first_gate': None,
               }
az_attrs = {'standard_name': 'ray_azimuth_angle',
            'long_name': 'azimuth_angle_from_true_north',
            'units': 'degrees',
            'axis': 'radial_azimuth_coordinate'}
el_attrs = {'standard_name': 'ray_elevation_angle',
            'long_name': 'elevation_angle_from_horizontal_plane',
            'units': 'degrees',
            'axis': 'radial_elevation_coordinate'}

time_attrs = {'standard_name': 'time',
              'units': 'seconds since 1970-01-01T00:00:00Z',
              }

root_vars = {'volume_number', 'platform_type', 'instrument_type',
             'primary_axis', 'time_coverage_start', 'time_coverage_end',
             'latitude', 'longitude', 'altitude', 'fixed_angle',
             'status_xml'}

sweep_vars1 = {'sweep_number', 'sweep_mode', 'polarization_mode',
               'prt_mode', 'follow_mode', 'fixed_angle',
               'target_scan_rate', 'sweep_start_ray_index',
               'sweep_end_ray_index'}

sweep_vars2 = {'azimuth', 'elevation', 'pulse_width', 'prt',
               'nyquist_velocity', 'unambiguous_range',
               'antenna_transition', 'n_samples', 'r_calib_index',
               'scan_rate'}

sweep_vars3 = {'DBZ', 'VR', 'time', 'range', 'reflectivity_horizontal'}


@xr.register_dataset_accessor('gamic')
class GamicAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._radial_range = None
        self._azimuth_range = None
        self._elevation_range = None
        self._time_range = None
        self._sitecoords = None
        self._polcoords = None
        self._projection = None
        self._time = None

    @property
    def radial_range(self):
        """Return the radial range of this dataset."""
        if self._radial_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            ngates = self._obj.attrs['bin_count']
            # range_start = self._obj.attrs['range_start']
            range_samples = self._obj.attrs['range_samples']
            range_step = self._obj.attrs['range_step']
            bin_range = range_step * range_samples
            range_data = np.arange(bin_range / 2., bin_range * ngates,
                                   bin_range,
                                   dtype='float32')
            range_attrs['meters_to_center_of_first_gate'] = bin_range / 2.
            da = xr.DataArray(range_data, dims=['range'], attrs=range_attrs)
            self._radial_range = da
        return self._radial_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            azstart = self._obj['azimuth_start']
            azstop = self._obj['azimuth_stop']
            zero_index = np.where(azstop < azstart)
            azstop[zero_index[0]] += 360
            azimuth = (azstart + azstop) / 2.
            azimuth = azimuth.assign_attrs(az_attrs)
            self._azimuth_range = azimuth
        return self._azimuth_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            elstart = self._obj['elevation_start']
            elstop = self._obj['elevation_stop']
            elevation = (elstart + elstop) / 2.
            elevation = elevation.assign_attrs(el_attrs)
            self._elevation_range = elevation
        return self._elevation_range

    @property
    def time_range(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            da = self._obj['timestamp'] / 1e6
            da.assign_attrs(time_attrs)
            self._time_range = da
        return self._time_range

    @property
    def sitecoords(self):
        if self._sitecoords is None:
            self._sitecoords = (self._obj.longitude, self._obj.latitude,
                                self._obj.altitude)
        return self._sitecoords

    @property
    def polcoords(self):
        if self._polcoords is None:
            self.assign_xyz()
        return self._polcoords

    def assign_xyz(self):
        az = self._obj.azimuth
        rng = self._obj.range
        ele = self._obj.elevation
        xx, yy = np.meshgrid(rng, az)
        lon, lat, alt = self.sitecoords
        coords, rad = georef.spherical_to_xyz(xx, yy, ele[0],
                                              (lon, lat, alt),
                                              re=georef.get_earth_radius(lat))
        self._polcoords = coords
        self._projection = rad

    @property
    def projection(self):
        if self._projection is None:
            self.assign_xyz()
        return self._projection


@xr.register_dataset_accessor('odim')
class OdimAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._radial_range = None
        self._azimuth_range = None
        self._elevation_range = None
        self._time_range = None
        self._sitecoords = None
        self._polcoords = None
        self._projection = None
        self._time = None

    @property
    def radial_range(self):
        """Return the radial range of this dataset."""
        if self._radial_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            ngates = self._obj.attrs['nbins']
            range_start = self._obj.attrs['rstart'] * 1000.
            bin_range = self._obj.attrs['rscale']
            cent_first = range_start + bin_range / 2.
            range_data = np.arange(cent_first,
                                   range_start + bin_range * ngates,
                                   bin_range,
                                   dtype='float32')
            range_attrs[
                'meters_to_center_of_first_gate'] = cent_first
            range_attrs[
                'meters_between_gates'] = bin_range

            da = xr.DataArray(range_data, dims=['range'], attrs=range_attrs)
            self._radial_range = da
        return self._radial_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            nrays = self._obj.attrs['nrays']
            res = 360. / nrays
            azimuth_data = np.arange(res / 2.,
                                     360.,
                                     res,
                                     dtype='float32')

            da = xr.DataArray(azimuth_data, dims=['time'], attrs=az_attrs)
            self._azimuth_range = da
        return self._azimuth_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            nrays = self._obj.attrs['nrays']
            elangle = self._obj.attrs['elangle']
            elevation_data = np.ones(nrays, dtype='float32') * elangle
            da = xr.DataArray(elevation_data, dims=['time'], attrs=el_attrs)
            self._elevation_range = da
        return self._elevation_range

    @property
    def time_range(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            startazT = self._obj.attrs['startazT']

            attrs = {'units': 'seconds since 1970-01-01T00:00:00Z',
                     'standard_name': 'time'}
            da = xr.DataArray(startazT, dims=['time'], attrs=attrs)
            self._time_range = da
        return self._time_range

    @property
    def time_range2(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            startdate = self._obj.attrs['startdate']
            starttime = self._obj.attrs['starttime']
            enddate = self._obj.attrs['enddate']
            endtime = self._obj.attrs['endtime']

            start = dt.datetime.strptime(startdate + starttime, '%Y%m%d%H%M%S')
            end = dt.datetime.strptime(enddate + endtime, '%Y%m%d%H%M%S')
            start = start.replace(tzinfo=dt.timezone.utc)
            end = end.replace(tzinfo=dt.timezone.utc)

            # attrs = {'units': 'seconds since 1970-01-01 00:00',
            #         'standard_name': 'time'}
            # da = xr.DataArray(startazT, dims=['time'], attrs=attrs)
            self._time_range = (start.timestamp(), end.timestamp())
        return self._time_range

    @property
    def sitecoords(self):
        if self._sitecoords is None:
            self._sitecoords = (self._obj.longitude.values,
                                self._obj.latitude.values,
                                self._obj.altitude.values)
        return self._sitecoords

    @property
    def cartesian_coords(self):
        if self._polcoords is None:
            self.assign_xyz()
        return self._polcoords

    def assign_xyz(self, sitecoords):
        az = self._obj.azimuth
        rng = self._obj.range
        ele = self._obj.elevation
        xx, yy = np.meshgrid(rng, az)
        lon, lat, alt = sitecoords
        coords, rad = georef.spherical_to_xyz(xx, yy, ele[0],
                                              (lon, lat, alt),
                                              re=georef.get_earth_radius(lat))
        self._polcoords = coords
        self._projection = rad
        self._obj = self._obj.assign_coords(
            x=(['time', 'range'], coords[..., 0]),
            y=(['time', 'range'], coords[..., 1]),
            z=(['range'], coords[0, ..., 2]))
        return self._obj

    @property
    def projection(self):
        if self._projection is None:
            self.assign_xyz()
        return self._projection


def write_odim(src, dest):
    for key, value in src.items():
        if key in dest.attrs:
            continue
        if isinstance(value, str):
            tid = h5py.h5t.C_S1.copy()
            tid.set_size(len(value) + 1)
            H5T_C_S1_NEW = h5py.Datatype(tid)
            dest.attrs.create(key, value, dtype=H5T_C_S1_NEW)
        else:
            dest.attrs[key] = value


def write_odim_moments(src, dest):
    keys = [key for key in src if key in ODIM_NAMES]
    data_list = ['data{}'.format(i + 1) for i in range(len(keys))]
    data_idx = np.argsort(data_list)
    for idx in data_idx:
        value = src[keys[idx]]
        h5_data = dest.create_group(data_list[idx])
        enc = value.encoding

        # p. 21 ff
        h5_what = h5_data.create_group('what')
        what = {'quantity': value.name,
                'gain': float(enc['scale_factor']),
                'offset': float(enc['add_offset']),
                'nodata': float(enc['_FillValue']),
                'undetect': float(value._Undetect),
                }
        write_odim(what, h5_what)

        # moments
        val = value.values
        maxval = value.encoding['_FillValue'] * value.gain + value.offset
        val[np.isnan(val)] = maxval
        val = (val - value.offset) / value.gain
        val = np.rint(val).astype(enc['dtype'])
        ds = h5_data.create_dataset('data', data=val, compression='gzip',
                                    compression_opts=6,
                                    fillvalue=enc['_FillValue'])
        if enc['dtype'] == 'uint8':
            image = 'IMAGE'
            version = '1.2'
            tid1 = h5py.h5t.C_S1.copy()
            tid1.set_size(len(image) + 1)
            H5T_C_S1_IMG = h5py.Datatype(tid1)
            tid2 = h5py.h5t.C_S1.copy()
            tid2.set_size(len(version) + 1)
            H5T_C_S1_VER = h5py.Datatype(tid2)
            ds.attrs.create('CLASS', image, dtype=H5T_C_S1_IMG)
            ds.attrs.create('IMAGE_VERSION', version, dtype=H5T_C_S1_VER)


def open_ds(nch, grp=None):
    if grp is not None:
        nch = nch.groups.get(grp, False)
    if nch:
        nch = xr.open_dataset(xr.backends.NetCDF4DataStore(nch),
                              mask_and_scale=True)
    return nch


class XRadVol(collections.MutableMapping):
    """ BaseClass for Xarray based RadarVolumes

    """

    def __init__(self):
        self._source = dict()

    def __getitem__(self, key):
        return self._source[key]

    def __setitem__(self, key, value):
        self._source[key] = value

    def __delitem__(self, key):
        del self._source[key]

    def __iter__(self):
        return iter(self._source)

    def __len__(self):
        return len(self._source)

    def __repr__(self):
        return self._source.__repr__()

    def __del__(self):
        for h in self._ds_handles[::-1]:
            h.close()

    def to_cfradial2(self, filename):
        """ Save volume to Cf/Radial2.0 compliant file.
        """
        root = self['root'].copy(deep=True)
        root.attrs['Conventions'] = 'Cf/Radial'
        root.attrs['version'] = '2.0'
        root.to_netcdf(filename, mode='w', group='/')
        for key in root.sweep_group_name.values:
            self[key].to_netcdf(filename, mode='a', group=key)

    def to_odim(self, filename):
        """ Save volume to ODIM_H5/V2_2 compliant file.
        """
        root = self['root']

        h5 = h5py.File(filename, 'w')

        # root group, only Conventions for ODIM_H5
        write_odim({'Conventions': 'ODIM_H5/V2_2'}, h5)

        # how group
        # first try to use original data
        try:
            how = self['odim']['how'].attrs
        except KeyError:
            how = {}
        else:
            how.update({'_modification_program': 'wradlib'})

        h5_how = h5.create_group('how')
        write_odim(how, h5_how)

        sweepnames = self['root'].sweep_group_name.values

        # what group, object, version, date, time, source, mandatory
        # p. 10 f
        try:
            what = self['odim']['what'].attrs
        except KeyError:
            what = {}
            if len(sweepnames) > 1:
                what['object'] = 'PVOL'
            else:
                what['object'] = 'SCAN'
            what['version'] = 'H5rad 2.2'
            what['date'] = str(root.time_coverage_start.values)[:10].replace(
                '-', '')
            what['time'] = str(root.time_coverage_end.values)[11:19].replace(
                ':', '')
            what['source'] = root.attrs['instrument_name']

        h5_what = h5.create_group('what')
        write_odim(what, h5_what)

        # where group, lon, lat, height, mandatory
        where = {'lon': root.longitude.values,
                 'lat': root.latitude.values,
                 'height': root.altitude.values}
        h5_where = h5.create_group('where')
        write_odim(where, h5_where)

        # datasets
        ds_list = ['dataset{}'.format(i + 1) for i in range(len(sweepnames))]
        ds_idx = np.argsort(ds_list)
        for idx in ds_idx:
            ds = self['sweep_{}'.format(idx + 1)]
            h5_dataset = h5.create_group(ds_list[idx])

            # what group p. 21 ff.
            h5_ds_what = h5_dataset.create_group('what')
            ds_what = {}
            t = sorted(ds.time.values)
            start = dt.datetime.utcfromtimestamp(t[0].astype('O') / 1e9)
            end = dt.datetime.utcfromtimestamp(
                np.rint(t[-1].astype('O') / 1e9))
            ds_what['product'] = 'SCAN'
            ds_what['startdate'] = start.strftime('%Y%m%d')
            ds_what['starttime'] = start.strftime('%H%M%S')
            ds_what['enddate'] = end.strftime('%Y%m%d')
            ds_what['endtime'] = end.strftime('%H%M%S')
            write_odim(ds_what, h5_ds_what)

            # where group, p. 11 ff. mandatory
            h5_ds_where = h5_dataset.create_group('where')
            rscale = ds.range.values[1] / 1. - ds.range.values[0]
            rstart = rscale / 2. - ds.range.values[0]
            ds_where = {'elangle': ds.fixed_angle,
                        'nbins': ds.range.shape[0],
                        'rstart': rstart,
                        'rscale': rscale,
                        'nrays': ds.azimuth.shape[0],
                        'a1gate':
                            np.nonzero(np.argsort(ds.time.values) == 0)[0][0]
                        }
            write_odim(ds_where, h5_ds_where)

            # how group, p. 14 ff.
            h5_ds_how = h5_dataset.create_group('how')
            try:
                ds_how = self['odim']['dsets'][ds_list[idx]]['how'].attrs
            except KeyError:
                ds_how = {'scan_index': ds.sweep_number + 1,
                          'scan_count': len(sweepnames),
                          }
            write_odim(ds_how, h5_ds_how)

            # write moments
            write_odim_moments(ds, h5_dataset)

        h5.close()


class CfRadial(XRadVol):
    """ Class for Xarray based retrieval of Cf/Radial data files
    """

    def __init__(self, filename=None, flavour=None, **kwargs):
        super(CfRadial, self).__init__()
        self._filename = filename
        self._ds_handles = []
        self._ncf = nc.Dataset(filename, diskless=True, persist=False)
        self._ds_handles.append(self._ncf)
        self._disk_format = self._ncf.disk_format
        self._file_format = self._ncf.file_format
        self._data_model = self._ncf.data_model
        if flavour is None:
            try:
                self._Conventions = self._ncf.Conventions
                self._version = self._ncf.version
            except AttributeError as e:
                raise AttributeError(
                    'wradlib: Missing "Conventions" attribute in {} ./n'
                    'Use the "flavour" kwarg to specify your source '
                    'data.'.format(filename)) from e
            if "cf/radial" in self._Conventions.lower():
                if self._version == '2.0':
                    flavour = 'Cf/Radial2'
                else:
                    flavour = 'Cf/Radial'

        if flavour == "Cf/Radial2":
            self.assign_data_radial2()
        elif flavour == "Cf/Radial":
            self.assign_data_radial()
        else:
            raise AttributeError(
                'wradlib: Unknown "flavour" kwarg attribute: {} .'
                ''.format(flavour))

    def assign_data_radial2(self):
        """ Assign from CfRadial2 data structure.

        """
        self['root'] = open_ds(self._ncf)
        self._ds_handles.append(self['root'])
        setattr(self, 'root', self['root'])
        sweepnames = self.root.sweep_group_name.values
        for sw in sweepnames:
            self[sw] = open_ds(self._ncf, sw)
            self._ds_handles.append(self[sw])
            setattr(self, sw, self[sw])

    def assign_data_radial(self):
        """ Assign from CfRadial1 data structure.

        """
        root = open_ds(self._ncf)
        var = root.variables.keys()
        remove_root = var ^ root_vars
        remove_root &= var
        root1 = root.drop(remove_root).rename(
            {'fixed_angle': 'sweep_fixed_angle'})
        sweep_group_name = []
        for i in range(root1.dims['sweep']):
            sweep_group_name.append('sweep_{}'.format(i + 1))
        self['root'] = root1.assign(
            {'sweep_group_name': (['sweep'], sweep_group_name)})
        setattr(self, 'root', self['root'])

        keep_vars = sweep_vars1 | sweep_vars2 | sweep_vars3
        remove_vars = var ^ keep_vars
        remove_vars &= var
        data = root.drop(remove_vars)
        data.attrs = {}
        start_idx = data.sweep_start_ray_index.values
        end_idx = data.sweep_end_ray_index.values
        data = data.drop({'sweep_start_ray_index', 'sweep_end_ray_index'})
        for i, sw in enumerate(sweep_group_name):
            tslice = slice(start_idx[i], end_idx[i])
            self[sw] = data.isel(time=tslice,
                                 sweep=slice(i, i + 1)).squeeze('sweep')
            setattr(self, sw, self[sw])


class OdimH5(XRadVol):
    """ Class for Xarray based retrieval of ODIM_H5 data files
    """

    def __init__(self, filename=None, flavour=None, strict=True, **kwargs):
        super(OdimH5, self).__init__()
        self._filename = filename
        self._ds_handles = []
        self._ncf = nc.Dataset(filename, diskless=True, persist=False)
        self._ds_handles.append(self._ncf)
        self._disk_format = self._ncf.disk_format
        self._file_format = self._ncf.file_format
        self._data_model = self._ncf.data_model
        if self._disk_format != 'HDF5':
            raise TypeError(
                'wradlib: File {} is neither "NETCDF4" (using HDF5 groups) '
                'nor plain "HDF5".'.format(filename))
        if flavour is None:
            try:
                self._Conventions = self._ncf.Conventions
            except AttributeError as e:
                raise AttributeError(
                    'wradlib: Missing "Conventions" attribute in {} ./n'
                    'Use the "flavour" kwarg to specify your source '
                    'data.'.format(filename)) from e
            if "ODIM_H5" in self._Conventions:
                flavour = 'ODIM'
            else:
                raise AttributeError(
                    'wradlib: "Conventions" attribute "{}" in {} is unknown./n'
                    'Use the "flavour" kwarg to specify your source '
                    'data.'.format(self._Conventions, filename))

        if flavour == "ODIM":
            self.assign_data_odim(strict=strict)
        elif flavour == "GAMIC":
            self.assign_data_gamic(strict=strict)
        else:
            raise AttributeError(
                'wradlib: Unknown "flavour" kwarg attribute: {} .'
                ''.format(flavour))

    def _get_groups(self):
        return (open_ds(self._ncf), open_ds(self._ncf, 'how'),
                open_ds(self._ncf, 'what'), open_ds(self._ncf, 'where'))

    def _get_ds_groups(self, sweep):
        return (open_ds(self._ncf, sweep),
                open_ds(self._ncf[sweep], 'how'),
                open_ds(self._ncf[sweep], 'what'),
                open_ds(self._ncf[sweep], 'where'))

    def _get_moment_names(self, sweep, fmt=None, src=None):
        moments = [mom for mom in getattr(self._ncf[sweep], src).keys() if
                   fmt in mom]
        moments_idx = np.argsort([int(s[len(fmt):]) for s in moments])
        return np.array(moments)[moments_idx]

    def _get_group_moments(self, sweep, moments=None):
        datas = {}
        for mom in moments:
            dmom_what = open_ds(self._ncf[sweep][mom], 'what')
            name = dmom_what.attrs.pop('quantity')
            if name not in moments_mapping.keys():
                continue
            dsmom = open_ds(self._ncf[sweep], mom)
            dmom = dsmom.data.assign_attrs(dmom_what.attrs)
            dmom.attrs['scale_factor'] = dmom.attrs.get('gain')
            dmom.attrs['add_offset'] = dmom.attrs.get('offset')
            dmom.attrs['_FillValue'] = dmom.attrs.get('nodata')
            dmom.attrs['_Undetect'] = dmom.attrs.get('undetect')
            dmom.attrs['coordinates'] = 'elevation azimuth range'

            # add cfradial moment attributes
            for k, v in moments_mapping[name].items():
                dmom.attrs[k] = v

            # drop short_name
            dmom.attrs.pop('short_name')
            dmom.attrs.pop('gamic')

            # fix dimensions
            dims = dmom.dims
            datas.update({name: dmom.rename({dims[0]: 'time',
                                             dims[1]: 'range',
                                             })})
        return datas

    def _get_variables_moments(self, ds, moments=None):
        for mom in moments:
            # open dataX dataset
            dmom = ds[mom]
            name = dmom.moment.lower()
            if name not in GAMIC_NAMES.keys():
                ds = ds.drop(mom)
                continue
            cfname = GAMIC_NAMES[name]
            # assign attributes
            dmax = np.iinfo(dmom.dtype).max
            minval = dmom.dyn_range_min
            maxval = dmom.dyn_range_max
            dmom.attrs['gain'] = (maxval - minval) / dmax
            dmom.attrs['offset'] = minval
            dmom.attrs['scale_factor'] = (maxval - minval) / dmax
            dmom.attrs['add_offset'] = minval

            dmom.attrs['_FillValue'] = float(dmax)
            dmom.attrs['nodata'] = float(dmax)
            dmom.attrs['_Undetect'] = 0.
            dmom.attrs['undetect'] = 0.
            for k, v in moments_mapping[cfname].items():
                dmom.attrs[k] = v
            dname = dmom.attrs.pop('short_name')
            dmom.attrs.pop('gamic')
            #
            ds = ds.rename({mom: dname})
            # fix dimensions
            dims = dmom.dims
            dmom = dmom.rename({dims[0]: 'time',
                                dims[1]: 'range',
                                })
        return ds

    def _get_swp_grp_name(self, name):
        # sweep group handling
        src = [key for key in self._ncf.groups.keys() if name in key]
        src.sort(key=lambda x: int(x[len(name):]))
        swp_grp_name = ['sweep_{}'.format(i) for i in
                        range(1, len(src) + 1)]
        return src, swp_grp_name

    def assign_data_odim(self, strict=True):
        """ Assign from ODIM_H5 data structure.

        """

        # retrieve and assign global groups root and /how, /what, /where
        root, how, what, where = self._get_groups()

        # sweep group handling
        src_swp_grp_name, swp_grp_name = self._get_swp_grp_name('dataset')

        sweep_fixed_angle = []
        time_coverage_start = np.datetime64('2037-01-01')
        time_coverage_end = np.datetime64('1970-01-01')
        # iterate sweeps
        sweeps = {}
        for i, sweep in enumerate(src_swp_grp_name):
            swp = {}

            # retrieve ds and assign datasetX how/what/where group attributes
            ds, ds_how, ds_what, ds_where = self._get_ds_groups(sweep)

            # moments
            moments = self._get_moment_names(sweep, fmt='data', src='groups')
            for name, dmom in self._get_group_moments(sweep,
                                                      moments=moments).items():
                ds[name] = dmom

            # coordinates wrap-up
            ds = ds.assign_coords(azimuth=ds_where.odim.azimuth_range)
            ds = ds.assign_coords(elevation=ds_where.odim.elevation_range)
            ds = ds.assign({'range': ds_where.odim.radial_range})

            # time coordinate
            try:
                timevals = ds_how.odim.time_range.values
            except KeyError:
                # timehandling if only start and end time is given
                start, end = ds_what.odim.time_range2
                delta = (end - start) / ds_where.nrays
                timevals = np.arange(start + delta / 2., end, delta)
                # print(delta, timevals.shape)
                attrs = {'units': 'seconds since 1970-01-01T00:00:00Z',
                         'standard_name': 'time'}
                da = xr.DataArray(timevals, dims=['time'], attrs=attrs)
                timevals = da.values
                timevals = np.roll(timevals, shift=-ds_where.a1gate)
                # timevals = ds_how.odim.time_range2.values
            ds = ds.assign({'time': (['time'], timevals, time_attrs)})

            # assign global sweep attributes
            ds = ds.assign({'sweep_number': i,
                            'sweep_mode': 'azimuthal_surveillance',
                            'follow_mode': 'none',
                            'prt_mode': 'fixed',
                            'fixed_angle': ds_where.elangle,
                            })
            sweep_fixed_angle.append(ds_where.elangle)

            # decode dataset
            ds = xr.decode_cf(ds)

            # extract time coverage
            time_coverage_start = min(time_coverage_start,
                                      ds.time.values.min())
            time_coverage_end = max(time_coverage_end,
                                    ds.time.values.max())

            # assign to sweep dict
            if not strict:
                swp.update({'how': ds_how})
                swp.update({'what': ds_what})
                swp.update({'where': ds_where})
                sweeps[sweep] = swp

            # dataset only
            self[swp_grp_name[i]] = ds
            setattr(self, swp_grp_name[i], self[swp_grp_name[i]])
            self._ds_handles.append(self[swp_grp_name[i]])

        # assign root variables
        time_coverage_start = str(time_coverage_start)[:22] + 'Z'
        time_coverage_end = str(time_coverage_end)[:22] + 'Z'

        # assign root variables
        root = root.assign({'volume_number': 0,
                            'platform_type': 'fixed',
                            'instrument_type': 'radar',
                            'primary_axis': 'axis_z',
                            'time_coverage_start': time_coverage_start,
                            'time_coverage_end': time_coverage_end,
                            'latitude': where.attrs['lat'],
                            'longitude': where.attrs['lon'],
                            'altitude': where.attrs['height'],
                            'sweep_group_name': (['sweep'], swp_grp_name),
                            'sweep_fixed_angle': (
                                ['sweep'], sweep_fixed_angle),
                            })

        # assign root attributes
        root = root.assign_attrs({'version': what.attrs['version'],
                                  'title': 'None',
                                  'institution': what.attrs['source'],
                                  'references': 'None',
                                  'source': 'None',
                                  'history': 'None',
                                  'comment': 'imported/exported using wradlib',
                                  'instrument_name': what.attrs['source'],
                                  })

        # assign to source dict
        self['root'] = root
        self._ds_handles.append(self['root'])
        setattr(self, 'root', self['root'])
        if not strict:
            self['odim'] = {'how': how,
                            'what': what,
                            'where': where,
                            'dsets': sweeps}

    def assign_data_gamic(self, strict=True):
        """ Assign from GAMIC hdf5 data structure.

        """
        # retrieve and assign global groups root and /how, /what, /where
        root, how, what, where = self._get_groups()

        # sweep group handling
        src_swp_grp_name, swp_grp_name = self._get_swp_grp_name('scan')

        sweep_fixed_angle = []
        time_coverage_start = np.datetime64('2037-01-01')
        time_coverage_end = np.datetime64('1970-01-01')

        # iterate sweeps
        sweeps = {}
        for i, sweep in enumerate(src_swp_grp_name):
            swp = {}

            # retrieve ds and assign datasetX how/what/where group attributes
            ds, ds_how, ds_what, ds_where = self._get_ds_groups(sweep)

            # fix dimensions
            dims = list(ds.dims.keys())
            ds = ds.rename({dims[0]: 'time',
                            dims[1]: 'range',
                            })

            # retrieve and assign ray_header
            # ToDo: move rayheader into own dataset
            h5 = h5py.File(self._filename)
            ray_header = h5['scan{}/ray_header'.format(i)][:]
            for name in ray_header.dtype.names:
                rh = ray_header[name]
                attrs = None
                ds_what = ds_what.assign({name: (['time'], rh, attrs)})

            # coordinates wrap-up
            ds = ds.assign_coords(azimuth=ds_what.gamic.azimuth_range)
            ds = ds.assign_coords(elevation=ds_what.gamic.elevation_range)
            ds = ds.assign({'range': ds_how.gamic.radial_range})
            ds = ds.assign({'time': (['time'], ds_what.gamic.time_range.values,
                                     time_attrs)})
            # get moments
            moments = self._get_moment_names(sweep, fmt='moment_',
                                             src='variables')
            ds = self._get_variables_moments(ds, moments=moments)

            # assign global sweep attributes
            ds = ds.assign({'sweep_number': i,
                            'sweep_mode': 'azimuthal_surveillance',
                            'follow_mode': 'none',
                            'prt_mode': 'fixed',
                            'fixed_angle': ds_how.attrs['elevation'],
                            })

            sweep_fixed_angle.append(ds_how.attrs['elevation'])

            ds = xr.decode_cf(ds)

            # extract time coverage
            time_coverage_start = min(time_coverage_start,
                                      ds.time.values.min())
            time_coverage_end = max(time_coverage_end,
                                    ds.time.values.max())

            # assign to sweep dict
            if not strict:
                swp.update({'how': ds_how,
                            'what': ds_what,
                            'where': ds_where})
                sweeps[sweep] = swp

            # dataset only
            self[swp_grp_name[i]] = ds
            setattr(self, swp_grp_name[i], self[swp_grp_name[i]])
            self._ds_handles.append(self[swp_grp_name[i]])

        # assign root variables
        time_coverage_start = str(time_coverage_start)[:22] + 'Z'
        time_coverage_end = str(time_coverage_end)[:22] + 'Z'

        # assign root variables
        root = root.assign({'volume_number': 0,
                            'platform_type': 'fixed',
                            'instrument_type': 'radar',
                            'primary_axis': 'axis_z',
                            'time_coverage_start': time_coverage_start,
                            'time_coverage_end': time_coverage_end,
                            'latitude': where.attrs['lat'],
                            'longitude': where.attrs['lon'],
                            'altitude': where.attrs['height'],
                            'sweep_group_name': (['sweep'], swp_grp_name),
                            'sweep_fixed_angle': (
                                ['sweep'], sweep_fixed_angle),
                            })

        # assign root attributes
        root = root.assign_attrs({'version': what.attrs['version'],
                                  'title': how.attrs['template_name'],
                                  'institution': 'None',
                                  'references': 'None',
                                  'source': 'None',
                                  'history': 'None',
                                  'comment': 'imported/exported using wradlib',
                                  'instrument_name': how.attrs['host_name'],
                                  })

        # assign to source dict
        self['root'] = root
        setattr(self, 'root', self['root'])
        self._ds_handles.append(self['root'])
        if not strict:
            self['odim'] = {'how': how,
                            'what': what,
                            'where': where,
                            'dsets': sweeps}
