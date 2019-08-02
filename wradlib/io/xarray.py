#!/usr/bin/env python
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Xarray powered Data I/O
^^^^^^^^^^^^^^^^^^^^^^^

Reads data from netcdf-based CfRadial1, CfRadial2 and hdf5-based ODIM_H5 and
other hdf5-flavours (GAMIC).

Writes data to CfRadial2 and ODIM_H5 files.

This reader implementation uses

* `netcdf4 <http://unidata.github.io/netcdf4-python/>`_,
* `h5py <https://www.h5py.org/>`_ and
* `xarray <xarray.pydata.org/>`_.

The data is claimed using netcdf4-Dataset in a diskless non-persistent mode::

    nch = nc.Dataset(filename, diskless=True, persist=False)

Further the different netcdf/hdf groups are accessed via xarray open_dataset
and the NetCDF4DataStore::

    xr.open_dataset(xr.backends.NetCDF4DataStore(nch), mask_and_scale=True)

For hdf5 data scaling/masking properties will be added to the datasets before
decoding. For GAMIC data compound data will be read via h5py.

The data structure holds one or many ['sweep_X'] xarray datasets, holding the
sweep data. The root group xarray dataset which corresponds to the
CfRadial2 root-group is available via the `.root`-object. Since for data
handling xarray is utilized all xarray features can be exploited,
like lazy-loading, pandas-like indexing on N-dimensional data and vectorized
mathematical operations across multiple dimensions.

The writer implementation uses xarray for CfRadial2 output and relies on h5py
for the ODIM_H5 output.

Warning
-------
    This implementation is considered experimental. Changes in the API should
    be expected.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   XRadVol
   CfRadial
   OdimH5
   to_cfradial2
   to_odim

"""

import warnings
import collections
import numpy as np
import datetime as dt
import netCDF4 as nc
import h5py
import xarray as xr

from ..georef import xarray


def create_xarray_dataarray(*args, **kwargs):
    warnings.warn("WRADLIB: calling `wradlib.io.create_xarray_dataarray` is "
                  "deprecated, please use "
                  "`wradlib.georef.create_xarray_dataarray`.",
                  DeprecationWarning)
    return xarray.create_xarray_dataarray(*args, **kwargs)


# CfRadial 2.0 - ODIM_H5 mapping
moments_mapping = {
    'DBZH': {'standard_name': 'radar_equivalent_reflectivity_factor_h',
             'long_name': 'Equivalent reflectivity factor H',
             'short_name': 'DBZH',
             'units': 'dBZ',
             'gamic': ['zh']},
    'DBZV': {'standard_name': 'radar_equivalent_reflectivity_factor_v',
             'long_name': 'Equivalent reflectivity factor V',
             'short_name': 'DBZV',
             'units': 'dBZ',
             'gamic': ['zv']},
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
    'DBZ': {'standard_name': 'radar_equivalent_reflectivity_factor',
            'long_name': 'Equivalent reflectivity factor',
            'short_name': 'DBZ',
            'units': 'dBZ',
            'gamic': None},
    'DBTH': {'standard_name': 'radar_equivalent_reflectivity_factor_h',
             'long_name': 'Total power H (uncorrected reflectivity)',
             'short_name': 'DBTH',
             'units': 'dBZ',
             'gamic': ['uzh', 'uh']},
    'DBTV': {'standard_name': 'radar_equivalent_reflectivity_factor_v',
             'long_name': 'Total power V (uncorrected reflectivity)',
             'short_name': 'DBTV',
             'units': 'dBZ',
             'gamic': ['uzv', 'uv'],
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
        'gamic': ['vh']},
    'VRADV': {
        'standard_name': 'radial_velocity_of_scatterers_'
                         'away_from_instrument_v',
        'long_name': 'Radial velocity of scatterers away from instrument V',
        'short_name': 'VRADV',
        'units': 'meters per second',
        'gamic': ['vv'],
    },
    'VR': {
        'standard_name': 'radial_velocity_of_scatterers_away_'
                         'from_instrument',
        'long_name': 'Radial velocity of scatterers away from instrument',
        'short_name': 'VR',
        'units': 'meters per seconds',
        'gamic': None},
    'WRADH': {'standard_name': 'radar_doppler_spectrum_width_h',
              'long_name': 'Doppler spectrum width H',
              'short_name': 'WRADH',
              'units': 'meters per seconds',
              'gamic': ['wh']},
    'UWRADH': {'standard_name': 'radar_doppler_spectrum_width_h',
               'long_name': 'Doppler spectrum width H',
               'short_name': 'UWRADH',
               'units': 'meters per seconds',
               'gamic': ['uwh']},
    'WRADV': {'standard_name': 'radar_doppler_spectrum_width_v',
              'long_name': 'Doppler spectrum width V',
              'short_name': 'WRADV',
              'units': 'meters per second',
              'gamic': ['wv']},
    'ZDR': {'standard_name': 'radar_differential_reflectivity_hv',
            'long_name': 'Log differential reflectivity H/V',
            'short_name': 'ZDR',
            'units': 'dB',
            'gamic': ['zdr']},
    'UZDR': {'standard_name': 'radar_differential_reflectivity_hv',
             'long_name': 'Log differential reflectivity H/V',
             'short_name': 'UZDR',
             'units': 'dB',
             'gamic': ['uzdr']},
    'LDR': {'standard_name': 'radar_linear_depolarization_ratio',
            'long_name': 'Log-linear depolarization ratio HV',
            'short_name': 'LDR',
            'units': 'dB',
            'gamic': ['ldr']},
    'PHIDP': {'standard_name': 'radar_differential_phase_hv',
              'long_name': 'Differential phase HV',
              'short_name': 'PHIDP',
              'units': 'degrees',
              'gamic': ['phidp']},
    'UPHIDP': {'standard_name': 'radar_differential_phase_hv',
               'long_name': 'Differential phase HV',
               'short_name': 'UPHIDP',
               'units': 'degrees',
               'gamic': ['uphidp']},
    'KDP': {'standard_name': 'radar_specific_differential_phase_hv',
            'long_name': 'Specific differential phase HV',
            'short_name': 'KDP',
            'units': 'degrees per kilometer',
            'gamic': ['kdp']},
    'RHOHV': {'standard_name': 'radar_correlation_coefficient_hv',
              'long_name': 'Correlation coefficient HV',
              'short_name': 'RHOHV',
              'units': 'unitless',
              'gamic': ['rhohv']},
    'URHOHV': {'standard_name': 'radar_correlation_coefficient_hv',
               'long_name': 'Correlation coefficient HV',
               'short_name': 'URHOHV',
               'units': 'unitless',
               'gamic': ['urhohv']},
    'SNRH': {'standard_name': 'signal_noise_ratio_h',
             'long_name': 'Signal Noise Ratio H',
             'short_name': 'SNRH',
             'units': 'unitless',
             'gamic': None},
    'SNRV': {'standard_name': 'signal_noise_ratio_v',
             'long_name': 'Signal Noise Ratio V',
             'short_name': 'SNRV',
             'units': 'unitless',
             'gamic': None},
    'SQIH': {'standard_name': 'signal_quality_index_h',
             'long_name': 'Signal Quality H',
             'short_name': 'SQIH',
             'units': 'unitless',
             'gamic': None},
    'SQIV': {'standard_name': 'signal_quality_index_v',
             'long_name': 'Signal Quality V',
             'short_name': 'SQIV',
             'units': 'unitless',
             'gamic': None},
    'CCORH': {'standard_name': 'clutter_correction_h',
              'long_name': 'Clutter Correction H',
              'short_name': 'CCORH',
              'units': 'unitless',
              'gamic': None},
    'CCORV': {'standard_name': 'clutter_correction_v',
              'long_name': 'Clutter Correction V',
              'short_name': 'CCORV',
              'units': 'unitless',
              'gamic': None},

}

ODIM_NAMES = {value['short_name']: key for (key, value) in
              moments_mapping.items()}

GAMIC_NAMES = {v: key for (key, value) in moments_mapping.items()
               if value['gamic'] is not None for v in value['gamic']}

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

global_attrs = [('Conventions', 'Cf/Radial'),
                ('version', 'Cf/Radial version number'),
                ('title', 'short description of file contents'),
                ('institution', 'where the original data were produced'),
                ('references',
                 ('references that describe the data or the methods used '
                  'to produce it')),
                ('source', 'method of production of the original data'),
                ('history', 'list of modifications to the original data'),
                ('comment', 'miscellaneous information'),
                ('instrument_name', 'name of radar or lidar'),
                ('site_name', 'name of site where data were gathered'),
                ('scan_name', 'name of scan strategy used, if applicable'),
                ('scan_id',
                 'scan strategy id, if applicable. assumed 0 if missing'),
                ('platform_is_mobile',
                 '"true" or "false", assumed "false" if missing'),
                ('ray_times_increase',
                 ('"true" or "false", assumed "true" if missing. '
                  'This is set to true if ray times increase monotonically '
                  'thoughout all of the sweeps in the volume')),
                ('field_names',
                 'array of strings of field names present in this file.'),
                ('time_coverage_start',
                 'copy of time_coverage_start global variable'),
                ('time_coverage_end',
                 'copy of time_coverage_end global variable'),
                ('simulated data',
                 ('"true" or "false", assumed "false" if missing. '
                  'data in this file are simulated'))]

global_variables = dict([('volume_number', np.int),
                         ('platform_type', 'fixed'),
                         ('instrument_type', 'radar'),
                         ('primary_axis', 'axis_z'),
                         ('time_coverage_start', '1970-01-01T00:00:00Z'),
                         ('time_coverage_end', '1970-01-01T00:00:00Z'),
                         ('latitude', np.nan),
                         ('longitude', np.nan),
                         ('altitude', np.nan),
                         ('altitude_agl', np.nan),
                         ('sweep_group_name', (['sweep'], [np.nan])),
                         ('sweep_fixed_angle', (['sweep'], [np.nan])),
                         ('frequency', np.nan),
                         ('status_xml', 'None')])


@xr.register_dataset_accessor('gamic')
class GamicAccessor(object):
    """Dataset Accessor for handling GAMIC HDF5 data files
    """
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
            times = self._obj['timestamp'] / 1e6
            attrs = {'units': 'seconds since 1970-01-01T00:00:00Z',
                     'standard_name': 'time'}
            da = xr.DataArray(times, attrs=attrs)
            self._time_range = da
        return self._time_range


@xr.register_dataset_accessor('odim')
class OdimAccessor(object):
    """Dataset Accessor for handling ODIM_H5 data files
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._radial_range = None
        self._azimuth_range = None
        self._elevation_range = None
        self._time_range = None
        self._time_range2 = None

    @property
    def radial_range(self):
        """Return the radial range of this dataset."""
        if self._radial_range is None:
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
    def azimuth_range2(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            nrays = self._obj.attrs['nrays']
            res = 360. / nrays
            azimuth_data = np.arange(res / 2.,
                                     360.,
                                     res,
                                     dtype='float32')

            da = xr.DataArray(azimuth_data, attrs=az_attrs)
            self._azimuth_range = da
        return self._azimuth_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            startaz = self._obj.attrs['startazA']
            stopaz = self._obj.attrs['stopazA']
            zero_index = np.where(stopaz < startaz)
            stopaz[zero_index[0]] += 360
            azimuth_data = (startaz + stopaz) / 2.
            da = xr.DataArray(azimuth_data, attrs=az_attrs)
            self._azimuth_range = da
        return self._azimuth_range

    @property
    def elevation_range2(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            nrays = self._obj.attrs['nrays']
            elangle = self._obj.attrs['elangle']
            elevation_data = np.ones(nrays, dtype='float32') * elangle
            da = xr.DataArray(elevation_data, attrs=el_attrs)
            self._elevation_range = da
        return self._elevation_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            startel = self._obj.attrs['startelA']
            stopel = self._obj.attrs['stopelA']
            elevation_data = (startel + stopel) / 2.
            da = xr.DataArray(elevation_data, attrs=el_attrs)
            self._elevation_range = da
        return self._elevation_range

    @property
    def time_range(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            startT = self._obj.attrs['startazT']
            stopT = self._obj.attrs['stopazT']
            times = (startT + stopT) / 2.
            self._time_range = times

        return self._time_range

    @property
    def time_range2(self):
        """Return the time range of this dataset."""
        if self._time_range2 is None:
            startdate = self._obj.attrs['startdate']
            starttime = self._obj.attrs['starttime']
            enddate = self._obj.attrs['enddate']
            endtime = self._obj.attrs['endtime']

            start = dt.datetime.strptime(startdate + starttime, '%Y%m%d%H%M%S')
            end = dt.datetime.strptime(enddate + endtime, '%Y%m%d%H%M%S')
            start = start.replace(tzinfo=dt.timezone.utc)
            end = end.replace(tzinfo=dt.timezone.utc)

            self._time_range2 = (start.timestamp(), end.timestamp())
        return self._time_range2


def to_cfradial2(volume, filename):
    """ Save XRadVol to CfRadial2.0 compliant file.

    Parameters
    ----------
    volume : XRadVol object
    filename : str
        output filename
    """
    volume.root.load()
    root = volume.root.copy(deep=True)
    root.attrs['Conventions'] = 'Cf/Radial'
    root.attrs['version'] = '2.0'
    root.to_netcdf(filename, mode='w', group='/')
    for key in root.sweep_group_name.values:
        swp = volume[key]
        swp.load()
        dims = list(swp.dims)
        dims.remove('range')
        dim0 = dims[0]

        swp = swp.rename_dims({dim0: 'time'})
        swp.drop(['x', 'y', 'z', 'gr', 'rays', 'bins'], errors='ignore')
        swp.to_netcdf(filename, mode='a', group=key)


def to_odim(volume, filename):
    """ Save XRadVol to ODIM_H5/V2_2 compliant file.

    Parameters
    ----------
    volume : XRadVol object
    filename : str
        output filename
    """
    root = volume.root

    h5 = h5py.File(filename, 'w')

    # root group, only Conventions for ODIM_H5
    _write_odim({'Conventions': 'ODIM_H5/V2_2'}, h5)

    # how group
    how = {}
    how.update({'_modification_program': 'wradlib'})

    h5_how = h5.create_group('how')
    _write_odim(how, h5_how)

    sweepnames = root.sweep_group_name.values

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
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
    _write_odim(what, h5_what)

    # where group, lon, lat, height, mandatory
    where = {'lon': root.longitude.values,
             'lat': root.latitude.values,
             'height': root.altitude.values}
    h5_where = h5.create_group('where')
    _write_odim(where, h5_where)

    # datasets
    ds_list = ['dataset{}'.format(i + 1) for i in range(len(sweepnames))]
    ds_idx = np.argsort(ds_list)
    for idx in ds_idx:
        ds = volume['sweep_{}'.format(idx + 1)]
        h5_dataset = h5.create_group(ds_list[idx])

        # what group p. 21 ff.
        h5_ds_what = h5_dataset.create_group('what')
        ds_what = {}
        # skip NaT values
        valid_times = ~np.isnat(ds.time.values)
        t = sorted(ds.time.values[valid_times])
        start = dt.datetime.utcfromtimestamp(np.rint(t[0].astype('O') / 1e9))
        end = dt.datetime.utcfromtimestamp(
            np.rint(t[-1].astype('O') / 1e9))
        ds_what['product'] = 'SCAN'
        ds_what['startdate'] = start.strftime('%Y%m%d')
        ds_what['starttime'] = start.strftime('%H%M%S')
        ds_what['enddate'] = end.strftime('%Y%m%d')
        ds_what['endtime'] = end.strftime('%H%M%S')
        _write_odim(ds_what, h5_ds_what)

        # where group, p. 11 ff. mandatory
        h5_ds_where = h5_dataset.create_group('where')
        rscale = ds.range.values[1] / 1. - ds.range.values[0]
        rstart = (ds.range.values[0] - rscale / 2.) / 1000.
        a1gate = np.argsort(ds.sortby('time').azimuth.values)[0]
        ds_where = {'elangle': ds.fixed_angle,
                    'nbins': ds.range.shape[0],
                    'rstart': rstart,
                    'rscale': rscale,
                    'nrays': ds.azimuth.shape[0],
                    'a1gate': a1gate,
                    }
        _write_odim(ds_where, h5_ds_where)

        # how group, p. 14 ff.
        h5_ds_how = h5_dataset.create_group('how')
        tout = [tx.astype('O') / 1e9 for tx in ds.sortby('azimuth').time]
        difft = np.diff(tout) / 2.
        difft = np.insert(difft, 0, difft[0])
        azout = ds.sortby('azimuth').azimuth
        diffa = np.diff(azout) / 2.
        diffa = np.insert(diffa, 0, diffa[0])
        elout = ds.sortby('azimuth').elevation
        diffe = np.diff(elout) / 2.
        diffe = np.insert(diffe, 0, diffe[0])
        ds_how = {'scan_index': ds.sweep_number + 1,
                  'scan_count': len(sweepnames),
                  'startazT': tout - difft,
                  'stopazT': tout + difft,
                  'startazA': azout - diffa,
                  'stopazA': azout + diffa,
                  'startelA': elout - diffe,
                  'stopelA': elout + diffe,
                  }
        _write_odim(ds_how, h5_ds_how)

        # write moments
        _write_odim_dataspace(ds, h5_dataset)

    h5.close()


class XRadVolFile(object):
    """BaseClass for holding netCDF4.Dataset handles

    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        self._filename = filename
        self._nch = None
        self._flavour = None
        self._nch, self._flavour = self._check_file(filename, flavour)

    def _check_file(self, filename, flavour):
        raise NotImplementedError

    def __del__(self):
        if self._nch is not None:
            self._nch.close()

    @property
    def filename(self):
        return self._filename

    @property
    def nch(self):
        return self._nch

    @property
    def flavour(self):
        return self._flavour


class OdimH5File(XRadVolFile):
    """Class for holding netCDF4.Dataset handles of OdimH5 files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.
    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        super(OdimH5File, self).__init__(filename=filename,
                                         flavour=flavour, **kwargs)

    def _check_file(self, filename, flavour):
        nch = nc.Dataset(filename, diskless=True, persist=False)
        if nch.disk_format != 'HDF5':
            raise TypeError(
                'wradlib: File {} is neither "NETCDF4" (using HDF5 groups) '
                'nor plain "HDF5".'.format(filename))
        if flavour is None:
            try:
                flavour = nch.Conventions
            except AttributeError as e:
                raise AttributeError(
                    'wradlib: Missing "Conventions" attribute in {} ./n'
                    'Use the "flavour" kwarg to specify your source '
                    'data.'.format(filename)) from e
            if 'ODIM' not in flavour:
                raise AttributeError(
                    'wradlib: "Conventions" attribute "{}" in {} is unknown./n'
                    'Use the "flavour" kwarg to specify your source '
                    'data.'.format(flavour, filename))

        if "ODIM" in flavour:
            self._dsdesc = 'dataset'
            self._swmode = 'product'
            self._mfmt = 'data'
            self._msrc = 'groups'
        elif "GAMIC" in flavour:
            self._dsdesc = 'scan'
            self._swmode = 'scan_type'
            self._mfmt = 'moment_'
            self._msrc = 'variables'
        else:
            raise AttributeError(
                'wradlib: Unknown "flavour" kwarg attribute: {} .'
                ''.format(flavour))

        return nch, flavour

    @property
    def flavour(self):
        flv = ['ODIM', 'GAMIC']
        return [s for s in flv if s in self._flavour][0]


class NetCDF4File(XRadVolFile):
    """Class for holding netCDF4.Dataset handles of Cf/Radial files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of flavour ('Cf/Radial' or 'Cf/Radial2').
    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        super(NetCDF4File, self).__init__(filename=filename,
                                          flavour=flavour,
                                          **kwargs)

    def _check_file(self, filename, flavour):
        nch = nc.Dataset(filename, diskless=True, persist=False)
        if flavour is None:
            try:
                Conventions = nch.Conventions
                version = nch.version
            except AttributeError as e:
                raise AttributeError(
                    'wradlib: Missing "Conventions" attribute in {} ./n'
                    'Use the "flavour" kwarg to specify your source'
                    'data.'.format(filename)) from e
            if "cf/radial" in Conventions.lower():
                if version == '2.0':
                    flavour = 'Cf/Radial2'
                else:
                    flavour = 'Cf/Radial'

        if flavour not in ['Cf/Radial', 'Cf/Radial2']:
            raise AttributeError(
                'wradlib: Unknown "flavour" kwarg attribute: {} .'
                ''.format(flavour))

        return nch, flavour


class XRadVol(collections.abc.MutableMapping):
    """BaseClass for xarray based RadarVolumes

    Implements `collections.MutableMapping` dictionary.
    """

    def __init__(self, init_root=False):
        self._sweeps = dict()
        self._nch = list()
        self.root = None
        if init_root:
            self._init_root()

    def __getitem__(self, key):
        if key == 'root':
            warnings.warn("WRADLIB: Use of `obj['root']` is deprecated, "
                          "please use obj.root instead.", DeprecationWarning)
            return self._root

        return self._sweeps[key]

    def __setitem__(self, key, value):
        if key in self._sweeps:
            self._sweeps[key] = value
        else:
            warnings.warn("WRADLIB: Use class methods to add data. "
                          "Direct setting is not allowed.", UserWarning)

    def __delitem__(self, key):
        del self._sweeps[key]

    def __iter__(self):
        return iter(self._sweeps)

    def __len__(self):
        return len(self._sweeps)

    def __repr__(self):
        return self._sweeps.__repr__()

    def __del__(self):
        del self._root
        for k in list(self._sweeps):
            del k
        for k in self._nch:
            del k

    def _init_root(self):
        self.root = xr.Dataset(data_vars=global_variables,
                               attrs=global_attrs)

    @property
    def root(self):
        """ Return `root` dataset.
        """
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def sweep(self):
        """ Return sweep dimension count.
        """
        return self.root.dims['sweep']

    @property
    def sweeps(self):
        """ Return zip sweep names, sweep_angles
        """
        names = list(self.root.sweep_group_name.values)
        angles = list(self.root.sweep_fixed_angle.values)
        return zip(names, angles)

    @property
    def location(self):
        """ Return location of data source.
        """
        return (self.root.longitude.values.item(),
                self.root.latitude.values.item(),
                self.root.altitude.values.item())

    @property
    def Conventions(self):
        """ Return CF/ODIM `Conventions`.
        """
        return self.root.Conventions

    @property
    def version(self):
        """ Return CF/ODIM version
        """
        return self.root.version

    def to_cfradial2(self, filename):
        """ Save volume to CfRadial2.0 compliant file.

        Parameter
        ---------
        filename : str
            Name of the output file
        """
        if self.root:
            to_cfradial2(self, filename)
        else:
            warnings.warn("WRADLIB: No CfRadial2-compliant data structure "
                          "available. Not saving.", UserWarning)

    def to_odim(self, filename):
        """ Save volume to ODIM_H5/V2_2 compliant file.

        Parameter
        ---------
        filename : str
            Name of the output file
        """
        if self.root:
            to_odim(self, filename)
        else:
            warnings.warn("WRADLIB: No OdimH5-compliant data structure "
                          "available. Not saving.", UserWarning)

    def georeference(self, sweeps=None):
        """Georeference sweeps

        Parameter
        ---------
        sweeps : list
            list with sweep keys to georeference, defaults to all sweeps
        """
        if sweeps is None:
            sweeps = self

        for swp in sweeps:
            self[swp] = self[swp].pipe(xarray.georeference_dataset)


class CfRadial(XRadVol):
    """ Class for xarray based retrieval of CfRadial data files

    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize xarray structure from Cf/Radial data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of flavour ('Cf/Radial' or 'Cf/Radial2').

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int | dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        super(CfRadial, self).__init__()
        if not isinstance(filename, list):
            filename = [filename]
        for i, f in enumerate(filename):
            nch = NetCDF4File(f, flavour=flavour)
            self._nch.append(nch)
            if nch.flavour == "Cf/Radial2":
                self.assign_data_radial2(nch, **kwargs)
            else:
                self.assign_data_radial(nch, **kwargs)

    def assign_data_radial2(self, nch, **kwargs):
        """ Assign from CfRadial2 data structure.

        Parameter
        ---------
        nch : NetCDF4File object

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int | dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        # keyword argument handling
        georef = kwargs.pop('georef', False)
        dim0 = kwargs.pop('dim0', 'time')

        self.root = _open_dataset(nch.nch, grp=None, **kwargs)
        sweepnames = self.root.sweep_group_name.values
        for sw in sweepnames:
            ds = _open_dataset(nch.nch, grp=sw, **kwargs)
            ds = ds.rename_dims({'time': dim0})
            coords = {'longitude': self.root.longitude,
                      'latitude': self.root.latitude,
                      'altitude': self.root.altitude,
                      'azimuth': ds.azimuth,
                      'elevation': ds.elevation,
                      }
            ds = ds.assign_coords(**coords)

            # adding xyz aeqd-coordinates
            if georef:
                ds = xarray.georeference_dataset(ds)

            self._sweeps[sw] = ds

    def assign_data_radial(self, nch, **kwargs):
        """ Assign from CfRadial1 data structure.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int | dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        # keyword argument handling
        georef = kwargs.pop('georef', False)
        dim0 = kwargs.pop('dim0', 'time')

        root = _open_dataset(nch.nch, grp=None, **kwargs)
        var = root.variables.keys()
        remove_root = var ^ root_vars
        remove_root &= var
        root1 = root.drop(remove_root).rename(
            {'fixed_angle': 'sweep_fixed_angle'})
        sweep_group_name = []
        for i in range(root1.dims['sweep']):
            sweep_group_name.append('sweep_{}'.format(i + 1))
        self.root = root1.assign(
            {'sweep_group_name': (['sweep'], sweep_group_name)})

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
            ds = data.isel(time=tslice,
                           sweep=slice(i, i + 1)).squeeze('sweep')
            ds = ds.rename_dims({'time': dim0})
            ds.sweep_mode.load()
            coords = {'longitude': self.root.longitude,
                      'latitude': self.root.latitude,
                      'altitude': self.root.altitude,
                      'azimuth': ds.azimuth,
                      'elevation': ds.elevation,
                      'sweep_mode': ds.sweep_mode.item().decode(),
                      }
            ds = ds.assign_coords(**coords)

            # adding xyz aeqd-coordinates
            if georef:
                ds = xarray.georeference_dataset(ds)

            self._sweeps[sw] = ds


class OdimH5(XRadVol):
    """ Class for xarray based retrieval of ODIM_H5 data files
    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize xarray structure from hdf5 data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int | dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        standard : str
            * `none` - data is read as verbatim as possible, no metadata
            * `odim` - data is read, odim metadata added to datasets
            * `cf-mandatory` - data is read according to cfradial2 standard
              importing mandatory metadata
            * `cf-full` - data is read according to cfradial2 standard
              importing all available cfradial2 metadata (not fully
              implemented)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        super(OdimH5, self).__init__()

        if not isinstance(filename, list):
            filename = [filename]

        for f in filename:
            self.assign_data(f, flavour=flavour, **kwargs)

    def assign_data(self, filename, flavour=None, **kwargs):
        """Assign xarray dataset from hdf5 data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int | dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy).
            Defaults to True.
        standard : str
            * `none` - data is read as verbatim as possible, no metadata
            * `odim` - data is read, odim metadata added to datasets
            * `cf-mandatory` - data is read according to cfradial2 standard
              importing mandatory metadata, default value
            * `cf-full` - data is read according to cfradial2 standard
              importing all available cfradial2 metadata (not fully
              implemented)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard, default value
                * `azimuth` - better for working with xarray
        """
        nch = OdimH5File(filename, flavour=flavour)
        self._nch.append(nch)

        # keyword argument handling
        decode_times = kwargs.get('decode_times', True)
        decode_coords = kwargs.get('decode_coords', True)
        mask_and_scale = kwargs.get('mask_and_scale', True)
        georef = kwargs.get('georef', False)
        standard = kwargs.get('standard', 'cf-mandatory')
        dim0 = kwargs.get('dim0', 'time')

        # retrieve and assign global groups /how, /what, /where
        groups = ['how', 'what', 'where']
        how, what, where = _get_odim_groups(nch.nch, groups)
        rt_grps = {'how': how,
                   'what': what,
                   'where': where}

        # sweep group handling
        (src_swp_grp_name,
         swp_grp_name) = _get_odim_sweep_group_names(nch.nch,
                                                     nch._dsdesc)

        sweep_fixed_angle = []
        if 'cf' in standard:
            time_coverage_start = np.datetime64('2037-01-01')
            time_coverage_end = np.datetime64('1970-01-01')
            if not decode_times:
                epoch = np.datetime64('1970-01-01T00:00:00Z')
                time_coverage_start = ((time_coverage_start - epoch) /
                                       np.timedelta64(1, 's'))
                time_coverage_end = ((time_coverage_end - epoch) /
                                     np.timedelta64(1, 's'))

        # iterate sweeps
        for i, sweep in enumerate(src_swp_grp_name):
            # retrieve ds and assign datasetX how/what/where group attributes
            groups = [None, 'how', 'what', 'where']
            ds, ds_how, ds_what, ds_where = _get_odim_groups(nch.nch[sweep],
                                                             groups)
            ds_grps = {'how': ds_how,
                       'what': ds_what,
                       'where': ds_where}

            # moments
            # possible bug in netcdf-c reader assuming only one dimension when
            # dimensions have same size
            # see https://github.com/Unidata/netcdf4-python/issues/945
            if len(ds.dims) == 1:
                # we just read the contents without standard and decoding
                decode_times = False
                decode_coords = False
                georef = False
                standard = 'None'
            else:
                # need to reclaim kwargs
                decode_times = kwargs.get('decode_times', True)
                decode_coords = kwargs.get('decode_coords', True)
                georef = kwargs.get('georef', False)
                standard = kwargs.get('standard', 'cf-mandatory')
                ds = _assign_odim_moments(ds, nch, sweep, **kwargs)

            # retrieve and assign gamic ray_header
            if nch.flavour == 'GAMIC':
                rh = _get_gamic_ray_header(nch.filename, i)
                ds_grps['what'] = ds_grps['what'].assign(rh)

            # coordinates wrap-up

            vars = collections.OrderedDict()
            coords = collections.OrderedDict()
            if 'cf' in standard or georef:
                coords['longitude'] = rt_grps['where'].attrs['lon']
                coords['latitude'] = rt_grps['where'].attrs['lat']
                coords['altitude'] = rt_grps['where'].attrs['height']
            if 'cf' in standard or georef:
                sweep_mode = _get_odim_sweep_mode(nch, ds_grps)
                coords['sweep_mode'] = sweep_mode

            if 'cf' in standard or decode_coords or georef:
                vars.update(_get_odim_coordinates(nch, ds_grps))
                vars['azimuth'] = vars['azimuth'].rename({'dim_0': dim0})
                vars['elevation'] = vars['elevation'].rename({'dim_0': dim0})
                # georeference needs coordinate variables
                if georef:
                    geods = xr.Dataset(vars, coords)
                    geods = xarray.georeference_dataset(geods)
                    coords.update(geods.coords)

            # time coordinate
            if 'cf' in standard or decode_times:
                timevals = _get_odim_timevalues(nch, ds_grps)
                if decode_times:
                    coords['time'] = ([dim0], timevals, time_attrs)
                else:
                    coords['time'] = ([dim0], timevals)

            # assign global sweep attributes
            fixed_angle = _get_odim_fixed_angle(nch, ds_grps)
            sweep_fixed_angle.append(fixed_angle)
            if 'cf' in standard:
                vars.update({'sweep_number': i,
                             'sweep_mode': sweep_mode,
                             'follow_mode': 'none',
                             'prt_mode': 'fixed',
                             'fixed_angle': fixed_angle})

            # assign variables and coordinates
            ds = ds.assign(vars)
            ds = ds.assign_coords(**coords)

            # decode dataset if requested
            if decode_times or decode_coords or mask_and_scale:
                ds = xr.decode_cf(ds, decode_times=decode_times,
                                  decode_coords=decode_coords,
                                  mask_and_scale=mask_and_scale)

            # extract time coverage
            if 'cf' in standard:
                time_coverage_start = min(ds.time.values.min(),
                                          time_coverage_start)
                time_coverage_end = max(ds.time.values.max(),
                                        time_coverage_end)

            # dataset only
            is_new_sweep = False
            # first file
            if self.root is None:
                self._sweeps[swp_grp_name[i]] = ds
            # all other files
            else:
                # sort sweeps by angles
                rt = self.root
                rt = rt.assign_coords(sweep=rt['sweep_fixed_angle'])
                rt = (rt.sortby('sweep_fixed_angle').
                      sel(sweep=slice(fixed_angle, fixed_angle)))
                # determine if same sweep
                if fixed_angle == rt['sweep_fixed_angle']:
                    dictkey = rt['sweep_group_name'].item()
                    # merge datasets
                    self._sweeps[dictkey] = xr.merge([self._sweeps[dictkey],
                                                      ds])
                # not same sweep (new sweep)
                else:
                    nidx = len(self._sweeps) + 1
                    swp_grp_name[i] = f'sweep_{nidx}'
                    self._sweeps[swp_grp_name[i]] = ds
                    is_new_sweep = True

        # assign root variables
        if 'cf' in standard:
            time_coverage_start_str = str(time_coverage_start)[:19] + 'Z'
            time_coverage_end_str = str(time_coverage_end)[:19] + 'Z'

            # create root group from scratch
            root = xr.Dataset(data_vars=global_variables,
                              attrs=global_attrs)

            # assign root variables
            root = root.assign({'volume_number': 0,
                                'platform_type': str('fixed'),
                                'instrument_type': 'radar',
                                'primary_axis': 'axis_z',
                                'time_coverage_start': time_coverage_start_str,
                                'time_coverage_end': time_coverage_end_str,
                                'latitude': rt_grps['where'].attrs['lat'],
                                'longitude': rt_grps['where'].attrs['lon'],
                                'altitude': rt_grps['where'].attrs['height'],
                                'sweep_group_name': (['sweep'], swp_grp_name),
                                'sweep_fixed_angle': (
                                    ['sweep'], sweep_fixed_angle),
                                })
            # assign root attributes
            attrs = _get_odim_root_attributes(nch, rt_grps)
            root = root.assign_attrs(attrs)

            if self.root is None:
                self.root = root
            else:
                if is_new_sweep:
                    # fix time coverage
                    tcs = self.root['time_coverage_start'].values.item()
                    tcs = np.datetime64(tcs)
                    tce = self.root['time_coverage_end'].values.item()
                    tce = np.datetime64(tce)
                    self.root = xr.concat([self.root, root], 'sweep',
                                          data_vars='different')
                    tcs = str(min(time_coverage_start, tcs))[:19] + 'Z'
                    tce = str(max(time_coverage_end, tce))[:19] + 'Z'
                    self.root['time_coverage_start'] = tcs
                    self.root['time_coverage_end'] = tce


def _write_odim(src, dest):
    """ Writes Odim Attributes.

    Parameters
    ----------
    src : dict
        Attributes to write
    dest : handle
        h5py-group handle
    """
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


def _write_odim_dataspace(src, dest):
    """ Writes Odim Dataspaces.

    Parameters
    ----------
    src : dict
        Moments to write
    dest : handle
        h5py-group handle
    """
    keys = [key for key in src if key in ODIM_NAMES]
    data_list = ['data{}'.format(i + 1) for i in range(len(keys))]
    data_idx = np.argsort(data_list)
    for idx in data_idx:
        value = src[keys[idx]]
        h5_data = dest.create_group(data_list[idx])
        enc = value.encoding

        # p. 21 ff
        h5_what = h5_data.create_group('what')
        try:
            undetect = float(value._Undetect)
        except AttributeError:
            undetect = np.finfo(np.float).max
        what = {'quantity': value.name,
                'gain': float(enc['scale_factor']),
                'offset': float(enc['add_offset']),
                'nodata': float(enc['_FillValue']),
                'undetect': undetect,
                }
        _write_odim(what, h5_what)

        # moments handling
        val = value.sortby('azimuth').values
        fillval = enc['_FillValue'] * enc['scale_factor']
        fillval += enc['add_offset']
        val[np.isnan(val)] = fillval
        val = (val - enc['add_offset']) / enc['scale_factor']
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


def _open_dataset(nch, grp=None, **kwargs):
    """ Open netcdf4/hdf5 group as xarray dataset.

    Parameters
    ----------
    nch : handle
        netcdf4-file handle
    grp : str
        group to access

    Returns
    -------
    nch : handle
        xarray Dataset handle
    """
    if grp is not None:
        nch = nch.groups.get(grp, False)
    if nch:
        nch = xr.open_dataset(xr.backends.NetCDF4DataStore(nch), **kwargs)
    return nch


def _get_gamic_ray_header(filename, scan):
    """Returns GAMIC ray header dictionary.

    Parameters
    ----------
    filename : str
        filename of GAMIC file
    scan : int
        Number of scan in file

    Returns
    -------
    vars : dict
        OrderedDict of ray header items
    """
    # ToDo: move rayheader into own dataset
    h5 = h5py.File(filename)
    ray_header = h5['scan{}/ray_header'.format(scan)][:]
    h5.close()
    vars = collections.OrderedDict()
    for name in ray_header.dtype.names:
        rh = ray_header[name]
        attrs = None
        vars.update({name: (['dim_0'], rh, attrs)})
    return vars


def _get_odim_sweep_group_names(nch, name):
    """ Return sweep names.

    Returns source names and cfradial names.

    Parameters
    ----------
    nch : handle
        netCDF4 Dataset handle
    name : str
        Common part of source dataset names.

    Returns
    -------
    src : list
        list of source dataset names
    swg_grp_name : list
        list of corresponding cfradial sweep_group_name
    """
    src = [key for key in nch.groups.keys() if name in key]
    src.sort(key=lambda x: int(x[len(name):]))
    swp_grp_name = ['sweep_{}'.format(i) for i in
                    range(1, len(src) + 1)]
    return src, swp_grp_name


def _get_odim_variables_moments(ds, moments=None, **kwargs):
    """ Retrieve radar moments from dataset variables.

    Parameters
    ----------
    ds : xarray dataset
        source dataset
    moments : list
        list of moment strings

    Returns
    -------
    ds : xarray Dataset
        altered dataset
    """

    standard = kwargs.get('standard', 'cf-mandatory')
    mask_and_scale = kwargs.get('mask_and_scale', True)
    decode_coords = kwargs.get('decode_coords', True)
    dim0 = kwargs.get('dim0', 'time')

    # fix dimensions
    dims = sorted(list(ds.dims.keys()),
                  key=lambda x: int(x[len('phony_dim_'):]))
    ds = ds.rename({dims[0]: dim0,
                    dims[1]: 'range',
                    })

    for mom in moments:
        # open dataX dataset
        dmom = ds[mom]
        name = dmom.moment.lower()
        if 'cf' in standard and name not in GAMIC_NAMES.keys():
            ds = ds.drop(mom)
            continue

        # extract attributes
        dmax = np.iinfo(dmom.dtype).max
        dmin = np.iinfo(dmom.dtype).min
        minval = dmom.dyn_range_min
        maxval = dmom.dyn_range_max
        gain = (maxval - minval) / dmax
        offset = minval
        fillval = float(dmax)
        undetect = float(dmin)

        # create attribute dict
        attrs = collections.OrderedDict()
        # clean moment attributes
        if standard != 'none':
            dmom.attrs = collections.OrderedDict()

        if standard in ['odim']:
            attrs['gain'] = gain
            attrs['offset'] = offset
            attrs['nodata'] = fillval
            attrs['undetect'] = undetect

        # add cfradial moment attributes
        if 'cf' in standard or mask_and_scale:
            attrs['scale_factor'] = gain
            attrs['add_offset'] = minval
            attrs['_FillValue'] = float(dmax)

        if 'cf' in standard or decode_coords:
            attrs['coordinates'] = 'elevation azimuth range'

        if 'full' in standard:
            attrs['_Undetect'] = undetect

        if 'cf' in standard:
            cfname = GAMIC_NAMES[name]
            for k, v in moments_mapping[cfname].items():
                attrs[k] = v
            name = attrs.pop('short_name')
            attrs.pop('gamic')

        # assign attributes to moment
        dmom.attrs.update(attrs)

        # keep original dataset name
        if standard != 'none':
            ds = ds.rename({mom: name.upper()})

    return ds


def _get_odim_group_moments(nch, sweep, moments=None, **kwargs):
    """ Retrieve radar moments from hdf groups.

    Parameters
    ----------
    nch : netCDF Dataset handle
    sweep : str
        sweep key
    moments : list
        list of moment strings

    Returns
    -------
    ds : dictionary
        moment datasets
    """

    standard = kwargs.get('standard', 'cf-mandatory')
    mask_and_scale = kwargs.get('mask_and_scale', True)
    decode_coords = kwargs.get('decode_coords', True)
    dim0 = kwargs.get('dim0', 'time')
    chunks = kwargs.get('chunks', None)

    datas = {}
    for mom in moments:
        dmom_what = _open_dataset(nch[sweep][mom], 'what', chunks=chunks)
        name = dmom_what.attrs.pop('quantity')
        if 'cf' in standard and name not in moments_mapping.keys():
            continue
        dsmom = _open_dataset(nch[sweep], mom, chunks=chunks)

        # create attribute dict
        attrs = collections.OrderedDict()

        if standard in ['odim']:
            attrs.update(dmom_what.attrs)

        # add cfradial moment attributes
        if 'cf' in standard or mask_and_scale:
            attrs['scale_factor'] = dmom_what.attrs.get('gain')
            attrs['add_offset'] = dmom_what.attrs.get('offset')
            attrs['_FillValue'] = dmom_what.attrs.get('nodata')
        if 'cf' in standard or decode_coords:
            attrs['coordinates'] = 'elevation azimuth range'
        if 'cf' in standard:
            for k, v in moments_mapping[name].items():
                attrs[k] = v
            # drop short_name
            attrs.pop('short_name')
            attrs.pop('gamic')
        if 'full' in standard:
            attrs['_Undetect'] = dmom_what.attrs.get('undetect')

        # assign attributes
        dmom = dsmom.data.assign_attrs(attrs)

        # fix dimensions
        dims = dmom.dims

        # keep original dataset name
        if standard == 'none':
            name = mom

        datas.update({name: dmom.rename({dims[0]: dim0,
                                         dims[1]: 'range'
                                         })})
    return datas


def _get_odim_groups(ncf, groups, **kwargs):
    """ Get hdf groups.

    Parameters
    ----------
    ncf : netCDf4 Dataset handle
    groups : list
        list of groups-keys

    Returns
    -------
    out : tuple
        tuple of xarray datasets
    """
    return tuple(map(lambda x: _open_dataset(ncf, x, **kwargs), groups))


def _get_odim_moment_names(sweep, fmt=None, src=None):
    """Get moment names.

    Parameters
    ----------
    sweep : netCDf4 Group handle
    fmt : str
        dataset descriptor format
    src : str
        dataset location

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        array of moment names
    """
    moments = [mom for mom in getattr(sweep, src).keys() if
               fmt in mom]
    moments_idx = np.argsort([int(s[len(fmt):]) for s in moments])
    return np.array(moments)[moments_idx]


def _assign_odim_moments(ds, nch, sweep, **kwargs):
    """Assign radar moments to dataset.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    ds : xarray dataset
        destination dataset
    sweep : str
        netcdf group name

    Keyword Arguments
    -----------------
    decode_times : bool
        If True, decode cf times to np.datetime64. Defaults to True.
    decode_coords : bool
        If True, use the ‘coordinates’ attribute on variable
        to assign coordinates. Defaults to True.
    mask_and_scale : bool
        If True, lazily scale (using scale_factor and add_offset)
        and mask (using _FillValue). Defaults to True.
    georef : bool
        If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
        2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
    standard : str
        * `none` - data is read as verbatim as possible, no metadata
        * `odim` - data is read, odim metadata added to datasets
        * `cf-mandatory` - data is read according to cfradial2 standard
          importing mandatory metadata
        * `cf-full` - data is read according to cfradial2 standard
          importing all available cfradial2 metadata (not fully
          implemented)
    dim0 : str
        name of the ray-dimension of DataArrays and Dataset:
            * `time` - cfradial2 standard
            * `azimuth` - better for working with xarray

    Returns
    -------
    ds : xarray dataset
        Dataset with assigned radar moments
    """
    moments = _get_odim_moment_names(nch.nch[sweep], fmt=nch._mfmt,
                                     src=nch._msrc)
    if nch.flavour == 'ODIM':
        for name, dmom in _get_odim_group_moments(nch.nch, sweep,
                                                  moments=moments,
                                                  **kwargs).items():
            ds[name] = dmom
    if nch.flavour == 'GAMIC':
        ds = _get_odim_variables_moments(ds, moments=moments, **kwargs)

    return ds


def _get_odim_timevalues(nch, grps):
    """Retrieve TimeArray from source data.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    timevals : :class:`numpy:numpy.ndarray`
            array of time values
    """
    if nch.flavour == 'ODIM':
        try:
            timevals = grps['how'].odim.time_range
        except (KeyError, AttributeError):
            # timehandling if only start and end time is given
            start, end = grps['what'].odim.time_range2
            delta = (end - start) / grps['where'].nrays
            timevals = np.arange(start + delta / 2., end, delta)
            timevals = np.roll(timevals, shift=-grps['where'].a1gate)
    if nch.flavour == 'GAMIC':
        timevals = grps['what'].gamic.time_range.values

    return timevals


def _get_odim_coordinates(nch, grps):
    """Retrieve coordinates according OdimH5 standard.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    coords : dict
        Dictionary of coordinate arrays
    """
    flavour = nch.flavour.lower()
    coords = collections.OrderedDict()
    if flavour == 'odim':
        rng = grps['where']
        az = el = grps['how']
    if flavour == 'gamic':
        az = el = grps['what']
        rng = grps['how']
    try:
        coords['azimuth'] = getattr(az, flavour).azimuth_range
        coords['elevation'] = getattr(el, flavour).elevation_range
    except (KeyError, AttributeError):
        az = el = grps['where']
        coords['azimuth'] = getattr(az, flavour).azimuth_range2
        coords['elevation'] = getattr(el, flavour).elevation_range2
    coords['range'] = getattr(rng, flavour).radial_range

    return coords


def _get_odim_sweep_mode(nch, grp):
    """Retrieve sweep mode

    Parameters
    ----------
    nch : netCDF4.Dataset handle
    grp : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    out : str
        'azimuth_surveillance' or 'rhi'

    """
    odim_mode = grp['what'].attrs[nch._swmode]
    return ('rhi' if odim_mode == 'RHI' else 'azimuth_surveillance')


def _get_odim_fixed_angle(nch, grps):
    """Retrieve fixed angle from source data.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    fixed-angle : float
        fixed angle of specific scan
    """
    mode = _get_odim_sweep_mode(nch, grps)
    if nch.flavour == 'ODIM':
        ang = {'azimuth_surveillance': 'elangle', 'rhi': 'azangle'}
        fixed_angle = getattr(grps['where'], ang[mode])
    if nch.flavour == 'GAMIC':
        ang = {'azimuth_surveillance': 'elevation', 'rhi': 'azimuth'}
        fixed_angle = grps['how'].attrs[ang[mode]]
    return fixed_angle


def _get_odim_root_attributes(nch, grps):
    """Retrieve root attributes according CfRadial2 standard.

    Parameters
    ----------
    grps : dict
        Dictionary of root hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    attrs : dict
        Dictionary of root attributes
    """
    attrs = collections.OrderedDict()
    attrs.update({'version': 'None',
                  'title': 'None',
                  'institution': 'None',
                  'references': 'None',
                  'source': 'None',
                  'history': 'None',
                  'comment': 'im/exported using wradlib',
                  'instrument_name': 'None',
                  })

    attrs['version'] = grps['what'].attrs['version']

    if nch.flavour == 'ODIM':
        attrs['institution'] = grps['what'].attrs['source']
        attrs['instrument'] = grps['what'].attrs['source']
    if nch.flavour == 'GAMIC':
        attrs['title'] = grps['how'].attrs['template_name']
        attrs['instrument'] = grps['how'].attrs['host_name']

    return attrs
