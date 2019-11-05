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

   {}
"""
__all__ = ['XRadVol', 'CfRadial', 'OdimH5', 'to_cfradial2', 'to_odim',
           'create_xarray_dataarray']
__doc__ = __doc__.format('\n   '.join(__all__))

import collections
import datetime as dt
import warnings

import deprecation
import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(val, **kwargs):
        print("wradlib: Please wait for completion of time consuming task! \n"
              "wradlib: Please install 'tqdm' for showing a progress bar "
              "instead.")
        return val

from wradlib.georef import xarray
from wradlib import version


@deprecation.deprecated(deprecated_in="1.5", removed_in="2.0",
                        current_version=version.version,
                        details="Use `wradlib.georef.create_xarray_dataarray` "
                                "instead.")
def create_xarray_dataarray(*args, **kwargs):
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

cf_full_vars = {'prt': 'prf', 'n_samples': 'pulse'}

global_attrs = dict([('Conventions', 'Cf/Radial'),
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
                     (
                     'scan_name', 'name of scan strategy used, if applicable'),
                     ('scan_id',
                      'scan strategy id, if applicable. assumed 0 if missing'),
                     ('platform_is_mobile',
                      '"true" or "false", assumed "false" if missing'),
                     ('ray_times_increase',
                      ('"true" or "false", assumed "true" if missing. '
                       'This is set to true if ray times increase '
                       'monotonically thoughout all of the sweeps '
                       'in the volume')),
                     ('field_names',
                      'array of strings of field names present in this file.'),
                     ('time_coverage_start',
                      'copy of time_coverage_start global variable'),
                     ('time_coverage_end',
                      'copy of time_coverage_end global variable'),
                     ('simulated data',
                      ('"true" or "false", assumed "false" if missing. '
                       'data in this file are simulated'))])

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
            da = xr.DataArray(range_data, dims=['dim_1'], attrs=range_attrs)
            self._radial_range = da
        return self._radial_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            azstart = self._obj['azimuth_start']
            azstop = self._obj['azimuth_stop']
            zero_index_diff = azstop - azstart
            zero_index = np.where(zero_index_diff < -0.1)
            azstop[zero_index[0]] += 360
            azimuth = np.round((azstart + azstop) / 2., decimals=10)
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
        self._prt = None
        self._n_samples = None

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

            da = xr.DataArray(range_data, dims=['dim_1'], attrs=range_attrs)
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

            da = xr.DataArray(azimuth_data, dims=['dim_0'], attrs=az_attrs)
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
            da = xr.DataArray(elevation_data, dims=['dim_0'], attrs=el_attrs)
            self._elevation_range = da
        return self._elevation_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            startel = self._obj.attrs['startelA']
            stopel = self._obj.attrs['stopelA']
            elevation_data = (startel + stopel) / 2.
            da = xr.DataArray(elevation_data, dims=['dim_0'], attrs=el_attrs)
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

    @property
    def prt(self):
        if self._prt is None:
            try:
                prt = 1. / self._obj.attrs['prf']
                da = xr.DataArray(prt, dims=['dim_0'])
                self._prt = da
            except KeyError:
                pass
        return self._prt

    @property
    def n_samples(self):
        if self._n_samples is None:
            try:
                da = xr.DataArray(self._obj.attrs['pulse'], dims=['dim_0'])
                self._n_samples = da
            except KeyError:
                pass
        return self._n_samples


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
        ds_how = {'scan_index': idx,
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


class XRadFile(collections.abc.MutableSequence):
    """BaseClass for holding Xarray backed netCDF4.Dataset structures
    """

    def __init__(self, filename=None, flavour=None, **kwargs):
        self._netcdf = None
        self._xarray = None
        self._sweeps = None
        self._flavour = flavour
        self._sweep_angles = None
        self._sweep_times = None
        self._standard = kwargs.get('standard', 'none')
        self._mask_and_scale = kwargs.get('mask_and_scale', True)
        self._decode_coords = kwargs.get('decode_coords', True)
        self._decode_times = kwargs.get('decode_times', True)
        self._chunks = kwargs.get('chunks', None)
        self._georef = kwargs.get('georef', False)
        self._dim0 = kwargs.get('dim0', 'time')

        self._netcdf = nc.Dataset(filename, diskless=True, persist=False)

        self._claim_file()
        self.reload()

    def __getitem__(self, index):
        return self._sweeps[index]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def insert(self, pos, val):
        self._sweeps.insert(pos, val)

    def __iter__(self):
        return iter(self._sweeps)

    def __len__(self):
        return len(self._sweeps)

    def __repr__(self):
        return self._sweeps.__repr__()

    def __del__(self):
        if self._netcdf is not None:
            self._netcdf.close()

    @property
    def xarray(self):
        return self._xarray

    @property
    def filename(self):
        return self._netcdf.filepath()

    @property
    def netcdf(self):
        return self._netcdf

    @property
    def flavour(self):
        return self._flavour

    @property
    def mask_and_scale(self):
        return self._mask_and_scale

    @property
    def standard(self):
        return self._standard

    @property
    def decode_coords(self):
        return self._decode_coords

    @property
    def decode_times(self):
        return self._decode_times

    @property
    def chunks(self):
        return self._chunks

    @property
    def georef(self):
        return self._georef

    @property
    def sweeps(self):
        raise NotImplementedError

    @property
    def sweep_angles(self):
        if self._sweep_angles is None:
            self._sweep_angles = [swp.fixed_angle.values.min()
                                  for swp in self.sweeps]
        return self._sweep_angles

    @property
    def sweep_times(self):
        if self._sweep_times is None:
            self._sweep_times = [swp.time.values.min() for swp in self.sweeps]
        return self._sweep_times

    def _claim_file(self):
        """Claims File using netCDF4.Dataset

        """
        if self._netcdf.disk_format != 'HDF5':
            raise TypeError(
                'wradlib: File {} is neither "NETCDF4" (using HDF5 groups) '
                'nor plain "HDF5".'.format(self.filename))

    def reload(self):
        self._sweeps = None
        self._sweep_angles = None
        self._sweep_times = None
        self._xarray = _get_file_groups(self._netcdf, chunks=self.chunks)


class H5RadFile(XRadFile):
    """Class for holding netCDF4.Dataset handles of HDF5 Radar Files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.
    """

    def __init__(self, filename=None, flavour=None, **kwargs):
        kwargs['standard'] = kwargs.get('standard', 'odim')
        kwargs['dim0'] = kwargs.get('dim0', 'azimuth')
        super(H5RadFile, self).__init__(filename=filename,
                                        flavour=flavour, **kwargs)

    @property
    def sweeps(self):
        if self._sweeps is None:
            src = [key for key in self._xarray.keys() if
                   self._dsdesc in key]
            src.sort(key=lambda x: int(x[len(self._dsdesc):]))
            swps = [self._xarray[s] for s in src]
            self._sweeps = list(swps)
        return self._sweeps

    def _merge_moments(self):
        for k, sweep in enumerate(self.sweeps):
            if isinstance(sweep, xr.Dataset):
                continue
            ds = self._get_sweep_moments(sweep)
            ds = ds.assign(self._get_sweep_variables(k, sweep))
            ds = ds.assign_coords(**self._get_sweep_coordinates(sweep))
            ds = ds.rename_dims({'dim_0': self._dim0, 'dim_1': 'range'})
            if self.georef:
                ds = xarray.georeference_dataset(ds)
            if self.mask_and_scale | self.decode_coords | self.decode_times:
                ds = self._decode_cf(ds)
            self.sweeps[k] = ds

    def _decode_cf(self, sweep):
        ds = xr.decode_cf(sweep,
                          decode_times=self.decode_times,
                          mask_and_scale=self.mask_and_scale,
                          decode_coords=self.decode_coords,
                          )
        return ds

    def _get_sweep_coordinates(self, sweep):
        coords = collections.OrderedDict()

        if 'cf' in self.standard or self.georef:
            coords['longitude'] = self._xarray['where'].attrs['lon']
            coords['latitude'] = self._xarray['where'].attrs['lat']
            coords['altitude'] = self._xarray['where'].attrs['height']

        sweep_mode = sweep['what'].attrs[self._swmode]
        sweep_angle = self._get_sweep_angle(sweep)
        sweep_mode = 'rhi' if sweep_mode == 'RHI' else 'azimuth_surveillance'
        coords['sweep_mode'] = sweep_mode
        coords['fixed_angle'] = sweep_angle

        # time coordinate
        if 'cf' in self.standard or self.decode_coords or self.decode_times:
            timevals = self._get_time_coordinate(sweep)
            if self.decode_times:
                coords['time'] = ([self._dim0], timevals, time_attrs)
            else:
                coords['time'] = ([self._dim0], timevals)

        if 'cf' in self.standard or self.decode_coords or self.georef:
            coords.update(self._get_polar_coordinates(sweep))

        return coords

    def _get_sweep_variables(self, num, sweep):
        variables = collections.OrderedDict()
        variables.update({'fixed_angle': self._get_sweep_angle(sweep)})

        # Todo: might need this later
        # if 'cf' in 'standard' or self.georef:
        #     sweep_mode = sweep['what'].attrs[self._swmode].lower()
        #     if 'rhi' not in sweep_mode:
        #         sweep_mode = 'azimuth_surveillance'
        #     variables.update({'sweep_mode': sweep_mode})

        if 'cf' in self.standard:
            variables.update({'sweep_number': num,
                              'follow_mode': 'none',
                              'prt_mode': 'fixed',
                              })

        if 'cf-full' in self.standard:
            full_vars = self._get_sweep_full_vars(sweep)
            variables.update(full_vars)

        return variables


class GamicFile(H5RadFile):
    """Class for holding netCDF4.Dataset handles of Gamic HDF5 files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.
    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        super(GamicFile, self).__init__(filename=filename, flavour=flavour,
                                        **kwargs)

    def _claim_file(self):
        super(GamicFile, self)._claim_file()
        filename = self.filename
        flavour = self.flavour
        nch = self._netcdf
        if flavour is None:
            try:
                how = nch['how']
            except IndexError:
                raise AttributeError(
                    'wradlib: Missing attributes in {} ./n'
                    'This is no GAMIC HDF file.'.format(filename))
            else:
                if hasattr(how, 'sdp_manufacturer'):
                    flavour = how.sdp_manufacturer
                elif hasattr(how, 'sdp_name'):
                    if 'ENIGMA' in how.sdp_name:
                        flavour = 'GAMIC'
                elif hasattr(how, 'software'):
                    if 'MURAN' in how.software:
                        flavour = 'GAMIC'
                else:
                    raise AttributeError(
                        'wradlib: Missing attributes in {} ./n'
                        'This is no GAMIC HDF file.'.format(filename))
        self._dsdesc = 'scan'
        self._swmode = 'scan_type'
        self._mfmt = 'moment_'
        self._msrc = 'variables'

        self._flavour = flavour

    def reload(self):
        super(GamicFile, self).reload()
        self._assign_ray_header()
        self._merge_moments()

    def _assign_ray_header(self):
        for k, sweep in enumerate(self.sweeps):
            key = [k for k in sweep if self._dsdesc in k]
            rh = _get_gamic_ray_header(self.filename, key[0])
            sweep['what'] = sweep['what'].assign(rh)

    def _get_sweep_angle(self, sweep):
        try:
            angle = sweep['how']['how'].elevation
        except AttributeError:
            angle = sweep['how']['how'].azimuth
        return angle

    # possible bug in netcdf-c reader assuming only one dimension when
    # dimensions have same size
    # see https://github.com/Unidata/netcdf4-python/issues/945
    # see https://github.com/Unidata/netcdf-c/issues/1484
    # fixed by dimension reassignment in _get_sweep_moments and
    def _get_sweep_moments(self, sweep):
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

        standard = self.standard
        mask_and_scale = self.mask_and_scale
        decode_coords = self.decode_coords

        ds = [v for k, v in sweep.items() if 'scan' in k][0]

        # fix dimensions
        dims = sorted(list(ds.dims.keys()),
                      key=lambda x: int(x[len('phony_dim_'):]))

        ds = ds.rename({dims[0]: 'dim_0'})
        if len(dims) > 1:
            ds = ds.rename({dims[1]: 'dim_1'})

        for mom in ds.variables:
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

            # fix dimensions, when dim_0 == dim_1
            dims = dmom.dims
            if dims[0] == dims[1]:
                ds.update({mom: (['dim_0', 'dim_1'], dmom)})

            # keep original dataset name
            if standard != 'none':
                ds = ds.rename({mom: name.upper()})

        return ds

    def _get_polar_coordinates(self, sweep):
        """Retrieve coordinates according OdimH5 standard.

        Parameters
        ----------
        sweep : dict
            Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        coords : dict
            Dictionary of coordinate arrays
        """
        coords = collections.OrderedDict()
        coords['azimuth'] = sweep['what'].gamic.azimuth_range
        coords['elevation'] = sweep['what'].gamic.elevation_range
        coords['range'] = sweep['how']['how'].gamic.radial_range

        return coords

    def _get_time_coordinate(self, sweep):
        """Retrieve TimeArray from source data.

        Parameters
        ----------
        sweep : dict
             Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        timevals : :class:`numpy:numpy.ndarray`
                array of time values
        """
        timevals = sweep['what'].gamic.time_range.values

        return timevals

    # todo fix for gamic to import from all groups
    def _get_sweep_full_vars(self, sweep):
        """Retrieve available non mandatory variables from source data.

        Parameters
        ----------
        sweep : dict
             Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        full_vars : dict
            full cf-variables
        """
        full_vars = collections.OrderedDict()
        for k, v in cf_full_vars.items():
            full_vars[k] = getattr(sweep['how'].gamic, k)
        return full_vars

    def _get_root_attributes(self):
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

        attrs['version'] = self._xarray['what'].attrs['version']

        attrs['title'] = self._xarray['how'].attrs['template_name']
        attrs['instrument'] = self._xarray['how'].attrs['host_name']

        return attrs


class OdimH5File(H5RadFile):
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

    def _claim_file(self):
        super(OdimH5File, self)._claim_file()
        filename = self.filename
        flavour = self.flavour
        nch = self._netcdf
        if flavour is None:
            try:
                flavour = nch.Conventions
            except AttributeError as e:
                raise AttributeError(
                    'wradlib: Missing "Conventions" attribute in {} ./n'
                    'This is no ODIM H5 data file. '
                    ''.format(filename)) from e
        if 'ODIM' not in flavour:
            raise AttributeError(
                'wradlib: "Conventions" attribute "{}" in {} is unknown.'
                ''.format(flavour, filename))

        flavour = 'ODIM'
        self._dsdesc = 'dataset'
        self._swmode = 'product'
        self._mfmt = 'data'
        self._msrc = 'groups'

        self._flavour = flavour

    def reload(self):
        super(OdimH5File, self).reload()
        self._merge_moments()

    def _get_sweep_angle(self, sweep):
        try:
            angle = sweep['where'].elangle
        except AttributeError:
            angle = sweep['where'].azangle
        return angle

    def _get_sweep_moments(self, sweep):
        """ Retrieve radar moments from hdf groups.

        Parameters
        ----------
        sweep : dict
             Dictionary containing sweep hdf5 groups

        Returns
        -------
        ds : dictionary
            moment datasets
        """

        standard = self.standard
        mask_and_scale = self.mask_and_scale
        decode_coords = self.decode_coords

        # get moment dicts/ sort etc
        moments = [mom for mom in sweep if self._mfmt in mom]
        moments_idx = np.argsort([int(s[len(self._mfmt):]) for s in moments])
        moments = np.array(moments)[moments_idx].tolist()
        moments = {mom: sweep[mom] for mom in moments}

        datas = {}
        for key, mom in moments.items():
            dmom_what = mom['what']
            name = dmom_what.attrs.get('quantity')
            if 'cf' in standard and name not in moments_mapping.keys():
                continue
            dsmom = mom[key]

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

            # keep original dataset name
            if standard == 'none':
                name = key

            # fix dimensions
            dims = dmom.dims
            if dims[0] == dims[1]:
                datas.update({name: (['dim_0', 'dim_1'],
                                     dmom.rename({dims[0]: 'dim_0'}))})
            else:
                datas.update({name: dmom.rename({dims[0]: 'dim_0',
                                                 dims[1]: 'dim_1'
                                                 })})

        return xr.Dataset(datas)

    def _get_polar_coordinates(self, sweep):
        """Retrieve coordinates according OdimH5 standard.

        Parameters
        ----------
        sweep : dict
            Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        coords : dict
            Dictionary of coordinate arrays
        """
        coords = collections.OrderedDict()
        try:
            coords['azimuth'] = sweep['how'].odim.azimuth_range
            coords['elevation'] = sweep['how'].odim.elevation_range
        except (KeyError, AttributeError):
            coords['azimuth'] = sweep['where'].odim.azimuth_range2
            coords['elevation'] = sweep['where'].odim.elevation_range2
        coords['range'] = sweep['where'].odim.radial_range

        return coords

    def _get_time_coordinate(self, sweep):
        """Retrieve TimeArray from source data.

        Parameters
        ----------
        sweep : dict
             Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        timevals : :class:`numpy:numpy.ndarray`
                array of time values
        """
        try:
            timevals = sweep['how'].odim.time_range
        except (KeyError, AttributeError):
            # timehandling if only start and end time is given
            start, end = sweep['what'].odim.time_range2
            if start == end:
                warnings.warn(
                    "WRADLIB: Equal ODIM `starttime` and `endtime` "
                    "values. Can't determine correct sweep start-, "
                    "end- and raytimes.", UserWarning)
                timevals = np.ones(sweep['where'].nrays) * start
            else:
                delta = (end - start) / sweep['where'].nrays
                timevals = np.arange(start + delta / 2., end, delta)
                timevals = np.roll(timevals, shift=-sweep['where'].a1gate)

        return timevals

    def _get_sweep_full_vars(self, sweep):
        """Retrieve available non-mandatory variables from source data.

        Parameters
        ----------
        sweep : dict
             Dictionary containing sweep hdf5 groups ('how', 'what', 'where')

        Returns
        -------
        full_vars : dict
            full cf-variables
        """
        full_vars = collections.OrderedDict()
        for k, v in cf_full_vars.items():
            full_vars[k] = getattr(sweep['how'].odim, k)
        return full_vars

    def _get_root_attributes(self):
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

        attrs['version'] = self._xarray['what'].attrs['version']
        attrs['institution'] = self._xarray['what'].attrs['source']
        attrs['instrument'] = self._xarray['what'].attrs['source']

        return attrs


class NetCDF4File(XRadFile):
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

    def _claim_file(self):
        filename = self.filename
        flavour = self.flavour
        nch = self._netcdf

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

        self._flavour = flavour


class XRadVol(collections.abc.MutableMapping):
    """BaseClass for Xarray backed RadarVolumes/TimeSeries

    Implements `collections.MutableMapping` dictionary.
    """

    def __init__(self, init_root=False):
        # Todo: make self._sweeps a list for better handling
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
        """Return `root` dataset.
        """
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def sweep_angles(self):
        """Return list of sweep angles
        """
        if self.root is None:
            return self._sweep_angles
        return list(self.root.sweep_fixed_angle.values)

    @sweep_angles.setter
    def sweep_angles(self, value):
        if self.root is None:
            self._sweep_angles.append(value)

    @property
    def sweep_names(self):
        """Return list of sweep names
        """
        if self.root is None:
            return self._sweep_names
        else:
            return list(self.root.sweep_group_name.values)

    @sweep_names.setter
    def sweep_names(self, value):
        if self.root is None:
            self._sweep_names.append(value)

    @property
    def sweep(self):
        """Return sweep dimension count.
        """
        return self.root.dims['sweep']

    @property
    def sweeps(self):
        """Return list of sweeps
        """
        return list(self._sweeps.values())

    @property
    def location(self):
        """Return location of data source.
        """
        return (self.root.longitude.values.item(),
                self.root.latitude.values.item(),
                self.root.altitude.values.item())

    @property
    def Conventions(self):
        """Return CF/ODIM `Conventions`.
        """
        return self.root.Conventions

    @property
    def version(self):
        """Return CF/ODIM version
        """
        return self.root.version

    def to_cfradial2(self, filename):
        """Save volume to CfRadial2.0 compliant file.

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
        """Save volume to ODIM_H5/V2_2 compliant file.

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
    """Class for Xarray backed CfRadial data files

    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize Xarray structure from Cf/Radial data structure.

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
                self._assign_data_radial2(nch, **kwargs)
            else:
                self._assign_data_radial(nch, **kwargs)

    def _assign_data_radial2(self, nch, **kwargs):
        """Assign from CfRadial2 data structure.

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

        # uses subgroups, need to get 'root' group
        self.root = self._nch[0].xarray['root']
        sweepnames = self.root.sweep_group_name.values
        for sw in sweepnames:
            ds = self._nch[0].xarray[sw]
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

    def _assign_data_radial(self, nch, **kwargs):
        """ Assign from CfRadial1 data structure.

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

        # only one group in xarray, equals root-group
        root = self._nch[0].xarray
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
    """ Class for Xarray backed ODIM_H5/Gamic HDF5 data files.
    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize Xarray structure from hdf5 data structure.

        Parameters
        ----------
        filename : iterable
            iterable of source data file names.
        flavour : str
            Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to None,
            which checks file for type.

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
            Defaults to False
        standard : str
            * `none` - data is read as verbatim as possible, no metadata
            * `odim` - data is read, odim metadata added to datasets
            * `cf-mandatory` - data is read according to cfradial2 standard
              importing mandatory metadata
            * `cf-full` - data is read according to cfradial2 standard
              importing all available cfradial2 metadata (not fully
              implemented)
            Defaults to 'odim'.
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
            Defaults to `azimuth`, need to be `azimuth` if a timeseries is
            imported.
        """
        super(OdimH5, self).__init__()
        georeference = kwargs.pop('georef', False)
        self.merge_files(filename, flavour=flavour, **kwargs)
        self.assign_root()
        if georeference:
            self.georeference()

    def merge_files(self, filename, flavour=None, **kwargs):
        """Merge given files into Xarray backed structure

       Parameters
       ----------
       obj : iterable
           iterable containing filenames to be imported
       """
        filename = np.array(filename).flatten()

        nch = None
        if flavour is not None:
            if 'ODIM' in flavour:
                fc = OdimH5File
            elif 'GAMIC' in flavour:
                fc = GamicFile
            else:
                raise AttributeError(
                    'wradlib: Unknown "flavour" kwarg attribute: {} .'
                    ''.format(flavour))
            nch = [fc(f, flavour=flavour, **kwargs)
                   for f in tqdm(filename, desc='Open  ', unit=' Files')]
        else:
            try:
                nch = [OdimH5File(f, **kwargs)
                       for f in tqdm(filename, desc='Open  ', unit=' Files')]
            except AttributeError:
                try:
                    nch = [GamicFile(f, **kwargs) for f in tqdm(filename,
                                                                desc='Open  ',
                                                                unit=' Files')]
                except AttributeError:
                    raise IOError('wradlib: Unknown file types while reading: '
                                  '{}.'.format(filename))

        # init sweep_ds with possibly existing data
        sweep_ds = list(self._sweeps.values())

        if nch:
            self._nch.extend(nch)

            # gather sweeps
            sweep_ds += [k for n in nch for k in n]

            # merge by equal time
            sweep_ds = self._merge_by_time(sweep_ds, **kwargs)

            # concat by equal angle
            sweep_ds = self._merge_by_angle(sweep_ds, **kwargs)

            # sort sweeps by time of first occurrence
            # todo: sortby time or sort by angle
            sweep_ds.sort(key=lambda x: x.time.values.min())

            # sort timeseries of single sweeps
            for i, sw in enumerate(sweep_ds):
                self._sweeps[f'sweep_{i+1}'] = sw.sortby('time')

    # todo: make this a function for outside use
    def _merge_by_time(self, obj, **kwargs):
        """Merge Datasets having same time

        Parameters
        ----------
        obj : list
            list of Xarray Datasets

        Returns
        -------
        out : list
            list of merged xarray.Datasets
        """
        out = []
        times = [ds.time.values.min() for ds in obj]
        unique_times = np.unique(times)
        if len(unique_times) == len(obj):
            out.extend([ob for ob in obj])
        else:
            for t in tqdm(unique_times, desc='Merge ', unit=' Datasets'):
                idx = np.argwhere(times == t).flatten()
                out.append(xr.combine_by_coords([obj[i] for i in idx]))
        return out

    # todo: make this a function for outside use
    def _merge_by_angle(self, obj, **kwargs):
        """Concatenate Datasets having same sweep angle

        Parameters
        ----------
        obj : list
            list of Xarray Datasets

        Keyword Arguments
        -----------------
        fix_coords : list
            list of coords strings, which should be treated/fixed.

        Returns
        -------
        out : list
            list of concatenated xarray.Datasets
        """
        fix_coords = kwargs.pop('fix_coords', [])
        include_coords = set(kwargs.pop('coords', []))
        if 'elevation' in fix_coords and 'elevation' in include_coords:
            include_coords.remove('elevation')
        if 'time' in fix_coords:
            include_coords.add('rtime')
        odim_vars = set(ODIM_NAMES)
        out = []
        angles = [ds.fixed_angle.values.item() for ds in obj]
        unique_angles = np.unique(angles)
        if len(unique_angles) == len(obj):
            out.extend([ob for ob in obj])
        else:
            try:
                for a in unique_angles:
                    idx = np.argwhere(angles == a).flatten()
                    merge_list = [obj[i].pipe(_fix_coords, fix_coords)
                                  for i in tqdm(idx, desc='Concat',
                                                unit=' Timesteps', leave=None)]
                    current_coords = set(merge_list[0].coords)
                    current_vars = set(merge_list[0].variables)
                    data_vars = odim_vars & current_vars
                    coords = include_coords & current_coords

                    # todo: make this more flexible and reliable
                    #  in terms of merging
                    out.append(xr.concat(merge_list,
                                         dim="time",
                                         data_vars=list(data_vars),
                                         coords=list(coords),
                                         compat='equals'))
            except ValueError as e:
                if e.args[0] == ('time already exists as coordinate '
                                 'or variable name.'):
                    e.args = ('wradlib: {} Please add "time" to keyword '
                              'argument "fix_coords" to fix this.'
                              ''.format(e.args[0]),)
                    raise
                else:
                    raise
        return out

    def assign_root(self):
        """(Re-)Create root object according CfRadial2 standard
        """
        # assign root variables
        sweep_group_names = list(self._sweeps.keys())
        sweep_fixed_angles = [ds.fixed_angle.values.min()
                              for ds in self._sweeps.values()]
        # extract time coverage
        try:
            tmin = [ds.time.values.min() for ds in self._sweeps.values()]
        except AttributeError:
            tmin = [ds.rtime.values.min() for ds in self._sweeps.values()]
        time_coverage_start = min(tmin)
        try:
            tmax = [ds.time.values.max() for ds in self._sweeps.values()]
        except AttributeError:
            tmax = [ds.rtime.values.max() for ds in self._sweeps.values()]
        time_coverage_end = max(tmax)

        time_coverage_start_str = str(time_coverage_start)[:19] + 'Z'
        time_coverage_end_str = str(time_coverage_end)[:19] + 'Z'

        # create root group from scratch
        root = xr.Dataset(data_vars=global_variables,
                          attrs=global_attrs)

        # take first dataset/file for retrieval of location
        nch = self._nch[0]

        # assign root variables
        root = root.assign({'volume_number': 0,
                            'platform_type': str('fixed'),
                            'instrument_type': 'radar',
                            'primary_axis': 'axis_z',
                            'time_coverage_start': time_coverage_start_str,
                            'time_coverage_end': time_coverage_end_str,
                            'latitude': nch.xarray['where'].attrs['lat'],
                            'longitude': nch.xarray['where'].attrs['lon'],
                            'altitude': nch.xarray['where'].attrs['height'],
                            'sweep_group_name': (
                                ['sweep'], sweep_group_names),
                            'sweep_fixed_angle': (
                                ['sweep'], sweep_fixed_angles),
                            })

        # assign root attributes
        attrs = nch._get_root_attributes()
        root = root.assign_attrs(attrs)
        self.root = root


def _fix_coords(ds, fix_coords):
    """Small wrapper around Dataset fix functions

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
    """
    if 'time' in fix_coords:
        ds = ds.pipe(_fix_time)
    if 'azimuth' in fix_coords:
        ds = ds.pipe(_fix_azimuth)
    if 'elevation' in fix_coords:
        ds = ds.pipe(_fix_elevation)
    return ds


def _fix_time(ds):
    """Rename time coordinate/variable and add new time dimension/coordinate

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
    """
    try:
        ds = ds.rename({'time': 'rtime'})
        ds = ds.assign_coords({'time': (['time'], [ds['rtime'].min().values])})
    except ValueError as e:
        if e.args[0] != "the new name 'rtime' conflicts":
            raise
    return ds


def _fix_azimuth(ds):
    """reindex azimuth dimension

    Fixes missing and/or double rays

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
    """
    dim = ds.azimuth.dims[0]
    res = ds.azimuth.diff(dim).median().round(decimals=1)
    azr = np.arange(res/2., 360, res)
    ds = ds.sortby('azimuth').reindex(azimuth=azr, method='nearest',
                                      tolerance=res/4.)
    return ds


def _fix_elevation(ds):
    """Recreate elevation array using a rounded value.

    Fixes glitches in the elevation data

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
    """
    dim = ds.elevation.dims[0]
    nlen = ds.elevation.shape[0]
    ele = np.ones(nlen) * ds.elevation.median(dim).round(decimals=1).values
    ds = ds.assign_coords({'elevation': ([dim], ele)})
    return ds


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


def _get_file_groups(ncid, chunks=None, **kwargs):
    """Reads netcdf (nested) hdf-groups into python dictionary with
    corresponding structure.

    """
    try:
        name = ncid.name
    except KeyError:
        name = 'root'
    out = dict()
    if ncid.__dict__.items() or ncid.variables:
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ncid),
                             chunks=chunks, **kwargs)
        if ncid.groups:
            out[name] = ds
        else:
            return ds

    # groups
    if ncid.groups:
        for k, v in ncid.groups.items():
            out[k] = _get_file_groups(v, chunks=chunks, **kwargs)

    return out


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
    h5 = h5py.File(filename, mode='r')
    ray_header = h5[f'{scan}/ray_header'][:]
    h5.close()
    vars = collections.OrderedDict()
    for name in ray_header.dtype.names:
        rh = ray_header[name]
        attrs = None
        vars.update({name: (['dim_0'], rh, attrs)})
    return vars
