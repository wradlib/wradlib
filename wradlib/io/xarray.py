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

    ncf = nc.Dataset(filename, diskless=True, persist=False)

Further the different netcdf/hdf groups are accessed via xarray open_dataset
and the NetCDF4DataStore::

    xr.open_dataset(xr.backends.NetCDF4DataStore(ncf), mask_and_scale=True)

For hdf5 data scaling/masking properties will be added to the datasets before
decoding. For GAMIC data compound data will be read via h5py.

The data structure holds one ['root'] xarray dataset which corresponds to the
CfRadial2 root-group and one or many ['sweep_X'] xarray datasets, holding the
sweep data. Since for data handling xarray is utilized all xarray features can
be exploited, like lazy-loading, pandas-like indexing on N-dimensional data
and vectorized mathematical operations across multiple dimensions.

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
   open_dataset
   to_cfradial2
   to_odim
   write_odim
   write_odim_dataspace
   get_sweep_group_name
   get_variables_moments
   get_group_moments
   get_groups
   get_moment_names

"""

import collections
import numpy as np
import datetime as dt
import netCDF4 as nc
import h5py
import xarray as xr
from osgeo import osr

from ..georef import spherical_to_xyz, spherical_to_proj

# CfRadial 2.0 - ODIM_H5 mapping
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
    'DBZ': {'standard_name': 'radar_equivalent_reflectivity_factor',
            'long_name': 'Equivalent reflectivity factor',
            'short_name': 'DBZ',
            'units': 'dBZ',
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

global_variables = dict([('volume_number', ''),
                         ('platform_type', ''),
                         ('instrument_type', ''),
                         ('primary_axis', ''),
                         ('time_coverage_start', ''),
                         ('time_coverage_end', ''),
                         ('latitude', ''),
                         ('longitude', ''),
                         ('altitude', ''),
                         ('altitude_agl', ''),
                         ('sweep_group_name', (['sweep'], [''])),
                         ('sweep_fixed_angle', (['sweep'], [''])),
                         ('frequency', ''),
                         ('status_xml', '')])


def as_xarray_dataarray(data, dims, coords):
    """Create Xarray DataArray from NumPy Array

        .. versionadded:: 1.3

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        data array
    dims : dictionary
        dictionary describing xarray dimensions
    coords : dictionary
        dictionary describing xarray coordinates

    Returns
    -------
    dataset : xr.DataArray
        DataArray
    """
    da = xr.DataArray(data, coords=dims.values(), dims=dims.keys())
    da = da.assign_coords(**coords)
    return da


def create_xarray_dataarray(data, r=None, phi=None, theta=None, proj=None,
                            site=None, sweep_mode='PPI', rf=1.0, **kwargs):
    """Create Xarray DataArray from Polar Radar Data

        .. versionadded:: 1.3

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        The data array. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : :class:`numpy:numpy.ndarray`
        The ranges. Units may be chosen arbitrarily, m preferred.
    phi : :class:`numpy:numpy.ndarray`
        The azimuth angles in degrees.
    theta : :class:`numpy:numpy.ndarray`
        The elevation angles in degrees.
    proj : osr object
        Destination Spatial Reference System (Projection).
    site : tuple
        Tuple of coordinates of the radar site.
    sweep_mode : str
        Defaults to 'PPI'.

    Returns
    -------
    dataset : xr.DataArray
        DataArray
    """
    if (r is None) or (phi is None) or (theta is None):
        raise TypeError("wradlib: function `create_xarray_dataarray` requires "
                        "r, phi and theta keyword-arguments.")

    r = r.copy()
    phi = phi.copy()
    theta = theta.copy()

    # create bins, rays 2D arrays for curvelinear coordinates
    if sweep_mode == 'PPI':
        bins, rays = np.meshgrid(r, phi, indexing='xy')
    else:
        bins, rays = np.meshgrid(r, theta, indexing='xy')

    # setup for spherical earth calculations
    re = kwargs.pop('re', None)
    ke = kwargs.pop('ke', 4. / 3.)
    if site is None:
        site = (0., 0., 0.)
        re = 6378137.

    # GDAL OSR, convert to this proj
    if isinstance(proj, osr.SpatialReference):
        xyz = spherical_to_proj(r, phi, theta, site, proj=proj, re=re, ke=ke)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
    # other proj, convert to aeqd
    elif proj:
        xyz, proj = spherical_to_xyz(r, phi, theta, site, re=re, ke=ke,
                                     squeeze=True)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
    # proj, convert to aeqd and add offset
    else:
        xyz, proj = spherical_to_xyz(r, phi, theta, site, re=re, ke=ke,
                                     squeeze=True)
        x = xyz[..., 0] + site[0]
        y = xyz[..., 1] + site[1]
        z = xyz[..., 2] + site[2]

    # calculate center point
    center = np.mean(xyz[:, 0, :], axis=0)

    # calculate ground range
    gr = np.sqrt((xyz[..., 0] - center[0]) ** 2 +
                 (xyz[..., 1] - center[1]) ** 2)

    # retrieve projection information
    cs = []
    if proj.IsProjected():
        cs.append(proj.GetAttrValue('projcs'))
    cs.append(proj.GetAttrValue('geogcs'))
    projstr = ' - '.join(cs)

    dims = collections.OrderedDict()
    dims['time'] = np.arange(phi.shape[0])
    dims['range'] = r / rf
    coords = {'azimuth': (['time'], phi),
              'elevation': (['time'], theta),
              'bins': (['time', 'range'], bins / rf),
              'rays': (['time', 'range'], rays),
              'x': (['time', 'range'], x / rf),
              'y': (['time', 'range'], y / rf),
              'z': (['time', 'range'], z / rf),
              'gr': (['time', 'range'], gr / rf),
              'longitude': (site[0]),
              'latitude': (site[1]),
              'altitude': (site[2]),
              'sweep_mode': sweep_mode,
              'projection': projstr,
              }

    # create xarray dataarray
    da = as_xarray_dataarray(data, dims=dims, coords=coords)

    return da


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
            da = xr.DataArray(times, dims=['time'], attrs=attrs)
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
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
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
            startazT = self._obj.attrs['startazT']

            attrs = {'units': 'seconds since 1970-01-01T00:00:00Z',
                     'standard_name': 'time'}
            da = xr.DataArray(startazT, dims=['time'], attrs=attrs)
            self._time_range = da
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


def write_odim(src, dest):
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


def write_odim_dataspace(src, dest):
    """ Writes Odim Dataspaces

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
        write_odim(what, h5_what)

        # moments
        val = value.values
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


def open_dataset(nch, grp=None):
    """ Open netcdf/hdf5 group as xarray dataset.

    Parameters
    ----------
    nch : handle
        netcdf4-file handle
    grp : str
        group to access

    Returns
    -------
    nch : handle
        netcdf4 group handle
    """
    if grp is not None:
        nch = nch.groups.get(grp, False)
    if nch:
        nch = xr.open_dataset(xr.backends.NetCDF4DataStore(nch),
                              mask_and_scale=True)
    return nch


def to_cfradial2(volume, filename):
    """ Save XRadVol to CfRadial2.0 compliant file.

    Parameters
    ----------
    volume : XRadVol object
    filename : str
        output filename
    """
    root = volume.root.copy(deep=True)
    root.attrs['Conventions'] = 'Cf/Radial'
    root.attrs['version'] = '2.0'
    root.to_netcdf(filename, mode='w', group='/')
    for key in root.sweep_group_name.values:
        volume[key].to_netcdf(filename, mode='a', group=key)


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
    write_odim({'Conventions': 'ODIM_H5/V2_2'}, h5)

    # how group
    # first try to use original data
    try:
        how = volume['odim']['how'].attrs
    except KeyError:
        how = {}
    else:
        how.update({'_modification_program': 'wradlib'})

    h5_how = h5.create_group('how')
    write_odim(how, h5_how)

    sweepnames = root.sweep_group_name.values

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
    try:
        what = volume['odim']['what'].attrs
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
        ds = volume['sweep_{}'.format(idx + 1)]
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
        rstart = (ds.range.values[0] - rscale / 2.) / 1000.
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
            ds_how = volume['odim']['dsets'][ds_list[idx]]['how'].attrs
        except KeyError:
            ds_how = {'scan_index': ds.sweep_number + 1,
                      'scan_count': len(sweepnames),
                      }
        write_odim(ds_how, h5_ds_how)

        # write moments
        write_odim_dataspace(ds, h5_dataset)

    h5.close()


def get_sweep_group_name(ncf, name):
    """ Return sweep names.

    Returns source names and cfradial names.

    Parameters
    ----------
    ncf : handle
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
    src = [key for key in ncf.groups.keys() if name in key]
    src.sort(key=lambda x: int(x[len(name):]))
    swp_grp_name = ['sweep_{}'.format(i) for i in
                    range(1, len(src) + 1)]
    return src, swp_grp_name


def get_variables_moments(ds, moments=None):
    """ Retrieve radar moments from dataset variables.

    Parameters
    ----------
    ds : xarray dataset
        source dataset
    moments : list
        list of moment strings

    Returns
    -------
    ds : xarray dataset
        altered dataset
    """
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


def get_group_moments(ncf, sweep, moments=None):
    """ Retrieve radar moments from hdf groups.

    Parameters
    ----------
    ncf : netCDF Dataset handle
    sweep : str
        sweep key
    moments : list
        list of moment strings

    Returns
    -------
    ds : dictionary
        moment datasets
    """
    datas = {}
    for mom in moments:
        dmom_what = open_dataset(ncf[sweep][mom], 'what')
        name = dmom_what.attrs.pop('quantity')
        if name not in moments_mapping.keys():
            continue
        dsmom = open_dataset(ncf[sweep], mom)
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


def get_groups(ncf, groups):
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
    return tuple(map(lambda x: open_dataset(ncf, x), groups))


def get_moment_names(sweep, fmt=None, src=None):
    moments = [mom for mom in getattr(sweep, src).keys() if
               fmt in mom]
    moments_idx = np.argsort([int(s[len(fmt):]) for s in moments])
    return np.array(moments)[moments_idx]


class XRadVol(collections.MutableMapping):
    """ BaseClass for xarray based RadarVolumes

    Implements `collections.MutableMapping` dictionary.
    """

    def __init__(self):
        self._source = dict()
        self._filename = None
        self._ncf = None
        self._disk_format = None
        self._file_format = None
        self._data_model = None
        self._init_root()

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
        for k in list(self._source):
            del self._source[k]
        self._ncf.close()

    def _init_root(self):
        self['root'] = xr.Dataset(data_vars=global_variables,
                                  attrs=global_attrs)

    @property
    def root(self):
        """ Return `root` dataset.
        """
        return self['root']

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
        """
        to_cfradial2(self, filename)

    def to_odim(self, filename):
        """ Save volume to ODIM_H5/V2_2 compliant file.
        """
        to_odim(self, filename)


class CfRadial(XRadVol):
    """ Class for xarray based retrieval of CfRadial data files

    """
    def __init__(self, filename=None, flavour=None, **kwargs):
        super(CfRadial, self).__init__()
        self._filename = filename
        self._ncf = nc.Dataset(filename, diskless=True, persist=False)
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
        self['root'] = open_dataset(self._ncf)
        sweepnames = self['root'].sweep_group_name.values
        for sw in sweepnames:
            self[sw] = open_dataset(self._ncf, sw)
            self[sw] = self[sw].assign_coords(longitude=self['root'].longitude)
            self[sw] = self[sw].assign_coords(latitude=self['root'].latitude)
            self[sw] = self[sw].assign_coords(altitude=self['root'].altitude)
            self[sw] = self[sw].assign_coords(azimuth=self[sw].azimuth)
            self[sw] = self[sw].assign_coords(elevation=self[sw].elevation)
            self[sw] = self[sw].assign_coords(sweep_mode=self[sw].sweep_mode)

            # adding xyz aeqd-coordinates
            ds = self[sw]
            site = (ds.longitude.values, ds.latitude.values,
                    ds.altitude.values)
            xyz, aeqd = spherical_to_xyz(ds.range,
                                         ds.azimuth,
                                         ds.elevation,
                                         site,
                                         squeeze=True)
            gr = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
            ds = ds.assign_coords(x=(['time', 'range'], xyz[..., 0]))
            ds = ds.assign_coords(y=(['time', 'range'], xyz[..., 1]))
            ds = ds.assign_coords(z=(['time', 'range'], xyz[..., 2]))
            ds = ds.assign_coords(gr=(['time', 'range'], gr))

            # what products?
            is_ppi = True
            if self[sw].sweep_mode != 'azimuth_surveillance':
                is_ppi = False

            # adding rays, bins coordinates
            if is_ppi:
                bins, rays = np.meshgrid(ds.range, ds.azimuth, indexing='xy')
            else:
                bins, rays = np.meshgrid(ds.range, ds.elevation, indexing='xy')
            ds = ds.assign_coords(rays=(['time', 'range'], rays))
            ds = ds.assign_coords(bins=(['time', 'range'], bins))
            self[sw] = ds

    def assign_data_radial(self):
        """ Assign from CfRadial1 data structure.

        """
        root = open_dataset(self._ncf)
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
            self[sw] = self[sw].assign_coords(longitude=self['root'].longitude)
            self[sw] = self[sw].assign_coords(latitude=self['root'].latitude)
            self[sw] = self[sw].assign_coords(altitude=self['root'].altitude)
            self[sw] = self[sw].assign_coords(azimuth=self[sw].azimuth)
            self[sw] = self[sw].assign_coords(elevation=self[sw].elevation)
            sweep_mode = self[sw].sweep_mode.values.item().decode()
            self[sw] = self[sw].assign_coords(sweep_mode=sweep_mode)
            # adding xyz aeqd-coordinates
            ds = self[sw]
            site = (ds.longitude.values, ds.latitude.values,
                    ds.altitude.values)
            xyz, aeqd = spherical_to_xyz(ds.range,
                                         ds.azimuth,
                                         ds.elevation,
                                         site,
                                         squeeze=True)
            gr = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
            ds = ds.assign_coords(x=(['time', 'range'], xyz[..., 0]))
            ds = ds.assign_coords(y=(['time', 'range'], xyz[..., 1]))
            ds = ds.assign_coords(z=(['time', 'range'], xyz[..., 2]))
            ds = ds.assign_coords(gr=(['time', 'range'], gr))

            # what products?
            is_ppi = True
            if sweep_mode != 'azimuth_surveillance':
                is_ppi = False

            # adding rays, bins coordinates
            if is_ppi:
                bins, rays = np.meshgrid(ds.range, ds.azimuth, indexing='xy')
            else:
                bins, rays = np.meshgrid(ds.range, ds.elevation, indexing='xy')
            ds = ds.assign_coords(rays=(['time', 'range'], rays))
            ds = ds.assign_coords(bins=(['time', 'range'], bins))
            self[sw] = ds


class OdimH5(XRadVol):
    """ Class for xarray based retrieval of ODIM_H5 data files
    """
    def __init__(self, filename=None, flavour=None, strict=True, **kwargs):
        super(OdimH5, self).__init__()
        self._filename = filename
        self._ncf = nc.Dataset(filename, diskless=True, persist=False)
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

    def assign_data_odim(self, strict=True):
        """ Assign from ODIM_H5 data structure.

        """

        # retrieve and assign global groups root and /how, /what, /where
        groups = [None, 'how', 'what', 'where']
        root, how, what, where = get_groups(self._ncf, groups)

        # sweep group handling
        src_swp_grp_name, swp_grp_name = get_sweep_group_name(self._ncf,
                                                              'dataset')

        sweep_fixed_angle = []
        time_coverage_start = np.datetime64('2037-01-01')
        time_coverage_end = np.datetime64('1970-01-01')
        # iterate sweeps
        sweeps = {}
        for i, sweep in enumerate(src_swp_grp_name):
            swp = {}

            # retrieve ds and assign datasetX how/what/where group attributes
            groups = [None, 'how', 'what', 'where']
            ds, ds_how, ds_what, ds_where = get_groups(self._ncf[sweep],
                                                       groups)

            # what products?
            is_ppi = True
            sweep_mode = 'azimuth_surveillance'
            if ds_what.attrs['product'] == 'RHI':
                sweep_mode = 'rhi'
                is_ppi = False

            # moments
            moments = get_moment_names(self._ncf[sweep], fmt='data',
                                       src='groups')
            for name, dmom in get_group_moments(self._ncf, sweep,
                                                moments=moments).items():
                ds[name] = dmom

            # coordinates wrap-up
            ds = ds.assign_coords(longitude=where.attrs['lon'],
                                  latitude=where.attrs['lat'],
                                  altitude=where.attrs['height'],
                                  sweep_mode=sweep_mode)

            ds = ds.assign_coords(azimuth=ds_where.odim.azimuth_range)
            ds = ds.assign_coords(elevation=ds_where.odim.elevation_range)
            ds = ds.assign({'range': ds_where.odim.radial_range})

            # adding xyz aeqd-coordinates
            site = (ds.longitude.values, ds.latitude.values,
                    ds.altitude.values)
            xyz, aeqd = spherical_to_xyz(ds.range,
                                         ds.azimuth,
                                         ds.elevation,
                                         site,
                                         squeeze=True)
            gr = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
            ds = ds.assign_coords(x=(['time', 'range'], xyz[..., 0]))
            ds = ds.assign_coords(y=(['time', 'range'], xyz[..., 1]))
            ds = ds.assign_coords(z=(['time', 'range'], xyz[..., 2]))
            ds = ds.assign_coords(gr=(['time', 'range'], gr))

            # adding rays, bins coordinates
            if is_ppi:
                bins, rays = np.meshgrid(ds.range, ds.azimuth, indexing='xy')
            else:
                bins, rays = np.meshgrid(ds.range, ds.elevation, indexing='xy')
            ds = ds.assign_coords(rays=(['time', 'range'], rays))
            ds = ds.assign_coords(bins=(['time', 'range'], bins))

            # time coordinate
            try:
                timevals = ds_how.odim.\
                    time_range.values
            except (KeyError, AttributeError):
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
            if is_ppi:
                fixed_angle = ds_where.elangle
            else:
                fixed_angle = ds_where.azangle
            ds = ds.assign({'sweep_number': i,
                            'sweep_mode': sweep_mode,
                            'follow_mode': 'none',
                            'prt_mode': 'fixed',
                            'fixed_angle': fixed_angle,
                            })
            sweep_fixed_angle.append(fixed_angle)

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

        # assign root variables
        time_coverage_start = str(time_coverage_start)[:19] + 'Z'
        time_coverage_end = str(time_coverage_end)[:19] + 'Z'

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
        if not strict:
            self['odim'] = {'how': how,
                            'what': what,
                            'where': where,
                            'dsets': sweeps}

    def assign_data_gamic(self, strict=True):
        """ Assign from GAMIC hdf5 data structure.

        """
        # retrieve and assign global groups root and /how, /what, /where
        groups = [None, 'how', 'what', 'where']
        root, how, what, where = get_groups(self._ncf, groups)

        # sweep group handling
        src_swp_grp_name, swp_grp_name = get_sweep_group_name(self._ncf,
                                                              'scan')

        sweep_fixed_angle = []
        time_coverage_start = np.datetime64('2037-01-01')
        time_coverage_end = np.datetime64('1970-01-01')

        # iterate sweeps
        sweeps = {}
        for i, sweep in enumerate(src_swp_grp_name):
            swp = {}

            # retrieve ds and assign datasetX how/what/where group attributes
            groups = [None, 'how', 'what', 'where']
            ds, ds_how, ds_what, ds_where = get_groups(self._ncf[sweep],
                                                       groups)

            # what products?
            is_ppi = True
            sweep_mode = 'azimuth_surveillance'
            if ds_what.attrs['scan_type'] == 'RHI':
                sweep_mode = 'rhi'
                is_ppi = False

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
            ds = ds.assign_coords(longitude=where.attrs['lon'],
                                  latitude=where.attrs['lat'],
                                  altitude=where.attrs['height'],
                                  sweep_mode=sweep_mode)

            ds = ds.assign_coords(azimuth=ds_what.gamic.azimuth_range)
            ds = ds.assign_coords(elevation=ds_what.gamic.elevation_range)
            ds = ds.assign({'range': ds_how.gamic.radial_range})
            ds = ds.assign({'time': (['time'], ds_what.gamic.time_range.values,
                                     time_attrs)})

            # adding xyz aeqd-coordinates
            site = (ds.longitude.values, ds.latitude.values,
                    ds.altitude.values)
            xyz, aeqd = spherical_to_xyz(ds.range,
                                         ds.azimuth,
                                         ds.elevation,
                                         site,
                                         squeeze=True)
            gr = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
            ds = ds.assign_coords(x=(['time', 'range'], xyz[..., 0]))
            ds = ds.assign_coords(y=(['time', 'range'], xyz[..., 1]))
            ds = ds.assign_coords(z=(['time', 'range'], xyz[..., 2]))
            ds = ds.assign_coords(gr=(['time', 'range'], gr))

            # adding rays, bins coordinates
            if is_ppi:
                bins, rays = np.meshgrid(ds.range, ds.azimuth, indexing='xy')
            else:
                bins, rays = np.meshgrid(ds.range, ds.elevation, indexing='xy')
            ds = ds.assign_coords(rays=(['time', 'range'], rays))
            ds = ds.assign_coords(bins=(['time', 'range'], bins))

            # get moments
            moments = get_moment_names(self._ncf[sweep], fmt='moment_',
                                       src='variables')
            ds = get_variables_moments(ds, moments=moments)

            # assign global sweep attributes
            if is_ppi:
                fixed_angle = ds_how.attrs['elevation']
            else:
                fixed_angle = ds_how.attrs['azimuth']
            ds = ds.assign({'sweep_number': i,
                            'sweep_mode': sweep_mode,
                            'follow_mode': 'none',
                            'prt_mode': 'fixed',
                            'fixed_angle': fixed_angle,
                            })
            sweep_fixed_angle.append(fixed_angle)

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

        # assign root variables
        time_coverage_start = str(time_coverage_start)[:19] + 'Z'
        time_coverage_end = str(time_coverage_end)[:19] + 'Z'

        # assign root variables
        root = root.assign({'volume_number': 0,
                            'platform_type': str('fixed'),
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
        if not strict:
            self['odim'] = {'how': how,
                            'what': what,
                            'where': where,
                            'dsets': sweeps}
