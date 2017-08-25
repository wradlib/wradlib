#!/usr/bin/env python
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Read IRIS Data Format
^^^^^^^^^^^^^^^^^^^^^

Reads data from Vaisala's IRIS data formats

IRIS (Vaisala Sigmet Interactive Radar Information System)

See M211318EN-F Programming Guide ftp://ftp.sigmet.com/outgoing/manuals/

.. autosummary::
   :nosignatures:
   :toctree: generated/

   IrisRecord
   IrisFile
   read_iris
"""

import numpy as np
import struct
from collections import OrderedDict
import warnings
import datetime as dt


RECORD_BYTES = 6144


class IrisRecord(object):
    """ Class holding a single record from a Sigmet IRIS file.
    """
    def __init__(self, record, recnum):
        self.record = record.copy()
        self._pos = 0
        self.recnum = recnum

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def recpos(self):
        return int(self._pos / 2)

    @recpos.setter
    def recpos(self, value):
        self._pos = value * 2

    def read(self, words, width=2):
        ret = self.record[self._pos:self._pos + words * width]
        self.pos += words * width
        return ret


class IrisFile(object):
    """ Class for retrieving data from Sigmet IRIS files.

    """
    def __init__(self, filename, loaddata=True, rawdata=False, debug=False):

        self._debug = debug
        self._rawdata = rawdata
        self._fh = np.memmap(filename)
        self._record_number = 0
        self._rh = IrisRecord(self._fh[0:RECORD_BYTES], 0)

        # read data headers
        self._product_hdr = _unpack_dictionary(self.read_record(0)
                                               [:LEN_PRODUCT_HDR],
                                               PRODUCT_HDR,
                                               rawdata)
        self._ingest_header = _unpack_dictionary(self.read_record(1)
                                                 [:LEN_INGEST_HEADER],
                                                 INGEST_HEADER,
                                                 rawdata)
        self.get_task_type_scan_info()
        self._raw_product_bhdrs = []

        # determine data types contained in the file
        self._data_types_numbers = self.get_data_types()

        self._sweeps = OrderedDict()
        if loaddata:
            self.get_sweeps()

    @property
    def fh(self):
        return self._fh

    @property
    def rh(self):
        return self._rh

    @property
    def filepos(self):
        return self._record_number * RECORD_BYTES + int(self._rh.recpos)

    @property
    def product_hdr(self):
        return self._product_hdr

    @property
    def ingest_header(self):
        return self._ingest_header

    @property
    def raw_product_bhdrs(self):
        return self._raw_product_bhdrs

    @property
    def sweeps(self):
        return self._sweeps

    @property
    def nsweeps(self):
        head = self._ingest_header['task_configuration']['task_scan_info']
        return head['sweep_number']

    @property
    def nbins(self):
        return self._product_hdr['product_end']['number_bins']

    @property
    def nrays(self):
        return self._ingest_header['ingest_configuration']['number_rays_sweep']

    @property
    def data_types(self):
        return [SIGMET_DATA_TYPES[i] for i in self._data_types_numbers]

    @property
    def data_types_count(self):
        return len(self._data_types_numbers)

    @property
    def data_types_names(self):
        return [SIGMET_DATA_TYPES[i]['name'] for i in self._data_types_numbers]

    def _check_record(self):
        chk = self._rh.record.shape[0] != RECORD_BYTES
        if chk:
            warnings.warn("Unexpected file end detected at record {0}"
                          "".format(self._record_number),
                          RuntimeWarning,
                          stacklevel=3)
        return chk

    def next_record(self):
        self._record_number += 1
        start = self._record_number * RECORD_BYTES
        stop = start + RECORD_BYTES
        self._rh = IrisRecord(self.fh[start:stop], self._record_number)
        return self._check_record()

    def read_record(self, recnum):
        start = recnum * RECORD_BYTES
        stop = start + RECORD_BYTES
        self._rh = IrisRecord(self.fh[start:stop], recnum)
        return self._rh.record

    def read(self, words=1, dtype='int16'):
        data = self._rh.read(words).view(dtype=dtype)
        words -= len(data)
        if words > 0:
            self.next_record()
            self.get_raw_prod_bhdr()
            data = np.append(data, self._rh.read(words).view(dtype=dtype))
        return data

    def get_compression_code(self):
        cmp_val = self.read(dtype='int16')[0]
        cmp_msb = np.sign(cmp_val) == -1
        if cmp_msb:
            cmp_val = cmp_val + 32768
        if self._debug:
            print("--- Sub CMP Code:", cmp_msb, cmp_val, self._rh.recpos - 1,
                  self._rh.recpos)
        return cmp_msb, cmp_val

    def get_data_types(self):
        """ Determine the available data types.
        """
        # determine the available fields
        task_config = self._ingest_header['task_configuration']
        task_dsp_info = task_config['task_dsp_info']
        word0 = task_dsp_info['dsp_data_mask0']['mask_word_0']
        word1 = task_dsp_info['dsp_data_mask0']['mask_word_1']
        word2 = task_dsp_info['dsp_data_mask0']['mask_word_2']
        word3 = task_dsp_info['dsp_data_mask0']['mask_word_3']

        return _data_types_from_dsp_mask([word0, word1, word2, word3])

    def get_task_type_scan_info(self):
        task_info = self._ingest_header['task_configuration']['task_scan_info']
        mode = task_info['antenna_scan_mode']
        key = 'task_type_scan_info'
        if mode in [1, 4]:
            task_info[key] = _unpack_dictionary(task_info[key],
                                                TASK_PPI_SCAN_INFO,
                                                self._rawdata)
        elif mode == 2:
            task_info[key] = _unpack_dictionary(task_info[key],
                                                TASK_RHI_SCAN_INFO,
                                                self._rawdata)
        elif mode == 3:
            task_info[key] = _unpack_dictionary(task_info[key],
                                                TASK_MANUAL_SCAN_INFO,
                                                self._rawdata)
        elif mode == 5:
            task_info[key] = _unpack_dictionary(task_info[key],
                                                TASK_FILE_SCAN_INFO,
                                                self._rawdata)
        else:
            pass

    def get_raw_prod_bhdr(self):
        self._raw_product_bhdrs.append(
            _unpack_dictionary(self._rh.read(LEN_RAW_PROD_BHDR, width=1),
                               RAW_PROD_BHDR))

    def get_ingest_data_headers(self):

        ingest_data_hdrs = OrderedDict()
        for i, dn in enumerate(self.data_types_names):
            ingest_data_hdrs[dn] = _unpack_dictionary(
                self._rh.read(LEN_INGEST_DATA_HEADER, width=1),
                INGEST_DATA_HEADER)

        return ingest_data_hdrs

    def get_ray(self, data):
        ray_pos = 0

        cmp_msb, cmp_val = self.get_compression_code()

        # ray is missing
        if (cmp_val == 1) & (cmp_msb == 0):
            if self._debug:
                print("ray missing")
            return None

        while not ((cmp_val == 1) & (not cmp_msb)):

            # data words follow
            if cmp_msb:
                if self._debug:
                    print(
                        "--- Add {0} WORDS at range {1}, record {2}:{3}:"
                        "".format(cmp_val, ray_pos, self._rh.recpos,
                                  self._rh.recpos + cmp_val))
                data[ray_pos:ray_pos + cmp_val] = self.read(cmp_val,
                                                            dtype='int16')
            # compressed zeros follow
            # can be skipped, if data array is created all zeros
            else:
                if self._debug:
                    print(
                        "--- Add {0} Zeros at range {1}, record {2}:{3}:"
                        "".format(cmp_val, ray_pos,
                                  self._rh.recpos, self._rh.recpos + 1))
                if cmp_val + ray_pos > self.nbins + 6:
                    return data
                data[ray_pos:ray_pos + cmp_val] = 0

            ray_pos += cmp_val

            # read next compression code
            cmp_msb, cmp_val = self.get_compression_code()

        return data

    def get_sweep(self):

        sweep = OrderedDict()

        self.get_raw_prod_bhdr()
        sweep['ingest_data_hdrs'] = self.get_ingest_data_headers()

        rays_per_data_type = [d['number_rays_file_expected'] for d
                              in sweep['ingest_data_hdrs'].values()]

        rays = sum(rays_per_data_type)
        bins = self._product_hdr['product_end']['number_bins']

        raw_data = np.zeros((rays, bins + 6), dtype='int16')

        for ray_i in range(rays):
            if self._debug:
                print("{0}: Ray started at {1}, file offset: {2}"
                      "".format(ray_i, int(self._rh.recpos) - 1,
                                self.filepos))
            ret = self.get_ray(raw_data[ray_i])
            if ret is not None:
                raw_data[ray_i] = ret

        sweep_data = OrderedDict()
        cnt = self.data_types_count
        for i, prod in enumerate(self.data_types):
            sweep_prod = OrderedDict()

            sweep_prod['data'] = self.decode_data(raw_data[i::cnt, 6:], prod)
            sweep_prod['azi_start'] = self.decode_data(raw_data[i::cnt, 0],
                                                       BIN2)
            sweep_prod['ele_start'] = self.decode_data(raw_data[i::cnt, 1],
                                                       BIN2)
            sweep_prod['azi_stop'] = self.decode_data(raw_data[i::cnt, 2],
                                                      BIN2)
            sweep_prod['ele_stop'] = self.decode_data(raw_data[i::cnt, 3],
                                                      BIN2)
            sweep_prod['rbins'] = raw_data[i::cnt, 4]
            sweep_prod['dtime'] = raw_data[i::cnt, 5]
            sweep_data[prod['name']] = sweep_prod

        sweep['sweep_data'] = sweep_data

        return sweep

    def decode_data(self, data, prod):
        if self._rawdata:
            return data
        kw = {}
        if prod['func']:
            try:
                kw.update(prod['kw'])
            except KeyError:
                pass
            try:
                rays, bins = data.shape
                data = data.view(prod['dtype']).reshape(rays, -1)[:, :bins]
            except ValueError:
                data = data.view(prod['dtype'])
            return prod['func'](data, **kw)
        else:
            return data

    def get_sweeps(self):
        self._record_number = 1
        for i in range(self.nsweeps):
            if self.next_record():
                break
            self._sweeps[i] = self.get_sweep()


def read_iris(filename, loaddata=True, rawdata=True, debug=False):
    fh = IrisFile(filename, loaddata=loaddata, rawdata=rawdata, debug=debug)
    data = OrderedDict()
    data['product_hdr'] = fh.product_hdr
    data['ingest_header'] = fh.ingest_header
    data['nsweeps'] = fh.nsweeps
    data['nrays'] = fh.nrays
    data['nbins'] = fh.nbins
    data['data_types'] = fh.data_types_names
    data['sweeps'] = fh.sweeps
    data['raw_product_bhdrs'] = fh.raw_product_bhdrs

    return data


def decode_bin_angle(bin_angle, mode=None):
    """ Decode BIN angle
    """
    return 360. * bin_angle / 2 ** (mode * 8)


def decode_array(data, scale=1., offset=0, offset2=0):
    return (data + offset) / scale + offset2


def decode_rainrate2(data):
    print("not decoding data")
    return data


def decode_kdp(data):
    zero = data[data == -128]
    data = -0.25 * np.sign(data) * 600 ** ((127 - np.abs(data)) / 126.)
    data[zero] = 0
    return data


def decode_phidp(data, **kwargs):
    return 180. * decode_array(data, **kwargs)


def decode_phidp2(data, **kwargs):
    return 360. * decode_array(data, **kwargs)


def decode_sqi(data, **kwargs):
    return np.sqrt(decode_array(data, **kwargs))


def decode_time(data):
    time = _unpack_dictionary(data, YMDS_TIME)
    return (dt.datetime(time['year'], time['month'], time['day']) +
            dt.timedelta(seconds=time['seconds'],
                         milliseconds=time['milliseconds']))


# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters

SINT1 = {'fmt': 'b'}
SINT2 = {'fmt': 'h'}
SINT4 = {'fmt': 'i'}
UINT1 = {'fmt': 'B'}
UINT2 = {'fmt': 'H'}
UINT4 = {'fmt': 'I'}
FLT4 = {'fmt': 'f'}
FLT8 = {'fmt': 'd'}
BIN1 = {'name': 'BIN1', 'dtype': 'uint8', 'size': 'B',
        'func': decode_bin_angle, 'fkw': {'mode': 1}}
BIN2 = {'name': 'BIN2', 'dtype': 'uint16', 'size': 'H',
        'func': decode_bin_angle, 'fkw': {'mode': 2}}
BIN4 = {'name': 'BIN4', 'dtype': 'uint32', 'size': 'I',
        'func': decode_bin_angle, 'fkw': {'mode': 4}}
MESSAGE = {'fmt': 'I'}
UINT16_T = {'fmt': 'H'}


def _get_fmt_string(dictionary, retsub=False):
    fmt = ''
    sub = OrderedDict()
    for k, v in dictionary.items():
        try:
            fmt += v['fmt']
        except KeyError:
            # remember sub-structures
            sub[k] = v
            try:
                fmt += '{}s'.format(struct.calcsize(_get_fmt_string(v)))
            except TypeError:
                fmt += v['size']
    if retsub:
        return fmt, sub
    else:
        return fmt


def _unpack_dictionary(buffer, dictionary, rawdata=False):
    """ Unpacks binary data using the given dictionary structure.

    Parameters
    ----------
    buffer : array-like
    dictionary : data structure in dictionary, keys are names and values are
        structure formats

    Returns
    -------
    Ordered Dictionary with unpacked data

    """
    # get format and substructures of dictionary
    fmt, sub = _get_fmt_string(dictionary, retsub=True)

    # unpack into OrderedDict
    data = OrderedDict(zip(dictionary, struct.unpack(fmt, buffer)))

    # remove spares
    if not rawdata:
        keys_to_remove = [k for k in data.keys() if k.startswith('spare')]
        for k in keys_to_remove:
            data.pop(k, None)

    # iterate over sub dictionary and unpack/read/decode
    for k, v in sub.items():
        if not rawdata:
            # read/decode data
            for k1 in ['read', 'func']:
                try:
                    data[k] = v[k1](data[k], **v[k1[0] + 'kw'])
                except KeyError:
                    pass
        # unpack sub dictionary
        try:
            data[k] = _unpack_dictionary(data[k], v, rawdata=rawdata)
        except TypeError:
            pass

    return data


def _data_types_from_dsp_mask(words):
    """
    Return a list of the data types from the words in the data_type mask.
    """
    data_types = []
    for i, word in enumerate(words):
        data_types += [j+(i*32) for j in range(32) if word >> j & 1]
    return data_types


# IRIS Data Structures

# structure_header Structure
# 4.3.48 page 55

STRUCTURE_HEADER = OrderedDict([('structure_identifier', SINT2),
                                ('format_version', SINT2),
                                ('bytes_in_structure', SINT4),
                                ('reserved', SINT2),
                                ('flag', SINT2)])

# ymds_time Structure
# 4.3.77, page 70

YMDS_TIME = OrderedDict([('seconds', SINT4),
                         ('milliseconds', UINT2),
                         ('year', SINT2),
                         ('month', SINT2),
                         ('day', SINT2)])

LEN_YMDS_TIME = struct.calcsize(_get_fmt_string(YMDS_TIME))

_YMDS_TIME = {'size': '{}s'.format(LEN_YMDS_TIME),
              'func': decode_time,
              'fkw': {}}

# color_scale_def Structure
# 4.3.5, page 26

COLOR_SCALE_DEF = OrderedDict([('iflags', UINT4),
                               ('istart', SINT4),
                               ('istep', SINT4),
                               ('icolcnt', SINT2),
                               ('iset_and_scale', UINT2),
                               ('ilevel_seams', {'size': '32s',
                                                 'read': np.fromstring,
                                                 'rkw': {'dtype': 'uint16'}})])

# product_configuration Structure
# 4.3.24, page 36

PRODUCT_CONFIGURATION = OrderedDict([('structure_header', STRUCTURE_HEADER),
                                     ('product_type_code', UINT2),
                                     ('scheduling_code', UINT2),
                                     ('seconds_between_runs', SINT4),
                                     ('generation_time', _YMDS_TIME),
                                     ('sweep_ingest_time', _YMDS_TIME),
                                     ('file_ingest_time', _YMDS_TIME),
                                     ('spare_0', {'fmt': '6s'}),
                                     ('product_name', {'fmt': '12s'}),
                                     ('task_name', {'fmt': '12s'}),
                                     ('flag', UINT2),
                                     ('x_scale', SINT4),
                                     ('y_scale', SINT4),
                                     ('z_scale', SINT4),
                                     ('x_size', SINT4),
                                     ('y_size', SINT4),
                                     ('z_size', SINT4),
                                     ('x_location', SINT4),
                                     ('y_location', SINT4),
                                     ('z_location', SINT4),
                                     ('maximum_range', SINT4),
                                     ('spare_1', {'fmt': '2s'}),
                                     ('data_type', UINT2),
                                     ('projection_name', {'fmt': '12s'}),
                                     ('input_data_type', UINT2),
                                     ('projection_type', UINT1),
                                     ('spare_2', {'fmt': '1s'}),
                                     ('radial_smoother', SINT2),
                                     ('times_run', SINT2),
                                     ('zr_constant', SINT4),
                                     ('zr_exponent', SINT4),
                                     ('x_smoother', SINT2),
                                     ('y_smoother', SINT2),
                                     ('product_specific_info', {'fmt': '80s'}),
                                     ('minor_task_suffixes', {'fmt': '16s'}),
                                     ('spare_3', {'fmt': '12s'}),
                                     ('color_scale_def', COLOR_SCALE_DEF)])
# product_end Structure
# 4.3.25, page 39

PRODUCT_END = OrderedDict([('site_name', {'fmt': '16s'}),
                           ('iris_version_created', {'fmt': '8s'}),
                           ('ingest_iris_version', {'fmt': '8s'}),
                           ('ingest_time', _YMDS_TIME),
                           ('spare_0', {'fmt': '28s'}),
                           ('GMT_minute_offset_local', SINT2),
                           ('ingest_hardware_name_', {'fmt': '16s'}),
                           ('ingest_site_name_', {'fmt': '16s'}),
                           ('GMT_minute_offset_standard', SINT2),
                           ('latitude', BIN4),
                           ('longitude', BIN4),
                           ('ground_height', SINT2),
                           ('radar_height', SINT2),
                           ('prf', SINT4),
                           ('pulse_width', SINT4),
                           ('signal_processor_type', UINT2),
                           ('trigger_rate', UINT2),
                           ('samples_used', SINT2),
                           ('clutter_filter', {'fmt': '12s'}),
                           ('number_linear_filter', UINT2),
                           ('wavelength', SINT4),
                           ('truncation_height', SINT4),
                           ('first_bin_range', SINT4),
                           ('last_bin_range', SINT4),
                           ('number_bins', SINT4),
                           ('flag', UINT2),
                           ('number_ingest', SINT2),
                           ('polarization', UINT2),
                           ('horizontal_calibration_i0', SINT2),
                           ('horizontal_calibration_noise', SINT2),
                           ('horizontal_radar_constant', SINT2),
                           ('receiver_bandwidth', UINT2),
                           ('horizontal_current_noise', SINT2),
                           ('vertical_current_noise', SINT2),
                           ('ldr_offset', SINT2),
                           ('zdr_offset', SINT2),
                           ('tcf_cal_flags_1', UINT16_T),
                           ('tcf_cal_flags_2', UINT16_T),
                           ('spare_bit1', UINT1),
                           ('spare_bit2', UINT1),
                           ('spare_bit3', UINT1),
                           ('spare_bit4', UINT1),
                           ('spare_bit5', UINT1),
                           ('spare_bit6', UINT1),
                           ('spare_1', {'fmt': '12s'}),
                           ('standard_parallel_1', BIN4),
                           ('standard_parallel_2', BIN4),
                           ('earth_radius', UINT4),
                           ('inverse_flatting', UINT4),
                           ('fault_status', UINT4),
                           ('input_mask', UINT4),
                           ('number_log_filter', UINT2),
                           ('cluttermap', UINT2),
                           ('latitude_projection', BIN4),
                           ('longitude_projection', BIN4),
                           ('product_sequence_number', SINT2),
                           ('spare_2', {'fmt': '32s'}),
                           ('melting_level', SINT2),
                           ('radar_height_above_reference', SINT2),
                           ('number_elements', SINT2),
                           ('mean_wind_speed', UINT1),
                           ('mean_wind_direction', BIN1),
                           ('spare_3', {'fmt': '2s'}),
                           ('tz_name', {'fmt': '8s'}),
                           ('extended_product_header_offset', UINT4),
                           ('spare_4', {'fmt': '4s'})])

# _product_hdr Structure
# 4.3.26 page 41
PRODUCT_HDR = OrderedDict([('structure_header', STRUCTURE_HEADER),
                           ('product_configuration', PRODUCT_CONFIGURATION),
                           ('product_end', PRODUCT_END)])

# ingest_configuration Structure
# 4.3.14, page 31

INGEST_CONFIGURATION = OrderedDict([('filename', {'fmt': '80s'}),
                                    ('number_files', SINT2),
                                    ('number_sweeps_completed', SINT2),
                                    ('total_size', SINT4),
                                    ('volume_scan_start_time', _YMDS_TIME),
                                    ('spare_0', {'fmt': '12s'}),
                                    ('ray_header_bytes', SINT2),
                                    ('extended_ray_header_bytes', SINT2),
                                    ('number_task_config_table', SINT2),
                                    ('playback_version', SINT2),
                                    ('spare_1', {'fmt': '4s'}),
                                    ('iris_version', {'fmt': '8s'}),
                                    ('hardware_site', {'fmt': '16s'}),
                                    ('gmt_offset_minutes_local', SINT2),
                                    ('site_name', {'fmt': '16s'}),
                                    ('gmt_offset_minutes_standard', SINT2),
                                    ('latitude_radar', BIN4),
                                    ('longitude_radar', BIN4),
                                    ('height_site', SINT2),
                                    ('height_radar', SINT2),
                                    ('resolution_rays', UINT2),
                                    ('first_ray_index', UINT2),
                                    ('number_rays_sweep', UINT2),
                                    ('gparam_bytes', SINT2),
                                    ('altitude_radar', SINT4),
                                    ('velocity_east', SINT4),
                                    ('velocity_north', SINT4),
                                    ('velocity_up', SINT4),
                                    ('antenna_offset_starboard', SINT4),
                                    ('antenna_offset_bow', SINT4),
                                    ('antenna_offset_up', SINT4),
                                    ('fault_status', UINT4),
                                    ('melting_layer', SINT2),
                                    ('spare_2', {'fmt': '2s'}),
                                    ('local_timezone', {'fmt': '8s'}),
                                    ('flags', UINT4),
                                    ('configuration_name', {'fmt': '16s'}),
                                    ('spare_3', {'fmt': '228s'})])

# task_sched Structure
# 4.3.62, page 63

TASK_SCHED_INFO = OrderedDict([('start_time', SINT4),
                               ('stop_time', SINT4),
                               ('skip_time', SINT4),
                               ('time_last_run', SINT4),
                               ('time_used_last_run', SINT4),
                               ('day_last_run', SINT4),
                               ('flag', UINT2),
                               ('spare_0', {'fmt': '94s'})])

# dsp_data_mask Structure
# 4.3.7, page 28

DSP_DATA_MASK = OrderedDict([('mask_word_0', UINT4),
                             ('extended_header_type', UINT4),
                             ('mask_word_1', UINT4),
                             ('mask_word_2', UINT4),
                             ('mask_word_3', UINT4),
                             ('mask_word_4', UINT4)])

# task_dsp_mode_batch Structure
# 4.3.53, page 59

TASK_DSP_MODE_BATCH = OrderedDict([('low_prf', UINT2),
                                   ('low_prf_fraction_part', UINT2),
                                   ('low_prf_sample_size', SINT2),
                                   ('low_prf_range_averaging_bins', SINT2),
                                   ('reflectivity_unfolding_threshold', SINT2),
                                   ('velocity_unfolding_threshold', SINT2),
                                   ('width_unfolding_threshold', SINT2),
                                   ('spare_0', {'fmt': '18s'})])

# task_dsp_info Structure
# 4.3.52, page 57f

TASK_DSP_INFO = OrderedDict([('major_mode', UINT2),
                             ('dsp_type', UINT2),
                             ('dsp_data_mask0', DSP_DATA_MASK),
                             ('dsp_data_mask1', DSP_DATA_MASK),
                             ('task_dsp_mode', TASK_DSP_MODE_BATCH),
                             ('spare_0', {'fmt': '52s'}),
                             ('prf', SINT4),
                             ('pulse_width', SINT4),
                             ('multi_prf_mode_flag', UINT2),
                             ('dual_prf_delay', SINT2),
                             ('agc_feedback_code', UINT2),
                             ('sample_size', SINT2),
                             ('gain_control_flag', UINT2),
                             ('clutter_filter_name', {'fmt': '12s'}),
                             ('linear_filter_num_first_bin', UINT1),
                             ('log_filter_num_first_bin', UINT1),
                             ('attenuation_fixed_gain', SINT2),
                             ('gas_attenuation', UINT2),
                             ('cluttermap_flag', UINT2),
                             ('xmt_phase_sequence', UINT2),
                             ('ray_header_config_mask', UINT4),
                             ('playback_flags', UINT2),
                             ('spare_1', {'fmt': '2s'}),
                             ('custom_ray_header_name', {'fmt': '16s'}),
                             ('spare_2', {'fmt': '120s'})])

# task_calib_info Structure
# 4.3.50, page 56f

TASK_CALIB_INFO = OrderedDict([('reflectivity_slope', SINT2),
                               ('reflectivity_noise_threshold', SINT2),
                               ('clutter_correction_threshold', SINT2),
                               ('sqi_threshold', SINT2),
                               ('power_threshold', SINT2),
                               ('spare_0', {'fmt': '8s'}),
                               ('calibration_reflectivity', SINT2),
                               ('uncorrected_reflectivity_threshold_flags',
                                UINT2),
                               ('corrected_reflectivity_threshold_flags',
                                UINT2),
                               ('velocity_threshold_flags', UINT2),
                               ('width_threshold_flags', UINT2),
                               ('zdr_threshold_flags', UINT2),
                               ('spare_1', {'fmt': '6s'}),
                               ('flags_1', UINT2),
                               ('spare_2', {'fmt': '2s'}),
                               ('ldr_bias', SINT2),
                               ('zdr_bias', SINT2),
                               ('nexrad_point_clutter_threshold', SINT2),
                               ('nexrad_point_clutter_bin_skip', UINT2),
                               ('horizontal_io_cal_value', SINT2),
                               ('vertical_io_cal_value', SINT2),
                               ('horizontal_noise_calibration', SINT2),
                               ('vertical_noise_calibration', SINT2),
                               ('horizontal_radar_constant', SINT2),
                               ('vertical_radar_constant', SINT2),
                               ('receiver_bandwidth', SINT2),
                               ('flags_2', UINT16_T),
                               ('spare_3', {'fmt': '256s'})])

# task_range_info Structure
# 4.3.59, page 61

TASK_RANGE_INFO = OrderedDict([('range_first_bin', SINT4),
                               ('range_last_bin', SINT4),
                               ('number_input_bins', SINT2),
                               ('number_output_bins', SINT2),
                               ('step_input_bins', SINT4),
                               ('step_output_bins', SINT4),
                               ('variable_range_bin_spacing_flag', UINT2),
                               ('range_bin_averaging_flag', SINT2),
                               ('spare_0', {'fmt': '136s'})])

# task_rhi_scan_info Structure
# 4.3.60, page 61

_ANGLE_LIST = {'size': '80s',
               'read': np.fromstring,
               'rkw': {'dtype': 'uint16'},
               'func': decode_bin_angle,
               'fkw': {'mode': 2}}

TASK_RHI_SCAN_INFO = OrderedDict([('lower_elevation_limit', UINT2),
                                  ('upper_elevation_limit', UINT2),
                                  ('list_of_azimuths', _ANGLE_LIST),
                                  ('spare_0', {'fmt': '115s'}),
                                  ('start_first_sector_sweep', UINT1)])
# task_ppi_scan_info Structure
# 4.3.58, page 61

TASK_PPI_SCAN_INFO = OrderedDict([('left_azimuth_limit', BIN2),
                                  ('right_azimuth_limit', BIN2),
                                  ('list_of_elevations', _ANGLE_LIST),
                                  ('spare_0', {'fmt': '115s'}),
                                  ('start_first_sector_sweep', UINT1)])

# task_file_scan_info Structure
# 4.3.55, page 60

TASK_FILE_SCAN_INFO = OrderedDict([('first_azimuth_angle', UINT2),
                                   ('first_elevation_angle', UINT2),
                                   ('filename_antenna_control',
                                    {'fmt': '12s'}),
                                   ('spare_0', {'fmt': '184s'})])

# task_manual_scan_info Structure
# 4.3.56, page 60

TASK_MANUAL_SCAN_INFO = OrderedDict([('flags', UINT2),
                                     ('spare_0', {'fmt': '198s'})])


# task_scan_info Structure
# 4.3.61, page 62

TASK_SCAN_INFO = OrderedDict([('antenna_scan_mode', UINT2),
                              ('desired_angular_resolution', SINT2),
                              ('spare_0', {'fmt': '2s'}),
                              ('sweep_number', SINT2),
                              ('task_type_scan_info', {'fmt': '200s'}),
                              ('spare_1', {'fmt': '112s'})])

# task_misc_info Structure
# 4.3.57, page 60

TASK_MISC_INFO = OrderedDict([('wavelength', SINT4),
                              ('tr_serial_number', {'fmt': '16s'}),
                              ('transmit_power', SINT4),
                              ('flags', UINT2),
                              ('polarization_type', UINT2),
                              ('truncation_height', SINT4),
                              ('spare_0', {'fmt': '18s'}),
                              ('spare_1', {'fmt': '12s'}),
                              ('number_comment_bytes', SINT2),
                              ('horizontal_beam_width', BIN4),
                              ('vertical_beam_width', BIN4),
                              ('customer_storage', {'fmt': '40s'}),
                              ('spare_2', {'fmt': '208s'})])

# task_end_info Structure
# 4.3.54, page 59

TASK_END_INFO = OrderedDict([('task_major_number', SINT2),
                             ('task_minor_number', SINT2),
                             ('task_configuration_file_name', {'fmt': '12s'}),
                             ('task_description', {'fmt': '80s'}),
                             ('number_tasks', SINT4),
                             ('task_state', UINT2),
                             ('spare_0', {'fmt': '2s'}),
                             ('task_data_time', _YMDS_TIME),
                             ('echo_class_identifiers', {'fmt': '6s'}),
                             ('spare_1', {'fmt': '198s'})])

# task_configuration Structure
# 4.3.51, page 57

TASK_CONFIGURATION = OrderedDict([('structure_header', STRUCTURE_HEADER),
                                  ('task_sched_info', TASK_SCHED_INFO),
                                  ('task_dsp_info', TASK_DSP_INFO),
                                  ('task_calib_info', TASK_CALIB_INFO),
                                  ('task_range_info', TASK_RANGE_INFO),
                                  ('task_scan_info', TASK_SCAN_INFO),
                                  ('task_misc_info', TASK_MISC_INFO),
                                  ('task_end_info', TASK_END_INFO),
                                  ('comments', {'fmt': '720s'})])

# _ingest_header Structure
# 4.3.16, page 33

INGEST_HEADER = OrderedDict([('structure_header', STRUCTURE_HEADER),
                             ('ingest_configuration', INGEST_CONFIGURATION),
                             ('task_configuration', TASK_CONFIGURATION),
                             ('spare_0', {'fmt': '732s'}),
                             ('gparm', {'fmt': '128s'}),
                             ('reserved', {'fmt': '920s'})])


# raw_prod_bhdr Structure
# 4.3.31, page 45

RAW_PROD_BHDR = OrderedDict([('_record_number', SINT2),
                             ('sweep_number', SINT2),
                             ('first_ray_byte_offset', SINT2),
                             ('sweep_ray_number', SINT2),
                             ('flags', UINT2),
                             ('spare', {'fmt': '2s'})])

# ingest_data_header Structure
# 4.3.15, page 32

INGEST_DATA_HEADER = OrderedDict([('structure_header', STRUCTURE_HEADER),
                                  ('sweep_start_time', _YMDS_TIME),
                                  ('sweep_number', SINT2),
                                  ('number_rays_per_sweep', SINT2),
                                  ('first_ray_index', SINT2),
                                  ('number_rays_file_expected', SINT2),
                                  ('number_rays_file_written', SINT2),
                                  ('fixed_angle', BIN2),
                                  ('bits_per_bin', SINT2),
                                  ('data_type', UINT2),
                                  ('spare_0', {'fmt': '36s'})])


# some length's of data structures
LEN_PRODUCT_HDR = struct.calcsize(_get_fmt_string(PRODUCT_HDR))
LEN_INGEST_HEADER = struct.calcsize(_get_fmt_string(INGEST_HEADER))
LEN_RAW_PROD_BHDR = struct.calcsize(_get_fmt_string(RAW_PROD_BHDR))
LEN_INGEST_DATA_HEADER = struct.calcsize(_get_fmt_string(INGEST_DATA_HEADER))

# Sigmet data types
# 4.9 Constants, Table 17

SIGMET_DATA_TYPES = OrderedDict([
    # Extended Headers
    (0, {'name': 'DB_XHDR', 'func': None}),
    # Total H power (1 byte)
    (1, {'name': 'DB_DBT', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 2., 'offset': -64.}}),
    # Clutter Corrected H reflectivity (1 byte)
    (2, {'name': 'DB_DBZ', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 2., 'offset': -64.}}),
    # Velocity (1 byte)
    (3, {'name': 'DB_VEL', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 127., 'offset': -128.}}),
    # Width (1 byte)
    (4, {'name': 'DB_WIDTH', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 256.}}),
    # Differential reflectivity (1 byte)
    (5, {'name': 'DB_ZDR', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 16., 'offset': -128.}}),
    # Old Rainfall rate (stored as dBZ), not used
    (6, {'name': 'DB_ORAIN', 'dtype': '(2,) uint8', 'func': None}),
    # Fully corrected reflectivity (1 byte)
    (7, {'name': 'DB_DBZC', 'dtype': '(2,) uint8', 'func': decode_array,
         'kw': {'scale': 2., 'offset': -64.}}),
    # Uncorrected reflectivity (2 byte)
    (8, {'name': 'DB_DBT2', 'dtype': 'uint16', 'func': decode_array,
         'kw': {'scale': 100., 'offset': -32768.}}),
    # Corrected reflectivity (2 byte)
    (9, {'name': 'DB_DBZ2', 'dtype': 'uint16', 'func': decode_array,
         'kw': {'scale': 100., 'offset': -32768.}}),
    # Velocity (2 byte)
    (10, {'name': 'DB_VEL2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Width (2 byte)
    (11, {'name': 'DB_WIDTH2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100.}}),
    # Differential reflectivity (2 byte)
    (12, {'name': 'DB_ZDR2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Rainfall rate (2 byte)
    (13, {'name': 'DB_RAINRATE2', 'dtype': 'uint16',
          'func': decode_rainrate2}),
    # Kdp (specific differential phase)(1 byte)
    (14, {'name': 'DB_KDP', 'dtype': '(2,) int8', 'func': decode_kdp,
          'kw': {}}),
    # Kdp (specific differential phase)(2 byte)
    (15, {'name': 'DB_KDP2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # PHIdp (differential phase)(1 byte)
    (16, {'name': 'DB_PHIDP', 'dtype': '(2,) uint8', 'func': decode_phidp,
          'kw': {'scale': 254., 'offset': -1}}),
    # Corrected Velocity (1 byte)
    (17, {'name': 'DB_VELC', 'dtype': '(2,) uint8', 'func': None}),
    # SQI (1 byte)
    (18, {'name': 'DB_SQI', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # RhoHV(0) (1 byte)
    (19, {'name': 'DB_RHOHV', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # RhoHV(0) (2 byte)
    (20, {'name': 'DB_RHOHV2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # Fully corrected reflectivity (2 byte)
    (21, {'name': 'DB_DBZC2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Corrected Velocity (2 byte)
    (22, {'name': 'DB_VELC2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # SQI (2 byte)
    (23, {'name': 'DB_SQI2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # PHIdp (differential phase)(2 byte)
    (24, {'name': 'DB_PHIDP2', 'dtype': 'uint16', 'func': decode_phidp2}),
    # LDR H to V (1 byte)
    (25, {'name': 'DB_LDRH', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 5., 'offset': -1., 'offset2': -45.0}}),
    # LDR H to V (2 byte)
    (26, {'name': 'DB_LDRH2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # LDR V to H (1 byte)
    (27, {'name': 'DB_LDRV', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 5., 'offset': -1., 'offset2': -45.0}}),
    # LDR V to H (2 byte)
    (28, {'name': 'DB_LDRV2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Individual flag bits for each bin
    (29, {'name': 'DB_FLAGS', 'func': None}),
    # (See bit definitions below)
    (30, {'name': 'DB_FLAGS2', 'func': None}),
    # Test of floating format
    (31, {'name': 'DB_FLOAT32', 'func': None}),
    # Height (1/10 km) (1 byte)
    (32, {'name': 'DB_HEIGHT', 'dtype': '(2,) uint8',
          'func': decode_array,
          'kw': {'scale': 10., 'offset': -1.}}),
    # Linear liquid (.001mm) (2 byte)
    (33, {'name': 'DB_VIL2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 1000., 'offset': -1.}}),
    # Data type is not applicable
    (34, {'name': 'DB_NULL', 'func': None}),
    # Wind Shear (1 byte)
    (35, {'name': 'DB_SHEAR', 'dtype': '(2,) int8', 'func': decode_array,
          'kw': {'scale': 5., 'offset': -128.}}),
    # Divergence (.001 10**-4) (2-byte)
    (36, {'name': 'DB_DIVERGE2', 'dtype': 'int16', 'func': decode_array,
          'kw': {'scale': 10e-7}}),
    # Floated liquid (2 byte)
    (37, {'name': 'DB_FLIQUID2', 'dtype': 'uint16', 'func': None}),
    # User type, unspecified data (1 byte)
    (38, {'name': 'DB_USER', 'func': None}),
    # Unspecified data, no color legend
    (39, {'name': 'DB_OTHER', 'func': None}),
    # Deformation (.001 10**-4) (2-byte)
    (40, {'name': 'DB_DEFORM2', 'dtype': 'int16', 'func': decode_array,
          'kw': {'scale': 10e-7}}),
    # Vertical velocity (.01 m/s) (2-byte)
    (41, {'name': 'DB_VVEL2', 'dtype': 'int16', 'func': decode_array,
          'kw': {'scale': 100.}}),
    # Horizontal velocity (.01 m/s) (2-byte)
    (41, {'name': 'DB_HVEL2', 'func': decode_array,
          'kw': {'scale': 100.}}),
    # Horizontal wind direction (.1 degree) (2-byte)
    (43, {'name': 'DB_HDIR2', 'dtype': 'int16', 'func': decode_array,
          'kw': {'scale': 10.}}),
    # Axis of Dilation (.1 degree) (2-byte)
    (44, {'name': 'DB_AXDIL2', 'dtype': 'int16', 'func': decode_array,
          'kw': {'scale': 10.}}),
    # Time of data (seconds) (2-byte)
    (45, {'name': 'DB_TIME2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 60., 'offset': -32768}}),
    # Rho H to V (1 byte)
    (46, {'name': 'DB_RHOH', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # Rho H to V (2 byte)
    (47, {'name': 'DB_RHOH2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # Rho V to H (1 byte)
    (48, {'name': 'DB_RHOV', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # Rho V to H (2 byte)
    (49, {'name': 'DB_RHOV2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # Phi H to V (1 byte)
    (50, {'name': 'DB_PHIH', 'dtype': '(2,) uint8', 'func': decode_phidp}),
    # Phi H to V (2 byte)
    (51, {'name': 'DB_PHIH2', 'dtype': 'uint16', 'func': decode_phidp2}),
    # Phi V to H (1 byte)
    (52, {'name': 'DB_PHIV', 'dtype': '(2,) uint8', 'func': decode_phidp}),
    # Phi V to H (2 byte)
    (53, {'name': 'DB_PHIV2', 'dtype': 'uint16', 'func': decode_phidp2}),
    # User type, unspecified data (2 byte)
    (54, {'name': 'DB_USER2', 'dtype': 'uint16', 'func': None}),
    # Hydrometeor class (1 byte)
    (55, {'name': 'DB_HCLASS', 'dtype': '(2,) uint8', 'func': None}),
    # Hydrometeor class (2 byte)
    (56, {'name': 'DB_HCLASS2', 'dtype': 'uint16', 'func': None}),
    # Corrected Differential reflectivity (1 byte)
    (57, {'name': 'DB_ZDRC', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 16., 'offset': -128.}}),
    # Corrected Differential reflectivity (2 byte)
    (58, {'name': 'DB_ZDRC2', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Temperature (2 byte)
    (59, {'name': 'DB_TEMPERATURE16', 'dtype': 'uint16', 'func': None}),
    # Vertically Integrated Reflectivity (2 byte)
    (60, {'name': 'DB_VIR16', 'dtype': 'uint16', 'func': None}),
    # Total V Power (1 byte)
    (61, {'name': 'DB_DBTV8', 'dtype': '(2,) uint8',
          'func': decode_array,
          'kw': {'scale': 2., 'offset': -64.}}),
    # Total V Power (2 byte)
    (62, {'name': 'DB_DBTV16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Clutter Corrected V Reflectivity (1 byte)
    (63, {'name': 'DB_DBZV8', 'dtype': '(2,) uint8',
          'func': decode_array,
          'kw': {'scale': 2., 'offset': -64.}}),
    # Clutter Corrected V Reflectivity (2 byte)
    (64, {'name': 'DB_DBZV16', 'dtype': 'uint16',
          'func': decode_array,
          'kw': {'scale': 100., 'offset': -32768.}}),
    # Signal to Noise ratio (1 byte)
    (65, {'name': 'DB_SNR8', 'dtype': '(2,) uint8',
          'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Signal to Noise ratio (2 byte)
    (66, {'name': 'DB_SNR16', 'dtype': 'uint16',
          'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Albedo (1 byte)
    (67, {'name': 'DB_ALBEDO8', 'dtype': '(2,) uint8', 'func': None}),
    # Albedo (2 byte)
    (68, {'name': 'DB_ALBEDO16', 'dtype': 'uint16', 'func': None}),
    # VIL Density (2 byte)
    (69, {'name': 'DB_VILD16', 'dtype': 'uint16', 'func': None}),
    # Turbulence (2 byte)
    (70, {'name': 'DB_TURB16', 'dtype': 'uint16', 'func': None}),
    # Total Power Enhanced (via H+V or HV) (1 byte)
    (71, {'name': 'DB_DBTE8', 'dtype': '(2,) uint8', 'func': None}),
    # Total Power Enhanced (via H+V or HV) (2 byte)
    (72, {'name': 'DB_DBTE16', 'dtype': 'uint16', 'func': None}),
    # Clutter Corrected Reflectivity Enhanced (1 byte)
    (73, {'name': 'DB_DBZE8', 'dtype': '(2,) uint8', 'func': None}),
    # Clutter Corrected Reflectivity Enhanced (2 byte)
    (74, {'name': 'DB_DBZE16', 'dtype': 'uint16', 'func': None}),
    # Polarimetric meteo index (1 byte)
    (75, {'name': 'DB_PMI8', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # Polarimetric meteo index (2 byte)
    (76, {'name': 'DB_PMI16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # The log receiver signal-to-noise ratio (1 byte)
    (77, {'name': 'DB_LOG8', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # The log receiver signal-to-noise ratio (2 byte)
    (78, {'name': 'DB_LOG16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Doppler channel clutter signal power (-CSR) (1 byte)
    (79, {'name': 'DB_CSP8', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Doppler channel clutter signal power (-CSR) (2 byte)
    (80, {'name': 'DB_CSP16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Cross correlation, uncorrected rhohv (1 byte)
    (81, {'name': 'DB_CCOR8', 'dtype': '(2,) uint8', 'func': decode_sqi,
          'kw': {'scale': 253., 'offset': -1}}),
    # Cross correlation, uncorrected rhohv (2 byte)
    (82, {'name': 'DB_CCOR16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 65536., 'offset': -1.}}),
    # Attenuation of Zh (1 byte)
    (83, {'name': 'DB_AH8', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Attenuation of Zh (2 byte)
    (84, {'name': 'DB_AH16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Attenuation of Zv (1 byte)
    (85, {'name': 'DB_AV8', 'dtype': '(2,) uint8', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Attenuation of Zv (2 byte)
    (86, {'name': 'DB_AV16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Attenuation of Zzdr (1 byte)
    (87, {'name': 'DB_AZDR8', 'dtype': '(2,) uint8',
          'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}}),
    # Attenuation of Zzdr (2 byte)
    (88, {'name': 'DB_AZDR16', 'dtype': 'uint16', 'func': decode_array,
          'kw': {'scale': 2., 'offset': -63.}})
    ])
