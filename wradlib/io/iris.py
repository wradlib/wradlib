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

   read_iris
"""

import numpy as np
import struct
from collections import OrderedDict


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
    def __init__(self, filename, debug=False):

        self.debug = debug
        self._fh = np.memmap(filename)
        self.record_number = 0
        start = self.record_number * RECORD_BYTES
        stop = start + RECORD_BYTES
        self.rh = IrisRecord(self.fh[start:stop], self.record_number)

        # read data headers
        self.product_hdr = _unpack_dictionary(self.read_record(0)
                                              [:LEN_PRODUCT_HDR],
                                              PRODUCT_HDR)
        # TODO: unpack task_{scantype}_scan_info based on scan type
        # from task_scan_info substructure
        self.ingest_header = _unpack_dictionary(self.read_record(1)
                                                [:LEN_INGEST_HEADER],
                                                INGEST_HEADER)
        self.raw_product_bhdrs = []

        # determine data types contained in the file
        self._data_types = self.get_data_types()

        self.sweeps = OrderedDict()

    def next_record(self):
        self.record_number += 1
        start = self.record_number * RECORD_BYTES
        stop = start + RECORD_BYTES
        self.rh = IrisRecord(self.fh[start:stop], self.record_number)

    def read_record(self, recnum):
        start = recnum * RECORD_BYTES
        stop = start + RECORD_BYTES
        self.rh = IrisRecord(self.fh[start:stop], recnum)
        return self.rh.record

    def read(self, words=1, dtype='int16'):
        dta = self.rh.read(words).view(dtype=dtype)
        words -= len(dta)
        if words > 0:
            self.next_record()
            self.get_raw_prod_bhdr()
            dta = np.append(dta, self.rh.read(words).view(dtype=dtype))
        return dta

    @property
    def fh(self):
        return self._fh

    @property
    def nsweeps(self):
        head = self.ingest_header['task_configuration']['task_scan_info']
        return head['sweep_number']

    @property
    def nbins(self):
        return self.product_hdr['product_end']['number_bins']

    @property
    def nrays(self):
        return self.ingest_header['ingest_configuration']['number_rays_sweep']

    @property
    def data_types(self):
        return self._data_types

    @property
    def data_types_count(self):
        return len(self.data_types)

    @property
    def data_types_names(self):
        return [SIGMET_DATA_TYPES[i] for i in self.data_types]

    def get_compression_code(self):
        cmp_val = self.read(dtype='int16')[0]
        cmp_msb = np.sign(cmp_val) == -1
        if cmp_msb:
            cmp_val = cmp_val + 32768
        if self.debug:
            print("--- Sub CMP Code:", cmp_msb, cmp_val, self.rh.recpos - 1,
                  self.rh.recpos)
        return cmp_msb, cmp_val

    def get_data_types(self):
        """ Determine the available data types.
        """
        # determine the available fields
        task_config = self.ingest_header['task_configuration']
        task_dsp_info = task_config['task_dsp_info']
        word0 = task_dsp_info['dsp_data_mask0']['mask_word_0']
        word1 = task_dsp_info['dsp_data_mask0']['mask_word_1']
        word2 = task_dsp_info['dsp_data_mask0']['mask_word_2']
        word3 = task_dsp_info['dsp_data_mask0']['mask_word_3']

        return _data_types_from_dsp_mask([word0, word1, word2, word3])

    def get_raw_prod_bhdr(self):
        self.raw_product_bhdrs.append(
            _unpack_dictionary(self.rh.read(LEN_RAW_PROD_BHDR, width=1),
                               RAW_PROD_BHDR))

    def get_ingest_data_headers(self):

        ingest_data_hdrs = OrderedDict()
        for i, dt in enumerate(self.data_types_names):
            ingest_data_hdrs[dt] = _unpack_dictionary(
                self.rh.read(LEN_INGEST_DATA_HEADER, width=1),
                INGEST_DATA_HEADER)

        return ingest_data_hdrs

    def get_ray(self):
        ray_pos = 0

        data = np.zeros((self.nbins + 6), dtype='int16')

        cmp_msb, cmp_val = self.get_compression_code()

        # ray is missing
        if (cmp_val == 1) & (cmp_msb == 0):
            if self.debug:
                print("ray missing")
            return None

        while not ((cmp_val == 1) & (not cmp_msb)):

            # data words follow
            if cmp_msb:
                if self.debug:
                    print(
                        "--- Add {0} WORDS at range {1}, record {2}:{3}:"
                        "".format(cmp_val, ray_pos, self.rh.recpos,
                                  self.rh.recpos + cmp_val))
                data[ray_pos:ray_pos + cmp_val] = self.read(cmp_val,
                                                            dtype='int16')
            # compressed zeros follow
            # can be skipped, if data array is created all zeros
            else:
                if self.debug:
                    print(
                        "--- Add {0} Zeros at range {1}, record {2}:{3}:"
                        "".format(cmp_val, ray_pos,
                                  self.rh.recpos, self.rh.recpos + 1))
                if cmp_val + ray_pos > self.nbins + 6:
                    return data[:6], data[6:]
                data[ray_pos:ray_pos + cmp_val] = 0

            ray_pos += cmp_val

            # read next compression code
            cmp_msb, cmp_val = self.get_compression_code()

        return data[:6], data[6:]

    def get_sweep(self):

        sweep = OrderedDict()

        self.get_raw_prod_bhdr()
        sweep['ingest_data_hdrs'] = self.get_ingest_data_headers()

        rays_per_data_type = [d['number_rays_file_expected'] for d
                              in sweep['ingest_data_hdrs'].values()]

        rays = sum(rays_per_data_type)
        bins = self.product_hdr['product_end']['number_bins']

        raw_sweep_data = np.zeros((rays, bins), dtype='int16')
        azi_start = np.zeros(rays, dtype='int16')
        azi_stop = np.zeros(rays, dtype='int16')
        ele_start = np.zeros(rays, dtype='int16')
        ele_stop = np.zeros(rays, dtype='int16')
        rbins = np.zeros(rays, dtype='int16')
        dtime = np.zeros(rays, dtype='uint16')

        for ray_i in range(rays):
            if self.debug:
                print("{0}: Ray started at {1}"
                      "".format(ray_i, int(self.rh.recpos) - 1))
            ret = self.get_ray()
            if ret is None:
                continue
            else:
                header = ret[0]
                azi_start[ray_i] = header[0]
                ele_start[ray_i] = header[1]
                azi_stop[ray_i] = header[2]
                ele_stop[ray_i] = header[3]
                rbins[ray_i] = header[4]
                dtime[ray_i] = header[5]
                raw_sweep_data[ray_i] = ret[1]

        sweep['raw_sweep_data'] = raw_sweep_data
        sweep['azi_start'] = azi_start
        sweep['ele_start'] = ele_start
        sweep['azi_stop'] = azi_stop
        sweep['ele_stop'] = ele_stop
        sweep['rbins'] = rbins
        sweep['dtime'] = dtime

        return sweep

    def get_sweeps(self):
        self.record_number = 1
        for i in range(self.nsweeps):
            self.next_record()
            self.sweeps[i] = self.get_sweep()


def read_iris(filename, loaddata=True, debug=False):
    fh = IrisFile(filename, debug=debug)
    data = OrderedDict()
    data['product_hdr'] = fh.product_hdr
    data['ingest_header'] = fh.ingest_header
    if loaddata:
        fh.get_sweeps()
        data['sweeps'] = fh.sweeps
    data['raw_product_bhdrs'] = fh.raw_product_bhdrs

    return data


def decode_bin_angle(bin_angle, width):
    """ Decode BIN angle
    """
    return 360. * bin_angle / 2**(width*8)




# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters

SINT1 = 'b'
SINT2 = 'h'
SINT4 = 'i'
UINT1 = 'B'
UINT2 = 'H'
UINT4 = 'I'
FLT4 = 'f'
FLT8 = 'd'
BIN1 = 'B'
BIN2 = 'H'
BIN4 = 'I'
MESSAGE = 'I'
UINT16_T = 'H'


def _unpack_dictionary(buffer, dictionary):
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
    fmt = ''
    sub = OrderedDict()

    # iterate over dictionary
    for k, v in dictionary.items():
        try:
            fmt += v
        except TypeError as e:
            # remember sub-structures
            sub[k] = v
            # check for type
            if e.args[0].startswith('must be str'):
                fmt += '{}s'.format(_calcsize(v))
            else:
                fmt += '{}s'.format(struct.calcsize(v[0]))

    # unpack into OrderedDict
    data = OrderedDict(zip(dictionary, struct.unpack(fmt, buffer)))

    # iterate over sub dictionaries
    for k, v in sub.items():
        try:
            data[k] = _unpack_dictionary(data[k], v)
        except AttributeError:
            data[k] = np.fromstring(data[k], dtype=v[1])
    return data


def _data_types_from_dsp_mask(words):
    """
    Return a list of the data types from the words in the data_type mask.
    """
    data_types = []
    for i, word in enumerate(words):
        data_types += [j+(i*32) for j in range(32) if word >> j & 1]
    return data_types


def _calcsize(structure):
    size = 0
    fmt = ''
    for v in structure.values():
        try:
            fmt += v
        except TypeError as e:
            if e.args[0].startswith('must be str'):
                size += _calcsize(v)
            else:
                size += struct.calcsize(v[0])
    size += struct.calcsize(fmt)
    return size


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

# color_scale_def Structure
# 4.3.5, page 26

COLOR_SCALE_DEF = OrderedDict([('iflags', UINT4),
                               ('istart', SINT4),
                               ('istep', SINT4),
                               ('icolcnt', SINT2),
                               ('iset_and_scale', UINT2),
                               ('ilevel_seams', np.array(['32s', 'uint16']))])

# product_configuration Structure
# 4.3.24, page 36

PRODUCT_CONFIGURATION = OrderedDict([('structure_header', STRUCTURE_HEADER),
                                     ('product_type_code', UINT2),
                                     ('scheduling_code', UINT2),
                                     ('seconds_between_runs', SINT4),
                                     ('generation_time', YMDS_TIME),
                                     ('sweep_ingest_time', YMDS_TIME),
                                     ('file_ingest_time', YMDS_TIME),
                                     ('spare_0', '6s'),
                                     ('product_name', '12s'),
                                     ('task_name', '12s'),
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
                                     ('spare_1', '2s'),
                                     ('data_type', UINT2),
                                     ('projection_name', '12s'),
                                     ('input_data_type', UINT2),
                                     ('projection_type', UINT1),
                                     ('spare_2', '1s'),
                                     ('radial_smoother', SINT2),
                                     ('times_run', SINT2),
                                     ('zr_constant', SINT4),
                                     ('zr_exponent', SINT4),
                                     ('x_smoother', SINT2),
                                     ('y_smoother', SINT2),
                                     ('product_specific_info', '80s'),
                                     ('minor_task_suffixes', '16s'),
                                     ('spare_3', '12s'),
                                     ('color_scale_def', COLOR_SCALE_DEF)])
# product_end Structure
# 4.3.25, page 39

PRODUCT_END = OrderedDict([('site_name', '16s'),
                           ('iris_version_created', '8s'),
                           ('ingest_iris_version', '8s'),
                           ('ingest_time', YMDS_TIME),
                           ('spare_0', '28s'),
                           ('GMT_minute_offset_local', SINT2),
                           ('ingest_hardware_name_', '16s'),
                           ('ingest_site_name_', '16s'),
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
                           ('clutter_filter', '12s'),
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
                           ('spare_1', '12s'),
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
                           ('spare_2', '32s'),
                           ('melting_level', SINT2),
                           ('radar_height_above_reference', SINT2),
                           ('number_elements', SINT2),
                           ('mean_wind_speed', UINT1),
                           ('mean_wind_direction', BIN1),
                           ('spare_3', '2s'),
                           ('tz_name', '8s'),
                           ('extended_product_header_offset', UINT4),
                           ('spare_4', '4s')])

# product_hdr Structure
# 4.3.26 page 41
PRODUCT_HDR = OrderedDict([('structure_header', STRUCTURE_HEADER),
                           ('product_configuration', PRODUCT_CONFIGURATION),
                           ('product_end', PRODUCT_END)])

# ingest_configuration Structure
# 4.3.14, page 31

INGEST_CONFIGURATION = OrderedDict([('filename', '80s'),
                                    ('number_files', SINT2),
                                    ('number_sweeps_completed', SINT2),
                                    ('total_size', SINT4),
                                    ('volume_scan_start_time', YMDS_TIME),
                                    ('spare_0', '12s'),
                                    ('ray_header_bytes', SINT2),
                                    ('extended_ray_header_bytes', SINT2),
                                    ('number_task_config_table', SINT2),
                                    ('playback_version', SINT2),
                                    ('spare_1', '4s'),
                                    ('iris_version', '8s'),
                                    ('hardware_site', '16s'),
                                    ('gmt_offset_minutes_local', SINT2),
                                    ('site_name', '16s'),
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
                                    ('spare_2', '2s'),
                                    ('local_timezone', '8s'),
                                    ('flags', UINT4),
                                    ('configuration_name', '16s'),
                                    ('spare_3', '228s')])

# task_sched Structure
# 4.3.62, page 63

TASK_SCHED_INFO = OrderedDict([('start_time', SINT4),
                               ('stop_time', SINT4),
                               ('skip_time', SINT4),
                               ('time_last_run', SINT4),
                               ('time_used_last_run', SINT4),
                               ('day_last_run', SINT4),
                               ('flag', UINT2),
                               ('spare_0', '94s')])

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
                                   ('spare_0', '18s')])

# task_dsp_info Structure
# 4.3.52, page 57f

TASK_DSP_INFO = OrderedDict([('major_mode', UINT2),
                             ('dsp_type', UINT2),
                             ('dsp_data_mask0', DSP_DATA_MASK),
                             ('dsp_data_mask1', DSP_DATA_MASK),
                             ('task_dsp_mode', TASK_DSP_MODE_BATCH),
                             ('spare_0', '52s'),
                             ('prf', SINT4),
                             ('pulse_width', SINT4),
                             ('multi_prf_mode_flag', UINT2),
                             ('dual_prf_delay', SINT2),
                             ('agc_feedback_code', UINT2),
                             ('sample_size', SINT2),
                             ('gain_control_flag', UINT2),
                             ('clutter_filter_name', '12s'),
                             ('linear_filter_num_first_bin', UINT1),
                             ('log_filter_num_first_bin', UINT1),
                             ('attenuation_fixed_gain', SINT2),
                             ('gas_attenuation', UINT2),
                             ('cluttermap_flag', UINT2),
                             ('xmt_phase_sequence', UINT2),
                             ('ray_header_config_mask', UINT4),
                             ('playback_flags', UINT2),
                             ('spare_1', '2s'),
                             ('custom_ray_header_name', '16s'),
                             ('spare_2', '120s')])

# task_calib_info Structure
# 4.3.50, page 56f

TASK_CALIB_INFO = OrderedDict([('reflectivity_slope', SINT2),
                               ('reflectivity_noise_threshold', SINT2),
                               ('clutter_correction_threshold', SINT2),
                               ('sqi_threshold', SINT2),
                               ('power_threshold', SINT2),
                               ('spare_0', '8s'),
                               ('calibration_reflectivity', SINT2),
                               ('uncorrected_reflectivity_threshold_flags',
                                UINT2),
                               ('corrected_reflectivity_threshold_flags',
                                UINT2),
                               ('velocity_threshold_flags', UINT2),
                               ('width_threshold_flags', UINT2),
                               ('zdr_threshold_flags', UINT2),
                               ('spare_1', '6s'),
                               ('flags_1', UINT2),
                               ('spare_2', '2s'),
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
                               ('spare_3', '256s')])

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
                               ('spare_0', '136s')])

# task_rhi_scan_info Structure
# 4.3.60, page 61

TASK_RHI_SCAN_INFO = OrderedDict([('lower_elevation_limit', UINT2),
                                  ('upper_elevation_limit', UINT2),
                                  ('list_of_azimuths', '80s'),
                                  ('spare_0', '115s'),
                                  ('start_first_sector_sweep', UINT1)])
# task_ppi_scan_info Structure
# 4.3.58, page 61

TASK_PPI_SCAN_INFO = OrderedDict([('left_azimuth_limit', BIN2),
                                  ('right_azimuth_limit', BIN2),
                                  ('list_of_elevations', '80s'),
                                  ('spare_0', '115s'),
                                  ('start_first_sector_sweep', UINT1)])

# task_file_scan_info Structure
# 4.3.55, page 60

TASK_FILE_SCAN_INFO = OrderedDict([('first_azimuth_angle', UINT2),
                                   ('first_elevation_angle', UINT2),
                                   ('filename_antenna_control', '12s'),
                                   ('spare_0', '184s')])

# task_manual_scan_info Structure
# 4.3.56, page 60

TASK_MANUAL_SCAN_INFO = OrderedDict([('flags', UINT2),
                                     ('spare_0', '198s')])


# task_scan_info Structure
# 4.3.61, page 62

TASK_SCAN_INFO = OrderedDict([('antenna_scan_mode', UINT2),
                              ('desired_angular_resolution', SINT2),
                              ('spare_0', '2s'),
                              ('sweep_number', SINT2),
                              ('task_scan_type_info', '200s'),
                              ('spare_1', '112s')])

# task_misc_info Structure
# 4.3.57, page 60

TASK_MISC_INFO = OrderedDict([('wavelength', SINT4),
                              ('tr_serial_number', '16s'),
                              ('transmit_power', SINT4),
                              ('flags', UINT2),
                              ('polarization_type', UINT2),
                              ('truncation_height', SINT4),
                              ('spare_0', '18s'),
                              ('spare_1', '12s'),
                              ('number_comment_bytes', SINT2),
                              ('horizontal_beam_width', BIN4),
                              ('vertical_beam_width', BIN4),
                              ('customer_storage', '40s'),
                              ('spare_2', '208s')])

# task_end_info Structure
# 4.3.54, page 59

TASK_END_INFO = OrderedDict([('task_major_number', SINT2),
                             ('task_minor_number', SINT2),
                             ('task_configuration_file_name', '12s'),
                             ('task_description', '80s'),
                             ('number_tasks', SINT4),
                             ('task_state', UINT2),
                             ('spare_0', '2s'),
                             ('task_data_time', YMDS_TIME),
                             ('echo_class_identifiers', '6s'),
                             ('spare_1', '198s')])

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
                                  ('comments', '720s')])

# ingest_header Structure
# 4.3.16, page 33

INGEST_HEADER = OrderedDict([('structure_header', STRUCTURE_HEADER),
                             ('ingest_configuration', INGEST_CONFIGURATION),
                             ('task_configuration', TASK_CONFIGURATION),
                             ('spare_0', '732s'),
                             ('gparm', '128s'),
                             ('reserved', '920s')])


# raw_prod_bhdr Structure
# 4.3.31, page 45

RAW_PROD_BHDR = OrderedDict([('record_number', SINT2),
                             ('sweep_number', SINT2),
                             ('first_ray_byte_offset', SINT2),
                             ('sweep_ray_number', SINT2),
                             ('flags', UINT2),
                             ('spare', '2s')])

# ingest_data_header Structure
# 4.3.15, page 32

INGEST_DATA_HEADER = OrderedDict([('structure_header', STRUCTURE_HEADER),
                                  ('sweep_start_time', YMDS_TIME),
                                  ('sweep_number', SINT2),
                                  ('number_rays_per_sweep', SINT2),
                                  ('first_ray_index', SINT2),
                                  ('number_rays_file_expected', SINT2),
                                  ('number_rays_file_written', SINT2),
                                  ('fixed_angle', BIN2),
                                  ('bits_per_bin', SINT2),
                                  ('data_type', UINT2),
                                  ('spare_0', '36s')])


# some length's of data structures
LEN_PRODUCT_HDR = _calcsize(PRODUCT_HDR)
LEN_INGEST_HEADER = _calcsize(INGEST_HEADER)
LEN_RAW_PROD_BHDR = _calcsize(RAW_PROD_BHDR)
LEN_INGEST_DATA_HEADER = _calcsize(INGEST_DATA_HEADER)

# Sigmet data types
# 4.9 Constants, Table 17

SIGMET_DATA_TYPES = OrderedDict(
    [(0, 'DB_XHDR'),  # Extended Headers
     (1, 'DB_DBT'),  # Total H power (1 byte)
     (2, 'DB_DBZ'),  # Clutter Corrected H reflectivity (1 byte)
     (3, 'DB_VEL'),  # Velocity (1 byte)
     (4, 'DB_WIDTH'),  # Width (1 byte)
     (5, 'DB_ZDR'),  # Differential reflectivity (1 byte)
     (6, 'DB_ORAIN'),  # Old Rainfall rate (stored as dBZ), not used
     (7, 'DB_DBZC'),  # Fully corrected reflectivity (1 byte)
     (8, 'DB_DBT2'),  # Uncorrected reflectivity (2 byte)
     (9, 'DB_DBZ2'),  # Corrected reflectivity (2 byte)
     (10, 'DB_VEL2'),  # Velocity (2 byte)
     (11, 'DB_WIDTH2'),  # Width (2 byte)
     (12, 'DB_ZDR2'),  # Differential reflectivity (2 byte)
     (13, 'DB_RAINRATE2'),  # Rainfall rate (2 byte)
     (14, 'DB_KDP'),  # Kdp (specific differential phase)(1 byte)
     (15, 'DB_KDP2'),  # Kdp (specific differential phase)(2 byte)
     (16, 'DB_PHIDP'),  # PHIdp (differential phase)(1 byte)
     (17, 'DB_VELC'),  # Corrected Velocity (1 byte)
     (18, 'DB_SQI'),  # SQI (1 byte)
     (19, 'DB_RHOHV'),  # RhoHV(0) (1 byte)
     (20, 'DB_RHOHV2'),  # RhoHV(0) (2 byte)
     (21, 'DB_DBZC2'),  # Fully corrected reflectivity (2 byte)
     (22, 'DB_VELC2'),  # Corrected Velocity (2 byte)
     (23, 'DB_SQI2'),  # SQI (2 byte)
     (24, 'DB_PHIDP2'),  # PHIdp (differential phase)(2 byte)
     (25, 'DB_LDRH'),  # LDR H to V (1 byte)
     (26, 'DB_LDRH2'),  # LDR H to V (2 byte)
     (27, 'DB_LDRV'),  # LDR V to H (1 byte)
     (28, 'DB_LDRV2'),  # LDR V to H (2 byte)
     (29, 'DB_FLAGS'),  # Individual flag bits for each bin
     (30, 'DB_FLAGS2'),  # (See bit definitions below)
     (31, 'DB_FLOAT32'),  # Test of floating format
     (32, 'DB_HEIGHT'),  # Height (1/10 km) (1 byte)
     (33, 'DB_VIL2'),  # Linear liquid (.001mm) (2 byte)
     (34, 'DB_NULL'),  # Data type is not applicable
     (35, 'DB_SHEAR'),  # Wind Shear (1 byte)
     (36, 'DB_DIVERGE2'),  # Divergence (.001 10**-4) (2-byte)
     (37, 'DB_FLIQUID2'),  # Floated liquid (2 byte)
     (38, 'DB_USER'),  # User type, unspecified data (1 byte)
     (39, 'DB_OTHER'),  # Unspecified data, no color legend
     (40, 'DB_DEFORM2'),  # Deformation (.001 10**-4) (2-byte)
     (41, 'DB_VVEL2'),  # Vertical velocity (.01 m/s) (2-byte)
     (41, 'DB_HVEL2'),  # Horizontal velocity (.01 m/s) (2-byte)
     (43, 'DB_HDIR2'),  # Horizontal wind direction (.1 degree) (2-byte)
     (44, 'DB_AXDIL2'),  # Axis of Dilation (.1 degree) (2-byte)
     (45, 'DB_TIME2'),  # Time of data (seconds) (2-byte)
     (46, 'DB_RHOH'),  # Rho H to V (1 byte)
     (47, 'DB_RHOH2'),  # Rho H to V (2 byte)
     (48, 'DB_RHOV'),  # Rho V to H (1 byte)
     (49, 'DB_RHOV2'),  # Rho V to H (2 byte)
     (50, 'DB_PHIH'),  # Phi H to V (1 byte)
     (51, 'DB_PHIH2'),  # Phi H to V (2 byte)
     (52, 'DB_PHIV'),  # Phi V to H (1 byte)
     (53, 'DB_PHIV2'),  # Phi V to H (2 byte)
     (54, 'DB_USER2'),  # User type, unspecified data (2 byte)
     (55, 'DB_HCLASS'),  # Hydrometeor class (1 byte)
     (56, 'DB_HCLASS2'),  # Hydrometeor class (2 byte)
     (57, 'DB_ZDRC'),  # Corrected Differential reflectivity (1 byte)
     (58, 'DB_ZDRC2'),  # Corrected Differential reflectivity (2 byte)
     (59, 'DB_TEMPERATURE16'),  # Temperature (2 byte)
     (60, 'DB_VIR16'),  # Vertically Integrated Reflectivity (2 byte)
     (61, 'DB_DBTV8'),  # Total V Power (1 byte)
     (62, 'DB_DBTV16'),  # Total V Power (2 byte)
     (63, 'DB_DBZV8'),  # Clutter Corrected V Reflectivity (1 byte)
     (64, 'DB_DBZV16'),  # Clutter Corrected V Reflectivity (2 byte)
     (65, 'DB_SNR8'),  # Signal to Noise ratio (1 byte)
     (66, 'DB_SNR16'),  # Signal to Noise ratio (2 byte)
     (67, 'DB_ALBEDO8'),  # Albedo (1 byte)
     (68, 'DB_ALBEDO16'),  # Albedo (2 byte)
     (69, 'DB_VILD16'),  # VIL Density (2 byte)
     (70, 'DB_TURB16'),  # Turbulence (2 byte)
     (71, 'DB_DBTE8'),  # Total Power Enhanced (via H+V or HV) (1 byte)
     (72, 'DB_DBTE16'),  # Total Power Enhanced (via H+V or HV) (2 byte)
     (73, 'DB_DBZE8'),  # Clutter Corrected Reflectivity Enhanced (1 byte)
     (74, 'DB_DBZE16'),  # Clutter Corrected Reflectivity Enhanced (2 byte)
     (75, 'DB_PMI8'),  # Polarimetric meteo index (1 byte)
     (76, 'DB_PMI16'),  # Polarimetric meteo index (2 byte)
     (77, 'DB_LOG8'),  # The log receiver signal-to-noise ratio (1 byte)
     (78, 'DB_LOG16'),  # The log receiver signal-to-noise ratio (2 byte)
     (79, 'DB_CSP8'),  # Doppler channel clutter signal power (-CSR) (1 byte)
     (80, 'DB_CSP16'),  # Doppler channel clutter signal power (-CSR) (2 byte)
     (81, 'DB_CCOR8'),  # Cross correlation, uncorrected rhohv (1 byte)
     (82, 'DB_CCOR16'),  # Cross correlation, uncorrected rhohv (2 byte)
     (83, 'DB_AH8'),  # Attenuation of Zh (1 byte)
     (84, 'DB_AH16'),  # Attenuation of Zh (2 byte)
     (85, 'DB_AV8'),  # Attenuation of Zv (1 byte)
     (86, 'DB_AV16'),  # Attenuation of Zv (2 byte)
     (87, 'DB_AZDR8'),  # Attenuation of Zzdr (1 byte)
     (88, 'DB_AZDR16'),  # Attenuation of Zzdr (2 byte)
     ])
