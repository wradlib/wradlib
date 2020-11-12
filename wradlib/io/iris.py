#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Read IRIS Data Format
^^^^^^^^^^^^^^^^^^^^^

Reads data from Vaisala's IRIS data formats

IRIS (Vaisala Sigmet Interactive Radar Information System)

See M211318EN-F Programming Guide ftp://ftp.sigmet.com/outgoing/manuals/

To read from IRIS files :class:`numpy:numpy.memmap` is used to get access to
the data. The IRIS header (`PRODUCT_HDR`, `INGEST_HEADER`) is read in any case
into dedicated OrderedDict's. Reading sweep data can be skipped by setting
`loaddata=False`. By default the data is decoded on the fly.
Using `rawdata=True` the data will be kept undecoded.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "IrisRecord",
    "IrisHeaderBase",
    "IrisStructureHeader",
    "IrisIngestHeader",
    "IrisProductHeader",
    "IrisIngestDataHeader",
    "IrisFileBase",
    "IrisFile",
    "IrisIngestHeaderFile",
    "IrisIngestDataFile",
    "IrisRecordFile",
    "IrisRawFile",
    "IrisProductFile",
    "IrisCartesianProductFile",
    "read_iris",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import copy
import datetime as dt
import io
import struct
import warnings
from collections import OrderedDict

import numpy as np


def get_dtype_size(dtype):
    """Return size in byte of given ``dtype``.

    Parameters
    ----------
    dtype : str
        dtype string

    Returns
    -------
    size : int
        dtype size in byte
    """
    return np.zeros(1, dtype=dtype).dtype.itemsize


def to_float(data):
    """Decode floating point value.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        encoded data

    Returns
    -------
    decoded : :class:`numpy:numpy.ndarray`
        decoded floating point data

    Note
    ----
    DB_FLIQUID2 decoding see IRIS manuals, 4.4.12 - Page 75

    """
    exp = data >> 12
    nz = exp > 0
    mantissa = (data & 0xFFF).astype(dtype="uint32")
    mantissa[nz] = (mantissa[nz] | 0x1000) << (exp[nz] - 1)
    return mantissa


def decode_bin_angle(bin_angle, mode=None):
    """Decode BIN angle.

    See 4.2 p.23

    Parameters
    ----------
    bin_angle : array-like
    mode : int
        number of bytes

    Returns
    -------
    out : array-like
        decoded angle
    """
    return 360.0 * bin_angle / 2 ** (mode * 8)


def decode_array(data, scale=1.0, offset=0, offset2=0, tofloat=False, mask=None):
    """Decode data array.

    .. math::

        decoded = \\frac{data + offset}{scale} + offset2

    Using the default values doesn't change the array.

    Parameters
    ----------
    data : array-like
    scale : int
    offset: int
    offset2: int
    tofloat: bool

    Returns
    -------
    data : array-like
        decoded data
    """
    if tofloat:
        data = to_float(data)
    if mask is not None:
        data = np.ma.masked_equal(data, mask)
    return (data + offset) / scale + offset2


def decode_vel(data, **kwargs):
    """Decode `DB_VEL`.

    See 4.4.46 p.85
    """
    nyquist = kwargs.pop("nyquist")
    # mask = kwargs.pop('mask')
    # data = np.ma.masked_equal(data, mask)
    return decode_array(data, **kwargs) * nyquist


def decode_width(data, **kwargs):
    """Decode `DB_WIDTH`.

    See 4.4.50 p.87
    """
    nyquist = kwargs.pop("nyquist")
    return decode_array(data, **kwargs) * nyquist


def decode_kdp(data, **kwargs):
    """Decode `DB_KDP`.

    See 4.4.20 p.77
    """
    wavelength = kwargs.pop("wavelength")
    zero = data[data == -128]
    data = -0.25 * np.sign(data) * 600 ** ((127 - np.abs(data)) / 126.0)
    data /= wavelength
    data[zero] = 0
    return data


def decode_phidp(data, **kwargs):
    """Decode `DB_PHIDP`.

    See 4.4.28 p.79
    """
    return 180.0 * decode_array(data, **kwargs)


def decode_phidp2(data, **kwargs):
    """Decode `DB_PHIDP2`.

    See 4.4.29 p.80
    """
    return 360.0 * decode_array(data, **kwargs)


def decode_sqi(data, **kwargs):
    """Decode `DB_SQI`

    See 4.4.41 p.83
    """
    return np.sqrt(decode_array(data, **kwargs))


def decode_time(data):
    """Decode `YMDS_TIME` into datetime object."""
    time = _unpack_dictionary(data, YMDS_TIME)
    try:
        return dt.datetime(time["year"], time["month"], time["day"]) + dt.timedelta(
            seconds=time["seconds"], milliseconds=time["milliseconds"]
        )
    except ValueError:
        return None


def decode_string(data):
    """Decode string and strip NULL-bytes from end."""
    return data.decode("utf-8").rstrip("\0")


# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters

SINT1 = {"fmt": "b"}
SINT2 = {"fmt": "h"}
SINT4 = {"fmt": "i"}
UINT1 = {"fmt": "B"}
UINT2 = {"fmt": "H"}
UINT4 = {"fmt": "I"}
FLT4 = {"fmt": "f"}
FLT8 = {"fmt": "d"}
BIN1 = {
    "name": "BIN1",
    "dtype": "uint8",
    "size": "B",
    "func": decode_bin_angle,
    "fkw": {"mode": 1},
}
BIN2 = {
    "name": "BIN2",
    "dtype": "uint16",
    "size": "H",
    "func": decode_bin_angle,
    "fkw": {"mode": 2},
}
BIN4 = {
    "name": "BIN4",
    "dtype": "uint32",
    "size": "I",
    "func": decode_bin_angle,
    "fkw": {"mode": 4},
}
MESSAGE = {"fmt": "I"}
UINT16_T = {"fmt": "H"}


def _get_fmt_string(dictionary, retsub=False):
    """Get Format String from given dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing data structure with fmt-strings.
    retsub : bool
        If True, return sub structures.

    Returns
    -------
    fmt : str
        struct format string
    sub : dict
        Dictionary containing substructure
    """
    fmt = ""
    if retsub:
        sub = OrderedDict()
    for k, v in dictionary.items():
        try:
            fmt += v["fmt"]
        except KeyError:
            # remember sub-structures
            if retsub:
                sub[k] = v
            if "size" in v:
                fmt += v["size"]
            else:
                fmt += "{}s".format(struct.calcsize(_get_fmt_string(v)))
    if retsub:
        return fmt, sub
    else:
        return fmt


def _unpack_dictionary(buffer, dictionary, rawdata=False):
    """Unpacks binary data using the given dictionary structure.

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
        keys_to_remove = [k for k in data.keys() if k.startswith("spare")]
        keys_to_remove.extend([k for k in data.keys() if k.startswith("reserved")])
        for k in keys_to_remove:
            data.pop(k, None)

    # iterate over sub dictionary and unpack/read/decode
    for k, v in sub.items():
        if not rawdata:
            # read/decode data
            for k1 in ["read", "func"]:
                try:
                    # print("K/V:", k, v)
                    data[k] = v[k1](data[k], **v[k1[0] + "kw"])
                except KeyError:
                    pass
                except UnicodeDecodeError:
                    pass
        # unpack sub dictionary
        try:
            data[k] = _unpack_dictionary(data[k], v, rawdata=rawdata)
        except TypeError:
            pass

    return data


def _data_types_from_dsp_mask(words):
    """Return a list of the data types from the words in the data_type mask."""
    data_types = []
    for i, word in enumerate(words):
        data_types += [j + (i * 32) for j in range(32) if word >> j & 1]
    return data_types


def _check_product(product_type):
    """Return IRIS File Class depending on product type."""
    if product_type in [
        "MAX",
        "TOPS",
        "HMAX",
        "BASE",
        "THICK",
        "PPI",
        "RHI",
        "CAPPI",
        "RAINN",
        "RAIN1",
        "CROSS",
        "SHEAR",
        "SRI",
        "RTI",
        "VIL",
        "LAYER",
        "BEAM",
        "MLHGT",
    ]:
        return IrisCartesianProductFile
    elif product_type in [
        "CATCH",
        "FCAST",
        "NDOP",
        "SLINE",
        "TDWR",
        "TRACK",
        "VAD",
        "VVP",
        "WARN",
        "WIND",
        "STATUS",
    ]:
        return IrisProductFile
    elif product_type in ["RAW"]:
        return IrisRawFile
    else:
        return False


def _check_identifier(identifier):
    """Return IRIS File Class depending on identifier."""
    if identifier == "INGEST_HEADER":
        return IrisIngestHeaderFile
    elif identifier == "INGEST_DATA_HEADER":
        return IrisIngestDataFile
    elif identifier == "PRODUCT_HDR":
        return IrisRecordFile
    else:
        return False


# IRIS Data Structures
_STRING = {"read": decode_string, "rkw": {}}


def string_dict(size):
    """Return _STRING dictionary"""
    dic = _STRING.copy()
    dic["size"] = "{0}s".format(size)
    return dic


_ARRAY = {"read": np.frombuffer, "rkw": {}}


def array_dict(size, dtype):
    """Return _ARRAY dictionary"""
    dic = _ARRAY.copy()
    dic["size"] = "{0}s".format(size * get_dtype_size(dtype))
    dic["rkw"]["dtype"] = dtype
    return copy.deepcopy(dic)


# structure_header Structure
# 4.3.48 page 55
STRUCTURE_HEADER = OrderedDict(
    [
        ("structure_identifier", SINT2),
        ("format_version", SINT2),
        ("bytes_in_structure", SINT4),
        ("reserved", SINT2),
        ("flag", SINT2),
    ]
)

# ymds_time Structure
# 4.3.77, page 70
YMDS_TIME = OrderedDict(
    [
        ("seconds", SINT4),
        ("milliseconds", UINT2),
        ("year", SINT2),
        ("month", SINT2),
        ("day", SINT2),
    ]
)

LEN_YMDS_TIME = struct.calcsize(_get_fmt_string(YMDS_TIME))

_YMDS_TIME = {"size": "{}s".format(LEN_YMDS_TIME), "func": decode_time, "fkw": {}}

# product_specific_info Structure(s) _PSI_ with connected _RESULTS

# beam_psi_struct
# 4.3.1, page 24
BEAM_PSI_STRUCT = OrderedDict(
    [
        ("min_range", UINT4),
        ("max_range", UINT4),
        ("left_azimuth", BIN4),
        ("right_azimuth", BIN4),
        ("lower_elevation", BIN4),
        ("upper_elevation", BIN4),
        ("azimuth_smoothing", BIN4),
        ("elevation_smoothing", BIN4),
        ("az_of_sun_at_start", BIN4),
        ("el_of_sun_at_start", BIN4),
        ("az_of_sun_at_end", BIN4),
        ("el_of_sun_at_end", BIN4),
        ("spare_0", {"fmt": "32s"}),
    ]
)

# cappi_psi_struct (if CAPPI)
CAPPI_PSI_STRUCT = OrderedDict(
    [
        ("shear_flags", UINT4),
        ("cappi_height", SINT4),
        ("flags", UINT2),
        ("azimuth_shear_smoothing", BIN2),
        ("vvp_shear_correction_name", string_dict(12)),
        ("vvp_shear_correction_max_age", UINT4),
        ("spare_0", {"fmt": "52s"}),
    ]
)

# catch_psi_struct (if CATCH)
CATCH_PSI_STRUCT = OrderedDict(
    [
        ("flags", UINT4),
        ("hours_accumulation", UINT4),
        ("threshold_offset", SINT4),
        ("threshold faction", SINT4),
        ("rain1_name", string_dict(12)),
        ("catchment_file", string_dict(16)),
        ("seconds_accumulation", UINT4),
        ("rain1_min_z", UINT4),
        ("rain1_span_seconds", UINT4),
        ("average_gage_correction_factor", UINT4),
        ("spare_0", {"fmt": "20s"}),
    ]
)

# catch_results Struct
# 4.3.4, page 25
CATCH_RESULTS = OrderedDict(
    [
        ("catchment_area_name", string_dict(16)),
        ("catchment_area_number", UINT4),
        ("latitude_of_label", BIN4),
        ("longitude_of_label", BIN4),
        ("catchment_area", SINT4),
        ("num_pixels", SINT4),
        ("num_pixels_scanned", SINT4),
        ("flags_0", UINT4),
        ("rainfall_accumulation", UINT2),
        ("rainfall_accumulation_warning_threshold", UINT2),
        ("spare_0", {"fmt": "52s"}),
        ("input_rainfall_accumulation", array_dict(96, "uint16")),
    ]
)

# cross_psi_struct (if XSECT or USERV)
CROSS_PSI_STRUCT = OrderedDict(
    [
        ("azimuth_angle", BIN2),
        ("spare_0", {"fmt": "10s"}),
        ("east_center_coordinate", SINT4),
        ("north_center_coordinate", SINT4),
        ("user_miscellaneous", SINT4),
        ("spare_1", {"fmt": "56s"}),
    ]
)

# fcast_psi_struct (if FCAST)
FCAST_PSI_STRUCT = OrderedDict(
    [
        ("correlation_threshold", UINT4),
        ("data_threshold", SINT4),
        ("mean_speed", SINT4),
        ("mean_speed_direction", BIN4),
        ("max_time_between_inputs", UINT4),
        ("max_allowable_velocity", SINT4),
        ("flags", UINT4),
        ("output_resolution", SINT4),
        ("input_product_type", UINT4),
        ("input_product_name", string_dict(12)),
        ("spare_0", {"fmt": "32s"}),
    ]
)

# maximum_psi_struct (if MAX)
MAX_PSI_STRUCT = OrderedDict(
    [
        ("spare_0", {"fmt": "4s"}),
        ("interval_bottom", SINT4),
        ("interval_top", SINT4),
        ("side_panels_num_pixels", SINT4),
        ("side_panels_hor_smoother", SINT4),
        ("side_panels_ver_smoother", SINT4),
        ("spare_1", {"fmt": "56s"}),
    ]
)

# mlhgt_psi_struct (if MLGHT)
# 4.3.18, page 34
MLHGT_PSI_STRUCT = OrderedDict(
    [
        ("flags", UINT4),
        ("averaged_ml_altitude", SINT2),
        ("interval_ml_altitudes", SINT2),
        ("vertical_grid_spacing", SINT2),
        ("num_az_sectors", UINT2),
        ("relaxation_time_mlhgt_confidence", UINT4),
        ("modeled_fraction_melt_classifications", UINT2),
        ("modeled_fraction_nomelt_classifications", UINT2),
        ("min_confidence", UINT2),
        ("spare_0", {"fmt": "58s"}),
    ]
)

# ndop_input Struct
# 4.3.19, page 34
NDOP_INPUT = OrderedDict(
    [("task_name", string_dict(12)), ("site_code", string_dict(3)), ("flags_0", UINT1)]
)

# ndop_psi_struct (if NDOP)
# 4.3.20, page 35
NDOP_PSI_STRUCT = OrderedDict(
    [
        ("ndop_input_0", NDOP_INPUT),
        ("ndop_input_1", NDOP_INPUT),
        ("ndop_input_2", NDOP_INPUT),
        ("time_window", SINT4),
        ("cappi_height", SINT4),
        ("output_resolution", SINT4),
        ("min_permitted_crossing_angle", BIN4),
        ("flags_0", UINT4),
        ("output_site_code", string_dict(4)),
        ("spare_0", {"fmt": "8s"}),
    ]
)

# ndop_results Struct
# 4.3.20, page 35
NDOP_RESULTS = OrderedDict(
    [
        ("velocity_east", UINT2),
        ("velocity_north", UINT2),
        ("change_rate", UINT2),
        ("signal_quality_index", UINT1),
        ("spare_0", {"fmt": "5s"}),
    ]
)

# ppi_psi_struct (if PPI)
PPI_PSI_STRUCT = OrderedDict([("elevation_angle", BIN2), ("spare_0", {"fmt": "78s"})])

# rain_psi_struct (if RAIN1 or RAINN)
RAIN_PSI_STRUCT = OrderedDict(
    [
        ("min_Z_to_accumulate", UINT4),
        ("average_gage_correction_factor", UINT2),
        ("seconds_accumulation", UINT2),
        ("flags", UINT2),
        ("hours_to_accumulate", SINT2),
        ("input_product_name", string_dict(12)),
        ("span_of input files", UINT4),
        ("spare_0", {"fmt": "52s"}),
    ]
)

# raw_psi_struct (if RAW)
RAW_PSI_STRUCT = OrderedDict(
    [
        ("data_type_mask0", UINT4),
        ("range_last_bin", SINT4),
        ("format_conversion_flag", UINT4),
        ("flags", UINT4),
        ("sweep_number", SINT4),
        ("xhdr_type", UINT4),
        ("data_type_mask1", UINT4),
        ("data_type_mask2", UINT4),
        ("data_type_mask3", UINT4),
        ("data_type_mask4", UINT4),
        ("playback_version", UINT4),
        ("spare_0", {"fmt": "36s"}),
    ]
)

# rhi_psi_struct (if RHI)
RHI_PSI_STRUCT = OrderedDict([("azimuth_angle", BIN2), ("spare_0", {"fmt": "78s"})])

# rti_psi-struct
# 4.3.35, page 46
RTI_PSI_STRUCT = OrderedDict(
    [
        ("nominal_sweep_angle", BIN4),
        ("starting_time_offset", UINT4),
        ("ending_time_offset", UINT4),
        ("azimuth_first_ray", BIN4),
        ("*elevation_first_ray", BIN4),
        ("spare_0", {"fmt": "60s"}),
    ]
)

# shear_psi_struct (if SHEAR)
SHEAR_PSI_STRUCT = OrderedDict(
    [
        ("azimuthal_smoothing_angle", BIN4),
        ("elevation_angle", BIN2),
        ("spare_0", {"fmt": "2s"}),
        ("flags", UINT4),
        ("vvp_product_name", string_dict(12)),
        ("vvp_product_max_age", UINT4),
        ("spare_1", {"fmt": "52s"}),
    ]
)

# sline_psi_struct (if SLINE)
SLINE_PSI_STRUCT = OrderedDict(
    [
        ("area", SINT4),
        ("shear_threshold", SINT4),
        ("bit_flags_protected_areas", UINT4),
        ("max_forecast_time", SINT4),
        ("max_age_motion_calculation", UINT4),
        ("max_allowed_velocity", SINT4),
        ("flags", UINT4),
        ("azimuthal_smoothing_angle", BIN4),
        ("elevation_binary_angle1", BIN4),
        ("elevation_binary_angle2", BIN4),
        ("name_vvp_task", string_dict(12)),
        ("vvp_max_age", UINT4),
        ("curve_fit_std_threshold", SINT4),
        ("min_length_sline", UINT4),
        ("spare_0", {"fmt": "16s"}),
    ]
)

# tdwr_psi_struct (if TDWR)
TDWR_PSI_STRUCT = OrderedDict(
    [
        ("flags", UINT4),
        ("max_range", UINT4),
        ("source_id", string_dict(4)),
        ("center_field_wind_direction", string_dict(3)),
        ("spare_0", {"fmt": "1s"}),
        ("center_field_wind_speed", string_dict(2)),
        ("center_field_gust_speed", string_dict(2)),
        ("mask_protected_areas_checked", UINT4),
        ("warning_count", UINT4),
        ("sline_count", UINT4),
        ("spare_1", {"fmt": "48s"}),
    ]
)

# top_psi_struct (if TOPS, BASE, HMAX, or THICK)
TOP_PSI_STRUCT = OrderedDict(
    [("flags", UINT4), ("z_treshold", SINT2), ("spare_0", {"fmt": "74s"})]
)

# track_psi_struct (if TRACK)
TRACK_PSI_STRUCT = OrderedDict(
    [
        ("centroid_area_threshold", SINT4),
        ("centroid_threshold level", SINT4),
        ("protected_area_mask", UINT4),
        ("max_forecast_time", SINT4),
        ("max_age_motion_calculation", UINT4),
        ("max_motion_allowed", SINT4),
        ("flags", UINT4),
        ("max_span_track_points", SINT4),
        ("input_product_type", UINT4),
        ("input_product_name", string_dict(12)),
        ("point_connecting_error_allowance", SINT4),
        ("spare_0", {"fmt": "28s"}),
    ]
)

# track_results_struct
# 4.3.68, page 65
TRACK_RESULTS = OrderedDict(
    [
        ("latitude", BIN4),
        ("longitude", BIN4),
        ("height", SINT4),
        ("flags_0", UINT4),
        ("centroid_area", SINT4),
        ("equal_area_ellipse_major_axis", SINT4),
        ("equal_area_ellipse_minor_axis", SINT4),
        ("ellipse_orientation_angle", BIN4),
        ("protected_area_mask_of_areas_hit", UINT4),
        ("max_value_within_area", SINT4),
        ("spare_0", {"fmt": "8s"}),
        ("average_value_within_area", SINT4),
        ("spare_1", {"fmt": "8s"}),
        ("input_data_scale_factor", SINT4),
        ("track_index_number", SINT4),
        ("text", string_dict(32)),
        ("time", _YMDS_TIME),
        ("eta_protected_areas", array_dict(32, "int32")),
        ("input_data_type", UINT4),
        ("spare_2", {"fmt": "8s"}),
        ("propagation_speed", SINT4),
        ("propagation_direction", BIN4),
        ("text_size", UINT4),
        ("color", UINT4),
        ("spare_3", {"fmt": "32s"}),
    ]
)

# vad_psi_struct (if VAD)
VAD_PSI_STRUCT = OrderedDict(
    [
        ("min_slant_range", SINT4),
        ("max_slant_range", SINT4),
        ("flags", UINT4),
        ("number_elevation_angles", UINT4),
        ("spare_0", {"fmt": "64s"}),
    ]
)

# vad_results Structure
VAD_RESULTS = OrderedDict(
    [
        ("elevation_angle", BIN2),
        ("azimuth_angle", BIN2),
        ("num_bins_averaged", UINT2),
        ("average_velocity", UINT2),
        ("standard_deviation", UINT2),
        ("elevation_index", UINT2),
        ("spare_0", {"fmt": "8s"}),
    ]
)

# vil_psi_struct (if VIL)
# missing in documentation

# vvp_psi_struct (if VVP)
# 4.3.71, page 67
VVP_PSI_STRUCT = OrderedDict(
    [
        ("min_range", SINT4),
        ("max_range", SINT4),
        ("min_height", SINT4),
        ("max_height", SINT4),
        ("num_intervals", SINT2),
        ("min_velocity", UINT2),
        ("quota_num_bins_interval", SINT4),
        ("mask_wind_parameters", SINT4),
        ("spare_0", {"fmt": "52s"}),
    ]
)

# vvp_results Struct
# 4.3.72, page 67
VVP_RESULTS = OrderedDict(
    [
        ("number_data_points", SINT4),
        ("center_interval_height", SINT4),
        ("number_reflectivity_data_points", SINT4),
        ("spare_0", {"fmt": "8s"}),
        ("wind_speed", SINT2),
        ("wind_speed_std", SINT2),
        ("wind_direction", SINT2),
        ("wind_direction_std", SINT2),
        ("vertical_wind_speed", SINT2),
        ("vertical_wind_speed_std", SINT2),
        ("horizontal_divergence", SINT2),
        ("horizontal_divergence_std", SINT2),
        ("radial_velocity_std", SINT2),
        ("linear_averaged_reflectivity", SINT2),
        ("log_averaged_reflectivity_std", SINT2),
        ("deformation", SINT2),
        ("deformation_std", SINT2),
        ("axis_of_dilatation", SINT2),
        ("axis_of_dilatation_std", SINT2),
        ("log_averaged_reflectivity", SINT2),
        ("linear_averaged_reflectivity_std", SINT2),
        ("spare_0", {"fmt": "30s"}),
    ]
)

# warn_psi_struct (if WARN)
# 4.3.73, page 68
WARN_PSI_STRUCT = OrderedDict(
    [
        ("centroid_area_threshold", SINT4),
        ("threshold_levels", array_dict(3, "int32")),
        ("data_valid_times", array_dict(3, "int16")),
        ("spare_0", {"fmt": "2s"}),
        ("symbol_to_display", string_dict(12)),
        ("product_file_names", string_dict(36)),
        ("product_types", array_dict(3, "uint8")),
        ("spare_1", {"fmt": "1s"}),
        ("protected_area_flag", UINT4),
    ]
)

# warning_results Struct
# 4.3.74, page 68
WARNING_RESULTS = OrderedDict(
    [
        ("latitude", BIN4),
        ("longitude", BIN4),
        ("height", SINT4),
        ("flags_0", UINT4),
        ("centroid_area", SINT4),
        ("equal_area_ellipse_major_axis", SINT4),
        ("equal_area_ellipse_minor_axis", SINT4),
        ("ellipse_orientation_angle", BIN4),
        ("protected_area_mask_of_areas_hit", UINT4),
        ("max_value_within_area", array_dict(3, "int32")),
        ("average_value_within_area", array_dict(3, "int32")),
        ("input_data_scale_factor", SINT4),
        ("spare_0", {"fmt": "4s"}),
        ("text", string_dict(16)),
        ("spare_1", {"fmt": "156s"}),
        ("input_data_type", array_dict(3, "uint32")),
        ("propagation_speed", SINT4),
        ("propagation_direction", BIN4),
        ("spare_2", {"fmt": "40s"}),
    ]
)

# wind_psi_struct (if WIND)
# 4.3.75, page 69
WIND_PSI_STRUCT = OrderedDict(
    [
        ("min_height", SINT4),
        ("max_height", SINT4),
        ("min_range", SINT4),
        ("max_range", SINT4),
        ("num_range_points", SINT4),
        ("num_panel_points", SINT4),
        ("sector_length", SINT4),
        ("sector_width_binary_angle", BIN4),
        ("spare_0", {"fmt": "48s"}),
    ]
)

# wind_results Struct
# 4.3.76, page 70
WIND_RESULTS = OrderedDict(
    [
        ("num_possible_hits", SINT4),
        ("num_data_points_used", SINT4),
        ("center_sector_range", SINT4),
        ("center_sector_azimuth", BIN2),
        ("east_velocity", SINT2),
        ("east_velocity_std", SINT2),
        ("north_velocity", SINT2),
        ("north_velocity_std", SINT2),
        ("spare_0", {"fmt": "10s"}),
    ]
)


# status_antenna_info Structure
# 4.3.40, page 51
STATUS_ANTENNA_INFO = OrderedDict(
    [
        ("azimuth_position", BIN4),
        ("elevation_position", BIN4),
        ("azimuth_velocity", UINT4),
        ("elevation_velocity", UINT4),
        ("command_bits", UINT4),
        ("command_bit_availability_mask", UINT4),
        ("status_bits", UINT4),
        ("status_bit_availability_mask", UINT4),
        ("bite_fault_flag", UINT4),
        ("lowest_field_num_generating_fault", SINT4),
        ("status_bits_which_cause_critical_faults", UINT4),
        ("mask_bite_fields_state", array_dict(3, "uint32")),
        ("mask_bite_fields_faulted", array_dict(3, "uint32")),
        ("spare_0", {"fmt": "32s"}),
    ]
)

# status_message_info Structure
# 4.3.42, page 52
STATUS_MESSAGE_INFO = OrderedDict(
    [
        ("message_count", SINT4),
        ("message_number", MESSAGE),
        ("message_repeat_number", SINT4),
        ("process_name", string_dict(16)),
        ("message_text", string_dict(80)),
        ("signal_name", string_dict(32)),
        ("message_time", _YMDS_TIME),
        ("message_type", UINT4),
        ("spare_0", {"fmt": "36s"}),
    ]
)

# status_misc_info Structure
# 4.3.43, page 52
STATUS_MISC_INFO = OrderedDict(
    [
        ("radar_status_configuration_name", string_dict(16)),
        ("task_configuration_name", string_dict(16)),
        ("product_scheduler_configuration_name", string_dict(16)),
        ("product_output_configuration_name", string_dict(16)),
        ("active_task_name", string_dict(16)),
        ("active_product_name", string_dict(16)),
        ("site_type", UINT4),
        ("num_incoming_network_connects", SINT4),
        ("num_iris_clients_connected", SINT4),
        ("spare_0", {"fmt": "4s"}),
        ("num_output_devices", SINT4),
        ("flags", UINT4),
        ("node_status_fault_site", string_dict(4)),
        ("time_of_active_task", _YMDS_TIME),
        ("spare_1", {"fmt": "64s"}),
    ]
)

# status_one_device Structure
# 4.3.44, page 53
STATUS_ONE_DEVICE = OrderedDict(
    [
        ("device_type", UINT4),
        ("unit_number", SINT4),
        ("status", UINT4),
        ("spare_0", {"fmt": "4s"}),
        ("process_table_mode", UINT4),
        ("string", string_dict(16)),
        ("spare_1", {"fmt": "4s"}),
    ]
)

# status_device_info Structure
# 4.3.41, page 52
STATUS_DEVICE_INFO = OrderedDict(
    [
        ("dsp", STATUS_ONE_DEVICE),
        ("antenna", STATUS_ONE_DEVICE),
        ("output_device_0", STATUS_ONE_DEVICE),
        ("output_device_1", STATUS_ONE_DEVICE),
        ("output_device_2", STATUS_ONE_DEVICE),
        ("output_device_3", STATUS_ONE_DEVICE),
        ("output_device_4", STATUS_ONE_DEVICE),
        ("output_device_5", STATUS_ONE_DEVICE),
        ("output_device_6", STATUS_ONE_DEVICE),
        ("output_device_7", STATUS_ONE_DEVICE),
        ("output_device_8", STATUS_ONE_DEVICE),
        ("output_device_9", STATUS_ONE_DEVICE),
        ("output_device_10", STATUS_ONE_DEVICE),
        ("output_device_11", STATUS_ONE_DEVICE),
        ("output_device_12", STATUS_ONE_DEVICE),
        ("output_device_13", STATUS_ONE_DEVICE),
        ("output_device_14", STATUS_ONE_DEVICE),
        ("output_device_15", STATUS_ONE_DEVICE),
        ("output_device_16", STATUS_ONE_DEVICE),
        ("output_device_17", STATUS_ONE_DEVICE),
    ]
)

# status_one_process Structure
# 4.3.45, page 54
STATUS_ONE_PROCESS = OrderedDict(
    [("command", UINT4), ("mode", UINT4), ("spare_0", {"fmt": "12s"})]
)

# status_process_info Structure
# 4.3.46, page 54
STATUS_PROCESS_INFO = OrderedDict(
    [
        ("ingest_process", STATUS_ONE_PROCESS),
        ("ingfio_process", STATUS_ONE_PROCESS),
        ("spare_0", {"fmt": "20s"}),
        ("output_master_process", STATUS_ONE_PROCESS),
        ("product_process", STATUS_ONE_PROCESS),
        ("watchdog_process", STATUS_ONE_PROCESS),
        ("reingest_process", STATUS_ONE_PROCESS),
        ("network_process", STATUS_ONE_PROCESS),
        ("nordrad_process", STATUS_ONE_PROCESS),
        ("server_process", STATUS_ONE_PROCESS),
        ("ribbuild_process", STATUS_ONE_PROCESS),
        ("spare_1", {"fmt": "180s"}),
    ]
)

# status_results Structure
# 4.3.47, page 54
STATUS_RESULTS = OrderedDict(
    [
        ("status_misc_info", STATUS_MISC_INFO),
        ("status_process_info", STATUS_PROCESS_INFO),
        ("status_device_info", STATUS_DEVICE_INFO),
        ("status_antenna_info", STATUS_ANTENNA_INFO),
        ("status_message_info", STATUS_MESSAGE_INFO),
    ]
)

# spare (if USER, OTHER, TEXT)
SPARE_PSI_STRUCT = OrderedDict([("spare_0", {"fmt": "80s"})])

# one_protected_region Structure
# 4.3.22, page 35
ONE_PROTECTED_REGION = OrderedDict(
    [
        ("east_center", SINT4),
        ("north_center", SINT4),
        ("east_west_size", SINT4),
        ("north_south_size", SINT4),
        ("angle_of_orientation", BIN2),
        ("spare_0", {"fmt": "2s"}),
        ("region_name", string_dict(12)),
    ]
)

# protect_setup Structure
# 4.3.29, page 44
PROTECT_SETUP = OrderedDict([("one_protected_region", {"fmt", "1024s"})])

# color_scale_def Structure
# 4.3.5, page 26

COLOR_SCALE_DEF = OrderedDict(
    [
        ("iflags", UINT4),
        ("istart", SINT4),
        ("istep", SINT4),
        ("icolcnt", SINT2),
        ("iset_and_scale", UINT2),
        ("ilevel_seams", array_dict(16, "uint16")),
    ]
)

# product_configuration Structure
# 4.3.24, page 36

PRODUCT_CONFIGURATION = OrderedDict(
    [
        ("structure_header", STRUCTURE_HEADER),
        ("product_type_code", UINT2),
        ("scheduling_code", UINT2),
        ("seconds_between_runs", SINT4),
        ("generation_time", _YMDS_TIME),
        ("sweep_ingest_time", _YMDS_TIME),
        ("file_ingest_time", _YMDS_TIME),
        ("spare_0", {"fmt": "6s"}),
        ("product_name", string_dict(12)),
        ("task_name", string_dict(12)),
        ("flag", UINT2),
        ("x_scale", SINT4),
        ("y_scale", SINT4),
        ("z_scale", SINT4),
        ("x_size", SINT4),
        ("y_size", SINT4),
        ("z_size", SINT4),
        ("x_location", SINT4),
        ("y_location", SINT4),
        ("z_location", SINT4),
        ("maximum_range", SINT4),
        ("spare_1", {"fmt": "2s"}),
        ("data_type", UINT2),
        ("projection_name", string_dict(12)),
        ("input_data_type", UINT2),
        ("projection_type", UINT1),
        ("spare_2", {"fmt": "1s"}),
        ("radial_smoother", SINT2),
        ("times_run", SINT2),
        ("zr_constant", SINT4),
        ("zr_exponent", SINT4),
        ("x_smoother", SINT2),
        ("y_smoother", SINT2),
        ("product_specific_info", {"fmt": "80s"}),
        ("minor_task_suffixes", string_dict(16)),
        ("spare_3", {"fmt": "12s"}),
        ("color_scale_def", COLOR_SCALE_DEF),
    ]
)

# product_end Structure
# 4.3.25, page 39

PRODUCT_END = OrderedDict(
    [
        ("site_name", string_dict(16)),
        ("iris_version_created", string_dict(8)),
        ("ingest_iris_version", string_dict(8)),
        ("ingest_time", _YMDS_TIME),
        ("spare_0", {"fmt": "28s"}),
        ("GMT_minute_offset_local", SINT2),
        ("ingest_hardware_name_", string_dict(16)),
        ("ingest_site_name_", string_dict(16)),
        ("GMT_minute_offset_standard", SINT2),
        ("latitude", BIN4),
        ("longitude", BIN4),
        ("ground_height", SINT2),
        ("radar_height", SINT2),
        ("prf", SINT4),
        ("pulse_width", SINT4),
        ("signal_processor_type", UINT2),
        ("trigger_rate", UINT2),
        ("samples_used", SINT2),
        ("clutter_filter", string_dict(12)),
        ("number_linear_filter", UINT2),
        ("wavelength", SINT4),
        ("truncation_height", SINT4),
        ("first_bin_range", SINT4),
        ("last_bin_range", SINT4),
        ("number_bins", SINT4),
        ("flag", UINT2),
        ("number_ingest", SINT2),
        ("polarization", UINT2),
        ("horizontal_calibration_i0", SINT2),
        ("horizontal_calibration_noise", SINT2),
        ("horizontal_radar_constant", SINT2),
        ("receiver_bandwidth", UINT2),
        ("horizontal_current_noise", SINT2),
        ("vertical_current_noise", SINT2),
        ("ldr_offset", SINT2),
        ("zdr_offset", SINT2),
        ("tcf_cal_flags_1", UINT16_T),
        ("tcf_cal_flags_2", UINT16_T),
        ("spare_bit1", UINT1),
        ("spare_bit2", UINT1),
        ("spare_bit3", UINT1),
        ("spare_bit4", UINT1),
        ("spare_bit5", UINT1),
        ("spare_bit6", UINT1),
        ("spare_1", {"fmt": "12s"}),
        ("standard_parallel_1", BIN4),
        ("standard_parallel_2", BIN4),
        ("earth_radius", UINT4),
        ("inverse_flatting", UINT4),
        ("fault_status", UINT4),
        ("input_mask", UINT4),
        ("number_log_filter", UINT2),
        ("cluttermap", UINT2),
        ("latitude_projection", BIN4),
        ("longitude_projection", BIN4),
        ("product_sequence_number", SINT2),
        ("spare_2", {"fmt": "32s"}),
        ("melting_level", SINT2),
        ("radar_height_above_reference", SINT2),
        ("number_elements", SINT2),
        ("mean_wind_speed", UINT1),
        ("mean_wind_direction", BIN1),
        ("spare_3", {"fmt": "2s"}),
        ("tz_name", string_dict(8)),
        ("extended_product_header_offset", UINT4),
        ("spare_4", {"fmt": "4s"}),
    ]
)

# _product_hdr Structure
# 4.3.26 page 41
PRODUCT_HDR = OrderedDict(
    [
        ("structure_header", STRUCTURE_HEADER),
        ("product_configuration", PRODUCT_CONFIGURATION),
        ("product_end", PRODUCT_END),
    ]
)

# ingest_configuration Structure
# 4.3.14, page 31

INGEST_CONFIGURATION = OrderedDict(
    [
        ("filename", string_dict(80)),
        ("number_files", SINT2),
        ("number_sweeps_completed", SINT2),
        ("total_size", SINT4),
        ("volume_scan_start_time", _YMDS_TIME),
        ("spare_0", {"fmt": "12s"}),
        ("ray_header_bytes", SINT2),
        ("extended_ray_header_bytes", SINT2),
        ("number_task_config_table", SINT2),
        ("playback_version", SINT2),
        ("spare_1", {"fmt": "4s"}),
        ("iris_version", string_dict(8)),
        ("hardware_site", string_dict(16)),
        ("gmt_offset_minutes_local", SINT2),
        ("site_name", string_dict(16)),
        ("gmt_offset_minutes_standard", SINT2),
        ("latitude_radar", BIN4),
        ("longitude_radar", BIN4),
        ("height_site", SINT2),
        ("height_radar", SINT2),
        ("resolution_rays", UINT2),
        ("first_ray_index", UINT2),
        ("number_rays_sweep", UINT2),
        ("gparam_bytes", SINT2),
        ("altitude_radar", SINT4),
        ("velocity_east", SINT4),
        ("velocity_north", SINT4),
        ("velocity_up", SINT4),
        ("antenna_offset_starboard", SINT4),
        ("antenna_offset_bow", SINT4),
        ("antenna_offset_up", SINT4),
        ("fault_status", UINT4),
        ("melting_layer", SINT2),
        ("spare_2", {"fmt": "2s"}),
        ("local_timezone", string_dict(8)),
        ("flags", UINT4),
        ("configuration_name", string_dict(16)),
        ("spare_3", {"fmt": "228s"}),
    ]
)

# task_sched Structure
# 4.3.62, page 63

TASK_SCHED_INFO = OrderedDict(
    [
        ("start_time", SINT4),
        ("stop_time", SINT4),
        ("skip_time", SINT4),
        ("time_last_run", SINT4),
        ("time_used_last_run", SINT4),
        ("day_last_run", SINT4),
        ("flag", UINT2),
        ("spare_0", {"fmt": "94s"}),
    ]
)

# dsp_data_mask Structure
# 4.3.7, page 28

DSP_DATA_MASK = OrderedDict(
    [
        ("mask_word_0", UINT4),
        ("extended_header_type", UINT4),
        ("mask_word_1", UINT4),
        ("mask_word_2", UINT4),
        ("mask_word_3", UINT4),
        ("mask_word_4", UINT4),
    ]
)

# task_dsp_mode_batch Structure
# 4.3.53, page 59

TASK_DSP_MODE_BATCH = OrderedDict(
    [
        ("low_prf", UINT2),
        ("low_prf_fraction_part", UINT2),
        ("low_prf_sample_size", SINT2),
        ("low_prf_range_averaging_bins", SINT2),
        ("reflectivity_unfolding_threshold", SINT2),
        ("velocity_unfolding_threshold", SINT2),
        ("width_unfolding_threshold", SINT2),
        ("spare_0", {"fmt": "18s"}),
    ]
)

# task_dsp_info Structure
# 4.3.52, page 57f

TASK_DSP_INFO = OrderedDict(
    [
        ("major_mode", UINT2),
        ("dsp_type", UINT2),
        ("dsp_data_mask0", DSP_DATA_MASK),
        ("dsp_data_mask1", DSP_DATA_MASK),
        ("task_dsp_mode", TASK_DSP_MODE_BATCH),
        ("spare_0", {"fmt": "52s"}),
        ("prf", SINT4),
        ("pulse_width", SINT4),
        ("multi_prf_mode_flag", UINT2),
        ("dual_prf_delay", SINT2),
        ("agc_feedback_code", UINT2),
        ("sample_size", SINT2),
        ("gain_control_flag", UINT2),
        ("clutter_filter_name", string_dict(12)),
        ("linear_filter_num_first_bin", UINT1),
        ("log_filter_num_first_bin", UINT1),
        ("attenuation_fixed_gain", SINT2),
        ("gas_attenuation", UINT2),
        ("cluttermap_flag", UINT2),
        ("xmt_phase_sequence", UINT2),
        ("ray_header_config_mask", UINT4),
        ("playback_flags", UINT2),
        ("spare_1", {"fmt": "2s"}),
        ("custom_ray_header_name", string_dict(16)),
        ("spare_2", {"fmt": "120s"}),
    ]
)

# task_calib_info Structure
# 4.3.50, page 56f

TASK_CALIB_INFO = OrderedDict(
    [
        ("reflectivity_slope", SINT2),
        ("reflectivity_noise_threshold", SINT2),
        ("clutter_correction_threshold", SINT2),
        ("sqi_threshold", SINT2),
        ("power_threshold", SINT2),
        ("spare_0", {"fmt": "8s"}),
        ("calibration_reflectivity", SINT2),
        ("uncorrected_reflectivity_threshold_flags", UINT2),
        ("corrected_reflectivity_threshold_flags", UINT2),
        ("velocity_threshold_flags", UINT2),
        ("width_threshold_flags", UINT2),
        ("zdr_threshold_flags", UINT2),
        ("spare_1", {"fmt": "6s"}),
        ("flags_1", UINT2),
        ("spare_2", {"fmt": "2s"}),
        ("ldr_bias", SINT2),
        ("zdr_bias", SINT2),
        ("nexrad_point_clutter_threshold", SINT2),
        ("nexrad_point_clutter_bin_skip", UINT2),
        ("horizontal_io_cal_value", SINT2),
        ("vertical_io_cal_value", SINT2),
        ("horizontal_noise_calibration", SINT2),
        ("vertical_noise_calibration", SINT2),
        ("horizontal_radar_constant", SINT2),
        ("vertical_radar_constant", SINT2),
        ("receiver_bandwidth", SINT2),
        ("flags_2", UINT16_T),
        ("spare_3", {"fmt": "256s"}),
    ]
)

# task_range_info Structure
# 4.3.59, page 61

TASK_RANGE_INFO = OrderedDict(
    [
        ("range_first_bin", SINT4),
        ("range_last_bin", SINT4),
        ("number_input_bins", SINT2),
        ("number_output_bins", SINT2),
        ("step_input_bins", SINT4),
        ("step_output_bins", SINT4),
        ("variable_range_bin_spacing_flag", UINT2),
        ("range_bin_averaging_flag", SINT2),
        ("spare_0", {"fmt": "136s"}),
    ]
)

# task_rhi_scan_info Structure
# 4.3.60, page 61

_ANGLE_LIST = {
    "size": "80s",
    "read": np.frombuffer,
    "rkw": {"dtype": "uint16"},
    "func": decode_bin_angle,
    "fkw": {"mode": 2},
}

TASK_RHI_SCAN_INFO = OrderedDict(
    [
        ("lower_elevation_limit", UINT2),
        ("upper_elevation_limit", UINT2),
        ("list_of_azimuths", _ANGLE_LIST),
        ("spare_0", {"fmt": "115s"}),
        ("start_first_sector_sweep", UINT1),
    ]
)
# task_ppi_scan_info Structure
# 4.3.58, page 61

TASK_PPI_SCAN_INFO = OrderedDict(
    [
        ("left_azimuth_limit", BIN2),
        ("right_azimuth_limit", BIN2),
        ("list_of_elevations", _ANGLE_LIST),
        ("spare_0", {"fmt": "115s"}),
        ("start_first_sector_sweep", UINT1),
    ]
)

# task_file_scan_info Structure
# 4.3.55, page 60

TASK_FILE_SCAN_INFO = OrderedDict(
    [
        ("first_azimuth_angle", UINT2),
        ("first_elevation_angle", UINT2),
        ("filename_antenna_control", string_dict(12)),
        ("spare_0", {"fmt": "184s"}),
    ]
)

# task_manual_scan_info Structure
# 4.3.56, page 60

TASK_MANUAL_SCAN_INFO = OrderedDict([("flags", UINT2), ("spare_0", {"fmt": "198s"})])

# task_scan_info Structure
# 4.3.61, page 62

TASK_SCAN_INFO = OrderedDict(
    [
        ("antenna_scan_mode", UINT2),
        ("desired_angular_resolution", SINT2),
        ("spare_0", {"fmt": "2s"}),
        ("sweep_number", SINT2),
        ("task_type_scan_info", {"fmt": "200s"}),
        ("spare_1", {"fmt": "112s"}),
    ]
)

# task_misc_info Structure
# 4.3.57, page 60

TASK_MISC_INFO = OrderedDict(
    [
        ("wavelength", SINT4),
        ("tr_serial_number", string_dict(16)),
        ("transmit_power", SINT4),
        ("flags", UINT2),
        ("polarization_type", UINT2),
        ("truncation_height", SINT4),
        ("spare_0", {"fmt": "18s"}),
        ("spare_1", {"fmt": "12s"}),
        ("number_comment_bytes", SINT2),
        ("horizontal_beam_width", BIN4),
        ("vertical_beam_width", BIN4),
        ("customer_storage", {"fmt": "40s"}),
        ("spare_2", {"fmt": "208s"}),
    ]
)

# task_end_info Structure
# 4.3.54, page 59

TASK_END_INFO = OrderedDict(
    [
        ("task_major_number", SINT2),
        ("task_minor_number", SINT2),
        ("task_configuration_file_name", string_dict(12)),
        ("task_description", string_dict(80)),
        ("number_tasks", SINT4),
        ("task_state", UINT2),
        ("spare_0", {"fmt": "2s"}),
        ("task_data_time", _YMDS_TIME),
        ("echo_class_identifiers", {"fmt": "6s"}),
        ("spare_1", {"fmt": "198s"}),
    ]
)

# task_configuration Structure
# 4.3.51, page 57

TASK_CONFIGURATION = OrderedDict(
    [
        ("structure_header", STRUCTURE_HEADER),
        ("task_sched_info", TASK_SCHED_INFO),
        ("task_dsp_info", TASK_DSP_INFO),
        ("task_calib_info", TASK_CALIB_INFO),
        ("task_range_info", TASK_RANGE_INFO),
        ("task_scan_info", TASK_SCAN_INFO),
        ("task_misc_info", TASK_MISC_INFO),
        ("task_end_info", TASK_END_INFO),
        ("comments", string_dict(720)),
    ]
)

# _ingest_header Structure
# 4.3.16, page 33

INGEST_HEADER = OrderedDict(
    [
        ("structure_header", STRUCTURE_HEADER),
        ("ingest_configuration", INGEST_CONFIGURATION),
        ("task_configuration", TASK_CONFIGURATION),
        ("spare_0", {"fmt": "732s"}),
        ("gparm", {"fmt": "128s"}),
        ("reserved", {"fmt": "920s"}),
    ]
)

# raw_prod_bhdr Structure
# 4.3.31, page 45

RAW_PROD_BHDR = OrderedDict(
    [
        ("_record_number", SINT2),
        ("sweep_number", SINT2),
        ("first_ray_byte_offset", SINT2),
        ("sweep_ray_number", SINT2),
        ("flags", UINT2),
        ("spare", {"fmt": "2s"}),
    ]
)

# ingest_data_header Structure
# 4.3.15, page 32

INGEST_DATA_HEADER = OrderedDict(
    [
        ("structure_header", STRUCTURE_HEADER),
        ("sweep_start_time", _YMDS_TIME),
        ("sweep_number", SINT2),
        ("number_rays_per_sweep", SINT2),
        ("first_ray_index", SINT2),
        ("number_rays_file_expected", SINT2),
        ("number_rays_file_written", SINT2),
        ("fixed_angle", BIN2),
        ("bits_per_bin", SINT2),
        ("data_type", UINT2),
        ("spare_0", {"fmt": "36s"}),
    ]
)

# ray_header Structure
# 4.3.33, page 46

RAY_HEADER = OrderedDict(
    [
        ("azi_start", BIN2),
        ("ele_start", BIN2),
        ("azi_stop", BIN2),
        ("ele_stop", BIN2),
        ("rbins", SINT2),
        ("dtime", UINT2),
    ]
)

# some length's of data structures
LEN_STRUCTURE_HEADER = struct.calcsize(_get_fmt_string(STRUCTURE_HEADER))
LEN_PRODUCT_HDR = struct.calcsize(_get_fmt_string(PRODUCT_HDR))
LEN_INGEST_HEADER = struct.calcsize(_get_fmt_string(INGEST_HEADER))
LEN_RAW_PROD_BHDR = struct.calcsize(_get_fmt_string(RAW_PROD_BHDR))
LEN_INGEST_DATA_HEADER = struct.calcsize(_get_fmt_string(INGEST_DATA_HEADER))
LEN_RAY_HEADER = struct.calcsize(_get_fmt_string(RAY_HEADER))

# Sigmet structure header identifiers
# extracted from headers.h
STRUCTURE_HEADER_IDENTIFIERS = OrderedDict(
    [
        (20, {"name": "VERSION_STEP"}),
        (22, {"name": "TASK_CONFIGURATION"}),
        (23, {"name": "INGEST_HEADER"}),
        (24, {"name": "INGEST_DATA_HEADER"}),
        (25, {"name": "TAPE_INVENTORY"}),
        (26, {"name": "PRODUCT_CONFIGURATION"}),
        (27, {"name": "PRODUCT_HDR"}),
        (28, {"name": "TAPE_HEADER_RECORD"}),
    ]
)

STRUCTURE_HEADER_FORMAT_VERSION = OrderedDict(
    [
        (3, {"name": "INGEST_DATA_HEADER"}),
        (4, {"name": "INGEST_HEADER"}),
        (5, {"name": "TASK_CONFIGURATION"}),
        (6, {"name": "PRODUCT_CONFIGURATION"}),
        (8, {"name": "PRODUCT_HDR"}),
    ]
)

# Sigmet data types
# 4.9 Constants, Table 17

SIGMET_DATA_TYPES = OrderedDict(
    [
        # Extended Headers
        (0, {"name": "DB_XHDR", "func": None}),
        # Total H power (1 byte)
        (
            1,
            {
                "name": "DB_DBT",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -64.0},
            },
        ),
        # Clutter Corrected H reflectivity (1 byte)
        (
            2,
            {
                "name": "DB_DBZ",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -64.0},
            },
        ),
        # Velocity (1 byte)
        (
            3,
            {
                "name": "DB_VEL",
                "dtype": "uint8",
                "func": decode_vel,
                "fkw": {"scale": 127.0, "offset": -128.0, "mask": 0.0},
            },
        ),
        # Width (1 byte)
        (
            4,
            {
                "name": "DB_WIDTH",
                "dtype": "uint8",
                "func": decode_width,
                "fkw": {"scale": 256.0},
            },
        ),
        # Differential reflectivity (1 byte)
        (
            5,
            {
                "name": "DB_ZDR",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 16.0, "offset": -128.0},
            },
        ),
        # Old Rainfall rate (stored as dBZ), not used
        (6, {"name": "DB_ORAIN", "dtype": "uint8", "func": None}),
        # Fully corrected reflectivity (1 byte)
        (
            7,
            {
                "name": "DB_DBZC",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -64.0},
            },
        ),
        # Uncorrected reflectivity (2 byte)
        (
            8,
            {
                "name": "DB_DBT2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Corrected reflectivity (2 byte)
        (
            9,
            {
                "name": "DB_DBZ2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Velocity (2 byte)
        (
            10,
            {
                "name": "DB_VEL2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Width (2 byte)
        (
            11,
            {
                "name": "DB_WIDTH2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0},
            },
        ),
        # Differential reflectivity (2 byte)
        (
            12,
            {
                "name": "DB_ZDR2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Rainfall rate (2 byte)
        (
            13,
            {
                "name": "DB_RAINRATE2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 10000.0, "offset": -1.0, "tofloat": True},
            },
        ),
        # Kdp (specific differential phase)(1 byte)
        (14, {"name": "DB_KDP", "dtype": "int8", "func": decode_kdp, "fkw": {}}),
        # Kdp (specific differential phase)(2 byte)
        (
            15,
            {
                "name": "DB_KDP2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # PHIdp (differential phase)(1 byte)
        (
            16,
            {
                "name": "DB_PHIDP",
                "dtype": "uint8",
                "func": decode_phidp,
                "fkw": {"scale": 254.0, "offset": -1},
            },
        ),
        # Corrected Velocity (1 byte)
        (17, {"name": "DB_VELC", "dtype": "uint8", "func": None}),
        # SQI (1 byte)
        (
            18,
            {
                "name": "DB_SQI",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # RhoHV(0) (1 byte)
        (
            19,
            {
                "name": "DB_RHOHV",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # RhoHV(0) (2 byte)
        (
            20,
            {
                "name": "DB_RHOHV2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # Fully corrected reflectivity (2 byte)
        (
            21,
            {
                "name": "DB_DBZC2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Corrected Velocity (2 byte)
        (
            22,
            {
                "name": "DB_VELC2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # SQI (2 byte)
        (
            23,
            {
                "name": "DB_SQI2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # PHIdp (differential phase)(2 byte)
        (
            24,
            {
                "name": "DB_PHIDP2",
                "dtype": "uint16",
                "func": decode_phidp2,
                "fkw": {"scale": 65534.0, "offset": -1.0},
            },
        ),
        # LDR H to V (1 byte)
        (
            25,
            {
                "name": "DB_LDRH",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 5.0, "offset": -1.0, "offset2": -45.0},
            },
        ),
        # LDR H to V (2 byte)
        (
            26,
            {
                "name": "DB_LDRH2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # LDR V to H (1 byte)
        (
            27,
            {
                "name": "DB_LDRV",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 5.0, "offset": -1.0, "offset2": -45.0},
            },
        ),
        # LDR V to H (2 byte)
        (
            28,
            {
                "name": "DB_LDRV2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Individual flag bits for each bin
        (29, {"name": "DB_FLAGS", "func": None}),
        # (See bit definitions below)
        (30, {"name": "DB_FLAGS2", "func": None}),
        # Test of floating format
        (31, {"name": "DB_FLOAT32", "func": None}),
        # Height (1/10 km) (1 byte)
        (
            32,
            {
                "name": "DB_HEIGHT",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 10.0, "offset": -1.0},
            },
        ),
        # Linear liquid (.001mm) (2 byte)
        (
            33,
            {
                "name": "DB_VIL2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 1000.0, "offset": -1.0},
            },
        ),
        # Data type is not applicable
        (34, {"name": "DB_RAW", "func": None}),
        # Wind Shear (1 byte)
        (
            35,
            {
                "name": "DB_SHEAR",
                "dtype": "int8",
                "func": decode_array,
                "fkw": {"scale": 5.0, "offset": -128.0},
            },
        ),
        # Divergence (.001 10**-4) (2-byte)
        (
            36,
            {
                "name": "DB_DIVERGE2",
                "dtype": "int16",
                "func": decode_array,
                "fkw": {"scale": 10e-7},
            },
        ),
        # Floated liquid (2 byte)
        (
            37,
            {
                "name": "DB_FLIQUID2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 1000.0, "tofloat": True},
            },
        ),
        # User type, unspecified data (1 byte)
        (38, {"name": "DB_USER", "func": None}),
        # Unspecified data, no color legend
        (39, {"name": "DB_OTHER", "func": None}),
        # Deformation (.001 10**-4) (2-byte)
        (
            40,
            {
                "name": "DB_DEFORM2",
                "dtype": "int16",
                "func": decode_array,
                "fkw": {"scale": 10e-7},
            },
        ),
        # Vertical velocity (.01 m/s) (2-byte)
        (
            41,
            {
                "name": "DB_VVEL2",
                "dtype": "int16",
                "func": decode_array,
                "fkw": {"scale": 100.0},
            },
        ),
        # Horizontal velocity (.01 m/s) (2-byte)
        (42, {"name": "DB_HVEL2", "func": decode_array, "fkw": {"scale": 100.0}}),
        # Horizontal wind direction (.1 degree) (2-byte)
        (
            43,
            {
                "name": "DB_HDIR2",
                "dtype": "int16",
                "func": decode_array,
                "fkw": {"scale": 10.0},
            },
        ),
        # Axis of Dilation (.1 degree) (2-byte)
        (
            44,
            {
                "name": "DB_AXDIL2",
                "dtype": "int16",
                "func": decode_array,
                "fkw": {"scale": 10.0},
            },
        ),
        # Time of data (seconds) (2-byte)
        (
            45,
            {
                "name": "DB_TIME2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 60.0, "offset": -32768},
            },
        ),
        # Rho H to V (1 byte)
        (
            46,
            {
                "name": "DB_RHOH",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # Rho H to V (2 byte)
        (
            47,
            {
                "name": "DB_RHOH2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # Rho V to H (1 byte)
        (
            48,
            {
                "name": "DB_RHOV",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # Rho V to H (2 byte)
        (
            49,
            {
                "name": "DB_RHOV2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # Phi H to V (1 byte)
        (50, {"name": "DB_PHIH", "dtype": "uint8", "func": decode_phidp}),
        # Phi H to V (2 byte)
        (
            51,
            {
                "name": "DB_PHIH2",
                "dtype": "uint16",
                "func": decode_phidp2,
                "fkw": {"scale": 65534.0, "offset": -1.0},
            },
        ),
        # Phi V to H (1 byte)
        (52, {"name": "DB_PHIV", "dtype": "uint8", "func": decode_phidp}),
        # Phi V to H (2 byte)
        (
            53,
            {
                "name": "DB_PHIV2",
                "dtype": "uint16",
                "func": decode_phidp2,
                "fkw": {"scale": 65534.0, "offset": -1.0},
            },
        ),
        # User type, unspecified data (2 byte)
        (54, {"name": "DB_USER2", "dtype": "uint16", "func": None}),
        # Hydrometeor class (1 byte)
        (55, {"name": "DB_HCLASS", "dtype": "uint8", "func": None}),
        # Hydrometeor class (2 byte)
        (56, {"name": "DB_HCLASS2", "dtype": "uint16", "func": None}),
        # Corrected Differential reflectivity (1 byte)
        (
            57,
            {
                "name": "DB_ZDRC",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 16.0, "offset": -128.0},
            },
        ),
        # Corrected Differential reflectivity (2 byte)
        (
            58,
            {
                "name": "DB_ZDRC2",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Temperature (2 byte)
        (59, {"name": "DB_TEMPERATURE16", "dtype": "uint16", "func": None}),
        # Vertically Integrated Reflectivity (2 byte)
        (60, {"name": "DB_VIR16", "dtype": "uint16", "func": None}),
        # Total V Power (1 byte)
        (
            61,
            {
                "name": "DB_DBTV8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -64.0},
            },
        ),
        # Total V Power (2 byte)
        (
            62,
            {
                "name": "DB_DBTV16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Clutter Corrected V Reflectivity (1 byte)
        (
            63,
            {
                "name": "DB_DBZV8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -64.0},
            },
        ),
        # Clutter Corrected V Reflectivity (2 byte)
        (
            64,
            {
                "name": "DB_DBZV16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 100.0, "offset": -32768.0},
            },
        ),
        # Signal to Noise ratio (1 byte)
        (
            65,
            {
                "name": "DB_SNR8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Signal to Noise ratio (2 byte)
        (
            66,
            {
                "name": "DB_SNR16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Albedo (1 byte)
        (67, {"name": "DB_ALBEDO8", "dtype": "uint8", "func": None}),
        # Albedo (2 byte)
        (68, {"name": "DB_ALBEDO16", "dtype": "uint16", "func": None}),
        # VIL Density (2 byte)
        (69, {"name": "DB_VILD16", "dtype": "uint16", "func": None}),
        # Turbulence (2 byte)
        (70, {"name": "DB_TURB16", "dtype": "uint16", "func": None}),
        # Total Power Enhanced (via H+V or HV) (1 byte)
        (71, {"name": "DB_DBTE8", "dtype": "uint8", "func": None}),
        # Total Power Enhanced (via H+V or HV) (2 byte)
        (72, {"name": "DB_DBTE16", "dtype": "uint16", "func": None}),
        # Clutter Corrected Reflectivity Enhanced (1 byte)
        (73, {"name": "DB_DBZE8", "dtype": "uint8", "func": None}),
        # Clutter Corrected Reflectivity Enhanced (2 byte)
        (74, {"name": "DB_DBZE16", "dtype": "uint16", "func": None}),
        # Polarimetric meteo index (1 byte)
        (
            75,
            {
                "name": "DB_PMI8",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # Polarimetric meteo index (2 byte)
        (
            76,
            {
                "name": "DB_PMI16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # The log receiver signal-to-noise ratio (1 byte)
        (
            77,
            {
                "name": "DB_LOG8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # The log receiver signal-to-noise ratio (2 byte)
        (
            78,
            {
                "name": "DB_LOG16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Doppler channel clutter signal power (-CSR) (1 byte)
        (
            79,
            {
                "name": "DB_CSP8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Doppler channel clutter signal power (-CSR) (2 byte)
        (
            80,
            {
                "name": "DB_CSP16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Cross correlation, uncorrected rhohv (1 byte)
        (
            81,
            {
                "name": "DB_CCOR8",
                "dtype": "uint8",
                "func": decode_sqi,
                "fkw": {"scale": 253.0, "offset": -1},
            },
        ),
        # Cross correlation, uncorrected rhohv (2 byte)
        (
            82,
            {
                "name": "DB_CCOR16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 65536.0, "offset": -1.0},
            },
        ),
        # Attenuation of Zh (1 byte)
        (
            83,
            {
                "name": "DB_AH8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Attenuation of Zh (2 byte)
        (
            84,
            {
                "name": "DB_AH16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Attenuation of Zv (1 byte)
        (
            85,
            {
                "name": "DB_AV8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Attenuation of Zv (2 byte)
        (
            86,
            {
                "name": "DB_AV16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Attenuation of Zzdr (1 byte)
        (
            87,
            {
                "name": "DB_AZDR8",
                "dtype": "uint8",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
        # Attenuation of Zzdr (2 byte)
        (
            88,
            {
                "name": "DB_AZDR16",
                "dtype": "uint16",
                "func": decode_array,
                "fkw": {"scale": 2.0, "offset": -63.0},
            },
        ),
    ]
)


PRODUCT_DATA_TYPE_CODES = OrderedDict(
    [
        (0, {"name": "NULL", "struct": SPARE_PSI_STRUCT}),
        (1, {"name": "PPI", "struct": PPI_PSI_STRUCT}),
        (2, {"name": "RHI", "struct": RHI_PSI_STRUCT}),
        (3, {"name": "CAPPI", "struct": CAPPI_PSI_STRUCT}),
        (4, {"name": "CROSS", "struct": CROSS_PSI_STRUCT}),
        (5, {"name": "TOPS", "struct": TOP_PSI_STRUCT}),
        (
            6,
            {
                "name": "TRACK",
                "struct": TRACK_PSI_STRUCT,
                "result": TRACK_RESULTS,
                "protect_setup": True,
            },
        ),
        (7, {"name": "RAIN1", "struct": RAIN_PSI_STRUCT}),
        (8, {"name": "RAINN", "struct": RAIN_PSI_STRUCT}),
        (9, {"name": "VVP", "struct": VVP_PSI_STRUCT, "result": VVP_RESULTS}),
        (10, {"name": "VIL", "struct": SPARE_PSI_STRUCT}),
        (11, {"name": "SHEAR", "struct": SHEAR_PSI_STRUCT}),
        (
            12,
            {
                "name": "WARN",
                "struct": WARN_PSI_STRUCT,
                "result": WARNING_RESULTS,
                "protect_setup": True,
            },
        ),
        (13, {"name": "CATCH", "struct": CATCH_PSI_STRUCT, "result": CATCH_RESULTS}),
        (14, {"name": "RTI", "struct": RTI_PSI_STRUCT}),
        (15, {"name": "RAW", "struct": RAW_PSI_STRUCT}),
        (16, {"name": "MAX", "struct": MAX_PSI_STRUCT}),
        (17, {"name": "USER", "struct": SPARE_PSI_STRUCT}),
        (18, {"name": "USERV", "struct": CROSS_PSI_STRUCT}),
        (19, {"name": "OTHER", "struct": SPARE_PSI_STRUCT}),
        (20, {"name": "STATUS", "struct": SPARE_PSI_STRUCT, "result": STATUS_RESULTS}),
        (21, {"name": "SLINE", "struct": SLINE_PSI_STRUCT, "protect_setup": True}),
        (22, {"name": "WIND", "struct": WIND_PSI_STRUCT, "result": WIND_RESULTS}),
        (23, {"name": "BEAM", "struct": BEAM_PSI_STRUCT}),
        (24, {"name": "TEXT", "struct": SPARE_PSI_STRUCT}),
        (25, {"name": "FCAST", "struct": FCAST_PSI_STRUCT, "result": NDOP_RESULTS}),
        (26, {"name": "NDOP", "struct": NDOP_PSI_STRUCT, "result": NDOP_RESULTS}),
        (27, {"name": "IMAGE", "struct": SPARE_PSI_STRUCT}),
        (28, {"name": "COMP", "struct": SPARE_PSI_STRUCT}),
        (29, {"name": "TDWR", "struct": TDWR_PSI_STRUCT, "protect_setup": True}),
        (30, {"name": "GAGE", "struct": SPARE_PSI_STRUCT}),
        (31, {"name": "DWELL", "struct": SPARE_PSI_STRUCT}),
        (32, {"name": "SRI", "struct": SPARE_PSI_STRUCT}),
        (33, {"name": "BASE", "struct": TOP_PSI_STRUCT}),
        (34, {"name": "HMAX", "struct": TOP_PSI_STRUCT}),
        (35, {"name": "VAD", "struct": VAD_PSI_STRUCT, "result": VAD_RESULTS}),
        (36, {"name": "THICK", "struct": SPARE_PSI_STRUCT}),
        (37, {"name": "SATELLITE", "struct": SPARE_PSI_STRUCT}),
        (38, {"name": "LAYER", "struct": SPARE_PSI_STRUCT}),
        (39, {"name": "SWS", "struct": SPARE_PSI_STRUCT}),
        (40, {"name": "MLHGT", "struct": MLHGT_PSI_STRUCT}),
    ]
)


RECORD_BYTES = 6144


class IrisRecord(object):
    """Class holding a single record from a Sigmet IRIS file."""

    def __init__(self, record, recnum):
        """

        Parameters
        ----------
        record : array-like
            Slice into memory mapped file.
        recnum : int
        """
        self.record = record.copy()
        self._pos = 0
        self.recnum = recnum

    @property
    def pos(self):
        """Returns current byte offset."""
        return self._pos

    @pos.setter
    def pos(self, value):
        """Sets current byte offset."""
        self._pos = value

    @property
    def recpos(self):
        """Returns current word offset."""
        return int(self._pos / 2)

    @recpos.setter
    def recpos(self, value):
        """Sets current word offset."""
        self._pos = value * 2

    def read(self, words, width=2):
        """Reads from Record.

        Parameters
        ----------
        words : unsigned int
            Number of data words to be read.
        width : unsigned int
            Width (bytes) of data words to be read. Defaults to 2.

        Returns
        -------
        ret : array-like
        """
        ret = self.record[self._pos : self._pos + words * width]
        self.pos += words * width
        return ret


class IrisHeaderBase(object):
    """Base Class for Iris Headers."""

    def __init__(self, **kwargs):
        super(IrisHeaderBase, self).__init__()

    def init_header(self):
        pass


class IrisStructureHeader(IrisHeaderBase):
    """Iris Structure Header class."""

    len = LEN_STRUCTURE_HEADER
    structure = STRUCTURE_HEADER
    name = "_structure_header"

    def __init__(self, **kwargs):
        super(IrisStructureHeader, self).__init__(**kwargs)
        self._structure_header = None

    @property
    def structure_identifier(self):
        return STRUCTURE_HEADER_IDENTIFIERS[
            self._structure_header["structure_identifier"]
        ]["name"]

    @property
    def structure_format(self):
        return STRUCTURE_HEADER_FORMAT_VERSION[
            self._structure_header["format_version"]
        ]["name"]

    @property
    def structure_size(self):
        return self._structure_header["bytes_in_structure"]


class IrisIngestHeader(IrisHeaderBase):
    """Iris Ingest Header class."""

    len = LEN_INGEST_HEADER
    structure = INGEST_HEADER
    name = "_ingest_header"

    def __init__(self, **kwargs):
        super(IrisIngestHeader, self).__init__(**kwargs)
        self._ingest_header = None
        self._data_types_numbers = None

    @property
    def ingest_header(self):
        """Returns ingest_header dictionary."""
        return self._ingest_header

    @property
    def nsweeps(self):
        """Returns number of sweeps."""
        head = self._ingest_header["task_configuration"]["task_scan_info"]
        return head["sweep_number"]

    @property
    def nrays(self):
        """Returns number of rays."""
        return self._ingest_header["ingest_configuration"]["number_rays_sweep"]

    @property
    def data_types_dict(self):
        """Returns list of data type dictionaries."""
        return [
            SIGMET_DATA_TYPES.get(i, {"name": "DB_UNKNOWN_{}".format(i), "func": None})
            for i in self._data_types_numbers
        ]

    @property
    def data_types_count(self):
        """Returns number of data types."""
        return len(self._data_types_numbers)

    @property
    def data_types(self):
        """Returns list of data type names."""
        return [d["name"] for d in self.data_types_dict]

    def get_data_types_numbers(self):
        """Returns the available data types."""
        # determine the available fields
        task_config = self.ingest_header["task_configuration"]
        task_dsp_info = task_config["task_dsp_info"]
        word0 = task_dsp_info["dsp_data_mask0"]["mask_word_0"]
        word1 = task_dsp_info["dsp_data_mask0"]["mask_word_1"]
        word2 = task_dsp_info["dsp_data_mask0"]["mask_word_2"]
        word3 = task_dsp_info["dsp_data_mask0"]["mask_word_3"]

        return _data_types_from_dsp_mask([word0, word1, word2, word3])

    def get_task_type_scan_info(self, rawdata):
        """Retrieves task type info"""
        task_info = self.ingest_header["task_configuration"]["task_scan_info"]
        mode = task_info["antenna_scan_mode"]
        key = "task_type_scan_info"
        if mode in [1, 4]:
            task_info[key] = _unpack_dictionary(
                task_info[key], TASK_PPI_SCAN_INFO, rawdata
            )
        elif mode == 2:
            task_info[key] = _unpack_dictionary(
                task_info[key], TASK_RHI_SCAN_INFO, rawdata
            )
        elif mode == 3:
            task_info[key] = _unpack_dictionary(
                task_info[key], TASK_MANUAL_SCAN_INFO, rawdata
            )
        elif mode == 5:
            task_info[key] = _unpack_dictionary(
                task_info[key], TASK_FILE_SCAN_INFO, rawdata
            )
        else:
            pass

    def init_header(self, rawdata=False):
        self._data_types_numbers = self.get_data_types_numbers()
        self.get_task_type_scan_info(rawdata)


class IrisProductHeader(IrisHeaderBase):
    """Iris Product Header class."""

    len = LEN_PRODUCT_HDR
    structure = PRODUCT_HDR
    name = "_product_hdr"

    def __init__(self, **kwargs):
        super(IrisProductHeader, self).__init__(**kwargs)
        self._product_hdr = None
        self._product_type_code = None

    @property
    def product_hdr(self):
        """Returns ingest_header dictionary."""
        return self._product_hdr

    @property
    def nbins(self):
        """Returns number of bins."""
        return self._product_hdr["product_end"]["number_bins"]

    @property
    def product_type_code(self):
        """Returns product type code."""
        return self._product_type_code

    @property
    def product_type(self):
        """Returns product type."""
        return PRODUCT_DATA_TYPE_CODES[self.product_type_code]["name"]

    @property
    def product_type_dict(self):
        """Returns product type dictionary."""
        return PRODUCT_DATA_TYPE_CODES[self.product_type_code]

    @property
    def data_type(self):
        """Returns product configuration data type."""
        data_type = self.product_hdr["product_configuration"]["data_type"]
        return SIGMET_DATA_TYPES[data_type]

    def init_header(self, rawdata=False):
        self._product_type_code = self.get_product_type_code()
        self.get_product_specific_info(rawdata)

    def get_product_type_code(self):
        """Returns product type code."""
        prod_conf = self.product_hdr["product_configuration"]
        return prod_conf["product_type_code"]

    def get_product_specific_info(self, rawdata):
        """Retrieves product specific info"""
        config = self.product_hdr["product_configuration"]
        pt = self.product_type_dict
        key = "product_specific_info"
        try:
            config[key] = _unpack_dictionary(config[key], pt["struct"], rawdata)
        except KeyError:
            warnings.warn(
                "product type {0} not implemented, \n"
                "only header information "
                "available".format(pt["name"]),
                RuntimeWarning,
                stacklevel=3,
            )


class IrisIngestDataHeader(IrisHeaderBase):
    """Iris Ingest Data Header class."""

    len = LEN_INGEST_DATA_HEADER
    structure = INGEST_DATA_HEADER
    name = "_ingest_data_header"

    def __init__(self, **kwargs):
        super(IrisIngestDataHeader, self).__init__(**kwargs)
        self._ingest_data_header = None
        self._nrays_expected = None
        self._data_types_numbers = None
        self._rawdata = None

    @property
    def ingest_data_header(self):
        """Returns ingest_header dictionary."""
        return self._ingest_data_header

    @property
    def nrays(self):
        return self._nrays_expected

    @property
    def data_types_dict(self):
        """Returns list of data type dictionaries."""
        i = self._data_types_numbers

        return SIGMET_DATA_TYPES.get(
            i, {"name": "DB_UNKNOWN_{}".format(i), "func": None}
        )

    @property
    def data_types(self):
        """Returns list of data type names."""
        return self.data_types_dict["name"]

    def init_header(self):
        self._nrays_expected = self._ingest_data_header["number_rays_file_expected"]
        self._data_types_numbers = self.get_data_types_numbers()

    def get_data_types_numbers(self):
        """Returns the available data types."""
        return self.ingest_data_header["data_type"]


class IrisFileBase(object):
    """Base class for Iris Files."""

    def __init__(self, **kwargs):
        super(IrisFileBase, self).__init__()


class IrisFile(IrisFileBase, IrisStructureHeader):
    """IrisFile class"""

    identifier = ["PRODUCT_HDR", "INGEST_HEADER", "INGEST_DATA_HEADER"]

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)
        if isinstance(filename, str):
            self._fh = np.memmap(filename, mode="r")
        else:
            if isinstance(filename, io.BytesIO):
                filename = filename.read()
            self._fh = np.frombuffer(filename, dtype=np.uint8)
        self._filepos = 0
        self._data = None
        super(IrisFile, self).__init__(**kwargs)
        # read first structure header
        self.get_header(IrisStructureHeader)
        self._filepos = 0

    def check_identifier(self):
        if self.structure_identifier in self.identifier:
            return self.structure_identifier
        else:
            raise IOError(
                "Cannot read {0} with {1} class".format(
                    self.structure_identifier, self.__class__.__name__
                )
            )

    @property
    def data(self):
        return self._data

    @property
    def loaddata(self):
        """Returns `loaddata` switch."""
        return self._loaddata

    @property
    def rawdata(self):
        """Returns `rawdata` switch."""
        return self._rawdata

    @property
    def debug(self):
        return self._debug

    @property
    def fh(self):
        return self._fh

    @property
    def filepos(self):
        return self._filepos

    @filepos.setter
    def filepos(self, pos):
        self._filepos = pos

    def read_from_file(self, size):
        """Read from file.

        Parameters
        ----------
        words : int
            Number of data words to read.
        dtype : str
            dtype string specifying data format.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        start = self._filepos
        self._filepos += size
        return self._fh[start : self._filepos]

    def get_header(self, header):
        head = _unpack_dictionary(
            self.read_from_file(header.len), header.structure, self._rawdata
        )
        setattr(self, header.name, head)
        header.init_header(self)


class IrisIngestHeaderFile(IrisFile, IrisIngestHeader):
    """Iris Ingest Header File class."""

    identifier = "INGEST_HEADER"

    def __init__(self, filename, **kwargs):
        super(IrisIngestHeaderFile, self).__init__(filename=filename, **kwargs)
        self.check_identifier()
        self.get_header(IrisIngestHeader)


class IrisIngestDataFile(IrisFile, IrisIngestDataHeader):
    """Iris Ingest Data File class."""

    identifier = "INGEST_DATA_HEADER"

    def __init__(self, filename, **kwargs):
        super(IrisIngestDataFile, self).__init__(filename=filename, **kwargs)
        self.check_identifier()
        self.get_header(IrisIngestDataHeader)

        self.pointers = self.array_from_file(self.nrays, "int32")

        ingest_header = kwargs.get("ingest_header", None)
        if ingest_header:
            self.ingest_header = ingest_header
        else:
            raise TypeError("`ingest_header` missing in keyword parameters")

        if self.loaddata:
            self._data = self.get_sweep()

    def array_from_file(self, words, dtype):
        """Retrieve array from current record.

        Parameters
        ----------
        words : int
            Number of data words to read.
        width : int
            Size of the data word to read in bytes.
        dtype : str
            dtype string specifying data format.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        width = get_dtype_size(dtype)
        data = self.read_from_file(words * width)
        return data.view(dtype=dtype)

    def get_sweep(self):
        sweep_data = OrderedDict()
        sweep_prod = OrderedDict()
        sweep_prod["data"] = []
        sweep_prod["azi_start"] = []
        sweep_prod["ele_start"] = []
        sweep_prod["azi_stop"] = []
        sweep_prod["ele_stop"] = []
        sweep_prod["rbins"] = []
        sweep_prod["dtime"] = []

        start = self.filepos
        for i, p in enumerate(self.pointers):
            if not p:
                continue
            self.filepos = start - 1 + p
            ray_header = _unpack_dictionary(
                self.read_from_file(LEN_RAY_HEADER), RAY_HEADER, self.rawdata
            )
            data = self.decode_data(
                self.read_from_file(ray_header["rbins"]), self.data_types_dict
            )
            sweep_prod["data"].append(data)
            sweep_prod["azi_start"].append(ray_header["azi_start"])
            sweep_prod["ele_start"].append(ray_header["ele_start"])
            sweep_prod["azi_stop"].append(ray_header["azi_stop"])
            sweep_prod["ele_stop"].append(ray_header["ele_stop"])
            sweep_prod["rbins"].append(ray_header["rbins"])
            sweep_prod["dtime"].append(ray_header["dtime"])

        [sweep_prod.update({k: np.array(v)}) for k, v in sweep_prod.items()]

        sweep_data[self.data_types] = sweep_prod

        return sweep_data

    def decode_data(self, data, prod):
        """Decode data according given prod-dict.

        Parameters
        ----------
        data : data to decode
        prod : dict

        Returns
        -------
        data : decoded data

        """
        if self._rawdata:
            return data
        kw = {}
        if prod["func"]:
            try:
                kw.update(prod["fkw"])
            except KeyError:
                pass
            # this doesn't work for ingest data files
            # if get_dtype_size(prod['dtype']) == 1:
            #    dtype = '(2,) {0}'.format(prod['dtype'])
            # else:
            #    dtype = '{0}'.format(prod['dtype'])
            dtype = "{0}".format(prod["dtype"])
            try:
                rays, bins = data.shape
                data = data.view(dtype).reshape(rays, -1)[:, :bins]
            except ValueError:
                data = data.view(dtype)
            if prod["func"] in [decode_vel, decode_width, decode_kdp]:
                # wavelength is normally used from product_hdr
                # wavelength = self.product_hdr['product_end']['wavelength']
                # but we can retrieve it from TASK_MISC_INFO, too
                wavelength = self.ingest_header["task_configuration"]["task_misc_info"][
                    "wavelength"
                ]
                if prod["func"] == decode_kdp:
                    # get wavelength in cm
                    kw.update({"wavelength": wavelength / 100})
                    return prod["func"](data, **kw)
                # PRF is normally used from product_hdr
                # prf = self.product_hdr['product_end']['prf']
                # but we can retrieve it from TASK_DSP_INFO, too
                prf = self.ingest_header["task_configuration"]["task_dsp_info"]["prf"]
                # division by 10000 to get from 1/100 cm to m
                nyquist = wavelength * prf / (10000.0 * 4.0)
                if prod["func"] == decode_vel:
                    nyquist *= (
                        self.ingest_header["task_configuration"]["task_dsp_info"][
                            "multi_prf_mode_flag"
                        ]
                        + 1
                    )
                kw.update({"nyquist": nyquist})

            return prod["func"](data, **kw)
        else:
            return data


class IrisRecordFile(IrisFile, IrisProductHeader):
    """Iris Record File class"""

    identifier = ["PRODUCT_HDR"]
    product_identifier = [
        "MAX",
        "TOPS",
        "HMAX",
        "BASE",
        "THICK",
        "PPI",
        "RHI",
        "CAPPI",
        "RAINN",
        "RAIN1",
        "CROSS",
        "SHEAR",
        "SRI",
        "RTI",
        "VIL",
        "LAYER",
        "BEAM",
        "MLHGT",
        "CATCH",
        "FCAST",
        "NDOP",
        "SLINE",
        "TDWR",
        "TRACK",
        "VAD",
        "VVP",
        "WARN",
        "WIND",
        "STATUS",
        "RAW",
    ]

    def __init__(self, filename, **kwargs):
        super(IrisRecordFile, self).__init__(filename=filename, **kwargs)
        self._rh = None
        self._record_number = None
        self.get_header(IrisProductHeader)
        self.check_product_identifier()

    def check_product_identifier(self):
        if self.product_type in self.product_identifier:
            return self.product_type
        else:
            raise IOError(
                "Cannot read {0} with {1} class".format(
                    self.product_type, self.__class__.__name__
                )
            )

    @property
    def rh(self):
        """Returns current record object."""
        return self._rh

    @rh.setter
    def rh(self, value):
        """Sets current record object."""
        self._rh = value

    @property
    def record_number(self):
        """Returns current record number."""
        return self._record_number

    @record_number.setter
    def record_number(self, value):
        """Sets current record number."""
        self._record_number = value

    def _check_record(self):
        """Checks record for correct size.

        Need to be implemented in the derived classes
        """
        return True

    def init_record(self, recnum):
        """Initialize record using given number."""
        start = recnum * RECORD_BYTES
        stop = start + RECORD_BYTES
        self.record_number = recnum
        self.rh = IrisRecord(self.fh[start:stop], recnum)
        self.filepos = self.record_number * RECORD_BYTES
        return self._check_record()

    def init_next_record(self):
        """Get next record from file.

        This increases record_number count and initialises a new IrisRecord
        with the calculated start and stop file offsets.

        Returns
        -------
        chk : bool
            True, if record is truncated.
        """
        return self.init_record(self.record_number + 1)

    def array_from_record(self, words, width, dtype):
        """Retrieve array from current record.

        Parameters
        ----------
        words : int
            Number of data words to read.
        width : int
            Size of the data word to read in bytes.
        dtype : str
            dtype string specifying data format.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        return self.rh.read(words, width=width).view(dtype=dtype)

    def bytes_from_record(self, words, width):
        """Retrieve bytes from current record.

        Parameters
        ----------
        words : int
            Number of data words to read.
        width : int
            Size of the data word to read in bytes.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        return self.rh.read(words, width=width)

    def read_from_record(self, words, dtype):
        """Read from file.

        Parameters
        ----------
        words : int
            Number of data words to read.
        dtype : str
            dtype string specifying data format.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        width = get_dtype_size(dtype)
        data = self.array_from_record(words, width, dtype)
        words -= len(data)
        while words > 0:
            self.init_next_record()
            next_data = self.array_from_record(words, width, dtype)
            data = np.append(data, next_data)
            words -= len(next_data)
        return data


class IrisRawFile(IrisRecordFile, IrisIngestHeader):
    """Iris Raw File class."""

    product_identifier = ["RAW"]

    def __init__(self, filename, **kwargs):
        super(IrisRawFile, self).__init__(filename, **kwargs)

        self.check_product_identifier()

        self.init_record(1)
        self.get_header(IrisIngestHeader)

        # get RAW file specifics
        self._raw_product_bhdrs = []
        self._data = OrderedDict()
        if self.loaddata:
            self.get_data()
        else:
            self.get_data_headers()

    @property
    def raw_product_bhdrs(self):
        """Returns `raw_product_bhdrs` dictionary."""
        return self._raw_product_bhdrs

    def _check_record(self):
        """Checks record for correct size.

        Returns
        -------
        chk : bool
            False, if record is truncated.
        """
        # we do not know filesize before reading first record,
        # so we try and pass
        try:
            if self.record_number >= self.filesize / RECORD_BYTES:
                return False
        except AttributeError:
            pass
        chk = self._rh.record.shape[0] == RECORD_BYTES
        if not chk:
            raise EOFError(
                "Unexpected file end detected at " "record {}".format(self.rh.recnum)
            )
        return chk

    def read_from_record(self, words, dtype):
        """Read from record/file.

        Parameters
        ----------
        words : int
            Number of data words to read.
        dtype : str
            dtype string specifying data format.
        Returns
        -------
        data : array-like
            numpy array of data
        """
        width = get_dtype_size(dtype)
        data = self.array_from_record(words, width, dtype)
        words -= len(data)
        while words > 0:
            self.init_next_record()
            self.raw_product_bhdrs.append(self.get_raw_prod_bhdr())
            next_data = self.array_from_record(words, width, dtype)
            data = np.append(data, next_data)
            words -= len(next_data)
        return data

    def get_compression_code(self):
        """Read and return data compression code.

        Returns
        -------
        cmp_msb : bool
            True, if MSB is set.
        cmp_val : int
            Value of data compression code.
        """
        cmp_val = self.read_from_record(1, "int16")[0]
        cmp_msb = np.sign(cmp_val) == -1
        if cmp_msb:
            cmp_val = cmp_val + 32768
        if self._debug:
            print(
                "--- Sub CMP Code:",
                cmp_msb,
                cmp_val,
                self._rh.recpos - 1,
                self._rh.recpos,
            )
        return cmp_msb, cmp_val

    def get_raw_prod_bhdr(self):
        """Read and unpack raw product bhdr."""
        return _unpack_dictionary(
            self._rh.read(LEN_RAW_PROD_BHDR, width=1), RAW_PROD_BHDR, self._rawdata
        )

    def get_ingest_data_headers(self):
        """Read and return ingest data headers."""
        ingest_data_hdrs = OrderedDict()
        for i, dn in enumerate(self.data_types):
            ingest_data_hdrs[dn] = _unpack_dictionary(
                self._rh.read(LEN_INGEST_DATA_HEADER, width=1),
                INGEST_DATA_HEADER,
                self._rawdata,
            )

        return ingest_data_hdrs

    def get_ray(self, data):
        """Retrieve single ray.

        Returns
        -------
        data : array-like
            Numpy array containing data of one ray.
        """
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
                        "".format(
                            cmp_val, ray_pos, self._rh.recpos, self._rh.recpos + cmp_val
                        )
                    )
                data[ray_pos : ray_pos + cmp_val] = self.read_from_record(
                    cmp_val, "int16"
                )
            # compressed zeros follow
            # can be skipped, if data array is created all zeros
            else:
                if self._debug:
                    print(
                        "--- Add {0} Zeros at range {1}, record {2}:{3}:"
                        "".format(
                            cmp_val, ray_pos, self._rh.recpos, self._rh.recpos + 1
                        )
                    )
                if cmp_val + ray_pos > self.nbins + 6:
                    return data
                data[ray_pos : ray_pos + cmp_val] = 0

            ray_pos += cmp_val

            # read next compression code
            cmp_msb, cmp_val = self.get_compression_code()

        return data

    def get_sweep(self, moment):
        """Retrieve a single sweep.

        Parameters
        ----------
        moment : list of strings
            Data Types to retrieve.

        Returns
        -------
        sweep : OrderedDict
            Dictionary containing sweep data.
        """
        sweep = OrderedDict()

        sweep["ingest_data_hdrs"] = self.get_ingest_data_headers()

        # get boolean True for moment in available data_types
        skip = [
            True if k in moment else False for k in sweep["ingest_data_hdrs"].keys()
        ]

        # get rays per available data type
        rays_per_data_type = [
            d["number_rays_file_expected"] for d in sweep["ingest_data_hdrs"].values()
        ]

        # get rays per selected data type
        rays_per_selected_type = [
            d["number_rays_file_expected"] if k in moment else 0
            for k, d in sweep["ingest_data_hdrs"].items()
        ]

        # get available selected data types
        selected_type = []
        for i, k in enumerate(sweep["ingest_data_hdrs"].keys()):
            if k in moment:
                selected_type.append(self.data_types_dict[i])

        # get boolean True for selected available rays
        raylist = skip * rays_per_data_type[0]

        # get sum of rays for selected available data types
        rays = sum(rays_per_selected_type)
        bins = self._product_hdr["product_end"]["number_bins"]

        raw_data = np.zeros((rays, bins + 6), dtype="int16")
        single_data = np.zeros((bins + 6), dtype="int16")
        cnt = 0
        for i, ray_i in enumerate(raylist):
            if ray_i:
                ret = self.get_ray(raw_data[cnt])
                if ret is not None:
                    raw_data[cnt] = ret
                cnt += 1
            else:
                self.get_ray(single_data)

        sweep_data = OrderedDict()
        cnt = len(selected_type)
        for i, prod in enumerate(selected_type):
            sweep_prod = OrderedDict()
            sweep_prod["data"] = self.decode_data(raw_data[i::cnt, 6:], prod)
            sweep_prod["azi_start"] = self.decode_data(raw_data[i::cnt, 0], BIN2)
            sweep_prod["ele_start"] = self.decode_data(raw_data[i::cnt, 1], BIN2)
            sweep_prod["azi_stop"] = self.decode_data(raw_data[i::cnt, 2], BIN2)
            sweep_prod["ele_stop"] = self.decode_data(raw_data[i::cnt, 3], BIN2)
            sweep_prod["rbins"] = raw_data[i::cnt, 4]
            sweep_prod["dtime"] = raw_data[i::cnt, 5]
            sweep_data[prod["name"]] = sweep_prod

        sweep["sweep_data"] = sweep_data

        return sweep

    def decode_data(self, data, prod):
        """Decode data according given prod-dict.

        Parameters
        ----------
        data : data to decode
        prod : dict

        Returns
        -------
        data : decoded data

        """
        if self._rawdata:
            return data
        kw = {}
        if prod["func"]:
            try:
                kw.update(prod["fkw"])
            except KeyError:
                pass
            if get_dtype_size(prod["dtype"]) == 1:
                dtype = "(2,) {0}".format(prod["dtype"])
            else:
                dtype = "{0}".format(prod["dtype"])
            try:
                rays, bins = data.shape
                data = data.view(dtype).reshape(rays, -1)[:, :bins]
            except ValueError:
                data = data.view(dtype)
            if prod["func"] in [decode_vel, decode_width, decode_kdp]:
                wavelength = self.product_hdr["product_end"]["wavelength"]
                if prod["func"] == decode_kdp:
                    kw.update({"wavelength": wavelength / 100})
                    return prod["func"](data, **kw)

                prf = self.product_hdr["product_end"]["prf"]
                nyquist = wavelength * prf / (10000.0 * 4.0)
                if prod["func"] == decode_vel:
                    nyquist *= (
                        self.ingest_header["task_configuration"]["task_dsp_info"][
                            "multi_prf_mode_flag"
                        ]
                        + 1
                    )
                kw.update({"nyquist": nyquist})

            return prod["func"](data, **kw)
        else:
            return data

    def get_data(self):
        """Retrieve all sweeps from file."""
        dt_names = self.data_types  # [d['name'] for d in self.data_types]
        rsweeps = range(1, self.nsweeps + 1)

        loaddata = self.loaddata
        try:
            sweep = loaddata.copy().pop("sweep", rsweeps)
            moment = loaddata.copy().pop("moment", dt_names)
        except AttributeError:
            sweep = rsweeps
            moment = dt_names

        self.init_record(1)
        sw = 0
        ingest_conf = self.ingest_header["ingest_configuration"]
        sw_completed = ingest_conf["number_sweeps_completed"]
        while sw < sw_completed and self.init_next_record():
            raw_prod_bhdr = self.get_raw_prod_bhdr()
            sw = raw_prod_bhdr["sweep_number"]
            # continue to next record if not belonging to wanted sweeps
            if sw not in sweep:
                continue
            self.raw_product_bhdrs.append(raw_prod_bhdr)
            self._data[sw] = self.get_sweep(moment)

    def get_data_headers(self):
        """Retrieve all sweep `ingest_data_header` from file."""
        self.init_record(1)
        sw = 0
        ingest_conf = self.ingest_header["ingest_configuration"]
        sw_completed = ingest_conf["number_sweeps_completed"]
        while sw < sw_completed and self.init_next_record():
            # get raw_prod_bhdr
            raw_prod_bhdr = self.get_raw_prod_bhdr()
            # continue to next record if belonging to same sweep
            if raw_prod_bhdr["sweep_number"] == sw:
                continue
            # else set current sweep number
            else:
                sw = raw_prod_bhdr["sweep_number"]
            # read headers and add to _data
            self.raw_product_bhdrs.append(raw_prod_bhdr)
            sweep = OrderedDict()
            sweep["ingest_data_hdrs"] = self.get_ingest_data_headers()
            self._data[sw] = sweep


class IrisProductFile(IrisRecordFile):
    """Class for retrieving data from Sigmet IRIS Product files."""

    product_identifier = [
        "CATCH",
        "FCAST",
        "NDOP",
        "SLINE",
        "TDWR",
        "TRACK",
        "VAD",
        "VVP",
        "WARN",
        "WIND",
        "STATUS",
    ]

    def __init__(self, filename, **kwargs):
        """
        Parameters
        ----------
        irisfile : IrisWrapperFile class instance handle
            class instance handle
        """
        super(IrisProductFile, self).__init__(filename, **kwargs)

        self.check_product_identifier()
        self._protect_setup = None
        self._data = OrderedDict()
        if self.loaddata:
            self.get_data()

    @property
    def data(self):
        return self._data

    @property
    def protect_setup(self):
        return self._protect_setup

    def get_protect_setup(self):
        protected_setup = self.read_from_record(1024, "uint8")
        protected_regions = OrderedDict()
        for i in range(32):
            region = _unpack_dictionary(
                protected_setup[i * 32 : i * 32 + 32],
                ONE_PROTECTED_REGION,
                self._rawdata,
            )
            if not region["region_name"].isspace():
                protected_regions[i] = region

        return protected_regions

    def get_results(self, results, num, structure):
        cnt = struct.calcsize(_get_fmt_string(structure))
        for i in range(num):
            dta = self.read_from_record(cnt, "uint8")
            res = _unpack_dictionary(dta, structure, self._rawdata)
            results[i] = res

    def get_data(self):
        """Retrieves cartesian data from file."""
        # set filepointer accordingly
        self.init_record(0)
        self._rh.pos = 640
        if "protect_setup" in self.product_type_dict:
            self._protect_setup = self.get_protect_setup()
        product_end = self.product_hdr["product_end"]
        product_config = self.product_hdr["product_configuration"]
        num_elements = product_end["number_elements"]
        specific_info = product_config["product_specific_info"]

        result = OrderedDict()
        if self.product_type in ["FCAST", "NDOP"]:
            x_size = product_config.get("x_size")
            y_size = product_config.get("y_size")
            z_size = product_config.get("z_size", 1)

            cnt = struct.calcsize(_get_fmt_string(self.product_type_dict["result"]))
            z = []
            for zi in range(z_size):
                y = []
                for yi in range(y_size):
                    x = []
                    for xi in range(x_size):
                        dta = self.read_from_record(cnt, "uint8")
                        res = _unpack_dictionary(
                            dta, self.product_type_dict["result"], self._rawdata
                        )
                        x.append(res)
                    y.append(x)
                z.append(y)
            result[0] = np.array(z)
        # get vvp num_elements
        elif self.product_type in ["VVP"]:
            num_elements = specific_info["num_intervals"]
            self.get_results(result, num_elements, self.product_type_dict["result"])
        # get wind num_elements
        elif self.product_type in ["WIND"]:
            num_elements = (
                specific_info["num_panel_points"] * specific_info["num_range_points"]
            )
            cnt = struct.calcsize(_get_fmt_string(VVP_RESULTS))
            dta = self.read_from_record(cnt, "uint8")
            res = _unpack_dictionary(dta, VVP_RESULTS, self._rawdata)
            result["VVP"] = res
            self.get_results(result, num_elements, self.product_type_dict["result"])
        elif self.product_type in ["STATUS"]:
            self.get_results(result, num_elements, self.product_type_dict["result"])
        else:
            if num_elements:
                self.get_results(result, num_elements, self.product_type_dict["result"])
            else:
                warnings.warn(
                    "{0} - No product result "
                    "array(s) available".format(self.product_type),
                    RuntimeWarning,
                    stacklevel=3,
                )

        if self._protect_setup is not None:
            result["protect_setup"] = self._protect_setup

        self._data = result


class IrisCartesianProductFile(IrisRecordFile):
    """Class for retrieving data from Sigmet IRIS Cartesian Product files."""

    product_identifier = [
        "MAX",
        "TOPS",
        "HMAX",
        "BASE",
        "THICK",
        "PPI",
        "RHI",
        "CAPPI",
        "RAINN",
        "RAIN1",
        "CROSS",
        "SHEAR",
        "SRI",
        "RTI",
        "VIL",
        "LAYER",
        "BEAM",
        "MLHGT",
    ]

    def __init__(self, irisfile, **kwargs):
        """
        Parameters
        ----------
        irisfile : IrisWrapperFile class instance handle
            class instance handle
        """
        super(IrisCartesianProductFile, self).__init__(irisfile, **kwargs)

        self.check_product_identifier()

        self._data = OrderedDict()
        if self.loaddata:
            self.get_data()

    @property
    def data(self):
        return self._data

    def fix_ext_header(self, ext):
        prod_conf = self.product_hdr["product_configuration"]
        ext.update({"x_size": ext.pop("x_size", prod_conf.get("x_size"))})
        ext.update({"y_size": ext.pop("y_size", prod_conf.get("y_size"))})
        ext.update({"z_size": ext.pop("z_size", prod_conf.get("z_size", 1))})
        ext.update({"data_type": ext.pop("iris_type", prod_conf.get("data_type"))})

    def get_extended_header(self):
        # hack, get from actual position to end of record
        ext = self.rh.record[self.rh.pos :]
        if len(ext) == 0:
            return False
        # extended header token
        search = [0x00, 0xFF]
        ext = np.where((ext[:-1] == search[0]) & (ext[1:] == search[1]))[0][0]
        extended_header = OrderedDict([("extended_header", string_dict(ext))])
        ext_str = _unpack_dictionary(
            self.bytes_from_record(ext, 1), extended_header, self._rawdata
        )["extended_header"]
        # skip search bytes
        self.bytes_from_record(2, 1)
        ext_hdr = OrderedDict()
        for d in ext_str.split("\n"):
            kv = d.split("=")
            try:
                ext_hdr[kv[0]] = int(kv[1])
            except ValueError:
                ext_hdr[kv[0]] = kv[1]
            except Exception:
                pass
        self.fix_ext_header(ext_hdr)
        return ext_hdr

    def get_image(self, header):
        """Retrieve cartesian image.

        Parameters
        ----------
        header : dict
            header dictionary

        Returns
        -------
        data : :class:`numpy:numpy.ndarray`
            3D array of cartesian data

        """
        prod = SIGMET_DATA_TYPES[header.get("data_type")]
        x_size = header.get("x_size")
        y_size = header.get("y_size")
        z_size = header.get("z_size")
        cnt = x_size * y_size * z_size
        data = self.read_from_record(cnt, prod["dtype"])
        data = self.decode_data(data, prod=prod)
        data.shape = (z_size, y_size, x_size)
        return np.flip(data, axis=1)

    def get_data(self):
        """Retrieves cartesian data from file."""
        # set filepointer accordingly
        self.init_record(0)
        self.rh.pos = 640

        product_hdr = self.product_hdr
        product_end = product_hdr["product_end"]
        if product_hdr["product_end"]["number_elements"]:
            warnings.warn(
                "{0} Not Implemented - Product results "
                "array available \nnot loading "
                "dataset".format(self.product_type),
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            self._data[0] = self.get_image(product_hdr["product_configuration"])
            if product_end["extended_product_header_offset"]:
                ext_hdr = OrderedDict()
                i = 0
                ext = self.get_extended_header()
                while ext:
                    ext_hdr[i + 1] = ext
                    i += 1
                    self._data[i] = self.get_image(ext)
                    ext = self.get_extended_header()
                self.product_hdr["extended_header"] = ext_hdr

    def decode_data(self, data, prod):
        """Decode data according given prod-dict.

        Parameters
        ----------
        data : data to decode
        prod : dict

        Returns
        -------
        data : decoded data

        """
        if self._rawdata:
            return data
        kw = {}
        if prod["func"]:
            try:
                kw.update(prod["fkw"])
            except KeyError:
                pass
            if prod["func"] in [decode_vel, decode_width, decode_kdp]:
                wavelength = self.product_hdr["product_end"]["wavelength"]
                if prod["func"] == decode_kdp:
                    kw.update({"wavelength": wavelength / 100})
                    return prod["func"](data, **kw)

                prf = self.product_hdr["product_end"]["prf"]
                nyquist = wavelength * prf / (10000.0 * 4.0)
                kw.update({"nyquist": nyquist})
            return prod["func"](data.view(prod["dtype"]), **kw)
        else:
            return data


def read_iris(filename, loaddata=True, rawdata=False, debug=False, **kwargs):
    """Read Iris file and return dictionary.

    Parameters
    ----------
    filename : str or file-like
        Filename of data file or file-like object.
    loaddata : bool | kwdict
                If true, retrieves whole data section from file.
                If false, retrievs only ingest_data_headers, but no data.
                If kwdict, retrieves according to given kwdict::

                    loaddata = {'moment': ['DB_DBZ', 'DB_VEL'],
                                'sweep': [1, 3, 9]}

    rawdata : bool
        If true, returns raw unconverted/undecoded data.
    debug : bool
        If true, print debug messages.

    Returns
    -------
    data : OrderedDict
        Dictionary with data and metadata retrieved from file.
    """
    if not isinstance(filename, str):
        filename = filename.read()

    irisfile = IrisFile(filename)
    id = irisfile.check_identifier()
    ic = _check_identifier(irisfile.check_identifier())
    if id == "PRODUCT_HDR":
        irisfile = IrisRecordFile(filename)
        pi = irisfile.check_product_identifier()
        ic = _check_product(pi)

    if not ic:
        raise TypeError("Unknown File or Product Type {}".format(id))

    irisfile = ic(filename, loaddata=loaddata, rawdata=rawdata, debug=debug, **kwargs)

    properties = [
        "product_hdr",
        "product_type",
        "ingest_header",
        "ingest_data_header",
        "nrays_expected",
        "sweep",
        "nsweeps",
        "nrays",
        "nbins",
        "data_types",
        "data",
        "raw_product_bhdrs",
        "sweeps",
        "spare_0",
        "gparm",
    ]

    data = OrderedDict()
    for k in properties:
        item = getattr(irisfile, k, None)
        if item:
            data.update({k: item})

    return data
