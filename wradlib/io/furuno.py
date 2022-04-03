#!/usr/bin/env python
# Copyright (c) 2022, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Furuno binary Data I/O
^^^^^^^^^^^^^^^^^^^^^^

Reads data from Furuno's binary data formats

To read from Furuno files :class:`numpy:numpy.memmap` is used to get access to
the data. The Furuno header is read in any case into dedicated OrderedDict's.
Reading sweep data can be skipped by setting `loaddata=False`.
By default the data is decoded on the fly.
Using `rawdata=True` the data will be kept undecoded.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "FurunoFile",
    "open_furuno_dataset",
    "open_furuno_mfdataset",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import gzip
import io
import struct
from collections import OrderedDict

import numpy as np

# todo: move to something like "core" module
from wradlib.io.iris import (
    SINT2,
    SINT4,
    UINT1,
    UINT2,
    UINT4,
    _get_fmt_string,
    _unpack_dictionary,
)
from wradlib.io.xarray import (
    _calculate_angle_res,
    open_radar_dataset,
    open_radar_mfdataset,
    raise_on_missing_xarray_backend,
)
from wradlib.util import import_optional

lat_lon_parser = import_optional("lat_lon_parser")


def decode_time(data, time_struct=None):
    """Decode `YMDS_TIME` into datetime object."""
    time = _unpack_dictionary(data, time_struct)
    try:
        t = dt.datetime(
            time["year"],
            time["month"],
            time["day"],
            time["hour"],
            time["minute"],
            time["second"],
        )
        return t
    except ValueError:
        return None


def decode_geo_angle(data):
    angle = _unpack_dictionary(data, GEO_ANGLE)
    angle = lat_lon_parser.to_dec_deg(
        angle["degree"], angle["minute"], angle["second"] / 1000.0
    )
    return angle


def decode_altitude(data):
    alt = _unpack_dictionary(data, ALTITUDE)
    return alt["upper"] * 100 + alt["lower"] / 100


def decode_radar_constant(data):
    rc = _unpack_dictionary(data, RADAR_CONSTANT)
    return rc["mantissa"] * 10 ** rc["exponent"]


YMDS_TIME = OrderedDict(
    [
        ("year", UINT2),
        ("month", UINT1),
        ("day", UINT1),
        ("hour", UINT1),
        ("minute", UINT1),
        ("second", UINT1),
        ("spare", {"fmt": "1s"}),
    ]
)

YMDS2_TIME = OrderedDict(
    [
        ("year", UINT2),
        ("month", UINT2),
        ("day", UINT2),
        ("hour", UINT2),
        ("minute", UINT2),
        ("second", UINT2),
    ]
)

GEO_ANGLE = OrderedDict(
    [
        ("degree", SINT2),
        ("minute", UINT2),
        ("second", UINT2),
    ]
)

ALTITUDE = OrderedDict(
    [
        ("upper", UINT2),
        ("lower", UINT2),
    ]
)

RADAR_CONSTANT = OrderedDict(
    [
        ("mantissa", SINT4),
        ("exponent", SINT2),
    ]
)


LEN_YMDS_TIME = struct.calcsize(_get_fmt_string(YMDS_TIME))
LEN_YMDS2_TIME = struct.calcsize(_get_fmt_string(YMDS2_TIME))
LEN_GEO_ANGLE = struct.calcsize(_get_fmt_string(GEO_ANGLE))
LEN_ALTITUDE = struct.calcsize(_get_fmt_string(ALTITUDE))
LEN_RADAR_CONSTANT = struct.calcsize(_get_fmt_string(RADAR_CONSTANT))

_YMDS_TIME = {
    "size": f"{LEN_YMDS_TIME}s",
    "func": decode_time,
    "fkw": {"time_struct": YMDS_TIME},
}
_YMDS2_TIME = {
    "size": f"{LEN_YMDS2_TIME}s",
    "func": decode_time,
    "fkw": {"time_struct": YMDS2_TIME},
}
_GEO_ANGLE = {"size": f"{LEN_GEO_ANGLE}s", "func": decode_geo_angle, "fkw": {}}
_ALTITUDE = {"size": f"{LEN_ALTITUDE}s", "func": decode_altitude, "fkw": {}}
_RADAR_CONSTANT = {
    "size": f"{LEN_RADAR_CONSTANT}s",
    "func": decode_radar_constant,
    "fkw": {},
}

# Furuno Operator's Manual WR2120
# data file type 3 binary v10
# 7.3 pp. 61-66
HEADER_HEAD = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
    ]
)

SCNX_HEADER = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
        ("scan_start_time", _YMDS_TIME),
        ("scan_stop_time", _YMDS_TIME),
        ("time_zone", SINT2),
        ("product_number", UINT2),
        ("model_type", UINT2),
        ("latitude", SINT4),
        ("longitude", SINT4),
        ("altitude", SINT4),
        ("azimuth_offset", UINT2),
        ("tx_frequency", UINT4),
        ("polarization_mode", UINT2),
        ("antenna_gain_h", UINT2),
        ("antenna_gain_v", UINT2),
        ("half_power_beam_width_h", UINT2),
        ("half_power_beam_width_v", UINT2),
        ("tx_power_h", UINT2),
        ("tx_power_v", UINT2),
        ("radar_constant_h", SINT2),
        ("radar_constant_v", SINT2),
        ("noise_power_short_pulse_h", SINT2),
        ("noise_power_long_pulse_h", SINT2),
        ("threshold_power_short_pulse", SINT2),
        ("threshold_power_long_pulse", SINT2),
        ("tx_pulse_specification", UINT2),
        ("prf_mode", UINT2),
        ("prf_1", UINT2),
        ("prf_2", UINT2),
        ("prf_3", UINT2),
        ("nyquist_velocity", UINT2),
        ("sample_number", UINT2),
        ("tx_pulse_blind_length", UINT2),
        ("short_pulse_width", UINT2),
        ("short_pulse_modulation_bandwidth", UINT2),
        ("long_pulse_width", UINT2),
        ("long_pulse_modulation_bandwidth", UINT2),
        ("pulse_switch_point", UINT2),
        ("observation_mode", UINT2),
        ("antenna_rotation_speed", UINT2),
        ("number_sweep_direction_data", UINT2),
        ("number_range_direction_data", UINT2),
        ("resolution_range_direction", UINT2),
        ("current_scan_number", UINT2),
        ("total_number_scans_volume", UINT2),
        ("rainfall_intensity_estimation_method", UINT2),
        ("z_r_coefficient_b", UINT2),
        ("z_r_coefficient_beta", UINT2),
        ("kdp_r_coefficient_a", UINT2),
        ("kdp_r_coefficient_b", UINT2),
        ("kdp_r_coefficient_c", UINT2),
        ("zh_attenuation_correction_method", UINT2),
        ("zh_attenuation_coefficient_b1", UINT2),
        ("zh_attenuation_coefficient_b2", UINT2),
        ("zh_attenuation_coefficient_d1", UINT2),
        ("zh_attenuation_coefficient_d2", UINT2),
        ("air_attenuation_one_way", UINT2),
        ("output_threshold_rain", UINT2),
        ("record_item", UINT2),
        ("signal_processing_flag", UINT2),
        ("clutter_reference_file", _YMDS_TIME),
        ("reserved", {"fmt": "8s"}),
    ]
)

SCN_HEADER = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
        ("dpu_log_time", _YMDS2_TIME),
        ("latitude", _GEO_ANGLE),
        ("longitude", _GEO_ANGLE),
        ("altitude", _ALTITUDE),
        ("antenna_rotation_speed", UINT2),
        ("prf_1", UINT2),
        ("prf_2", UINT2),
        ("noise_level_pulse_modulation_h", SINT2),
        ("noise_level_frequency_modulation_h", SINT2),
        ("number_sweep_direction_data", UINT2),
        ("number_range_direction_data", UINT2),
        ("resolution_range_direction", UINT2),
        ("radar_constant_h", _RADAR_CONSTANT),
        ("radar_constant_v", _RADAR_CONSTANT),
        ("azimuth_offset", UINT2),
        ("scan_start_time", _YMDS2_TIME),
        ("record_item", UINT2),
        ("tx_pulse_blind_length", UINT2),
        ("tx_pulse_specification", UINT2),
    ]
)


class FurunoFile:
    """FurunoFile class"""

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)
        self._obsmode = kwargs.get("obsmode", None)
        self._fp = None
        self._filename = filename
        if isinstance(filename, str):
            if filename.endswith(".gz"):
                filename = gzip.open(filename)
        if isinstance(filename, str):
            self._fp = open(filename, "rb")
            self._fh = np.memmap(self._fp, mode="r")
        else:
            if isinstance(filename, (io.BytesIO, gzip.GzipFile)):
                filename.seek(0)
                filename = filename.read()
            self._fh = np.frombuffer(filename, dtype=np.uint8)
        self._filepos = 0
        self._data = None
        # read header
        len = struct.calcsize(_get_fmt_string(HEADER_HEAD))
        head = _unpack_dictionary(self.read_from_file(len), HEADER_HEAD)
        if head["format_version"] == 10:
            header = SCNX_HEADER
        elif head["format_version"] == 3:
            header = SCN_HEADER
        self._filepos = 0
        self.get_header(header)
        self._filepos = 0
        if self._loaddata:
            self.get_data()

    def get_data(self):
        if self._data is None:
            moments = [
                "RATE",
                "DBZH",
                "VRADH",
                "ZDR",
                "KDP",
                "PHIDP",
                "RHOHV",
                "WRADH",
                "QUAL",
                "RES1",
                "RES2",
                "RES3",
                "RES4",
                "RES5",
                "RES6",
                "FIX",
            ]
            # check available moments
            items = dict()
            for i in range(9):
                if (self.header["record_item"] & 2**i) == 2**i:
                    items[i] = moments[i]
            # claim available moments
            rays = self.header["number_sweep_direction_data"]
            rng = self.header["number_range_direction_data"]
            start = self.header["size_of_header"]
            cnt = len(items)
            raw_data = self._fh[start:].view(dtype="uint16").reshape(rays, -1)
            data = raw_data[:, 4:].reshape(rays, cnt, rng)
            self._data = dict()
            for i in range(cnt):
                self._data[items[i]] = data[:, i, :]
            # get angles
            angles = raw_data[:, :4].reshape(rays, 4)
            self._data["azimuth"] = np.fmod(
                angles[:, 1] + self.header["azimuth_offset"], 36000
            )
            if self.version == 3:
                dtype = "int16"
            else:
                dtype = "uint16"
            self._data["elevation"] = angles[:, 2].view(dtype=dtype)
        return self._data

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def header(self):
        """Returns ingest_header dictionary."""
        return self._header

    @property
    def version(self):
        return self.header["format_version"]

    @property
    def site_coords(self):
        if self.version == 3:
            lon = self.header["longitude"]
            lat = self.header["latitude"]
            alt = self.header["altitude"]
        else:
            lon = self.header["longitude"] * 1e-5
            lat = self.header["latitude"] * 1e-5
            alt = self.header["altitude"] * 1e-2
        return lon, lat, alt

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
    def filename(self):
        return self._filename

    @property
    def first_dimension(self):
        obs_mode = None
        if self.version == 3:
            # extract mode from filename
            if ".scn" in self.filename:
                obs_mode = 1
            elif ".sppi" in self.filename:
                obs_mode = 1
            elif ".rhi" in self.filename:
                obs_mode = 2
            elif isinstance(self.filename, io.BytesIO):
                if self._obsmode is None:
                    raise ValueError(
                        "Furuno `observation mode` can't be extracted from `io.BytesIO`. "
                        "Please use kwarg `obsmode=1` for PPI or `obsmode=2` "
                        "for RHI sweeps."
                    )
                obs_mode = self._obsmode
            else:
                pass
        else:
            obs_mode = self.header["observation_mode"]
        if obs_mode in [1, 3, 4]:
            return "azimuth"
        elif obs_mode == 2:
            return "elevation"
        else:
            raise TypeError(f"Unknown Furuno Observation Mode: {obs_mode}")

    @property
    def fixed_angle(self):
        dim = "azimuth" if self.first_dimension == "elevation" else "elevation"

        return self._data[dim][0] * 1e-2

    @property
    def a1gate(self):
        return np.argmin(self._data[self.first_dimension][::-1])

    @property
    def angle_resolution(self):
        return _calculate_angle_res(self._data[self.first_dimension] / 100.0)

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
        size : int
            Number of data words to read.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        start = self._filepos
        self._filepos += size
        return self._fh[start : self._filepos]

    def get_header(self, header):
        len = struct.calcsize(_get_fmt_string(header))
        self._header = _unpack_dictionary(
            self.read_from_file(len), header, self._rawdata
        )


def open_furuno_dataset(filename_or_obj, group=None, **kwargs):
    """Open and decode a Furuno radar sweep from a file or file-like object.

    This function uses :func:`~wradlib.io.open_radar_dataset`` under the hood.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.furuno.open_furuno_mfdataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_dataset(filename_or_obj, engine="furuno", **kwargs)


def open_furuno_mfdataset(filename_or_obj, group=None, **kwargs):
    """Open and decode a Furuno radar sweep from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_mfdataset` under the hood.
    Needs `dask` package to be installed.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    reindex_angle : bool or float
        Defaults to None (reindex angle with tol=0.4deg). If given a floating point
        number, it is used as tolerance. If False, no reindexing is performed.
        Only invoked if `decode_coord=True`.
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.furuno.open_furuno_dataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_mfdataset(filename_or_obj, engine="furuno", **kwargs)
