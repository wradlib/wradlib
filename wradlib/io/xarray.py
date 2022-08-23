#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray based Data I/O
^^^^^^^^^^^^^^^^^^^^^

Reads data from netcdf-based CfRadial1, CfRadial2,hdf5-based ODIM_H5 and
other hdf5-flavours (GAMIC), Iris/Sigmet and Rainbow5. More radar backends will be
implemented as needed.

Writes data to CfRadial2, ODIM_H5 or plain netCDF files.

This reader implementation uses

* `xarray <https://xarray.pydata.org/>`_,
* `netcdf4 <https://unidata.github.io/netcdf4-python/>`_,
* `h5py <https://www.h5py.org/>`_ and
* `h5netcdf <https://github.com/h5netcdf/h5netcdf>`_.

It utilizes the newly implemented :py:class:`xarray:xarray.backends.BackendEntrypoint`.
For every radar source (CfRadial1, CfRadial2, GAMIC, ODIM, IRIS, Rainbow5) a specific backend is
implemented in wradlib which returns an specific `sweep` as :py:class:`xarray:xarray.Dataset`.
Convenience functions (eg. :func:`wradlib.io.xarray.open_radar_dataset`) are available to read
volume data into shallow :class:`wradlib.io.xarray.RadarVolume`-wrapper.

Warning
-------
    This implementation is considered experimental. It will be based on CfRadial2, ODIM_H5
    and the new standard enforced by WMO JET-OWR `FM301 <https://community.wmo.int/wmo-jet-owr-seminar-series-weather-radar-data-exchange>.
    Changes in the API should be expected.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "WradlibVariable",
    "RadarVolume",
    "open_radar_dataset",
    "open_radar_mfdataset",
    "to_cfradial2",
    "to_odim",
    "to_netcdf",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import collections
import datetime as dt
import glob
import io
import os
import re
import warnings

import dateutil
import deprecation
import numpy as np
import xarray as xr
from packaging.version import Version
from xarray.core.variable import Variable

from wradlib import version
from wradlib.georef import xarray
from wradlib.util import has_import, import_optional

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(val, **kwargs):
        print(
            "wradlib: Please wait for completion of time consuming task! \n"
            "wradlib: Please install 'tqdm' for showing a progress bar "
            "instead."
        )
        return val


h5py = import_optional("h5py")
h5netcdf = import_optional("h5netcdf")
netCDF4 = import_optional("netCDF4")


def raise_on_missing_xarray_backend():
    """Raise errors if functionality isn't available."""
    if Version(xr.__version__) < Version("0.17.0"):
        raise ImportError(
            f"'xarray>=0.17.0' needed to perform this operation. "
            f"'xarray={xr.__version__}'  available.",
        )
    elif Version(xr.__version__) < Version("0.18.2"):
        xarray_backend_api = os.environ.get("XARRAY_BACKEND_API", None)
        if xarray_backend_api is None:
            os.environ["XARRAY_BACKEND_API"] = "v2"
        else:
            if xarray_backend_api != "v2":
                raise ValueError(
                    "Environment variable `XARRAY_BACKEND_API='v2'` needed to perform "
                    "this operation. "
                )
    else:
        pass


class WradlibVariable:
    """Minimal variable wrapper."""

    def __init__(self, dims, data, attrs):
        self._dimensions = dims
        self._data = data
        self._attrs = attrs

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def data(self):
        return self._data

    @property
    def attributes(self):
        return self._attrs


@deprecation.deprecated(
    deprecated_in="1.5",
    removed_in="2.0",
    current_version=version.version,
    details="Use `wradlib.georef.create_xarray_dataarray` " "instead.",
)
def create_xarray_dataarray(*args, **kwargs):
    return xarray.create_xarray_dataarray(*args, **kwargs)


moment_attrs = {"standard_name", "long_name", "units"}


iris_mapping = {
    "DB_DBT": "DBTH",
    "DB_DBT2": "DBTH",
    "DB_DBZ": "DBZH",
    "DB_DBZ2": "DBZH",
    "DB_VEL": "VRADH",
    "DB_VEL2": "VRADH",
    "DB_WIDTH": "WRADH",
    "DB_WIDTH2": "WRADH",
    "DB_ZDR": "ZDR",
    "DB_ZDR2": "ZDR",
    "DB_KDP": "KDP",
    "DB_KDP2": "KDP",
    "DB_PHIDP": "PHIDP",
    "DB_PHIDP2": "PHIDP",
    "DB_SQI": "SQIH",
    "DB_SQI2": "SQIH",
    "DB_SNR16": "SNRH",
    "DB_RHOHV": "RHOHV",
    "DB_RHOHV2": "RHOHV",
}


rainbow_mapping = {
    "dBuZ": "DBTH",
    "dBZ": "DBZH",
    "dBuZv": "DBTV",
    "dBZv": "DBZV",
    "V": "VRADH",
    "W": "WRADH",
    "ZDR": "ZDR",
    "KDP": "KDP",
    "PhiDP": "PHIDP",
    "SQI": "SQIH",
    "SNR": "SNR",
    "RhoHV": "RHOHV",
}

# CfRadial 2.0 - ODIM_H5 mapping
moments_mapping = {
    "DBZH": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Equivalent reflectivity factor H",
        "short_name": "DBZH",
        "units": "dBZ",
        "gamic": ["zh"],
    },
    "DBZH_CLEAN": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Equivalent reflectivity factor H",
        "short_name": "DBZH_CLEAN",
        "units": "dBZ",
        "gamic": None,
    },
    "DBZV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Equivalent reflectivity factor V",
        "short_name": "DBZV",
        "units": "dBZ",
        "gamic": ["zv"],
    },
    "ZH": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_h",
        "long_name": "Linear equivalent reflectivity factor H",
        "short_name": "ZH",
        "units": "unitless",
        "gamic": None,
    },
    "ZV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Linear equivalent reflectivity factor V",
        "short_name": "ZV",
        "units": "unitless",
        "gamic": None,
    },
    "DBZ": {
        "standard_name": "radar_equivalent_reflectivity_factor",
        "long_name": "Equivalent reflectivity factor",
        "short_name": "DBZ",
        "units": "dBZ",
        "gamic": None,
    },
    "DBTH": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Total power H (uncorrected reflectivity)",
        "short_name": "DBTH",
        "units": "dBZ",
        "gamic": ["uzh", "uh"],
    },
    "DBTV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Total power V (uncorrected reflectivity)",
        "short_name": "DBTV",
        "units": "dBZ",
        "gamic": ["uzv", "uv"],
    },
    "TH": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_h",
        "long_name": "Linear total power H (uncorrected reflectivity)",
        "short_name": "TH",
        "units": "unitless",
        "gamic": None,
    },
    "TV": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_v",
        "long_name": "Linear total power V (uncorrected reflectivity)",
        "short_name": "TV",
        "units": "unitless",
        "gamic": None,
    },
    "VRADH": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument_h",
        "long_name": "Radial velocity of scatterers away from instrument H",
        "short_name": "VRADH",
        "units": "meters per seconds",
        "gamic": ["vh"],
    },
    "VRADV": {
        "standard_name": "radial_velocity_of_scatterers_" "away_from_instrument_v",
        "long_name": "Radial velocity of scatterers away from instrument V",
        "short_name": "VRADV",
        "units": "meters per second",
        "gamic": ["vv"],
    },
    "VR": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument",
        "long_name": "Radial velocity of scatterers away from instrument",
        "short_name": "VR",
        "units": "meters per seconds",
        "gamic": None,
    },
    "VRAD": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument",
        "long_name": "Radial velocity of scatterers away from instrument",
        "short_name": "VRAD",
        "units": "meters per seconds",
        "gamic": None,
    },
    "VRADDH": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument_h",
        "long_name": "Radial velocity of scatterers away from instrument H",
        "short_name": "VRADDH",
        "units": "meters per seconds",
        "gamic": None,
    },
    "WRADH": {
        "standard_name": "radar_doppler_spectrum_width_h",
        "long_name": "Doppler spectrum width H",
        "short_name": "WRADH",
        "units": "meters per seconds",
        "gamic": ["wh"],
    },
    "UWRADH": {
        "standard_name": "radar_doppler_spectrum_width_h",
        "long_name": "Doppler spectrum width H",
        "short_name": "UWRADH",
        "units": "meters per seconds",
        "gamic": ["uwh"],
    },
    "WRADV": {
        "standard_name": "radar_doppler_spectrum_width_v",
        "long_name": "Doppler spectrum width V",
        "short_name": "WRADV",
        "units": "meters per second",
        "gamic": ["wv"],
    },
    "WRAD": {
        "standard_name": "radar_doppler_spectrum_width",
        "long_name": "Doppler spectrum width",
        "short_name": "WRAD",
        "units": "meters per second",
        "gamic": None,
    },
    "ZDR": {
        "standard_name": "radar_differential_reflectivity_hv",
        "long_name": "Log differential reflectivity H/V",
        "short_name": "ZDR",
        "units": "dB",
        "gamic": ["zdr"],
    },
    "UZDR": {
        "standard_name": "radar_differential_reflectivity_hv",
        "long_name": "Log differential reflectivity H/V",
        "short_name": "UZDR",
        "units": "dB",
        "gamic": ["uzdr"],
    },
    "LDR": {
        "standard_name": "radar_linear_depolarization_ratio",
        "long_name": "Log-linear depolarization ratio HV",
        "short_name": "LDR",
        "units": "dB",
        "gamic": ["ldr"],
    },
    "PHIDP": {
        "standard_name": "radar_differential_phase_hv",
        "long_name": "Differential phase HV",
        "short_name": "PHIDP",
        "units": "degrees",
        "gamic": ["phidp"],
    },
    "UPHIDP": {
        "standard_name": "radar_differential_phase_hv",
        "long_name": "Differential phase HV",
        "short_name": "UPHIDP",
        "units": "degrees",
        "gamic": ["uphidp"],
    },
    "KDP": {
        "standard_name": "radar_specific_differential_phase_hv",
        "long_name": "Specific differential phase HV",
        "short_name": "KDP",
        "units": "degrees per kilometer",
        "gamic": ["kdp"],
    },
    "RHOHV": {
        "standard_name": "radar_correlation_coefficient_hv",
        "long_name": "Correlation coefficient HV",
        "short_name": "RHOHV",
        "units": "unitless",
        "gamic": ["rhohv"],
    },
    "URHOHV": {
        "standard_name": "radar_correlation_coefficient_hv",
        "long_name": "Correlation coefficient HV",
        "short_name": "URHOHV",
        "units": "unitless",
        "gamic": ["urhohv"],
    },
    "SNRH": {
        "standard_name": "signal_noise_ratio_h",
        "long_name": "Signal Noise Ratio H",
        "short_name": "SNRH",
        "units": "unitless",
        "gamic": None,
    },
    "SNRV": {
        "standard_name": "signal_noise_ratio_v",
        "long_name": "Signal Noise Ratio V",
        "short_name": "SNRV",
        "units": "unitless",
        "gamic": None,
    },
    "SQIH": {
        "standard_name": "signal_quality_index_h",
        "long_name": "Signal Quality H",
        "short_name": "SQIH",
        "units": "unitless",
        "gamic": None,
    },
    "SQIV": {
        "standard_name": "signal_quality_index_v",
        "long_name": "Signal Quality V",
        "short_name": "SQIV",
        "units": "unitless",
        "gamic": None,
    },
    "CCORH": {
        "standard_name": "clutter_correction_h",
        "long_name": "Clutter Correction H",
        "short_name": "CCORH",
        "units": "unitless",
        "gamic": None,
    },
    "CCORV": {
        "standard_name": "clutter_correction_v",
        "long_name": "Clutter Correction V",
        "short_name": "CCORV",
        "units": "unitless",
        "gamic": None,
    },
    "CMAP": {
        "standard_name": "clutter_map",
        "long_name": "Clutter Map",
        "short_name": "CMAP",
        "units": "unitless",
        "gamic": ["cmap"],
    },
    "RATE": {
        "standard_name": "rainfall_rate",
        "long_name": "rainfall_rate",
        "short_name": "RATE",
        "units": "mm h-1",
        "gamic": None,
    },
}

ODIM_NAMES = {value["short_name"]: key for (key, value) in moments_mapping.items()}

GAMIC_NAMES = {
    v: key
    for (key, value) in moments_mapping.items()
    if value["gamic"] is not None
    for v in value["gamic"]
}

range_attrs = {
    "units": "meters",
    "standard_name": "projection_range_coordinate",
    "long_name": "range_to_measurement_volume",
    "spacing_is_constant": "true",
    "axis": "radial_range_coordinate",
    "meters_to_center_of_first_gate": None,
}

az_attrs_template = {
    "standard_name": "ray_azimuth_angle",
    "long_name": "azimuth_angle_from_true_north",
    "units": "degrees",
    "axis": "radial_azimuth_coordinate",
}

el_attrs_template = {
    "standard_name": "ray_elevation_angle",
    "long_name": "elevation_angle_from_horizontal_plane",
    "units": "degrees",
    "axis": "radial_elevation_coordinate",
}

time_attrs = {
    "standard_name": "time",
    "units": "seconds since 1970-01-01T00:00:00Z",
}

root_vars = {
    "volume_number",
    "platform_type",
    "instrument_type",
    "primary_axis",
    "time_coverage_start",
    "time_coverage_end",
    "latitude",
    "longitude",
    "altitude",
    "fixed_angle",
    "status_xml",
}

sweep_vars1 = {
    "sweep_number",
    "sweep_mode",
    "polarization_mode",
    "prt_mode",
    "follow_mode",
    "fixed_angle",
    "target_scan_rate",
    "sweep_start_ray_index",
    "sweep_end_ray_index",
}

sweep_vars2 = {
    "azimuth",
    "elevation",
    "pulse_width",
    "prt",
    "nyquist_velocity",
    "unambiguous_range",
    "antenna_transition",
    "n_samples",
    "r_calib_index",
    "scan_rate",
}

sweep_vars3 = {
    "DBZH",
    "DBZV",
    "VELH",
    "VELV",
    "DBZ",
    "VR",
    "time",
    "range",
    "reflectivity_horizontal",
}

cf_full_vars = {"prt": "prf", "n_samples": "pulse"}

global_attrs = [
    ("Conventions", "Cf/Radial"),
    ("version", "Cf/Radial version number"),
    ("title", "short description of file contents"),
    ("institution", "where the original data were produced"),
    (
        "references",
        ("references that describe the data or the methods used " "to produce it"),
    ),
    ("source", "method of production of the original data"),
    ("history", "list of modifications to the original data"),
    ("comment", "miscellaneous information"),
    ("instrument_name", "nameThe  of radar or lidar"),
    ("site_name", "name of site where data were gathered"),
    ("scan_name", "name of scan strategy used, if applicable"),
    ("scan_id", "scan strategy id, if applicable. assumed 0 if missing"),
    ("platform_is_mobile", '"true" or "false", assumed "false" if missing'),
    (
        "ray_times_increase",
        (
            '"true" or "false", assumed "true" if missing. '
            "This is set to true if ray times increase monotonically "
            "thoughout all of the sweeps in the volume"
        ),
    ),
    ("field_names", "array of strings of field names present in this file."),
    ("time_coverage_start", "copy of time_coverage_start global variable"),
    ("time_coverage_end", "copy of time_coverage_end global variable"),
    (
        "simulated data",
        (
            '"true" or "false", assumed "false" if missing. '
            "data in this file are simulated"
        ),
    ),
]

global_variables = dict(
    [
        ("volume_number", np.int_),
        ("platform_type", "fixed"),
        ("instrument_type", "radar"),
        ("primary_axis", "axis_z"),
        ("time_coverage_start", "1970-01-01T00:00:00Z"),
        ("time_coverage_end", "1970-01-01T00:00:00Z"),
        ("latitude", np.nan),
        ("longitude", np.nan),
        ("altitude", np.nan),
        ("altitude_agl", np.nan),
        ("sweep_group_name", (["sweep"], [np.nan])),
        ("sweep_fixed_angle", (["sweep"], [np.nan])),
        ("frequency", np.nan),
        ("status_xml", "None"),
    ]
)


def to_cfradial2(volume, filename, timestep=None, engine=None):
    """Save RadarVolume/XRadVol/XRadVolume to CfRadial2.0 compliant file.

    Parameters
    ----------
    volume : :class:`wradlib.io.xarray.RadarVolume`, :class:`wradlib.io.xarray.XRadVol` or :class:`wradlib.io.xarray.XRadVolume`
    filename : str
        output filename
    timestep : int
        timestep of wanted volume
    """
    if engine is None:
        if has_import(netCDF4):
            engine == "netcdf4"
        elif has_import(h5netcdf):
            engine == "h5netcdf"
        else:
            raise ImportError(
                "wradlib: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )
    volume.root.load()
    root = volume.root.copy(deep=True)
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    root.to_netcdf(filename, mode="w", group="/", engine=engine)
    for idx, key in enumerate(root.sweep_group_name.values):
        ds = volume[idx]
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        swp = ds.isel(time=timestep)
        swp.load()
        dim0 = list(set(swp.dims) & {"azimuth", "elevation"})[0]
        try:
            swp = swp.swap_dims({dim0: "time"})
        except ValueError:
            swp = swp.drop_vars("time").rename({"rtime": "time"})
            swp = swp.swap_dims({dim0: "time"})
        swp = swp.drop_vars(["x", "y", "z", "gr", "rays", "bins"], errors="ignore")
        swp = swp.sortby("time")
        swp.to_netcdf(filename, mode="a", group=key, engine=engine)


def to_netcdf(volume, filename, timestep=None, keys=None, engine=None):
    """Save RadarVolume/XRadVolume to netcdf compliant file.

    Parameters
    ----------
    volume : :class:`wradlib.io.xarray.RadarVolume`, :class:`wradlib.io.xarray.XRadVol` or :class:`wradlib.io.xarray.XRadVolume`
    filename : str
        output filename
    timestep : int, slice
        timestep/slice of wanted volume, defaults to full slice
    keys : list
        list of sweep_group_names which should be written to the file
    engine : str
        engine to save data, defaults to 'netcdf4' or 'h5netcdf' if
        found (in this order)
    """
    if engine is None:
        if has_import(netCDF4):
            engine == "netcdf4"
        elif has_import(h5netcdf):
            engine == "h5netcdf"
        else:
            raise ImportError(
                "wradlib: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )
    volume.root.load()
    root = volume.root.copy(deep=True)
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    root.to_netcdf(filename, mode="w", group="/", engine=engine)
    if keys is None:
        keys = root.sweep_group_name.values
    for idx, key in enumerate(root.sweep_group_name.values):
        if key in keys:
            try:
                swp = volume[idx].data.isel(time=timestep)
            except AttributeError:
                ds = volume[idx]
                if "time" not in ds.dims:
                    ds = ds.expand_dims("time")
                if timestep is None:
                    timestep = slice(None, None, None)
                swp = ds.isel(time=timestep)
            swp.to_netcdf(filename, mode="a", group=key, engine=engine)


def to_odim(volume, filename, timestep=0):
    """Save RadarVolume/XRadVol/XRadVolume to ODIM_H5/V2_2 compliant file.

    Parameters
    ----------
    volume : :class:`wradlib.io.xarray.RadarVolume`, :class:`wradlib.io.xarray.XRadVol` or :class:`wradlib.io.xarray.XRadVolume`
    filename : str
        output filename
    timestep : int
        timestep of wanted volume
    """
    root = volume.root

    h5 = h5py.File(filename, "w")

    # root group, only Conventions for ODIM_H5
    _write_odim({"Conventions": "ODIM_H5/V2_2"}, h5)

    # how group
    how = {}
    how.update({"_modification_program": "wradlib"})

    h5_how = h5.create_group("how")
    _write_odim(how, h5_how)

    sweepnames = root.sweep_group_name.values

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
    what = {}
    if len(sweepnames) > 1:
        what["object"] = "PVOL"
    else:
        what["object"] = "SCAN"
    what["version"] = "H5rad 2.2"
    what["date"] = str(root.time_coverage_start.values)[:10].replace("-", "")
    what["time"] = str(root.time_coverage_end.values)[11:19].replace(":", "")
    what["source"] = root.attrs["instrument_name"]

    h5_what = h5.create_group("what")
    _write_odim(what, h5_what)

    # where group, lon, lat, height, mandatory
    where = {
        "lon": root.longitude.values,
        "lat": root.latitude.values,
        "height": root.altitude.values,
    }
    h5_where = h5.create_group("where")
    _write_odim(where, h5_where)

    # datasets
    ds_list = [f"dataset{i + 1}" for i in range(len(sweepnames))]
    ds_idx = np.argsort(ds_list)
    for idx in ds_idx:
        ds = volume[idx]
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        ds = ds.isel(time=timestep, drop=True)
        ds = ds.drop_vars("time", errors="ignore").rename({"rtime": "time"})
        h5_dataset = h5.create_group(ds_list[idx])

        # what group p. 21 ff.
        h5_ds_what = h5_dataset.create_group("what")
        ds_what = {}
        # skip NaT values
        valid_times = ~np.isnat(ds.time.values)
        t = sorted(ds.time.values[valid_times])
        start = dt.datetime.utcfromtimestamp(np.rint(t[0].astype("O") / 1e9))
        end = dt.datetime.utcfromtimestamp(np.rint(t[-1].astype("O") / 1e9))
        ds_what["product"] = "SCAN"
        ds_what["startdate"] = start.strftime("%Y%m%d")
        ds_what["starttime"] = start.strftime("%H%M%S")
        ds_what["enddate"] = end.strftime("%Y%m%d")
        ds_what["endtime"] = end.strftime("%H%M%S")
        _write_odim(ds_what, h5_ds_what)

        # where group, p. 11 ff. mandatory
        h5_ds_where = h5_dataset.create_group("where")
        rscale = ds.range.values[1] / 1.0 - ds.range.values[0]
        rstart = (ds.range.values[0] - rscale / 2.0) / 1000.0
        # todo: make this work for RHI's
        a1gate = np.argsort(ds.sortby("azimuth").time.values)[0]
        try:
            fixed_angle = ds.fixed_angle
        except AttributeError:
            fixed_angle = ds.elevation.round(decimals=1).median().values
        ds_where = {
            "elangle": fixed_angle,
            "nbins": ds.range.shape[0],
            "rstart": rstart,
            "rscale": rscale,
            "nrays": ds.azimuth.shape[0],
            "a1gate": a1gate,
        }
        _write_odim(ds_where, h5_ds_where)

        # how group, p. 14 ff.
        h5_ds_how = h5_dataset.create_group("how")
        tout = [tx.astype("O") / 1e9 for tx in ds.sortby("azimuth").time.values]
        tout_sorted = sorted(tout)

        # handle non-uniform times (eg. only second-resolution)
        if np.count_nonzero(np.diff(tout_sorted)) < (len(tout_sorted) - 1):
            tout = np.roll(
                np.linspace(tout_sorted[0], tout_sorted[-1], len(tout)), a1gate
            )
            tout_sorted = sorted(tout)

        difft = np.diff(tout_sorted) / 2.0
        difft = np.insert(difft, 0, difft[0])
        azout = ds.sortby("azimuth").azimuth
        diffa = np.diff(azout) / 2.0
        diffa = np.insert(diffa, 0, diffa[0])
        elout = ds.sortby("azimuth").elevation
        diffe = np.diff(elout) / 2.0
        diffe = np.insert(diffe, 0, diffe[0])
        try:
            sweep_number = ds.sweep_number + 1
        except AttributeError:
            sweep_number = timestep
        ds_how = {
            "scan_index": sweep_number,
            "scan_count": len(sweepnames),
            "startazT": tout - difft,
            "stopazT": tout + difft,
            "startazA": azout - diffa,
            "stopazA": azout + diffa,
            "startelA": elout - diffe,
            "stopelA": elout + diffe,
        }
        _write_odim(ds_how, h5_ds_how)

        # write moments
        _write_odim_dataspace(ds, h5_dataset)

    h5.close()


def _calculate_angle_res(dim):
    # need to sort dim first
    angle_diff = np.diff(sorted(dim))
    angle_diff2 = np.abs(np.diff(angle_diff))

    # only select angle_diff, where angle_diff2 is less than 0.1 deg
    # Todo: currently 0.05 is working in most cases
    #  make this robust or parameterisable
    angle_diff_wanted = angle_diff[:-1][angle_diff2 < 0.05]
    return np.round(np.nanmean(angle_diff_wanted), decimals=2)


def _maybe_decode(attr):
    try:
        return attr.item().decode()
    except AttributeError:
        return attr


class _OdimH5NetCDFMetadata:
    """Wrapper around OdimH5 data fileobj for easy access of metadata.

    Parameters
    ----------
    fileobj : file-like
        h5netcdf filehandle.
    group : str
        odim group to acquire

    Returns
    -------
    object : metadata object
    """

    def __init__(self, fileobj, group):
        self._root = fileobj
        self._group = group

    @property
    def first_dim(self):
        dim, _ = self._get_fixed_dim_and_angle()
        return dim

    def get_variable_dimensions(self, dims):
        dimensions = []
        for n, _ in enumerate(dims):
            if n == 0:
                dimensions.append(self.first_dim)
            elif n == 1:
                dimensions.append("range")
            else:
                pass
        return tuple(dimensions)

    @property
    def coordinates(self):
        azimuth = self.azimuth
        elevation = self.elevation
        a1gate = self.a1gate
        rtime = self.ray_times
        dim, angle = self.fixed_dim_and_angle
        angle_res = _calculate_angle_res(locals()[dim])
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        az_attrs = az_attrs_template.copy()
        el_attrs = el_attrs_template.copy()
        az_attrs["a1gate"] = a1gate

        if dim == "azimuth":
            az_attrs["angle_res"] = angle_res
        else:
            el_attrs["angle_res"] = angle_res

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"

        rtime_attrs = {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "standard_name": "time",
        }

        range_data, cent_first, bin_range = self.range
        range_attrs["meters_to_center_of_first_gate"] = cent_first
        range_attrs["meters_between_gates"] = bin_range

        lon_attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
            "standard_name": "longitude",
        }
        lat_attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        }
        alt_attrs = {
            "long_name": "altitude",
            "units": "meters",
            "standard_name": "altitude",
        }

        lon, lat, alt = self.site_coords

        coordinates = {
            "azimuth": Variable((dims[0],), azimuth, az_attrs),
            "elevation": Variable((dims[0],), elevation, el_attrs),
            "rtime": Variable((dims[0],), rtime, rtime_attrs),
            "range": Variable(("range",), range_data, range_attrs),
            "time": Variable((), self.time, time_attrs),
            "sweep_mode": Variable((), sweep_mode),
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
        }
        return coordinates

    @property
    def site_coords(self):
        return self._get_site_coords()

    @property
    def time(self):
        return self._get_time()

    @property
    def fixed_dim_and_angle(self):
        return self._get_fixed_dim_and_angle()

    @property
    def range(self):
        return self._get_range()

    @property
    def what(self):
        return self._get_dset_what()

    def _get_azimuth_how(self):
        grp = self._group.split("/")[0]
        how = self._root[grp]["how"].attrs
        startaz = how["startazA"]
        stopaz = how.get("stopazA", False)
        if stopaz is False:
            # stopazA missing
            # create from startazA
            stopaz = np.roll(startaz, -1)
            stopaz[-1] += 360
        zero_index = np.where(stopaz < startaz)
        stopaz[zero_index[0]] += 360
        azimuth_data = (startaz + stopaz) / 2.0
        azimuth_data[azimuth_data >= 360] -= 360
        return azimuth_data

    def _get_azimuth_where(self):
        grp = self._group.split("/")[0]
        nrays = self._root[grp]["where"].attrs["nrays"]
        res = 360.0 / nrays
        azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")
        return azimuth_data

    def _get_fixed_dim_and_angle(self):
        grp = self._group.split("/")[0]
        dim = "elevation"

        # try RHI first
        angle_keys = ["az_angle", "azangle"]
        angle = None
        for ak in angle_keys:
            angle = self._root[grp]["where"].attrs.get(ak, None)
            if angle is not None:
                break
        if angle is None:
            dim = "azimuth"
            angle = self._root[grp]["where"].attrs["elangle"]

        angle = np.round(angle, decimals=1)
        return dim, angle

    def _get_elevation_how(self):
        grp = self._group.split("/")[0]
        how = self._root[grp]["how"].attrs
        startaz = how.get("startelA", False)
        stopaz = how.get("stopelA", False)
        if startaz is not False and stopaz is not False:
            elevation_data = (startaz + stopaz) / 2.0
        else:
            elevation_data = how["elangles"]
        return elevation_data

    def _get_elevation_where(self):
        grp = self._group.split("/")[0]
        nrays = self._root[grp]["where"].attrs["nrays"]
        elangle = self._root[grp]["where"].attrs["elangle"]
        elevation_data = np.ones(nrays, dtype="float32") * elangle
        return elevation_data

    def _get_time_how(self):
        grp = self._group.split("/")[0]
        startT = self._root[grp]["how"].attrs["startazT"]
        stopT = self._root[grp]["how"].attrs["stopazT"]
        time_data = (startT + stopT) / 2.0
        return time_data

    def _get_time_what(self, nrays=None):
        grp = self._group.split("/")[0]
        what = self._root[grp]["what"].attrs
        startdate = _maybe_decode(what["startdate"])
        starttime = _maybe_decode(what["starttime"])
        # take care for missing enddate/endtime
        # see https://github.com/wradlib/wradlib/issues/563
        enddate = _maybe_decode(what.get("enddate", what["startdate"]))
        endtime = _maybe_decode(what.get("endtime", what["starttime"]))
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        end = dt.datetime.strptime(enddate + endtime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        end = end.replace(tzinfo=dt.timezone.utc).timestamp()
        if nrays is None:
            nrays = self._root[grp]["where"].attrs["nrays"]
        if start == end:
            import warnings

            warnings.warn(
                "WRADLIB: Equal ODIM `starttime` and `endtime` "
                "values. Can't determine correct sweep start-, "
                "end- and raytimes.",
                UserWarning,
            )

            time_data = np.ones(nrays) * start
        else:
            delta = (end - start) / nrays
            time_data = np.arange(start + delta / 2.0, end, delta)
            time_data = np.roll(time_data, shift=+self.a1gate)
        return time_data

    def _get_ray_times(self, nrays=None):
        try:
            time_data = self._get_time_how()
            self._need_time_recalc = False
        except (AttributeError, KeyError, TypeError):
            time_data = self._get_time_what(nrays=nrays)
            self._need_time_recalc = True
        return time_data

    def _get_range(self):
        grp = self._group.split("/")[0]
        where = self._root[grp]["where"].attrs
        ngates = where["nbins"]
        range_start = where["rstart"] * 1000.0
        bin_range = where["rscale"]
        cent_first = range_start + bin_range / 2.0
        range_data = np.arange(
            cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
        )
        return range_data, cent_first, bin_range

    def _get_time(self, point="start"):
        grp = self._group.split("/")[0]
        what = self._root[grp]["what"].attrs
        startdate = _maybe_decode(what[f"{point}date"])
        starttime = _maybe_decode(what[f"{point}time"])
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_a1gate(self):
        grp = self._group.split("/")[0]
        a1gate = self._root[grp]["where"].attrs["a1gate"]
        return a1gate

    def _get_site_coords(self):
        lon = self._root["where"].attrs["lon"]
        lat = self._root["where"].attrs["lat"]
        alt = self._root["where"].attrs["height"]
        return lon, lat, alt

    def _get_dset_what(self):
        attrs = {}
        what = self._root[self._group]["what"].attrs
        attrs["scale_factor"] = what.get("gain", 1)
        attrs["add_offset"] = what.get("offset", 0)
        attrs["_FillValue"] = what.get("nodata", None)
        attrs["_Undetect"] = what.get("undetect", 0)
        # if no quantity is given, use the group-name
        attrs["quantity"] = _maybe_decode(
            what.get("quantity", self._group.split("/")[-1])
        )
        return attrs

    @property
    def a1gate(self):
        return self._get_a1gate()

    @property
    def azimuth(self):
        try:
            azimuth = self._get_azimuth_how()
        except (AttributeError, KeyError, TypeError):
            azimuth = self._get_azimuth_where()
        return azimuth

    @property
    def elevation(self):
        try:
            elevation = self._get_elevation_how()
        except (AttributeError, KeyError, TypeError):
            elevation = self._get_elevation_where()
        return elevation

    @property
    def ray_times(self):
        return self._get_ray_times()


class _GamicH5NetCDFMetadata:
    """Wrapper around OdimH5 data fileobj for easy access of metadata.

    Parameters
    ----------
    fileobj : file-like
        h5netcdf filehandle.
    group : str
        odim group to acquire

    Returns
    -------
    object : metadata object
    """

    def __init__(self, fileobj, group):
        self._root = fileobj
        self._group = group

    @property
    def first_dim(self):
        dim, _ = self._get_fixed_dim_and_angle()
        return dim

    def get_variable_dimensions(self, dims):
        dimensions = []
        for n, _ in enumerate(dims):
            if n == 0:
                dimensions.append(self.first_dim)
            elif n == 1:
                dimensions.append("range")
            else:
                pass
        return tuple(dimensions)

    def coordinates(self, dimensions, data, encoding):

        ray_header = _get_ray_header_data(dimensions, data, encoding)
        dim, angle = self.fixed_dim_and_angle
        angles = ray_header[dim]

        angle_res = _calculate_angle_res(angles)
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        sort_idx = np.argsort(angles)
        a1gate = np.argsort(ray_header["rtime"][sort_idx])[0]

        az_attrs = az_attrs_template.copy()
        el_attrs = el_attrs_template.copy()
        az_attrs["a1gate"] = a1gate

        if dim == "azimuth":
            az_attrs["angle_res"] = angle_res
        else:
            el_attrs["angle_res"] = angle_res

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"

        rtime_attrs = {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "standard_name": "time",
        }

        range_data, cent_first, bin_range = self.range
        range_attrs["meters_to_center_of_first_gate"] = cent_first
        range_attrs["meters_between_gates"] = bin_range

        lon_attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
            "standard_name": "longitude",
        }
        lat_attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        }
        alt_attrs = {
            "long_name": "altitude",
            "units": "meters",
            "standard_name": "altitude",
        }

        lon, lat, alt = self.site_coords

        coordinates = {
            "azimuth": Variable((dims[0],), ray_header["azimuth"], az_attrs),
            "elevation": Variable((dims[0],), ray_header["elevation"], el_attrs),
            "rtime": Variable((dims[0],), ray_header["rtime"], rtime_attrs),
            "range": Variable(("range",), range_data, range_attrs),
            "time": Variable((), self.time, time_attrs),
            "sweep_mode": Variable((), sweep_mode),
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
        }

        return coordinates

    @property
    def site_coords(self):
        return self._get_site_coords()

    @property
    def time(self):
        return self._get_time()

    @property
    def fixed_dim_and_angle(self):
        return self._get_fixed_dim_and_angle()

    @property
    def range(self):
        return self._get_range()

    @property
    def what(self):
        return self._get_dset_what()

    def _get_fixed_dim_and_angle(self):
        how = self._root[self._group]["how"].attrs
        dims = {0: "elevation", 1: "azimuth"}
        try:
            dim = 1
            angle = np.round(how[dims[0]], decimals=1)
        except KeyError:
            dim = 0
            angle = np.round(how[dims[1]], decimals=1)

        return dims[dim], angle

    def _get_range(self):
        how = self._root[self._group]["how"].attrs
        range_samples = how["range_samples"]
        range_step = how["range_step"]
        ngates = how["bin_count"]
        bin_range = range_step * range_samples
        cent_first = bin_range / 2.0
        range_data = np.arange(
            cent_first,
            bin_range * ngates,
            bin_range,
            dtype="float32",
        )
        return range_data, cent_first, bin_range

    def _get_time(self):
        start = self._root[self._group]["how"].attrs["timestamp"]
        start = dateutil.parser.parse(start)
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_site_coords(self):
        lon = self._root["where"].attrs["lon"]
        lat = self._root["where"].attrs["lat"]
        alt = self._root["where"].attrs["height"]
        return lon, lat, alt


def _fix_angle(da):
    # fix elevation outliers
    if len(set(da.values)) > 1:
        med = da.median(skipna=True)
        da = da.where(da == med).fillna(med)
    return da


def _remove_duplicate_rays(ds, store=None):
    dimname = list(ds.dims)[0]
    # find exact duplicates and remove
    _, idx = np.unique(ds[dimname], return_index=True)
    if len(idx) < len(ds[dimname]):
        ds = ds.isel({dimname: idx})
        # if ray_time was erroneously created from wrong dimensions
        # we need to recalculate it
        if store and store._need_time_recalc:
            ray_times = store._get_ray_times(nrays=len(idx))
            # need to decode only if ds is decoded
            if "units" in ds.rtime.encoding:
                ray_times = xr.decode_cf(xr.Dataset({"rtime": ray_times})).rtime
            ds = ds.assign({"rtime": ray_times})
    return ds


def _reindex_angle(ds, store=None, force=False, tol=None):
    # Todo: The current code assumes to have PPI's of 360deg and RHI's of 90deg,
    #       make this work also for sectorized (!!!) measurements
    #       this needs refactoring, it's too complex
    if tol is True or tol is None:
        tol = 0.4
    # disentangle different functionality
    full_range = {"azimuth": 360, "elevation": 90}
    dimname = list(ds.dims)[0]
    # sort in any case, to prevent unsorted errors
    ds = ds.sortby(dimname)
    # fix angle range for rhi
    if hasattr(ds, "elevation_upper_limit"):
        ul = np.rint(ds.elevation_upper_limit)
        full_range["elevation"] = ul

    secname = {"azimuth": "elevation", "elevation": "azimuth"}.get(dimname)
    dim = ds[dimname]
    diff = dim.diff(dimname)
    # this captures different angle spacing
    # catches also missing rays and double rays
    # and other erroneous ray alignments which result in different diff values
    diffset = set(diff.values)
    non_uniform_angle_spacing = len(diffset) > 1
    # this captures missing and additional rays in case the angle differences
    # are equal
    non_full_circle = False
    if not non_uniform_angle_spacing:
        res = list(diffset)[0]
        non_full_circle = ((res * ds.dims[dimname]) % full_range[dimname]) != 0

    # fix issues with ray alignment
    if force | non_uniform_angle_spacing | non_full_circle:
        # create new array and reindex
        if store and hasattr(store, "angle_resolution"):
            res = store.angle_resolution
        elif hasattr(ds[dimname], "angle_res"):
            res = ds[dimname].angle_res
        else:
            res = diff.median(dimname).values
        new_rays = int(np.round(full_range[dimname] / res, decimals=0))
        # find exact duplicates and remove
        ds = _remove_duplicate_rays(ds, store=store)

        # do we have all needed rays?
        if non_uniform_angle_spacing | len(ds[dimname]) != new_rays:
            # todo: check if assumption that beam center points to
            #       multiples of res/2. is correct in any case
            # it might fail for cfradial1 data which already points to beam centers
            azr = np.arange(res / 2.0, new_rays * res, res, dtype=diff.dtype)
            fill_value = {
                k: np.asarray(v._FillValue).astype(v.dtype)
                for k, v in ds.items()
                if hasattr(v, "_FillValue")
            }
            ds = ds.reindex(
                {dimname: azr},
                method="nearest",
                tolerance=tol,
                fill_value=fill_value,
            )

        # check other coordinates
        # check secondary angle coordinate (no nan)
        # set nan values to reasonable median
        if hasattr(ds, secname) and np.count_nonzero(np.isnan(ds[secname])):
            ds[secname] = ds[secname].fillna(ds[secname].median(skipna=True))
        # todo: rtime is also affected, might need to be treated accordingly

    return ds


def _get_h5group_names(filename, engine):
    if engine == "odim":
        groupname = "dataset"
    elif engine == "gamic":
        groupname = "scan"
    elif engine == "cfradial2":
        groupname = "sweep"
    else:
        raise ValueError(f"wradlib: unknown engine `{engine}`.")
    with h5netcdf.File(filename, "r", decode_vlen_strings=True) as fh:
        groups = ["/".join(["", grp]) for grp in fh.groups if groupname in grp.lower()]
    if isinstance(filename, io.BytesIO):
        filename.seek(0)
    return groups


def _get_nc4group_names(filename, engine):
    if engine == "cfradial2":
        groupname = "sweep"
    else:
        raise ValueError(f"wradlib: unknown engine `{engine}`.")
    with netCDF4.Dataset(filename, "r") as fh:
        groups = ["".join(["", grp]) for grp in fh.groups if groupname in grp.lower()]
    if isinstance(filename, io.BytesIO):
        filename.seek(0)
    return groups


def _get_iris_group_names(filename):
    from wradlib.io.iris import _check_iris_file

    sid, opener = _check_iris_file(filename)
    with opener(filename, loaddata=False) as ds:
        keys = list(ds.data.keys())
    return keys


def _get_rainbow_group_names(filename):
    from wradlib.io.rainbow import RainbowFile

    with RainbowFile(filename, loaddata=False) as fh:
        cnt = len(fh.slices)
    return list(range(cnt))


def _get_odim_variable_name_and_attrs(name, attrs):
    if "data" in name:
        name = attrs.pop("quantity")
        # handle non-standard moment names
        try:
            mapping = moments_mapping[name]
        except KeyError:
            pass
        else:
            attrs.update({key: mapping[key] for key in moment_attrs})
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
    return name, attrs


def _get_gamic_variable_name_and_attrs(attrs, dtype):
    name = attrs.pop("moment").lower()
    try:
        name = GAMIC_NAMES[name]
        mapping = moments_mapping[name]
    except KeyError:
        # ds = ds.drop_vars(mom)
        pass
    else:
        attrs.update({key: mapping[key] for key in moment_attrs})

    dmax = np.iinfo(dtype).max
    dmin = np.iinfo(dtype).min
    minval = attrs.pop("dyn_range_min")
    maxval = attrs.pop("dyn_range_max")
    dtype = minval.dtype
    dyn_range = maxval - minval
    if maxval != minval:
        gain = dyn_range / (dmax - 1)
        minval -= gain
    else:
        gain = (dmax - dmin) / dmax
        minval = dmin
    # ensure numpy type
    gain = np.array([gain])[0].astype(dtype)
    minval = np.array([minval])[0].astype(dtype)
    undetect = np.array([dmin])[0].astype(dtype)
    attrs["scale_factor"] = gain
    attrs["add_offset"] = minval
    attrs["_FillValue"] = undetect
    attrs["_Undetect"] = undetect

    attrs[
        "coordinates"
    ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"

    return name, attrs


def _get_ray_header_data(dimensions, data, encoding):
    ray_header = Variable(dimensions, data, {}, encoding)

    azstart = ray_header.values["azimuth_start"]
    azstop = ray_header.values["azimuth_stop"]
    zero_index = np.where(azstop < azstart)
    azstop[zero_index[0]] += 360
    azimuth = (azstart + azstop) / 2.0

    elstart = ray_header.values["elevation_start"]
    elstop = ray_header.values["elevation_stop"]
    elevation = (elstart + elstop) / 2.0

    rtime = ray_header.values["timestamp"] / 1e6

    return {"azimuth": azimuth, "elevation": elevation, "rtime": rtime}


def _unpack_netcdf_delta_units_ref_date(units):
    matches = re.match(r"(.+) since (.+)", units)
    if not matches:
        raise ValueError(f"invalid time units: {units}")
    return [s.strip() for s in matches.groups()]


def _rewrite_time_reference_units(ds):
    has_time_reference = "time_reference" in ds.variables
    if has_time_reference:
        ref_date = str(ds.variables["time_reference"].data)
        for v in ds.variables.values():
            attrs = v.attrs
            has_time_reference_units = (
                "units" in attrs
                and "since" in attrs["units"]
                and "time_reference" in attrs["units"]
            )
            if has_time_reference_units and has_time_reference:
                delta_units, _ = _unpack_netcdf_delta_units_ref_date(attrs["units"])
                v.attrs["units"] = " ".join([delta_units, "since", ref_date])
    return ds


def _assign_data_radial(root, sweep="sweep_1"):
    """Assign from CfRadial1 data structure.

    Parameters
    ----------
    root : xarray.Dataset
        Dataset of CfRadial1 file
    sweep : str, optional
        Sweep name to extract, default to first sweep. If None, all sweeps are
        extracted into a list.
    """
    var = root.variables.keys()
    remove_root = var ^ root_vars
    remove_root &= var
    root1 = root.drop_vars(remove_root).rename({"fixed_angle": "sweep_fixed_angle"})
    sweep_group_name = []
    for i in range(root1.dims["sweep"]):
        sweep_group_name.append(f"sweep_{i + 1}")

    # keep all vars for now
    # keep_vars = sweep_vars1 | sweep_vars2 | sweep_vars3
    # remove_vars = var ^ keep_vars
    # remove_vars &= var
    remove_vars = {}
    data = root.drop_vars(remove_vars)
    data.attrs = {}
    # explicitely cast to int, as there are floats out in the wild
    start_idx = data.sweep_start_ray_index.values.astype(int)
    end_idx = data.sweep_end_ray_index.values.astype(int)
    ray_n_gates = data.get("ray_n_gates", False)
    ray_start_index = data.get("ray_start_index", False)
    data = data.drop_vars(
        {
            "sweep_start_ray_index",
            "sweep_end_ray_index",
            "ray_n_gates",
            "ray_start_index",
        },
        errors="ignore",
    )
    sweeps = []
    for i, sw in enumerate(sweep_group_name):
        if sweep is not None and sweep != sw:
            continue
        tslice = slice(start_idx[i], end_idx[i] + 1)
        ds = data.isel(time=tslice, sweep=slice(i, i + 1)).squeeze("sweep")
        ds.sweep_mode.load()
        sweep_mode = _maybe_decode(ds.sweep_mode)
        dim0 = "elevation" if sweep_mode == "rhi" else "azimuth"
        ds = ds.swap_dims({"time": dim0})
        ds = ds.rename({"time": "rtime"})

        # check and extract for variable number of gates
        if ray_n_gates is not False:
            current_ray_n_gates = ray_n_gates.isel(time=tslice)
            current_rays_sum = current_ray_n_gates.sum().values
            nslice = slice(
                ray_start_index[start_idx[i]].values,
                ray_start_index[start_idx[i]].values + current_rays_sum,
            )
            rslice = slice(0, current_ray_n_gates[0].values)
            ds = ds.isel(range=rslice)
            ds = ds.isel(n_points=nslice).stack(n_points=["azimuth", "range"])
            ds = ds.unstack()
            # fix elevation/time additional range dimension in coordinate
            ds = ds.assign_coords({"elevation": ds.elevation.isel(range=0, drop=True)})

        ds.attrs["fixed_angle"] = np.round(ds.fixed_angle.values, decimals=1)
        time = ds.rtime[0].reset_coords(drop=True)
        # get and delete "comment" attribute for time variable
        key = [key for key in time.attrs.keys() if "comment" in key]
        for k in key:
            del time.attrs[k]
        coords = {
            "longitude": root1.longitude,
            "latitude": root1.latitude,
            "altitude": root1.altitude,
            "azimuth": ds.azimuth,
            "elevation": ds.elevation,
            "sweep_mode": sweep_mode,
            "time": time,
        }
        ds = ds.assign_coords(**coords)
        sweeps.append(ds)

    return sweeps


def _assign_data_radial2(ds):
    """Assign from CfRadial2 data structure.

    Parameters
    ----------
    ds : Dataset

    """
    ds.sweep_mode.load()
    sweep_mode = ds.sweep_mode.item()
    dim0 = "elevation" if sweep_mode == "rhi" else "azimuth"
    ds = ds.swap_dims({"time": dim0})
    ds = ds.rename({"time": "rtime"})
    time = ds.rtime[0].reset_coords(drop=True)
    # catch `decode_times=False` case
    try:
        time = time.dt.round("S")
    except TypeError:
        pass

    # todo: check use-case
    key = [key for key in time.attrs.keys() if "comment" in key]
    if key:
        del time.attrs[key[0]]
    coords = {
        "azimuth": ds.azimuth,
        "elevation": ds.elevation,
        "sweep_mode": sweep_mode,
        "time": time,
    }
    ds = ds.assign_coords(**coords)

    return ds


def open_radar_dataset(filename_or_obj, engine=None, **kwargs):
    """Open and decode a radar sweep or volume from a single file or file-like object.

    This function uses :py:func:`xarray:xarray.open_dataset` under the hood. Please refer for
    details to the documentation of :py:func:`xarray:xarray.open_dataset`.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or Datastore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    engine : {"odim", "furuno", "gamic", "cfradial1", "cfradial2", "iris", "rainbow"}
        Engine to use when reading files.

    Keyword Arguments
    -----------------
    group : str, optional
        Path to a sweep group in the given file to open.
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.xarray.open_radar_mfdataset`
    """
    if engine not in [
        "cfradial1",
        "cfradial2",
        "furuno",
        "gamic",
        "odim",
        "iris",
        "rainbow",
    ]:
        raise TypeError(f"Missing or unknown `engine` keyword argument '{engine}'.")

    group = kwargs.pop("group", None)
    groups = []
    backend_kwargs = kwargs.pop("backend_kwargs", {})

    if isinstance(group, (str, int)):
        groups = [group]
    elif isinstance(group, list):
        pass
    else:
        if engine == "cfradial1":
            groups = ["/"]
        elif engine == "cfradial2":
            groups = _get_nc4group_names(filename_or_obj, engine)
        elif engine in ["gamic", "odim"]:
            groups = _get_h5group_names(filename_or_obj, engine)
        elif engine == "iris":
            groups = _get_iris_group_names(filename_or_obj)
        elif engine in ["rainbow"]:
            groups = _get_rainbow_group_names(filename_or_obj)
        elif engine in ["furuno"]:
            groups = [group]
        elif isinstance(group, str):
            groups = [group]
        elif isinstance(group, int):
            groups = [group]
        else:
            pass

    if engine in ["gamic", "odim"]:
        keep_azimuth = kwargs.pop("keep_azimuth", False)
        backend_kwargs["keep_azimuth"] = keep_azimuth

    kwargs["backend_kwargs"] = backend_kwargs

    ds = [
        xr.open_dataset(filename_or_obj, group=grp, engine=engine, **kwargs)
        for grp in groups
    ]

    # cfradial1 backend always returns single group or root-object,
    # from above we get back root-object in any case
    if engine == "cfradial1" and not isinstance(group, str):
        ds = _assign_data_radial(ds[0], sweep=group)

    if group is None:
        vol = RadarVolume(engine=engine)
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds


def open_radar_mfdataset(paths, **kwargs):
    """Open multiple radar files as a single radar sweep dataset or radar volume.

    This function uses :py:func:`xarray:xarray.open_mfdataset` under the hood. Please
    refer for details to the documentation of :py:func:`xarray:xarray.open_mfdataset`.
    Needs ``dask`` package to be installed [1]_.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form ``"path/to/my/files/*"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see :py:func:`xarray:xarray.combine_nested` for details).
        (A string glob will be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_.
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='by_coords'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        :py:func:`xarray:xarray.merge`.
    combine : {"by_coords", "nested"}, optional
        Whether :py:func:`xarray:xarray.combine_by_coords` or :py:func:`xarray:xarray.combine_nested`
        is used to combine all the data. Default is to use :py:func:`xarray:xarray.combine_by_coords`.
    engine : {"odim", "gamic", "cfradial1", "cfradial2"}
        Engine to use when reading files.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_mfdataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`~wradlib.io.xarray.RadarVolume`

    See Also
    --------
    :func:`~wradlib.io.xarray.open_radar_dataset`

    References
    ----------
    .. [1] https://docs.dask.org/en/latest/
    .. [2] https://xarray.pydata.org/en/stable/user-guide/dask.html#chunking-and-performance
    """

    def _unpack_paths(paths):
        from pathlib import Path

        out = []
        for p in paths:
            if isinstance(p, list):
                out.append(_unpack_paths(p))
            else:
                if isinstance(p, io.BytesIO):
                    out.append(p)
                else:
                    if os.path.isfile(p):
                        if isinstance(p, Path):
                            out.append(str(p))
                        else:
                            out.append(p)
                    else:
                        out.append(sorted(glob.glob(p)))
        return out

    def _align_paths(paths):

        if isinstance(paths, str):
            paths = sorted(glob.glob(paths))
        else:
            paths = _unpack_paths(paths)
        patharr = np.array(paths)

        if patharr.ndim == 2 and len(patharr) == 1:
            patharr = patharr[0]

        return patharr

    patharr = _align_paths(paths)

    def _concat_combine(kwargs, patharr):
        concat_dim = kwargs.pop("concat_dim", "time")
        combine = kwargs.pop("combine", "nested")
        if concat_dim and patharr.ndim > 1:
            concat_dim = ["time"] + (patharr.ndim - 1) * [None]
        if concat_dim is None:
            combine = "by_coords"
        return concat_dim, combine

    concat_dim, combine = _concat_combine(kwargs, patharr)
    engine = kwargs.pop("engine")

    group = kwargs.pop("group", None)
    if group is None:
        if engine == "cfradial2":
            group = _get_nc4group_names(patharr.flat[0], engine)
        elif engine in ["gamic", "odim"]:
            group = _get_h5group_names(patharr.flat[0], engine)
        elif engine == "iris":
            group = _get_iris_group_names(patharr.flat[0])
        elif engine in ["rainbow"]:
            group = _get_rainbow_group_names(patharr.flat[0])
    elif isinstance(group, str):
        group = [group]
    elif isinstance(group, int):
        group = [group]
    else:
        pass

    ds = [
        xr.open_mfdataset(
            patharr.tolist(),
            engine=engine,
            group=grp,
            concat_dim=concat_dim,
            combine=combine,
            **kwargs,
        )
        for grp in tqdm(group)
    ]

    if len(ds) > 1:
        vol = RadarVolume(engine=engine)
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds


class XRadBase(collections.abc.MutableSequence):
    """Base Class for all XRad-classes."""

    def __init__(self, **kwargs):
        super().__init__()
        self._seq = []

    def __getitem__(self, index):
        return self._seq[index]

    def __setitem__(self, index, value):
        self._seq[index] = value

    def __delitem__(self, index):
        del self._seq[index]

    def insert(self, pos, val):
        self._seq.insert(pos, val)

    def __iter__(self):
        return iter(self._seq)

    def __len__(self):
        return len(self._seq)

    def __repr__(self):
        return self._seq.__repr__()

    def __del__(self):
        if self._seq:
            for i in range(len(self._seq)):
                del self._seq[0]
            self._seq = None

    def sort(self, **kwargs):
        self._seq.sort(**kwargs)


class RadarVolume(XRadBase):
    """Class for holding a volume of radar sweeps"""

    def __init__(self, **kwargs):
        super().__init__()
        self._data = None
        self._root = None
        self._engine = kwargs.pop("engine", "netcdf4")
        self._dims = {"azimuth": "elevation", "elevation": "azimuth"}

    def __repr__(self):
        summary = [f"<wradlib.{type(self).__name__}>"]
        dims = "Dimension(s):"
        dims_summary = f"sweep: {len(self)}"
        summary.append(f"{dims} ({dims_summary})")
        dim0 = list(set(self[0].dims) & {"azimuth", "elevation"})[0]
        angle = f"{self._dims[dim0].capitalize()}(s):"
        angle_summary = [f"{v.attrs['fixed_angle']:.1f}" for v in self]
        angle_summary = ", ".join(angle_summary)
        summary.append(f"{angle} ({angle_summary})")

        return "\n".join(summary)

    @property
    def root(self):
        """Return root object."""
        if self._root is None:
            self.assign_root()
        return self._root

    def get_attrs(self, sweep, group):
        for v in self[sweep].variables.values():
            if "source" in v.encoding:
                src = v.encoding["source"]
                break
        return xr.open_dataset(src, engine=self._engine, group=group).attrs

    def get_attr(self, sweep, group, attr):
        for v in self[sweep].variables.values():
            if "source" in v.encoding:
                src = v.encoding["source"]
                break
        return xr.open_dataset(src, engine=self._engine, group=group).attrs[attr]

    def assign_root(self):
        """(Re-)Create root object according CfRadial2 standard"""
        # assign root variables
        sweep_group_names = [f"sweep_{i}" for i in range(len(self))]

        sweep_fixed_angles = [ts.attrs["fixed_angle"] for ts in self]

        # extract time coverage
        times = np.array(
            [[ts.rtime.values.min(), ts.rtime.values.max()] for ts in self]
        ).flatten()
        time_coverage_start = min(times)
        time_coverage_end = max(times)

        time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
        time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

        # create root group from scratch
        root = xr.Dataset()  # data_vars=wrl.io.xarray.global_variables,
        # attrs=wrl.io.xarray.global_attrs)

        # take first dataset/file for retrieval of location
        # site = self.site

        # assign root variables
        root = root.assign(
            {
                "volume_number": 0,
                "platform_type": str("fixed"),
                "instrument_type": "radar",
                "primary_axis": "axis_z",
                "time_coverage_start": time_coverage_start_str,
                "time_coverage_end": time_coverage_end_str,
                "latitude": self[0]["latitude"],
                "longitude": self[0]["longitude"],
                "altitude": self[0]["altitude"],
                "sweep_group_name": (["sweep"], sweep_group_names),
                "sweep_fixed_angle": (["sweep"], sweep_fixed_angles),
            }
        )

        # assign root attributes
        attrs = {}
        attrs.update(
            {
                "version": "None",
                "title": "None",
                "institution": "None",
                "references": "None",
                "source": "None",
                "history": "None",
                "comment": "im/exported using wradlib",
                "instrument_name": "None",
            }
        )
        # attrs["version"] = self[0].attrs["version"]
        root = root.assign_attrs(attrs)
        # todo: pull in only CF attributes
        root = root.assign_attrs(self[0].attrs)
        self._root = root

    @property
    def site(self):
        """Return coordinates of radar site."""
        return self[0][["latitude", "longitude", "altitude"]]

    @property
    def Conventions(self):
        """Return Conventions string."""
        try:
            conv = self[0].attrs["Conventions"]
        except KeyError:
            conv = None
        return conv

    def to_odim(self, filename, timestep=0):
        """Save volume to ODIM_H5/V2_2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep of wanted volume
        """
        if self.root:
            to_odim(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No OdimH5-compliant data structure " "available. Not saving.",
                UserWarning,
            )

    def to_cfradial2(self, filename, timestep=0):
        """Save volume to CfRadial2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep wanted volume
        """
        if self.root:
            to_cfradial2(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No CfRadial2-compliant data structure "
                "available. Not saving.",
                UserWarning,
            )

    def to_netcdf(self, filename, timestep=None, keys=None):
        """Save volume to netcdf compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int, slice
            timestep/slice of wanted volume
        keys : list
            list of sweep_group_names which should be written to the file
        """
        if self.root:
            to_netcdf(self, filename, keys=keys, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No netcdf-compliant data structure " "available. Not saving.",
                UserWarning,
            )


def _write_odim(src, dest):
    """Writes Odim Attributes.

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
    """Writes Odim Dataspaces.

    Parameters
    ----------
    src : dict
        Moments to write
    dest : handle
        h5py-group handle
    """
    keys = [key for key in src if key in ODIM_NAMES]
    data_list = [f"data{i + 1}" for i in range(len(keys))]
    data_idx = np.argsort(data_list)
    for idx in data_idx:
        value = src[keys[idx]]
        h5_data = dest.create_group(data_list[idx])
        enc = value.encoding

        # p. 21 ff
        h5_what = h5_data.create_group("what")
        try:
            undetect = float(value._Undetect)
        except AttributeError:
            undetect = np.finfo(np.float_).max

        # set some defaults, if not available
        scale_factor = float(enc.get("scale_factor", 1.0))
        add_offset = float(enc.get("add_offset", 0.0))
        _fillvalue = float(enc.get("_FillValue", undetect))
        dtype = enc.get("dtype", value.dtype)
        what = {
            "quantity": value.name,
            "gain": scale_factor,
            "offset": add_offset,
            "nodata": _fillvalue,
            "undetect": undetect,
        }
        _write_odim(what, h5_what)

        # moments handling
        val = value.sortby("azimuth").values
        fillval = _fillvalue * scale_factor
        fillval += add_offset
        val[np.isnan(val)] = fillval
        val = (val - add_offset) / scale_factor
        if np.issubdtype(dtype, np.integer):
            val = np.rint(val).astype(dtype)
        ds = h5_data.create_dataset(
            "data",
            data=val,
            compression="gzip",
            compression_opts=6,
            fillvalue=_fillvalue,
            dtype=dtype,
        )
        if enc["dtype"] == "uint8":
            image = "IMAGE"
            version = "1.2"
            tid1 = h5py.h5t.C_S1.copy()
            tid1.set_size(len(image) + 1)
            H5T_C_S1_IMG = h5py.Datatype(tid1)
            tid2 = h5py.h5t.C_S1.copy()
            tid2.set_size(len(version) + 1)
            H5T_C_S1_VER = h5py.Datatype(tid2)
            ds.attrs.create("CLASS", image, dtype=H5T_C_S1_IMG)
            ds.attrs.create("IMAGE_VERSION", version, dtype=H5T_C_S1_VER)
