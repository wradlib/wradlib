#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray based Data I/O
^^^^^^^^^^^^^^^^^^^^^

Note
----
    The Xarray backend code has moved to xradar-package. Here we keep a
    backwards-compatibility by providing knwon API.

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
    Changes in the API should be expected. The development is continued at xradar-package.

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

import deprecation
import numpy as np
import xarray as xr
from datatree import DataTree
from packaging.version import Version
from xradar.io.backends.cfradial1 import _get_sweep_groups
from xradar.io.backends.common import _get_h5group_names
from xradar.io.backends.iris import _get_iris_group_names
from xradar.io.backends.rainbow import _get_rainbow_group_names
from xradar.io.export import to_odim

from wradlib import version
from wradlib.georef import xarray as geoxarray
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

backends = [
    "wradlib-cfradial1",
    "wradlib-cfradial2",
    "wradlib-furuno",
    "wradlib-gamic",
    "wradlib-odim",
    "wradlib-iris",
    "wradlib-rainbow",
]


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
    details="Use `wradlib.georef.create_xarray_dataarray` instead.",
)
def create_xarray_dataarray(*args, **kwargs):
    return geoxarray.create_xarray_dataarray(*args, **kwargs)


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
    time = ds.rtime.min().reset_coords(drop=True)
    # catch `decode_times=False` case
    try:
        time = time.dt.round("ns")
    except TypeError:
        pass

    if "fixed_angle" in ds.data_vars:
        ds = ds.rename({"fixed_angle": "sweep_fixed_angle"})

    # todo: check use-case
    key = [key for key in time.attrs.keys() if "comment" in key]
    if key:
        del time.attrs[key[0]]
    coords = {
        "azimuth": ds.azimuth,
        "elevation": ds.elevation,
        "latitude": ds.latitude,
        "longitude": ds.longitude,
        "altitude": ds.altitude,
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
    engine : str or xarray.backends.BackendEntrypoint
        Engine to use when reading files, eg. ``wradlib-odim`` or
        ``wradlib.io.backends.OdimBackendEntryPoint``.

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
    if not (
        (engine in backends) or (hasattr(engine, "name") and engine.name in backends)
    ):
        raise TypeError(f"Missing or unknown `engine` keyword argument '{engine}'.")

    group = kwargs.pop("group", None)
    groups = []
    backend_kwargs = kwargs.pop("backend_kwargs", {})

    # get engine name
    engine_name = engine if isinstance(engine, str) else engine.name
    engine_name = engine_name.split("-")[1]

    warnings.warn(
        f"`open_{engine_name}_dataset` functionality has been moved to `xradar`-package "
        f"and will be removed in 2.0. Use `open_{engine_name}_datatree` from "
        "`xradar`-package.",
        category=FutureWarning,
        stacklevel=2,
    )

    if isinstance(group, (str, int)):
        groups = [group]
    elif isinstance(group, list):
        pass
    else:
        if engine_name == "cfradial1":
            groups = ["/"]
            engine = "netcdf4"
        elif engine_name == "cfradial2":
            groups = _get_nc4group_names(filename_or_obj, engine_name)
        elif engine_name in ["gamic", "odim"]:
            groups = _get_h5group_names(filename_or_obj, engine_name)
        elif engine_name == "iris":
            groups = _get_iris_group_names(filename_or_obj)
        elif engine_name in ["rainbow"]:
            groups = _get_rainbow_group_names(filename_or_obj)
        elif engine_name in ["furuno"]:
            groups = [group]
        elif isinstance(group, str):
            groups = [group]
        elif isinstance(group, int):
            groups = [group]
        else:
            pass

    if engine_name in ["gamic", "odim"]:
        keep_azimuth = kwargs.pop("keep_azimuth", False)
        backend_kwargs["keep_azimuth"] = keep_azimuth

    kwargs["backend_kwargs"] = backend_kwargs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = [
            xr.open_dataset(filename_or_obj, group=grp, engine=engine, **kwargs)
            for grp in groups
        ]

    # cfradial1 backend always returns single group or root-object,
    # from above we get back root-object in any case
    if engine_name == "cfradial1" and not isinstance(group, str):
        dsn = list(_get_sweep_groups(ds[0], sweep=group).values())
        ds = []
        for dsx in dsn:
            dsx = dsx.rename({"time": "rtime"})
            dsx = dsx.assign_coords({"time": dsx.rtime.min()})
            # backwards compat
            dsx = dsx.assign_coords(
                {
                    "sweep_mode": dsx.sweep_mode.str.decode("ascii").reset_coords(
                        drop=True
                    )
                }
            )
            ds.append(dsx)

    if group is None:
        vol = RadarVolume(engine=engine_name)
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
    engine : str or xarray.backends.BackendEntrypoint
        Engine to use when reading files, eg. ``wradlib-odim`` or
        ``wradlib.io.backends.OdimBackendEntryPoint``.
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

    if not (
        (engine in backends) or (hasattr(engine, "name") and engine.name in backends)
    ):
        raise TypeError(f"Missing or unknown `engine` keyword argument '{engine}'.")

    # get engine name
    engine_name = engine if isinstance(engine, str) else engine.name
    engine_name = engine_name.split("-")[1]

    warnings.warn(
        f"`open_{engine_name}_mfdataset` is deprecated and will be removed in 2.0. "
        "Future development will take place in `xradar`-package.",
        category=FutureWarning,
        stacklevel=2,
    )

    group = kwargs.pop("group", None)
    if group is None:
        if engine_name == "cfradial2":
            group = _get_nc4group_names(patharr.flat[0], engine_name)
        elif engine_name in ["gamic", "odim"]:
            group = _get_h5group_names(patharr.flat[0], engine_name)
        elif engine_name == "iris":
            group = _get_iris_group_names(patharr.flat[0])
        elif engine_name in ["rainbow"]:
            group = _get_rainbow_group_names(patharr.flat[0])
        elif engine_name == "furuno":
            group = [group]
    elif isinstance(group, str):
        group = [group]
    elif isinstance(group, int):
        group = [group]
    else:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        vol = RadarVolume(engine=engine_name)
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
        # Todo: remove if fixed in xradar
        try:
            angle_summary = [f"{v['sweep_fixed_angle'].min():.1f}" for v in self]
        except KeyError:
            angle_summary = [f"{v.attrs['fixed_angle'].min():.1f}" for v in self]
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
        # todo: remove if fixed in xradar
        try:
            sweep_fixed_angles = [ts["sweep_fixed_angle"].min() for ts in self]
        except KeyError:
            sweep_fixed_angles = [ts.attrs["fixed_angle"].min() for ts in self]

        # extract time coverage
        times = np.array(
            [[ts.rtime.values.min(), ts.rtime.values.max()] for ts in self]
        ).flatten()
        time_coverage_start = min(times)
        time_coverage_end = max(times)
        time_coverage_end = dt.datetime.utcfromtimestamp(
            np.ceil(time_coverage_end.astype("O") / 1e9)
        )

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
                "platform_type": "fixed",
                "instrument_type": "radar",
                "primary_axis": "axis_z",
                "time_coverage_start": time_coverage_start_str,
                "time_coverage_end": time_coverage_end_str,
                "latitude": self[0]["latitude"].reset_coords(drop=True),
                "longitude": self[0]["longitude"].reset_coords(drop=True),
                "altitude": self[0]["altitude"].reset_coords(drop=True),
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

    def to_odim(self, filename, timestep=0, **kwargs):
        """Save volume to ODIM_H5/V2_2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep of wanted volume
        """
        if self.root:
            to_odim(self.tree(timestep), filename, **kwargs)
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
            self.tree(timestep, "cfradial2").to_netcdf(filename)
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

    def tree(self, timestep=0, datamodel=None):
        dtree = DataTree(data=self.root, name="root")
        for i, sw in enumerate(self):
            if "time" in sw.dims:
                sw = sw.isel(time=timestep)
            dim0 = list(set(sw.dims) & {"azimuth", "elevation"})[0]

            sw = sw.drop_vars("time")
            sw = sw.rename({"rtime": "time"})
            if datamodel == "cfradial2":
                sw = sw.swap_dims({dim0: "time"})

            if "fixed_angle" in sw:
                sw = sw.rename({"fixed_angle": "sweep_fixed_angle"})
            DataTree(sw, name=f"sweep_{i}", parent=dtree)
        return dtree
