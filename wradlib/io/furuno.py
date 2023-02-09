#!/usr/bin/env python
# Copyright (c) 2022-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Furuno binary Data I/O
^^^^^^^^^^^^^^^^^^^^^^

Reads data from Furuno's binary data formats. Former available code was ported to
xradar-package and is imported from there.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "open_furuno_dataset",
    "open_furuno_mfdataset",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import warnings

from xradar.io.backends import furuno as xfuruno

from wradlib.io.xarray import (
    open_radar_dataset,
    open_radar_mfdataset,
    raise_on_missing_xarray_backend,
)


def FurunoFile(*args, **kwargs):
    warnings.warn(
        "FurunoFile class has been moved to xradar-package. "
        "Importing from wradlib will be removed in v2.0.",
        category=FutureWarning,
        stacklevel=2,
    )
    return xfuruno.FurunoFile(*args, **kwargs)


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
    from wradlib.io.backends import FurunoBackendEntrypoint

    kwargs["group"] = group
    return open_radar_dataset(filename_or_obj, engine=FurunoBackendEntrypoint, **kwargs)


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
    from wradlib.io.backends import FurunoBackendEntrypoint

    kwargs["group"] = group
    return open_radar_mfdataset(
        filename_or_obj, engine=FurunoBackendEntrypoint, **kwargs
    )
