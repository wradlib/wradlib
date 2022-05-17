#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
HDF Data I/O
^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "open_gamic_dataset",
    "open_gamic_mfdataset",
    "open_gpm_dataset",
    "open_odim_dataset",
    "open_odim_mfdataset",
    "read_generic_hdf5",
    "read_opera_hdf5",
    "read_gamic_hdf5",
    "to_hdf5",
    "from_hdf5",
    "read_gpm",
    "read_trmm",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt

import numpy as np
import xarray as xr
from packaging.version import Version

from wradlib.io.xarray import (
    open_radar_dataset,
    open_radar_mfdataset,
    raise_on_missing_xarray_backend,
)
from wradlib.util import import_optional

h5py = import_optional("h5py")
h5netcdf = import_optional("h5netcdf")
nc = import_optional("netCDF4")


def open_odim_dataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an ODIM radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_dataset` and
    :py:func:`xarray.open_dataset` under the hood.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    keep_elevation : bool
        For PPI only. Keep original elevation data if True. Defaults to False,
        which fixes erroneous elevation data.
    keep_azimuth : bool
        For RHI only. Keep original azimuth data if True. Defaults to False,
        which fixes erroneous azimuth data.
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
    :func:`~wradlib.io.hdf.open_odim_mfdataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_dataset(filename_or_obj, engine="odim", **kwargs)


def open_gamic_dataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an GAMIC radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_dataset` and
    :py:func:`xarray.open_dataset` under the hood.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    keep_elevation : bool
        For PPI only. Keep original elevation data if True. Defaults to False,
        which fixes erroneous elevation data.
    keep_azimuth : bool
        For RHI only. Keep original azimuth data if True. Defaults to False,
        which fixes erroneous azimuth data.
    reindex_angle : bool or float
        Defaults to None (reindex angle with tol=0.4deg). If given a floating point
        number, it is used as tolerance. If False, no reindexing is performed.
        Only invoked if `decode_coord=True`.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.hdf.open_gamic_mfdataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_dataset(filename_or_obj, engine="gamic", **kwargs)


def open_odim_mfdataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an ODIM radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_mfdataset` and
    :py:func:`xarray:xarray.open_mfdataset` under the hood.
    Needs ``dask`` package to be installed [1]_.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    keep_elevation : bool
        For PPI only. Keep original elevation data if True. Defaults to False,
        which fixes erroneous elevation data.
    keep_azimuth : bool
        For RHI only. Keep original azimuth data if True. Defaults to False,
        which fixes erroneous azimuth data.
    reindex_angle : bool or float
        Defaults to None (reindex angle with tol=0.4deg). If given a floating point
        number, it is used as tolerance. If False, no reindexing is performed.
        Only invoked if `decode_coord=True`.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.hdf.open_odim_dataset`

    References
    ----------
    .. [1] https://docs.dask.org/en/latest/
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_mfdataset(filename_or_obj, engine="odim", **kwargs)


def open_gamic_mfdataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an GAMIC radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_mfdataset` and
    :py:func:`xarray:xarray.open_mfdataset` under the hood.
    Needs ``dask`` package to be installed [1]_.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    keep_elevation : bool
        For PPI only. Keep original elevation data if True. Defaults to False,
        which fixes erroneous elevation data.
    keep_azimuth : bool
        For RHI only. Keep original azimuth data if True. Defaults to False,
        which fixes erroneous azimuth data.
    reindex_angle : bool or float
        Defaults to None (reindex angle with tol=0.4deg). If given a floating point
        number, it is used as tolerance. If False, no reindexing is performed.
        Only invoked if `decode_coord=True`.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.hdf.open_gamic_dataset`

    References
    ----------
    .. [1] https://docs.dask.org/en/latest/
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_mfdataset(filename_or_obj, engine="gamic", **kwargs)


def read_generic_hdf5(fname):
    """Reads hdf5 files according to their structure

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.

    Parameters
    ----------
    fname : str or file-like
        a hdf5 file path or file-like object

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure

    Examples
    --------
    See :ref:`/notebooks/fileio/wradlib_radar_formats.ipynb#Generic-HDF5`.
    """
    fcontent = {}

    def filldict(x, y):
        # create a new container
        tmp = {}
        # add attributes if present
        if len(y.attrs) > 0:
            tmp["attrs"] = dict(y.attrs)
        # add data if it is a dataset
        if isinstance(y, h5py.Dataset):
            tmp["data"] = np.array(y)
        # only add to the dictionary, if we have something meaningful to add
        if tmp != {}:
            fcontent[x] = tmp

    with h5py.File(fname, "r") as f:
        f.visititems(filldict)

    return fcontent


def read_opera_hdf5(fname):
    """Reads hdf5 files according to OPERA conventions

    Please refer to the OPERA data model documentation :cite:`OPERA-data-model`
    in order to understand how an hdf5 file is organized that conforms to the
    OPERA ODIM_H5 conventions.

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.
    If the end member of a branch (or path) is "data", then the corresponding
    item of output dictionary is a numpy array with actual data.

    Any other end member (either *how*, *where*,
    and *what*) will contain the meta information applying to the corresponding
    level of the file hierarchy.

    Parameters
    ----------
    fname : str or file-like
        a hdf5 file path or file-like object

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure
    """
    # now we browse through all Groups and Datasets and store the info in one
    # dictionary
    fcontent = {}

    def filldict(x, y):
        if isinstance(y, h5py.Group):
            if len(y.attrs) > 0:
                fcontent[x] = dict(y.attrs)
        elif isinstance(y, h5py.Dataset):
            fcontent[x] = np.array(y)

    with h5py.File(fname, "r") as f:
        f.visititems(filldict)

    return fcontent


def read_gamic_scan_attributes(scan, scan_type):
    """Read attributes from one particular scan from a GAMIC hdf5 file

    Parameters
    ----------
    scan : object
        scan object from hdf5 file
    scan_type : str
        "PVOL" (plan position indicator) or "RHI" (range height indicator)

    Returns
    -------
    sattrs : dict
        dictionary of scan attributes
    """

    # global zero_index, el, az

    # placeholder for attributes
    sattrs = {}

    # link to scans 'how' hdf5 group
    sg1 = scan["how"]

    # get scan attributes
    for attrname in list(sg1.attrs):
        sattrs[attrname] = sg1.attrs.get(attrname)
    sattrs["bin_range"] = sattrs["range_step"] * sattrs["range_samples"]

    # get scan header
    ray_header = scan["ray_header"]

    # az, el, zero_index for PPI scans
    if scan_type == "PVOL":
        azi_start = ray_header["azimuth_start"]
        azi_stop = ray_header["azimuth_stop"]
        # Azimuth corresponding to 1st ray
        zero_index = np.where(azi_stop < azi_start)
        azi_stop[zero_index[0]] += 360
        zero_index = zero_index[0] + 1
        az = (azi_start + azi_stop) / 2
        az = np.roll(az, -zero_index, axis=0)
        az = np.round(az, 1)
        el = sg1.attrs.get("elevation")

    # az, el, zero_index for RHI scans
    if scan_type == "RHI":
        ele_start = np.round(ray_header["elevation_start"], 1)
        ele_stop = np.round(ray_header["elevation_stop"], 1)
        angle_step = np.round(sattrs["angle_step"], 1)
        angle_step = int(np.round(sattrs["ele_stop"], 1) / angle_step)
        # Elevation corresponding to 1st ray
        if ele_start[0] < 0:
            ele_start = ele_start[1:]
            ele_stop = ele_stop[1:]
        zero_index = np.where(ele_stop > ele_start)
        zero_index = zero_index[0]  # - 1
        el = (ele_start + ele_stop) / 2
        el = np.round(el, 1)
        el = el[-angle_step:]

        az = sg1.attrs.get("azimuth")

    # save zero_index (first ray) to scan attributes
    sattrs["zero_index"] = zero_index[0]

    # create range array
    r = np.arange(
        sattrs["bin_range"],
        sattrs["bin_range"] * sattrs["bin_count"] + sattrs["bin_range"],
        sattrs["bin_range"],
    )

    # save variables to scan attributes
    sattrs["az"] = az
    sattrs["el"] = el
    sattrs["r"] = r
    sattrs["Time"] = sattrs.pop("timestamp")
    sattrs["max_range"] = r[-1]

    return sattrs


def read_gamic_scan(scan, scan_type, wanted_moments):
    """Read data from one particular scan from GAMIC hdf5 file

    Parameters
    ----------
    scan : object
        scan object from hdf5 file
    scan_type : str
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    wanted_moments : sequence
        sequence of strings containing upper case names of moment(s) to
        be returned

    Returns
    -------
    data : dict
        dictionary of moment data (numpy arrays)
    sattrs : dict
        dictionary of scan attributes
    """

    # placeholder for data and attrs
    data = {}
    sattrs = {}

    # try to read wanted moments
    for mom in list(scan):
        if "moment" in mom:
            data1 = {}
            sg2 = scan[mom]
            actual_moment = sg2.attrs.get("moment")
            if Version(h5py.__version__) < Version("3.0.0"):
                actual_moment = actual_moment.decode()
            actual_moment = actual_moment.upper()
            if (actual_moment in wanted_moments) or (wanted_moments == "all"):
                # read attributes only once
                if not sattrs:
                    sattrs = read_gamic_scan_attributes(scan, scan_type)
                mdata = sg2[...]
                dyn_range_max = sg2.attrs.get("dyn_range_max")
                dyn_range_min = sg2.attrs.get("dyn_range_min")
                bin_format = sg2.attrs.get("format")
                if Version(h5py.__version__) < Version("3.0.0"):
                    bin_format = bin_format.decode()
                if bin_format == "UV8":
                    div = 254
                else:
                    div = 65534
                mdata = (
                    dyn_range_min + (mdata - 1) * (dyn_range_max - dyn_range_min) / div
                )

                if scan_type == "PVOL":
                    # rotate accordingly
                    mdata = np.roll(mdata, -1 * sattrs["zero_index"], axis=0)

                if scan_type == "RHI":
                    # remove first zero angles
                    sdiff = mdata.shape[0] - sattrs["el"].shape[0]
                    mdata = mdata[sdiff:, :]

                data1["data"] = mdata
                data1["dyn_range_max"] = dyn_range_max
                data1["dyn_range_min"] = dyn_range_min
                data[actual_moment] = data1

    return data, sattrs


def read_gamic_hdf5(filename, wanted_elevations=None, wanted_moments=None):
    """Data reader for hdf5 files produced by the commercial \
    GAMIC Enigma V3 MURAN software

    See GAMIC homepage for further info (https://www.gamic.com).

    Parameters
    ----------
    filename : str or file-like
        path of the gamic hdf5 file or file-like object
    wanted_elevations : sequence
        sequence of strings of elevation_angle(s) of scan (only needed for PPI)
    wanted_moments : sequence
        sequence of strings of moment name(s)

    Returns
    -------
    data : dict
        dictionary of scan and moment data (numpy arrays)
    attrs : dict
        dictionary of attributes

    Examples
    --------
    See :ref:`/notebooks/fileio/wradlib_radar_formats.ipynb#GAMIC-HDF5`.
    """

    # check elevations
    if wanted_elevations is None:
        wanted_elevations = "all"

    # check wanted_moments
    if wanted_moments is None:
        wanted_moments = "all"

    # read the data from file
    with h5py.File(filename, "r") as f:

        # placeholder for attributes and data
        attrs = {}
        vattrs = {}
        data = {}

        # check if GAMIC file and
        try:
            f["how"].attrs.get("software")
        except KeyError:
            print("WRADLIB: File is no GAMIC hdf5!")
            raise

        # get scan_type (PVOL or RHI)
        scan_type = f["what"].attrs.get("object")
        if Version(h5py.__version__) < Version("3.0.0"):
            scan_type = scan_type.decode()

        # single or volume scan
        if scan_type == "PVOL":
            # loop over 'main' hdf5 groups (how, scanX, what, where)
            for n in list(f):
                if "scan" in n:
                    g = f[n]
                    sg1 = g["how"]

                    # get scan elevation
                    el = sg1.attrs.get("elevation")
                    el = str(round(el, 2))

                    # try to read scan data and attrs
                    # if wanted_elevations are found
                    if (el in wanted_elevations) or (wanted_elevations == "all"):
                        sdata, sattrs = read_gamic_scan(
                            scan=g, scan_type=scan_type, wanted_moments=wanted_moments
                        )  # noqa
                        if sdata:
                            data[n.upper()] = sdata
                        if sattrs:
                            attrs[n.upper()] = sattrs

        # single rhi scan
        elif scan_type == "RHI":
            # loop over 'main' hdf5 groups (how, scanX, what, where)
            for n in list(f):
                if "scan" in n:
                    g = f[n]
                    # try to read scan data and attrs
                    sdata, sattrs = read_gamic_scan(
                        scan=g, scan_type=scan_type, wanted_moments=wanted_moments
                    )
                    if sdata:
                        data[n.upper()] = sdata
                    if sattrs:
                        attrs[n.upper()] = sattrs

        # collect volume attributes if wanted data is available
        if data:
            vattrs["Latitude"] = f["where"].attrs.get("lat")
            vattrs["Longitude"] = f["where"].attrs.get("lon")
            vattrs["Height"] = f["where"].attrs.get("height")
            # check whether its useful to implement that feature
            # vattrs['sitecoords'] = (vattrs['Longitude'], vattrs['Latitude'],
            #                         vattrs['Height'])
            attrs["VOL"] = vattrs

    return data, attrs


def to_hdf5(fpath, data, mode="w", metadata=None, dataset="data", compression="gzip"):
    """Quick storage of one <data> array and a <metadata> dict in an hdf5 file

    This is more efficient than pickle, cPickle or numpy.save. The data is
    stored in a subgroup named ``data`` (i.e. hdf5file["data").
    See :func:`~wradlib.io.hdf.from_hdf5` for retrieving stored data.

    Parameters
    ----------
    fpath : str or file-like
        path to the hdf5 file or file-like object
    data : :py:class:`numpy:numpy.ndarray`
    mode : str
        file open mode, defaults to "w" (create, truncate if exists)
    metadata : dict
        dictionary of data's attributes
    dataset : str
        describing dataset
    compression : str
        h5py compression type {"gzip"|"szip"|"lzf"}, see h5py documentation
        for details
    """
    with h5py.File(fpath, mode=mode) as f:
        dset = f.create_dataset(dataset, data=data, compression=compression)
        # store metadata
        if metadata:
            for key in metadata.keys():
                dset.attrs[key] = metadata[key]


def from_hdf5(fpath, dataset="data"):
    """Loading data from hdf5 files that was stored by \
    :func:`~wradlib.io.hdf.to_hdf5`

    Parameters
    ----------
    fpath : str or file-like
        path to the hdf5 file or file-like object
    dataset : str
        name of the Dataset in which the data is stored
    """
    with h5py.File(fpath, mode="r") as f:
        # Check whether Dataset exists
        if dataset not in f.keys():
            raise KeyError(
                f"WRADLIB: Cannot read Dataset <{dataset}> from hdf5 file <{f}>"
            )
        data = np.array(f[dataset][:])
        # get metadata
        metadata = {}
        for key in f[dataset].attrs.keys():
            metadata[key] = f[dataset].attrs[key]

    return data, metadata


def read_gpm(filename, bbox=None):
    """Reads GPM files for matching with GR

    Parameters
    ----------
    filename : str
        path of the GPM file
    bbox : dict
        dictionary with bounding box coordinates (lon, lat),
        defaults to None

    Returns
    -------
    gpm_data : dict
        dictionary of gpm data

    Examples
    --------
    See :ref:`/notebooks/match3d/wradlib_match_workflow.ipynb`.
    """
    pr_data = nc.Dataset(filename, mode="r")
    lon = pr_data["NS"].variables["Longitude"]
    lat = pr_data["NS"].variables["Latitude"]

    if bbox is not None:
        poly = [
            [bbox["left"], bbox["bottom"]],
            [bbox["left"], bbox["top"]],
            [bbox["right"], bbox["top"]],
            [bbox["right"], bbox["bottom"]],
            [bbox["left"], bbox["bottom"]],
        ]
        from wradlib.zonalstats import get_clip_mask

        mask = get_clip_mask(np.dstack((lon[:], lat[:])), poly)
    else:
        mask = np.ones_like(lon, dtype=bool, subok=False)

    mask = np.nonzero(np.count_nonzero(mask, axis=1))

    lon = lon[mask]
    lat = lat[mask]

    year = pr_data["NS"]["ScanTime"].variables["Year"][mask]
    month = pr_data["NS"]["ScanTime"].variables["Month"][mask]
    dayofmonth = pr_data["NS"]["ScanTime"].variables["DayOfMonth"][mask]
    # dayofyear = pr_data['NS']['ScanTime'].variables['DayOfYear'][mask]
    hour = pr_data["NS"]["ScanTime"].variables["Hour"][mask]
    minute = pr_data["NS"]["ScanTime"].variables["Minute"][mask]
    second = pr_data["NS"]["ScanTime"].variables["Second"][mask]
    # secondofday = pr_data['NS']['ScanTime'].variables['SecondOfDay'][mask]
    millisecond = pr_data["NS"]["ScanTime"].variables["MilliSecond"][mask]
    date_array = zip(
        year,
        month,
        dayofmonth,
        hour,
        minute,
        second,
        millisecond.astype(np.int32) * 1000,
    )
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in date_array]
    )

    sfc = pr_data["NS"]["PRE"].variables["landSurfaceType"][mask]
    pflag = pr_data["NS"]["PRE"].variables["flagPrecip"][mask]

    # bbflag = pr_data['NS']['CSF'].variables['flagBB'][mask]
    zbb = pr_data["NS"]["CSF"].variables["heightBB"][mask]
    # print(zbb.dtype)
    bbwidth = pr_data["NS"]["CSF"].variables["widthBB"][mask]
    qbb = pr_data["NS"]["CSF"].variables["qualityBB"][mask]
    qtype = pr_data["NS"]["CSF"].variables["qualityTypePrecip"][mask]
    ptype = pr_data["NS"]["CSF"].variables["typePrecip"][mask]

    quality = pr_data["NS"]["scanStatus"].variables["dataQuality"][mask]
    refl = pr_data["NS"]["SLV"].variables["zFactorCorrected"][mask]
    # print(pr_data['NS']['SLV'].variables['zFactorCorrected'])

    zenith = pr_data["NS"]["PRE"].variables["localZenithAngle"][mask]

    pr_data.close()

    # Check for bad data
    if max(quality) != 0:
        raise ValueError("GPM contains Bad Data")

    pflag = pflag.astype(np.int8)

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError(f"GPM Dimensions do not match! Needed 3, given {ndim}")

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    refl = np.flip(refl, axis=-1)

    # Change pflag=1 to pflag=2 to be consistent with 'Rain certain' in TRMM
    pflag[pflag == 1] = 2

    # Simplify the precipitation types
    ptype = (ptype / 1e7).astype(np.int16)

    # Simplify the surface types
    imiss = sfc == -9999
    sfc = (sfc / 1e2).astype(np.int16) + 1
    sfc[imiss] = 0

    # Set a quality indicator for the BB and precip type data
    # TODO: Why is the `quality` variable overwritten?

    quality = np.zeros((nscan, nray), dtype=np.uint8)

    i1 = ((qbb == 0) | (qbb == 1)) & (qtype == 1)
    quality[i1] = 1

    i2 = (qbb > 1) | (qtype > 2)
    quality[i2] = 2

    gpm_data = {}
    gpm_data.update(
        {
            "nscan": nscan,
            "nray": nray,
            "nbin": nbin,
            "date": pr_time,
            "lon": lon,
            "lat": lat,
            "pflag": pflag,
            "ptype": ptype,
            "zbb": zbb,
            "bbwidth": bbwidth,
            "sfc": sfc,
            "quality": quality,
            "refl": refl,
            "zenith": zenith,
        }
    )

    return gpm_data


def _get_gpm_group(filename, group, variables=None):
    """Return group as xarrax.Dataset from GPM file."""
    ds = xr.open_dataset(
        filename,
        group=group,
        decode_cf=False,
        engine="h5netcdf",
        backend_kwargs=dict(phony_dims="sort"),
    )
    for n, v in ds.items():
        dimnames = v.attrs.get("DimensionNames", False)
        if dimnames:
            vdims = v.attrs["DimensionNames"].split(",")
            dims = {dim: vdims[i] for i, dim in enumerate(v.dims)}
            ds[n] = v.swap_dims(dims)
    if variables is not None:
        keep = set(variables)
        ds = ds.drop_vars(set(ds.variables) ^ keep)
        if isinstance(variables, dict):
            ds = ds.rename(variables)
    return ds


def _get_gpm_time_group(filename, group):
    """Return time subgroup as xarrax.Dataset from GPM file."""
    variables = [
        "Year",
        "Month",
        "DayOfMonth",
        "Hour",
        "Minute",
        "Second",
        "MilliSecond",
    ]
    ds = _get_gpm_group(filename, group=group, variables=variables)
    date_array = zip(
        ds.Year.values,
        ds.Month.values,
        ds.DayOfMonth.values,
        ds.Hour.values,
        ds.Minute.values,
        ds.Second.values,
        ds.MilliSecond.values,
    )
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in date_array]
    )
    ds = ds.assign_coords({"date": (["nscan"], pr_time)})
    ds = ds.drop_vars(set(ds.variables) ^ set(["date"]))
    return ds


def open_gpm_dataset(filename, group):
    """Reads GPM files version `V07A`.

    Parameters
    ----------
    filename : str
        path of the GPM file
    group : str
        name of group

    Returns
    -------
    ds : xarray.Dataset
        xarray.Dataset representation of GPM file with requested `group`.
    """
    parents = group.split("/")[:-1]

    with h5netcdf.File(filename, mode="r", decode_vlen_strings=True) as f:
        grps = list(f[group].groups)

    root = _get_gpm_group(filename, group="/")
    root = root.rename_dims({list(root.dims)[0]: "nswath"})
    groups = {"root": root}
    if parents:
        for p in parents:
            pds = _get_gpm_group(filename, group=f"/{p}")
            groups[p] = pds
    subroot = _get_gpm_group(filename, group=f"/{group}")
    groups["subroot"] = subroot
    for grp in grps:
        gname = "/".join(["/" + group, grp])
        if grp == "ScanTime":
            groups[grp] = _get_gpm_time_group(filename, group=gname)
        else:
            groups[grp] = _get_gpm_group(filename, group=gname)
    ds = xr.merge(groups.values())
    return ds


def read_trmm(filename1, filename2, bbox=None):
    """Reads TRMM files for matching with GR

    Parameters
    ----------
    filename1 : str
        path of the TRMM 2A23 file
    filename2 : str
        path of the TRMM 2A25 file
    bbox : dict
        dictionary with bounding box coordinates (lon, lat),
        defaults to None

    Returns
    -------
    trmm_data : dict
        dictionary of trmm data

    Examples
    --------
    See :ref:`/notebooks/match3d/wradlib_match_workflow.ipynb`.
    """
    # trmm 2A23 and 2A25 data is hdf4
    pr_data1 = nc.Dataset(filename1, mode="r")
    pr_data2 = nc.Dataset(filename2, mode="r")

    lon = pr_data1.variables["Longitude"]
    lat = pr_data1.variables["Latitude"]

    if bbox is not None:
        poly = [
            [bbox["left"], bbox["bottom"]],
            [bbox["left"], bbox["top"]],
            [bbox["right"], bbox["top"]],
            [bbox["right"], bbox["bottom"]],
            [bbox["left"], bbox["bottom"]],
        ]
        from wradlib.zonalstats import get_clip_mask

        mask = get_clip_mask(np.dstack((lon[:], lat[:])), poly)
    else:
        mask = np.ones_like(lon, dtype=bool)

    mask = np.nonzero(np.count_nonzero(mask, axis=1))

    lon = pr_data1.variables["Longitude"][mask]
    lat = pr_data1.variables["Latitude"][mask]

    year = pr_data1.variables["Year"][mask]
    month = pr_data1.variables["Month"][mask]
    dayofmonth = pr_data1.variables["DayOfMonth"][mask]
    # dayofyear = pr_data1.variables['DayOfYear'][mask]
    hour = pr_data1.variables["Hour"][mask]
    minute = pr_data1.variables["Minute"][mask]
    second = pr_data1.variables["Second"][mask]
    # secondofday = pr_data1.variables['scanTime_sec'][mask]
    millisecond = pr_data1.variables["MilliSecond"][mask]
    date_array = zip(
        year,
        month,
        dayofmonth,
        hour,
        minute,
        second,
        millisecond.astype(np.int32) * 1000,
    )
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in date_array]
    )

    pflag = pr_data1.variables["rainFlag"][mask]
    ptype = pr_data1.variables["rainType"][mask]

    status = pr_data1.variables["status"][mask]
    zbb = pr_data1.variables["HBB"][mask].astype(np.float32)
    bbwidth = pr_data1.variables["BBwidth"][mask].astype(np.float32)

    quality = pr_data2.variables["dataQuality"][mask]
    refl = pr_data2.variables["correctZFactor"][mask] / 100.0
    zenith = pr_data2.variables["scLocalZenith"][mask]

    pr_data1.close()
    pr_data2.close()

    # mask array
    refl = np.ma.array(refl)

    # Ground clutter
    refl[refl == -8888.0] = np.ma.masked
    # Misssing data
    refl[refl == -9999.0] = np.ma.masked
    # Scaling
    refl /= 100.0

    # Check for bad data
    if max(quality) != 0:
        raise ValueError("TRMM contains Bad Data")

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError(f"TRMM Dimensions do not match! Needed 3, given {ndim}")

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    refl = np.flip(refl, axis=-1)

    # Simplify the precipitation flag
    ipos = (pflag >= 10) & (pflag <= 20)
    icer = pflag >= 20
    pflag[ipos] = 1
    pflag[icer] = 2

    # Simplify the precipitation types
    istr = (ptype >= 100) & (ptype <= 200)
    icon = (ptype >= 200) & (ptype <= 300)
    ioth = ptype >= 300
    inone = ptype == -88
    imiss = ptype == -99
    ptype[istr] = 1
    ptype[icon] = 2
    ptype[ioth] = 3
    ptype[inone] = 0
    ptype[imiss] = -1

    # Extract the surface type
    sfc = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = status == 168
    sfc[i0] = 0
    i1 = status % 10 == 0
    sfc[i1] = 1
    i2 = (status - 1) % 10 == 0
    sfc[i2] = 2
    i3 = (status - 3) % 10 == 0
    sfc[i3] = 3
    i4 = (status - 4) % 10 == 0
    sfc[i4] = 4
    i5 = (status - 5) % 10 == 0
    sfc[i5] = 5
    i9 = (status - 9) % 10 == 0
    sfc[i9] = 9

    # Extract 2A23 quality
    # TODO: Why is the `quality` variable overwritten?
    quality = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = status == 168
    quality[i0] = 0
    i1 = status < 50
    quality[i1] = 1
    i2 = (status >= 50) & (status < 109)
    quality[i2] = 2

    trmm_data = {}
    trmm_data.update(
        {
            "nscan": nscan,
            "nray": nray,
            "nbin": nbin,
            "date": pr_time,
            "lon": lon,
            "lat": lat,
            "pflag": pflag,
            "ptype": ptype,
            "zbb": zbb,
            "bbwidth": bbwidth,
            "sfc": sfc,
            "quality": quality,
            "refl": refl,
            "zenith": zenith,
        }
    )

    return trmm_data
