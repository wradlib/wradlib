#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# """
# Raw Data I/O
# ^^^^^^^^^^^^
#
# Please have a look at the tutorial
# :ref:`notebooks/fileio/wradlib_radar_formats.ipynb`
# for an introduction on how to deal with different file formats.
#
# .. autosummary::
#    :nosignatures:
#    :toctree: generated/
#
#    writePolygon2Text
#    read_EDGE_netcdf
#    read_generic_hdf5
#    read_generic_netcdf
#    read_OPERA_hdf5
#    read_GAMIC_hdf5
#    read_Rainbow
#    read_safnwc
#    write_raster_dataset
#    to_AAIGrid
#    to_GeoTIFF
#    to_hdf5
#    from_hdf5
#    read_raster_data
#    open_shape
#
# """

# standard libraries
from __future__ import absolute_import
import sys
import datetime as dt

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from StringIO import StringIO
    import io
except ImportError:
    from io import StringIO  # noqa
    import io

# from builtins import bytes, chr
from collections import OrderedDict
import re
import os
import warnings

# site packages
import h5py
import numpy as np
# ATTENTION: Needs to be imported AFTER h5py, otherwise ungraceful crash
import netCDF4 as nc
from osgeo import gdal, ogr, osr
from .. import util as util
from .. import georef as georef


def _write_polygon2txt(f, idx, vertices):
    f.write('%i %i\n' % idx)
    for i, vert in enumerate(vertices):
        f.write('%i ' % (i,))
        f.write('%f %f %f %f\n' % tuple(vert))


def writePolygon2Text(fname, polygons):
    """Writes Polygons to a Text file which can be interpreted by ESRI \
    ArcGIS's "Create Features from Text File (Samples)" tool.

    This is (yet) only a convenience function with limited functionality.
    E.g. interior rings are not yet supported.

    Parameters
    ----------
    fname : string
        name of the file to save the vertex data to
    polygons : list of lists
        list of polygon vertices.
        Each vertex itself is a list of 3 coordinate values and an
        additional value. The third coordinate and the fourth value may be nan.

    Returns
    -------
    None

    Note
    ----
    As Polygons are closed shapes, the first and the last vertex of each
    polygon **must** be the same!

    Examples
    --------
    Writes two triangle Polygons to a text file::

        poly1 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        poly2 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        polygons = [poly1, poly2]
        writePolygon2Text('polygons.txt', polygons)

    The resulting text file will look like this::

        Polygon
        0 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        1 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        END

    """
    with open(fname, 'w') as f:
        f.write('Polygon\n')
        count = 0
        for vertices in polygons:
            _write_polygon2txt(f, (count, 0), vertices)
            count += 1
        f.write('END\n')


def read_EDGE_netcdf(filename, enforce_equidist=False):
    """Data reader for netCDF files exported by the EDGE radar software

    The corresponding NetCDF files from the EDGE software typically contain
    only one variable (e.g. reflectivity) for one elevation angle (sweep).
    The elevation angle is specified in the attributes keyword "Elevation".

    Please note that the radar might not return data with equidistant azimuth
    angles. In case you need equidistant azimuth angles, please set
    enforce_equidist to True.

    Parameters
    ----------
    filename : string
        path of the netCDF file
    enforce_equidist : boolean
        Set True if the values of the azimuth angles should be forced to be
        equidistant; default value is False

    Returns
    -------
    output : :func:`numpy:numpy.array`
        of image data (dBZ), dictionary of attributes
    """
    try:
        # read the data from file
        dset = nc.Dataset(filename)
        data = dset.variables[dset.TypeName][:]
        # Check azimuth angles and rotate image
        az = dset.variables['Azimuth'][:]
        # These are the indices of the minimum and maximum azimuth angle
        ix_minaz = np.argmin(az)
        ix_maxaz = np.argmax(az)
        if enforce_equidist:
            az = np.linspace(np.round(az[ix_minaz], 2),
                             np.round(az[ix_maxaz], 2), len(az))
        else:
            az = np.roll(az, -ix_minaz)
        # rotate accordingly
        data = np.roll(data, -ix_minaz, axis=0)
        data = np.where(data == dset.getncattr('MissingData'), np.nan, data)
        # Ranges
        binwidth = ((dset.getncattr('MaximumRange-value') * 1000.) /
                    len(dset.dimensions['Gate']))
        r = np.arange(binwidth,
                      (dset.getncattr('MaximumRange-value') * 1000.) +
                      binwidth, binwidth)
        # collect attributes
        attrs = {}
        for attrname in dset.ncattrs():
            attrs[attrname] = dset.getncattr(attrname)
        # # Limiting the returned range
        # if range_lim and range_lim / binwidth <= data.shape[1]:
        #     data = data[:,:range_lim / binwidth]
        #     r = r[:range_lim / binwidth]
        # Set additional metadata attributes
        attrs['az'] = az
        attrs['r'] = r
        attrs['sitecoords'] = (attrs['Longitude'], attrs['Latitude'],
                               attrs['Height'])
        attrs['time'] = dt.datetime.utcfromtimestamp(attrs.pop('Time'))
        attrs['max_range'] = data.shape[1] * binwidth
    except:
        raise
    finally:
        dset.close()

    return data, attrs


def browse_hdf5_group(grp):
    """Browses one hdf5 file level
    """
    pass


def read_generic_hdf5(fname):
    """Reads hdf5 files according to their structure

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.

    Parameters
    ----------
    fname : string
        a hdf5 file path

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_radar_formats.ipynb#Generic-HDF5`.

    """
    f = h5py.File(fname, "r")
    fcontent = {}

    def filldict(x, y):
        # create a new container
        tmp = {}
        # add attributes if present
        if len(y.attrs) > 0:
            tmp['attrs'] = dict(y.attrs)
        # add data if it is a dataset
        if isinstance(y, h5py.Dataset):
            tmp['data'] = np.array(y)
        # only add to the dictionary, if we have something meaningful to add
        if tmp != {}:
            fcontent[x] = tmp

    f.visititems(filldict)

    f.close()

    return fcontent


def read_OPERA_hdf5(fname):
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
    fname : string
        a hdf5 file path

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure

    """
    f = h5py.File(fname, "r")

    # now we browse through all Groups and Datasets and store the info in one
    # dictionary
    fcontent = {}

    def filldict(x, y):
        if isinstance(y, h5py.Group):
            if len(y.attrs) > 0:
                fcontent[x] = dict(y.attrs)
        elif isinstance(y, h5py.Dataset):
            fcontent[x] = np.array(y)

    f.visititems(filldict)

    f.close()

    return fcontent


def read_gamic_scan_attributes(scan, scan_type):
    """Read attributes from one particular scan from a GAMIC hdf5 file

    Provided by courtesy of Kai Muehlbauer (University of Bonn).

    Parameters
    ----------
    scan : object
        scan object from hdf5 file
    scan_type : string
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
    sg1 = scan['how']

    # get scan attributes
    for attrname in list(sg1.attrs):
        sattrs[attrname] = sg1.attrs.get(attrname)
    sattrs['bin_range'] = sattrs['range_step'] * sattrs['range_samples']

    # get scan header
    ray_header = scan['ray_header']

    # az, el, zero_index for PPI scans
    if scan_type == 'PVOL':
        azi_start = ray_header['azimuth_start']
        azi_stop = ray_header['azimuth_stop']
        # Azimuth corresponding to 1st ray
        zero_index = np.where(azi_stop < azi_start)
        azi_stop[zero_index[0]] += 360
        zero_index = zero_index[0] + 1
        az = (azi_start + azi_stop) / 2
        az = np.roll(az, -zero_index, axis=0)
        az = np.round(az, 1)
        el = sg1.attrs.get('elevation')

    # az, el, zero_index for RHI scans
    if scan_type == 'RHI':
        ele_start = np.round(ray_header['elevation_start'], 1)
        ele_stop = np.round(ray_header['elevation_stop'], 1)
        angle_step = np.round(sattrs['angle_step'], 1)
        angle_step = int(np.round(sattrs['ele_stop'], 1) / angle_step)
        # Elevation corresponding to 1st ray
        if ele_start[0] < 0:
            ele_start = ele_start[1:]
            ele_stop = ele_stop[1:]
        zero_index = np.where(ele_stop > ele_start)
        zero_index = zero_index[0]  # - 1
        el = (ele_start + ele_stop) / 2
        el = np.round(el, 1)
        el = el[-angle_step:]

        az = sg1.attrs.get('azimuth')

    # save zero_index (first ray) to scan attributes
    sattrs['zero_index'] = zero_index[0]

    # create range array
    r = np.arange(sattrs['bin_range'],
                  sattrs['bin_range'] * sattrs['bin_count'] +
                  sattrs['bin_range'], sattrs['bin_range'])

    # save variables to scan attributes
    sattrs['az'] = az
    sattrs['el'] = el
    sattrs['r'] = r
    sattrs['Time'] = sattrs.pop('timestamp')
    sattrs['max_range'] = r[-1]

    return sattrs


def read_gamic_scan(scan, scan_type, wanted_moments):
    """Read data from one particular scan from GAMIC hdf5 file

    Provided by courtesy of Kai Muehlbauer (University of Bonn).

    Parameters
    ----------
    scan : object
        scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    wanted_moments : strings
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
        if 'moment' in mom:
            data1 = {}
            sg2 = scan[mom]
            actual_moment = sg2.attrs.get('moment').decode().upper()
            if (actual_moment in wanted_moments) or (wanted_moments == 'all'):
                # read attributes only once
                if not sattrs:
                    sattrs = read_gamic_scan_attributes(scan, scan_type)
                mdata = sg2[...]
                dyn_range_max = sg2.attrs.get('dyn_range_max')
                dyn_range_min = sg2.attrs.get('dyn_range_min')
                bin_format = sg2.attrs.get('format').decode()
                if bin_format == 'UV8':
                    div = 256.0
                else:
                    div = 65536.0
                mdata = (dyn_range_min + mdata *
                         (dyn_range_max - dyn_range_min) / div)

                if scan_type == 'PVOL':
                    # rotate accordingly
                    mdata = np.roll(mdata, -1 * sattrs['zero_index'], axis=0)

                if scan_type == 'RHI':
                    # remove first zero angles
                    sdiff = mdata.shape[0] - sattrs['el'].shape[0]
                    mdata = mdata[sdiff:, :]

                data1['data'] = mdata
                data1['dyn_range_max'] = dyn_range_max
                data1['dyn_range_min'] = dyn_range_min
                data[actual_moment] = data1

    return data, sattrs


def read_GAMIC_hdf5(filename, wanted_elevations=None, wanted_moments=None):
    """Data reader for hdf5 files produced by the commercial
    GAMIC Enigma V3 MURAN software

    Provided by courtesy of Kai Muehlbauer (University of Bonn). See GAMIC
    homepage for further info (http://www.gamic.com).

    Parameters
    ----------
    filename : string
        path of the gamic hdf5 file
    wanted_elevations : strings
        sequence of strings of elevation_angle(s) of scan (only needed for PPI)
    wanted_moments : strings
        sequence of strings of moment name(s)

    Returns
    -------
    data : dict
        dictionary of scan and moment data (numpy arrays)
    attrs : dict
        dictionary of attributes

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_radar_formats.ipynb#GAMIC-HDF5`.

    """

    # check elevations
    if wanted_elevations is None:
        wanted_elevations = 'all'

    # check wanted_moments
    if wanted_moments is None:
        wanted_moments = 'all'

    # read the data from file
    f = h5py.File(filename, 'r')

    # placeholder for attributes and data
    attrs = {}
    vattrs = {}
    data = {}

    # check if GAMIC file and
    try:
        f['how'].attrs.get('software')
    except KeyError:
        print("WRADLIB: File is no GAMIC hdf5!")
        raise

    # get scan_type (PVOL or RHI)
    scan_type = f['what'].attrs.get('object').decode()

    # single or volume scan
    if scan_type == 'PVOL':
        # loop over 'main' hdf5 groups (how, scanX, what, where)
        for n in list(f):
            if 'scan' in n:
                g = f[n]
                sg1 = g['how']

                # get scan elevation
                el = sg1.attrs.get('elevation')
                el = str(round(el, 2))

                # try to read scan data and attrs
                # if wanted_elevations are found
                if (el in wanted_elevations) or (wanted_elevations == 'all'):
                    sdata, sattrs = read_gamic_scan(scan=g,
                                                    scan_type=scan_type,
                                                    wanted_moments=wanted_moments)  # noqa
                    if sdata:
                        data[n.upper()] = sdata
                    if sattrs:
                        attrs[n.upper()] = sattrs

    # single rhi scan
    elif scan_type == 'RHI':
        # loop over 'main' hdf5 groups (how, scanX, what, where)
        for n in list(f):
            if 'scan' in n:
                g = f[n]
                # try to read scan data and attrs
                sdata, sattrs = read_gamic_scan(scan=g, scan_type=scan_type,
                                                wanted_moments=wanted_moments)
                if sdata:
                    data[n.upper()] = sdata
                if sattrs:
                    attrs[n.upper()] = sattrs

    # collect volume attributes if wanted data is available
    if data:
        vattrs['Latitude'] = f['where'].attrs.get('lat')
        vattrs['Longitude'] = f['where'].attrs.get('lon')
        vattrs['Height'] = f['where'].attrs.get('height')
        # check whether its useful to implement that feature
        # vattrs['sitecoords'] = (vattrs['Longitude'], vattrs['Latitude'],
        #                         vattrs['Height'])
        attrs['VOL'] = vattrs

    f.close()

    return data, attrs


def find_key(key, dictionary):
    """Searches for given key in given (nested) dictionary.

    Returns all found parent dictionaries in a list.

    Parameters
    ----------
    key : string
        the key to be searched for in the nested dict
    dictionary : dict
        the dictionary to be searched

    Returns
    -------
    output : dict
        a dictionary or list of dictionaries

    """
    for k, v in dictionary.items():
        if k == key:
            yield dictionary
        elif isinstance(v, dict):
            for result in find_key(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for result in find_key(key, d):
                        yield result


def decompress(data):
    """Decompression of data

    Parameters
    ----------
    data : string
        (from xml) data string containing compressed data.
    """
    zlib = util.import_optional('zlib')
    return zlib.decompress(data)


def get_RB_data_layout(datadepth):
    """Calculates DataWidth and DataType from given DataDepth of
    RAINBOW radar data

    Parameters
    ----------
    datadepth : int
        DataDepth as read from the Rainbow xml metadata.

    Returns
    -------
    datawidth : int
        Width in Byte of data.

    datatype : string
        conversion string .
    """

    if sys.byteorder != 'big':
        byteorder = '>'
    else:
        byteorder = '<'

    datawidth = int(datadepth / 8)

    if datawidth in [1, 2, 4]:
        datatype = byteorder + 'u' + str(datawidth)
    else:
        raise ValueError("Wrong DataDepth: %d. "
                         "Conversion only for depth 8, 16, 32" % datadepth)

    return datawidth, datatype


def get_RB_data_attribute(xmldict, attr):
    """Get Attribute `attr` from dict `xmldict`

    Parameters
    ----------
    xmldict : dict
        Blob Description Dictionary
    attr : string
        Attribute key

    Returns
    -------
    sattr : int
        Attribute Values

    """

    try:
        sattr = int(xmldict['@' + attr])
    except KeyError:
            raise KeyError('Attribute @{0} is missing from '
                           'Blob Description. There may be some '
                           'problems with your file'.format(attr))
    return sattr


def get_RB_blob_attribute(blobdict, attr):
    """Get Attribute `attr` from dict `blobdict`

    Parameters
    ----------
    blobdict : dict
        Blob Description Dictionary
    attr : string
        Attribute key

    Returns
    -------
    ret : Attribute Value

    """
    try:
        value = blobdict['BLOB']['@' + attr]
    except KeyError:
        raise KeyError('Attribute @' + attr + ' is missing from Blob.' +
                       'There may be some problems with your file')

    return value


def get_RB_blob_data(datastring, blobid):
    """ Read BLOB data from datastring and return it

    Parameters
    ----------
    datastring : string
        Blob Description String

    blobid : int
        Number of requested blob

    Returns
    -------
    data : string
        Content of blob

    """
    xmltodict = util.import_optional('xmltodict')

    start = 0
    searchString = '<BLOB blobid="{0}"'.format(blobid)
    start = datastring.find(searchString.encode(), start)
    if start == -1:
        raise EOFError('Blob ID {0} not found!'.format(blobid))
    end = datastring.find(b'>', start)
    xmlstring = datastring[start:end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring.decode() + '</BLOB>')
    cmpr = get_RB_blob_attribute(xmldict, 'compression')
    size = int(get_RB_blob_attribute(xmldict, 'size'))
    data = datastring[end + 2:end + 2 + size]  # read blob data to string

    # decompress if necessary
    # the first 4 bytes are neglected for an unknown reason
    if cmpr == "qt":
        data = decompress(data[4:])

    return data


def map_RB_data(data, datadepth):
    """ Map BLOB data to correct DataWidth and Type and convert it
    to numpy array

    Parameters
    ----------
    data : string
        Blob Data
    datadepth : int
        bit depth of Blob data

    Returns
    -------
    data : numpy array
        Content of blob
    """
    flagdepth = None
    if datadepth < 8:
        flagdepth = datadepth
        datadepth = 8

    datawidth, datatype = get_RB_data_layout(datadepth)

    # import from data buffer well aligned to data array
    data = np.ndarray(shape=(int(len(data) / datawidth),),
                      dtype=datatype, buffer=data)

    if flagdepth:
        data = np.unpackbits(data)

    return data


def get_RB_data_shape(blobdict):
    """
    Retrieve correct BLOB data shape from blobdict

    Parameters
    ----------
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    tuple : shape
        shape of data
    """
    # this is a bit hacky, but we do not know beforehand,
    # so we extract this on the run
    try:
        dim0 = get_RB_data_attribute(blobdict, 'rays')
        dim1 = get_RB_data_attribute(blobdict, 'bins')
        # if rays and bins are found, return both
        return dim0, dim1
    except KeyError as e1:
        try:
            # if only rays is found, return rays
            return dim0
        except UnboundLocalError:
            try:
                # if both rays and bins not found assuming pixmap
                dim0 = get_RB_data_attribute(blobdict, 'rows')
                dim1 = get_RB_data_attribute(blobdict, 'columns')
                dim2 = get_RB_data_attribute(blobdict, 'depth')
                if dim2 < 8:
                    # if flagged data return rows x columns x depth
                    return dim0, dim1, dim2
                else:
                    # otherwise just rows x columns
                    return dim0, dim1
            except KeyError as e2:
                # if no some keys are missing, print errors and raise
                print(e1)
                print(e2)
                raise


def get_RB_blob_from_string(datastring, blobdict):
    """
    Read BLOB data from datastring and return it as numpy array with correct
    dataWidth and shape

    Parameters
    ----------
    datastring : string
        Blob Description String
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    data : numpy array
        Content of blob as numpy array
    """

    blobid = get_RB_data_attribute(blobdict, 'blobid')
    data = get_RB_blob_data(datastring, blobid)

    # map data to correct datatype and width
    datadepth = get_RB_data_attribute(blobdict, 'depth')
    data = map_RB_data(data, datadepth)

    # reshape data
    data.shape = get_RB_data_shape(blobdict)

    return data


def get_RB_blob_from_file(f, blobdict):
    """
    Read BLOB data from file and return it with correct
    dataWidth and shape

    Parameters
    ----------
    f : string or file handle
        File handle of or path to Rainbow file
    blobdict : dict
        Blob Dict

    Returns
    -------
    data : numpy array
        Content of blob as numpy array
    """

    # Try to read the data from a file handle
    try:
        f.seek(0, 0)
        fid = f
        datastring = fid.read()
    except AttributeError:
        # If we did not get a file handle, assume that we got a filename,
        # get a file handle and read the data
        try:
            fid = open(f, "rb")
            datastring = fid.read()
            fid.close()
        except IOError:
            print("WRADLIB: Error opening Rainbow file ", f)
            raise IOError

    data = get_RB_blob_from_string(datastring, blobdict)

    return data


def get_RB_file_as_string(fid):
    """ Read Rainbow File Contents in dataString

    Parameters
    ----------
    fid : file handle
        File handle of Data File

    Returns
    -------
    dataString : string
        File Contents as dataString
    """

    try:
        dataString = fid.read()
    except:
        raise IOError('Could not read from file handle')

    return dataString


def get_RB_blobs_from_file(fid, rbdict):
    """Read all BLOBS found in given nested dict, loads them from file
    given by filename and add them to the dict at the appropriate position.

    Parameters
    ----------
    fid : file handle
        File handle of Data File
    rbdict : dict
        Rainbow file Contents

    Returns
    -------
    ret : dict
        Rainbow File Contents
    """

    blobs = list(find_key('@blobid', rbdict))

    datastring = get_RB_file_as_string(fid)
    for blob in blobs:
        data = get_RB_blob_from_string(datastring, blob)
        blob['data'] = data

    return rbdict


def get_RB_header(fid):
    """Read Rainbow Header from filename, converts it to a dict and returns it

    Parameters
    ----------
    fid : file handle
        File handle of Data File

    Returns
    -------
    object : dictionary
        Rainbow File Contents

    """

    # load the header lines, i.e. the XML part
    endXMLmarker = b"<!-- END XML -->"
    header = b""
    line = b""

    try:
        while not line.startswith(endXMLmarker):
            header += line[:-1]
            line = fid.readline()
            if len(line) == 0:
                break
    except:
        raise IOError('Could not read from file handle')

    xmltodict = util.import_optional('xmltodict')

    return xmltodict.parse(header)


def read_Rainbow(f, loaddata=True):
    """Reads Rainbow files files according to their structure

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the XML
    outline in the original data file.

    The radar data will be extracted from the data blobs, converted and added
    to the dict with key 'data' at the place where the @blobid was pointing
    from.

    Parameters
    ----------
    f : string or file handle
        a rainbow file path or file handle of rainbow file
    loaddata : bool
        True | False, If False function returns only metadata

    Returns
    -------
    rbdict : dict
        a dictionary that contains both data and metadata according to the
        original rainbow file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_load_rainbow_example.ipynb`.

    .. versionchanged 0.10.0
       Added reading from file handles.

    """

    # Check if a file handle has been passed
    try:
        f.seek(0, 0)
        fid = f
    except AttributeError:
        # If we did not get a file handle, assume that we got a filename and
        #  get a file handle for the corresponding file
        try:
            fid = open(f, "rb")
        except IOError:
            print("WRADLIB: Error opening Rainbow file ", f)
            raise IOError

    rbdict = get_RB_header(fid)

    if loaddata:
        rbdict = get_RB_blobs_from_file(fid, rbdict)

    return rbdict


def to_pickle(fpath, obj):
    """Pickle object <obj> to file <fpath>
    """
    output = open(fpath, 'wb')
    pickle.dump(obj, output)
    output.close()


def from_pickle(fpath):
    """Return pickled object from file <fpath>
    """
    pkl_file = open(fpath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    return obj


def to_hdf5(fpath, data, mode="w", metadata=None,
            dataset="data", compression="gzip"):
    """Quick storage of one <data> array and a <metadata> dict in an hdf5 file

    This is more efficient than pickle, cPickle or numpy.save. The data is
    stored in a subgroup named ``data`` (i.e. hdf5file["data").
    See :meth:`~wradlib.io.from_hdf5` for retrieving stored data.

    Parameters
    ----------
    fpath : string
        path to the hdf5 file
    data : :func:`numpy:numpy.array`
    mode : string
        file open mode, defaults to "w" (create, truncate if exists)
    metadata : dict
        dictionary of data's attributes
    dataset : string
        describing dataset
    compression : string
        h5py compression type {"gzip"|"szip"|"lzf"}, see h5py documentation
        for details
    """
    f = h5py.File(fpath, mode=mode)
    dset = f.create_dataset(dataset, data=data, compression=compression)
    # store metadata
    if metadata:
        for key in metadata.keys():
            dset.attrs[key] = metadata[key]
    # close hdf5 file
    f.close()


def from_hdf5(fpath, dataset="data"):
    """Loading data from hdf5 files that was stored by \
    :meth:`~wradlib.io.to_hdf5`

    Parameters
    ----------
    fpath : string
        path to the hdf5 file
    dataset : string
        name of the Dataset in which the data is stored
    """
    f = h5py.File(fpath, mode="r")
    # Check whether Dataset exists
    if dataset not in f.keys():
        print("Cannot read Dataset <%s> from hdf5 file <%s>" % (dataset, f))
        f.close()
        sys.exit()
    data = np.array(f[dataset][:])
    # get metadata
    metadata = {}
    for key in f[dataset].attrs.keys():
        metadata[key] = f[dataset].attrs[key]
    f.close()
    return data, metadata


def read_safnwc(filename):
    """Read MSG SAFNWC hdf5 file into a gdal georeferenced object

    Parameters
    ----------
    filename : string
        satellite file name

    Returns
    -------
    ds : gdal.DataSet
        with satellite data
    """

    root = gdal.Open(filename)
    ds1 = gdal.Open('HDF5:' + filename + '://CT')
    ds = gdal.GetDriverByName('MEM').CreateCopy('out', ds1, 0)

    # name = os.path.basename(filename)[7:11]
    try:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(ds.GetMetadata()["PROJECTION"])
    except Exception:
        raise NameError("No metadata for satellite file %s" % filename)
    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj.ExportToWkt())
    ds.SetGeoTransform([float(x) for x in geotransform])
    return ds


def read_netcdf_group(ncid):
    """Reads netcdf (nested) groups into python dictionary with corresponding
    structure.

    Note
    ----
    The returned dictionary could be quite big, depending on the content of
    the file.

    Parameters
    ----------
    ncid : object
        nc/group id from netcdf file

    Returns
    -------
    out : ordered dict
        an ordered dictionary that contains both data and metadata
        according to the original netcdf file structure
    """
    out = OrderedDict()

    # attributes
    for k, v in ncid.__dict__.items():
        out[k] = v

    # groups
    if ncid.groups:
        for k, v in ncid.groups.items():
            out[k] = read_netcdf_group(v)

    # dimensions
    dimids = np.array([])
    if ncid.dimensions:
        dim = OrderedDict()
        for k, v in ncid.dimensions.items():
            tmp = OrderedDict()
            try:
                tmp['data_model'] = v._data_model
            except AttributeError:
                pass
            try:
                tmp['size'] = v.__len__()
            except AttributeError:
                pass
            tmp['dimid'] = v._dimid
            dimids = np.append(dimids, v._dimid)
            tmp['grpid'] = v._grpid
            tmp['isunlimited'] = v.isunlimited()
            dim[k] = tmp
        # Usually, the dimensions should be ordered by dimid automatically
        # in case netcdf used OrderedDict. However, we should double check
        if np.array_equal(dimids, np.sort(dimids)):
            # is already sorted
            out['dimensions'] = dim
        else:
            # need to sort
            dim2 = OrderedDict()
            keys = dim.keys()
            for dimid in np.sort(dimids):
                dim2[keys[dimid]] = dim[keys[dimid]]
            out["dimensions"] = dim2

    # variables
    if ncid.variables:
        var = OrderedDict()
        for k, v in ncid.variables.items():
            tmp = OrderedDict()
            for k1 in v.ncattrs():
                tmp[k1] = v.getncattr(k1)
            if v[:].dtype.kind == 'S':
                try:
                    tmp['data'] = nc.chartostring(v[:])
                except:
                    tmp['data'] = v[:]
            else:
                tmp['data'] = v[:]
            var[k] = tmp
        out['variables'] = var

    return out


def read_generic_netcdf(fname):
    """Reads netcdf files and returns a dictionary with corresponding
    structure.

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.

    Please see the examples below on how to browse through a return object. The
    most important keys are the "dimensions" which define the shape of the data
    arrays, and the "variables" which contain the actual data and typically
    also the data that define the dimensions (e.g. sweeps, azimuths, ranges).
    These keys should be present in any netcdf file.

    Note
    ----
    The returned dictionary could be quite big, depending on the content of
    the file.

    Parameters
    ----------
    fname : string
        a netcdf file path

    Returns
    -------
    out : ordered dict
        an ordered dictionary that contains both data and metadata according
        to the original netcdf file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_generic_netcdf_example.ipynb`.

    """
    try:
        ncid = nc.Dataset(fname, 'r')
    except RuntimeError:
        print("wradlib1: Could not read %s." % fname)
        print("Check whether file exists, and whether it is a netCDF file.")
        print("Raising exception...")
        raise

    out = read_netcdf_group(ncid)

    ncid.close()

    return out


def _check_arguments(fpath, data):
    """Helper function to check input arguments for GIS export function
    """
    # Check arguments
    if not type(data) == np.ndarray:
        raise Exception("Argument 'data' in has to be of type numpy.ndarray. "
                        "Found argument of %s instead" % str(type(data)))

    if not data.ndim == 2:
        raise Exception("Argument 'data' has to be 2-dimensional. "
                        "Found %d dimensions instead" % data.ndim)

    if not os.path.exists(os.path.dirname(fpath)):
        raise Exception("Directory does not exist: %s" %
                        os.path.dirname(fpath))


def write_raster_dataset(fpath, dataset, format, options=None, remove=False):
    """ Write raster dataset to file format

    .. versionadded 0.10.0

    Parameters
    ----------
    fpath : string
        A file path - should have file extension corresponding to format.
    dataset : gdal.Dataset
        gdal raster dataset
    format : string
        gdal raster format string
    options : list
        List of option strings for the corresponding format.
    remove : bool
        if True, existing gdal.Dataset will be
        removed before creation

    Note
    ----
    For format and options refer to
    `formats_list <http://www.gdal.org/formats_list.html>`_.

    Examples
    --------
    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.
    """
    # check for option list
    if options is None:
        options = []

    driver = gdal.GetDriverByName(format)
    metadata = driver.GetMetadata()

    # check driver capability
    if 'DCAP_CREATECOPY' in metadata and metadata['DCAP_CREATECOPY'] != 'YES':
        assert "Driver %s doesn't support CreateCopy() method.".format(format)

    if remove:
        if os.path.exists(fpath):
            driver.Delete(fpath)

    target = driver.CreateCopy(fpath, dataset, 0, options)
    del target


@util.deprecated(write_raster_dataset)
def to_AAIGrid(fpath, data, xllcorner, yllcorner, cellsize,
               nodata=-9999, proj=None, fmt="%.2f", to_esri=True):
    """Write a cartesian grid to an Arc/Info ASCII grid file.

    .. versionadded:: 0.6.0

    The function writes a text file to ``fpath`` that contains the header info
    and the grid data passed with the argument ``data``. Find details on ESRI
    grids (or Arc/Info ASCII grids) on wikipedia :cite:`ESRI-grid`.
    This should work for most GIS software systems
    (tested for QGIS and ESRI ArcGIS).

    In case a GDAL SpatialReference object (argument ``proj``) is passed,
    the function will also try to write an accompanying projection (``.prj``)
    file that has the same file name, but a different extension.

    Please refer to :mod:`wradlib.georef`
    to see how to create SpatialReference objects from e.g.
    EPSG codes :meth:`~wradlib.georef.epsg_to_osr`,
    PROJ.4 strings :meth:`~wradlib.georef.proj4_to_osr`,
    or WKT strings :meth:`~wradlib.georef.wkt_to_osr`. Other projections
    are addressed by :meth:`~wradlib.georef.create_osr`.

    Parameters
    ----------
    fpath : string
        a file path - must have a ".txt" or ".asc" extension.
    data : :func:`numpy:numpy.array`
        two dimensional numpy array of type integer or float
    xllcorner : float
        x coordinate of the lower left corner of the grid
    yllcorner : float
        y coordinate of the lower left corner of the grid
    cellsize : float
        size of the grid cells - needs to be consistent with proj
    nodata : float
        no data flag
    proj : osr.SpatialReference
        a SpatialReference of class 'osr.SpatialReference'
    fmt : string
        format string
    to_esri : bool
        set True if the prj file should be made ESRI compatible

    Note
    ----
    Has been tested with ESRI ArcGIS 9.3 and QGIS 2.8.

    Examples
    --------
    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.

    """
    # Check input data
    _check_arguments(fpath, data)

    ext = os.path.splitext(fpath)[-1]
    if ext not in [".txt", ".asc"]:
        raise Exception("File name extension should be either "
                        "'.txt' or '.asc'. Found extension instead: %s" % ext)

    # Define header
    header = ("\n"
              "ncols         %d\n"
              "nrows         %d\n"
              "xllcorner     %.4f\n"
              "yllcorner     %.4f\n"
              "cellsize      %.4f\n"
              "NODATA_value  %.1f\n" % (data.shape[0], data.shape[1],
                                        xllcorner, yllcorner, cellsize,
                                        nodata))

    # Replace NaNs by NoData
    # ...but we do not want to manipulate the original array!
    data = data.copy()
    data[np.isnan(data)] = nodata

    # Write grid file
    # with open(fpath, "w") as f:
    #    f.write(header)
    np.savetxt(fpath, np.flipud(data), fmt=fmt, header=header, comments='')

    if proj is None:
        # No prj file will be written
        return 0
    elif not type(proj) == osr.SpatialReference:
        raise Exception("Expected 'proj' argument of type "
                        "'osr.SpatialReference', but got %s. See library "
                        "reference for wradlib.georef on how to create "
                        "SpatialReference objects from different sources "
                        "(proj4, WKT, EPSG, ...)." % type(proj))

    if to_esri:
        # Make a copy before manipulation
        proj = proj.Clone()
        proj.MorphToESRI()

    # Write projection file
    prjpath = os.path.splitext(fpath)[0] + ".prj"
    with open(prjpath, "w") as f:
        f.write(proj.ExportToWkt())

    return 0


@util.deprecated(write_raster_dataset)
def to_GeoTIFF(fpath, data, geotransform, nodata=-9999, proj=None):
    """Write a cartesian grid to a GeoTIFF file.

    .. versionadded:: 0.6.0

    The function writes a GeoTIFF file to ``fpath`` that contains the grid data
    passed with the argument ``data``. For details on the GeoTIFF format
    see e.g. wikipedia :cite:`GeoTIFF`.

    Warning
    -------
    The GeoTIFF files produced by this function might not work with ESRI
    ArcGIS, depending on the projection. Problems are particularly expected
    with the RADOLAN projection, due to inconsistencies in the definition of
    polar stereographic projections between GDAL and ESRI ArcGIS.

    The projection information (argument ``proj``) needs to be passed as a GDAL
    SpatialReference object. Please refer to :mod:`wradlib.georef`
    to see how to create SpatialReference objects from e.g.
    EPSG codes :meth:`~wradlib.georef.epsg_to_osr`,
    PROJ.4 strings :meth:`~wradlib.georef.proj4_to_osr`,
    or WKT strings :meth:`~wradlib.georef.wkt_to_osr`. Other projections
    are addressed by :meth:`~wradlib.georef.create_osr`.

    Writing a GeoTIFF file requires a ``geotransform`` list to define how to
    compute map coordinates from grid indices. The list needs to contain the
    following items: top left x, w-e pixel resolution, rotation, top left y,
    rotation, n-s pixel resolution. The unit of the pixel resolution has to be
    consistent with the projection information. **BE CAREFUL**: You need to
    consider whether your grid coordinates define the corner (typically lower
    left) or the center of your pixels.
    And since the ``geotransform`` is used to define the grid from the top-left
    corner, the n-s pixel resolution is usually a negative value.

    Here is an example of the ``geotransform`` that worked e.g. with RADOLAN
    grids. Notice that the RADOLAN coordinates returned by wradlib refer to the
    lower left pixel corners, so you have to add another pixel unit to the top
    left y coordinate in order to define the top left corner of the
    bounding box::

        import wradlib
        xy = wradlib.georef.get_radolan_grid(900,900)
        # top left x, w-e pixel size, rotation, top left y, rotation,
        # n-s pixel size
        geotransform = [xy[0,0,0], 1., 0, xy[-1,-1,1]+1., 0, -1.]

    Parameters
    ----------
    fpath : string
        a file path - must have a ".txt" or ".asc" extension.
    data : :func:`numpy:numpy.array`
        two dimensional numpy array of type integer or float
    geotransform : sequence
        sequence of length 6 (# top left x, w-e pixel size, rotation,
        top left y, rotation, n-s pixel size)
    nodata : float
        no data flag
    proj : osr.SpatialReference
        a SpatialReference of class 'osr.SpatialReference'

    Note
    ----
    Has been tested with ESRI ArcGIS 9.3 and QGIS 2.8.

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.

    """
    # Check input data
    _check_arguments(fpath, data)
    ext = os.path.splitext(fpath)[-1]
    if ext not in [".tif", ".tiff"]:
        raise Exception("File name extension should be either '.tif' or "
                        "'.tiff'. Found extension instead: %s" % ext)

    # Set up our export object
    driver = gdal.GetDriverByName("GTiff")

    # Mapping ur data type to GDAL data types
    if data.dtype == "float64":
        gdal_dtype = gdal.GDT_Float64
    elif data.dtype == "float32":
        gdal_dtype = gdal.GDT_Float32
    elif data.dtype == "int32":
        gdal_dtype = gdal.GDT_Int32
    elif data.dtype == "int16":
        gdal_dtype = gdal.GDT_Int16
    else:
        raise Exception("The data type of your input array data should be one "
                        "of the following: float64, float32, int32, int16. "
                        "You can use numpy's 'astype' method to convert "
                        "your array to the desired data type.")

    # Creat our export object
    ds = driver.Create(fpath, data.shape[0], data.shape[1], 1, gdal_dtype)

    # set the reference info
    if proj is None:
        pass
    elif not isinstance(proj, osr.SpatialReference):
        raise Exception("Expected 'proj' argument of type "
                        "'osr.SpatialReference', but got %s. See library "
                        "reference for wradlib.georef on how to create "
                        "SpatialReference objects from different sources "
                        "(proj4, WKT, EPSG, ...)." % type(proj))
    else:
        ds.SetProjection(proj.ExportToWkt())

    # top left x, w-e pixel resolution, rotation, top left y, rotation,
    # n-s pixel resolution
    ds.SetGeoTransform(geotransform)

    # Replace NaNs by NoData
    # ...but we do not want to manipulate the original array!
    data = data.copy()
    data[np.isnan(data)] = nodata
    # and replace them by NoData flag
    ds.GetRasterBand(1).SetNoDataValue(nodata)

    # Write data
    ds.GetRasterBand(1).WriteArray(np.flipud(data))

    # This is how we close the export file
    ds = None


@util.deprecated(georef.extract_raster_dataset)
def read_raster_data(filename, driver=None, **kwargs):
    """Read raster data

    Reads raster data files supported by GDAL. If driver is not given,
    GDAL tries to autodetect the file format. Works well in most cases.

    .. seealso:: http://www.gdal.org/formats_list.html

    Resamples data on the fly if special keyword arguments are given

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        filename of raster file
    driver : string
        GDAL Raster Format Code
        see: http://www.gdal.org/formats_list.html
        if no driver is given gdal is autodetecting which may fail

    Keyword Arguments
    -----------------
    spacing : float or tuple of two floats
        pixel spacing of resampled dataset, same unit as pixel coordinates
    size : tuple of two ints
        X/YRasterSize of resampled dataset
    resample : GDALResampleAlg
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12

    Returns
    -------
    coords : :func:`numpy:numpy.array`
        numpy ndarray of raster coordinates
    values : :func:`numpy:numpy.array`
        numpy 2darray of raster values

    Examples
    --------

    See :ref:`notebooks/beamblockage/wradlib_beamblock.ipynb` and
    :ref:`notebooks/visualisation/wradlib_overlay.ipynb`

    """

    dataset = open_raster(filename, driver=driver)

    if 'spacing' in kwargs or 'size' in kwargs:
        dataset1 = georef.resample_raster_dataset(dataset, **kwargs)
    else:
        dataset1 = dataset

    # we have to flipud data, because raster data is origin "upper left"
    values = np.flipud(georef.read_gdal_values(dataset1))
    coords = np.flipud(georef.read_gdal_coordinates(dataset1,
                                                    mode='centers',
                                                    z=False))

    # close dataset
    dataset1 = None

    return coords, values


def open_shape(filename, driver=None):
    """
    Open shapefile, return gdal.Dataset and OGR.Layer

    .. warning:: dataset and layer have to live in the same context,
                 if dataset is deleted all layer references will get lost

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        shapefile name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : gdal.Dataset
        dataset
    layer : ogr.Layer
        layer
    """

    if driver is None:
        driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(filename)
    if dataset is None:
        print('Could not open file')
        raise IOError
    layer = dataset.GetLayer()
    return dataset, layer


def open_raster(filename, driver=None):
    """
    Open raster file, return gdal.Dataset

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        raster file name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : gdal.Dataset
        dataset
    """

    dataset = gdal.Open(filename)

    if driver:
        gdal.GetDriverByName(driver)

    return dataset


if __name__ == '__main__':
    print('wradlib: Calling module <io> as main...')
