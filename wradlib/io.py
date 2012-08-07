#-------------------------------------------------------------------------------
# Name:         clutter
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Raw Data I/O
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   getDXTimestamp
   readDX
   writePolygon2Text
   read_EDGE_netcdf
   read_BUFR

"""

import re
import numpy as np
import netCDF4 as nc
import datetime as dt
import pytz
import wradlib.bufr as bufr


# current DWD file naming pattern (2008) for example:
# raa00-dx_10488-200608050000-drs---bin
dwdpattern = re.compile('raa..-(..)[_-]([0-9]{5})-([0-9]*)-(.*?)---bin')


def _getTimestampFromFilename(filename):
    """Helper function doing the actual work of getDXTimestamp"""
    time = dwdpattern.search(filename).group(3)
    if len(time) == 10:
        time = '20' + time
    return dt.datetime.strptime(time, '%Y%m%d%H%M')


def getDXTimestamp(name, tz=pytz.utc):
    """converts a dx-timestamp (as part of a dx-product filename) to a python
    datetime.object.

    Parameters
    ----------
    name : string representing a DWD product name

    tz : timezone object (see pytz package or datetime module for explanation)
         in case the timezone of the data is not UTC

    opt : currently unused

    Returns
    -------
    time : timezone-aware datetime.datetime object
    """
    return _getTimestampFromFilename(name).replace(tzinfo=tz)


def unpackDX(raw):
    """function removes DWD-DX-product bit-13 zero packing"""
    # data is encoded in the first 12 bits
    data = 4095
    # the zero compression flag is bit 13
    flag = 4096

    beam = []

##    # naive version
##    # 49193 function calls in 0.772 CPU seconds
##    # 20234 function calls in 0.581 CPU seconds
##    for item in raw:
##        if item & flag:
##            beam.extend([0]* (item & data))
##        else:
##            beam.append(item & data)

    # performance version - hopefully
    # 6204 function calls in 0.149 CPU seconds

    # get all compression cases
    flagged = np.where(raw & flag)[0]

    # if there is no zero in the whole data, we can return raw as it is
    if flagged.size == 0:
        assert raw.size == 128
        return raw

    # everything until the first flag is normal data
    beam.extend(raw[0:flagged[0]])

    # iterate over all flags except the last one
    for this, next in zip(flagged[:-1],flagged[1:]):
        # create as many zeros as there are given within the flagged
        # byte's data part
        beam.extend([0]* (raw[this] & data))
        # append the data until the next flag
        beam.extend(raw[this+1:next])

    # process the last flag
    # add zeroes
    beam.extend([0]* (raw[flagged[-1]] & data))

    # add remaining data
    beam.extend(raw[flagged[-1]+1:])

    # return the data
    return np.array(beam)


def readDX(filename):
    r"""Data reader for German Weather Service DX raw radar data files
    developed by Thomas Pfaff.

    The algorith basically unpacks the zeroes and returns a regular array of
    360 x 128 data values.

    Parameters
    ----------
    filename : binary file of DX raw data

    Returns
    -------
    data : numpy array of image data [dBZ]; shape (360,128)

    attributes : dictionary of attributes - currently implemented keys:

        - 'azim' - azimuths np.array of shape (360,)
        - 'elev' - elevations (1 per azimuth); np.array of shape (360,)
        - 'clutter' - clutter mask; boolean array of same shape as `data`;
            corresponds to bit 15 set in each dataset.
    """

    azimuthbitmask = 2**(14-1)
    databitmask = 2**(13-1) - 1
    clutterflag = 2**15
    dataflag = 2**13 -1
    # open the DX file in binary mode for reading
    if type(filename) == file:
        f = filename
    else:
        f = open(filename, 'rb')

    # the static part of the DX Header is 68 bytes long
    # after that a variable message part is appended, which apparently can
    # become quite long. Therefore we do it the dynamic way.
    staticheadlen = 68
    statichead = f.read(staticheadlen)

    # find MS and extract following number
    msre = re.compile('MS([ 0-9]{3})')
    mslen = int(msre.search(statichead).group(1))
    # add to headlength and read that
    headlen = staticheadlen + mslen + 1

    # this is now our first header length guess
    # however, some files have an additional 0x03 byte after the first one
    # (older files or those from the Uni Hannover don't, newer have it, if
    # the header would end after an uneven number of bytes)
    #headlen = headend
    f.seek(headlen)
    # so we read one more byte
    void = f.read(1)
    # and check if this is also a 0x03 character
    if void == chr(3):
        headlen = headlen + 1

    # rewind the file
    f.seek(0)

    # read the actual header
    header = f.read(headlen)

    # we can interpret the rest directly as a 1-D array of 16 bit unsigned ints
    raw = np.fromfile(f, dtype='uint16')

    # reading finished, close file.
    f.close()

    # a new ray/beam starts with bit 14 set
    # careful! where always returns its results in a tuple, so in order to get
    # the indices we have to retrieve element 0 of this tuple
    newazimuths = np.where( raw == azimuthbitmask )[0]  ###Thomas kontaktieren!!!!!!!!!!!!!!!!!!!

    # for the following calculations it is necessary to have the end of the data
    # as the last index
    newazimuths = np.append(newazimuths,len(raw))

    # initialize our list of rays/beams
    beams = []
    # initialize our list of elevations
    elevs = []
    # initialize our list of azimuths
    azims = []

    # iterate over all beams
    for i in range(newazimuths.size-1):
        # unpack zeros
        beam = unpackDX(raw[newazimuths[i]+3:newazimuths[i+1]])
        # the beam may regularly only contain 128 bins, so we
        # explicitly cut that here to get a rectangular data array
        beams.append(beam[0:128])
        elevs.append((raw[newazimuths[i]+2] & databitmask)/10.)
        azims.append((raw[newazimuths[i]+1] & databitmask)/10.)

    beams = np.array(beams)

    attrs =  {}
    attrs['elev']  = np.array(elevs)
    attrs['azim'] = np.array(azims)
    attrs['clutter'] = (beams & clutterflag) != 0

    # converting the DWD rvp6-format into dBZ data and return as numpy array together with attributes
    return (beams & dataflag) * 0.5 - 32.5, attrs


def _write_polygon2txt(f, idx, vertices):
    f.write('%i %i\n'%idx)
    for i, vert in enumerate(vertices):
        f.write('%i '%(i,))
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

    Notes
    -----
    As Polygons are closed shapes, the first and the last vertex of each
    polygon **must** be the same!

    Examples
    --------
    Writes two triangle Polygons to a text file

    >>> poly1 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
    >>> poly2 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
    >>> polygons = [poly1, poly2]
    >>> writePolygon2Text('polygons.txt', polygons)

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


def read_EDGE_netcdf(filename, range_lim = 200000.):
    """Data reader for netCDF files exported by the EDGE radar software

    Parameters
    ----------
    filename : path of the netCDF file
    range_lim : range limitation [m] of the returned radar data
                (200000 per default)

    Returns
    -------
    output : numpy array of image data (dBZ), dictionary of attributes

    """
    # read the data from file
    dset = nc.Dataset(filename)
    data = dset.variables[dset.TypeName][:]
    # Azimuth corresponding to 1st slice
    theta0 = int(round(dset.variables['Azimuth'][0]))
    # rotate accordingly
    data = np.roll(data, theta0, axis=0)
#    data = np.flipud(data)
    data = np.where(data==dset.getncattr('MissingData'), np.nan, data)
    # Azimuth
    az = dset.variables['Azimuth'][:]
    az = np.round(az, 0)
    az = np.roll(az, theta0)
    # Ranges
    binwidth = (dset.getncattr('MaximumRange-value') * 1000.) / len(dset.dimensions['Gate'])
    r = np.arange(binwidth, (dset.getncattr('MaximumRange-value') * 1000.) + binwidth, binwidth)
    # collect attributes
    attrs =  {}
    for attrname in dset.ncattrs():
        attrs[attrname] = dset.getncattr(attrname)
    # Limiting the returned range
    if range_lim and range_lim / binwidth <= data.shape[1]:
        data = data[:,:range_lim / binwidth]
        r = r[:range_lim / binwidth]

    attrs['az'] = az
    attrs['r']  = r
    attrs['sitecoords'] = (attrs['Latitude'], attrs['Longitude'], attrs['Height'])
    attrs['time'] = dt.datetime.utcfromtimestamp(attrs.pop('Time'))
    attrs['max_range'] = data.shape[1] * binwidth
    dset.close()

    return data, attrs

def read_BUFR(buffile):
    """Main BUFR interface: Decodes BUFR file and returns metadata and values

    The actual function refererence is contained in :doc:`wradlib.bufr.decodebufr`.

    The BUFR file format is a self-describing binary format for meteorological
    data. wradlib uses the decoding software from the OPERA 3 program. All
    background information is available under http://www.knmi.nl/opera/bufr.html.

    Basically, a BUFR file consists of a set of *descriptors* which contain all
    the relevant metadata and a data section.

    This decoding function returns a three element tuple. The first element is a
    dictionary which relates the *descriptor identifiers* to comprehensible
    *descriptor names*. The second element is a dictionary which relates the
    *descriptor names* to *descriptor values*. E.g. if the *descriptor identifier*
    was (0, 30, 21), the *descriptor name* would be 'Number of pixels per row' and
    the *descriptor value* could be an integer which actually specifies the number
    of rows of a grid. The third element of the return tuple is the actual value
    array. It is a multi-dimensional numpy array of which the shape depends on
    the descriptor specifications (mostly it will be 2-dimensional).

    Parameters
    ----------
    buffile : Path to a BUFR file

    Returns
    -------
    output: a tuple with three elements (descnames, descrvals, data)

        - descnames: a dictionary of descriptor names

        - descvals: dictionary of descriptor values

        - data: the actual data as a multidimensional numpy array

    Examples
    --------
    >>> import wradlib.bufr as bufr
    >>> buffile = "wradlib/examples/data/test.buf"
    >>> descnames, descvals, data = bufr.decodebufr(buffile)
    >>> print descnames
    >>> print descvals
    >>> print data.shape

    """
    return bufr.decodebufr(buffile)

if __name__ == '__main__':
    print 'wradlib: Calling module <io> as main...'
    import doctest
    doctest.testmod()
