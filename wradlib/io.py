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

   readDX

"""

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


def readDX(filename, elevations=None, azimuths=None):
    r"""Data reader for German Weather Service DX raw radar data files
    developed by Thomas Pfaff.

    The algorith basically unpacks the zeroes and returns a regular array of
    128 by 360 data values.

    Parameters
    ----------
    filename : binary file of DX raw data

    Returns
    -------
    output : numpy array of image data.

    """

    azimuthbitmask = 2**(14-1)
    databitmask = 2**(13-1) - 1
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

    if type(elevations) is list:
        elevations.extend(elevs)
    if type(azimuths) is list:
        azimuths.extend(azims)

    return np.array(beams)

def purgeDX(data, options):
    clutterflag = 2**15
    dataflag = 2**13 -1

    result = np.ma.MaskedArray(data & dataflag, data & clutterflag)

    if 'mask' in options:
        return result
    if 'nan' in options:
        result.fill_value = np.nan
        result = result.filled()
    if 'negative' in options:
        result.fill_value = -1
        result = result.filled()

    return result

if __name__ == '__main__':
    print 'wradlib: Calling module <io> as main...'
