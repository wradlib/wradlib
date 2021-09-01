#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Miscellaneous Data I/O
^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "write_polygon_to_text",
    "to_pickle",
    "from_pickle",
    "get_radiosonde",
    "get_membership_functions",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import io
import pickle
import urllib
import warnings

import numpy as np

from wradlib import util


def _write_polygon_to_txt(f, idx, vertices):
    f.write("%i %i\n" % idx)
    for i, vert in enumerate(vertices):
        f.write("%i " % (i,))
        f.write("%f %f %f %f\n" % tuple(vert))


def write_polygon_to_text(fname, polygons):
    """Writes Polygons to a Text file which can be interpreted by ESRI \
    ArcGIS's "Create Features from Text File (Samples)" tool.

    This is (yet) only a convenience function with limited functionality.
    E.g. interior rings are not yet supported.

    Parameters
    ----------
    fname : str
        name of the file to save the vertex data to
    polygons : list
        list of lists of polygon vertices.
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
        write_polygon_to_text('polygons.txt', polygons)
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
    with open(fname, "w") as f:
        f.write("Polygon\n")
        count = 0
        for vertices in polygons:
            _write_polygon_to_txt(f, (count, 0), vertices)
            count += 1
        f.write("END\n")


def to_pickle(fpath, obj):
    """Pickle object <obj> to file <fpath>"""
    output = open(fpath, "wb")
    pickle.dump(obj, output)
    output.close()


def from_pickle(fpath):
    """Return pickled object from file <fpath>"""
    pkl_file = open(fpath, "rb")
    obj = pickle.load(pkl_file)
    pkl_file.close()
    return obj


def get_radiosonde(wmoid, date, cols=None):
    """Download radiosonde data from internet.

    Based on http://weather.uwyo.edu/upperair/sounding.html.

    Parameters
    ----------
    wmoid : int
        WMO radiosonde ID
    date : :py:class:`datetime.datetime`
        Date and Time

    Keyword Arguments
    -----------------
    cols : tuple
        tuple of int or strings describing the columns to consider,
        defaults to None (all columns)

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Structured array of radiosonde data
    meta : dict
        radiosonde metadata
    """
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    hour = date.strftime("%H")

    # Radiosondes are only at noon and midnight
    hour = "12" if (6 < int(hour) < 18) else "00"

    # url
    url_str = (
        "http://weather.uwyo.edu/cgi-bin/sounding?"
        "TYPE=TEXT%3ALIST&"
        "YEAR={0}&MONTH={1}&"
        "FROM={2}{3}&TO={2}{3}&STNM={4}&"
        "ICE=1".format(year, month, day, hour, wmoid)
    )

    # html request
    with urllib.request.urlopen(url_str) as url_request:
        response = url_request.read()

    # decode string
    url_text = response.decode("utf-8")

    # first line (eg errormessage)
    if url_text.find("<H2>") == -1:
        err = url_text.split("\n", 1)[1].split("\n", 1)[0]
        raise ValueError(err)

    # extract relevant information
    url_data = url_text.split("<PRE>")[1].split("</PRE>")[0]
    url_meta = url_text.split("<PRE>")[2].split("</PRE>")[0]

    # extract empty lines, names, units and data
    _, _, names, units, _, url_data = url_data.split("\n", 5)

    names = names.split()
    units = units.split()

    unitdict = {name: unit for (name, unit) in zip(names, units)}

    # read data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        data = np.genfromtxt(
            io.StringIO(url_data),
            names=names,
            dtype=float,
            usecols=cols,
            autostrip=True,
            invalid_raise=False,
        )

    # read metadata
    meta = {}
    for i, row in enumerate(io.StringIO(url_meta)):
        if i == 0:
            continue
        k, v = row.split(":")
        k = k.strip()
        v = v.strip()
        if i == 2:
            v = int(v)
        elif i == 3:
            v = dt.datetime.strptime(v, "%y%m%d/%H%M")
        elif i > 3:
            v = float(v)
        meta[k] = v

    meta["quantity"] = {item: unitdict[item] for item in data.dtype.names}

    return data, meta


def get_membership_functions(filename):
    """Reads membership function parameters from wradlib-data file.

    Parameters
    ----------
    filename : str
        Filename of wradlib-data file

    Returns
    -------
    msf : :py:class:`numpy:numpy.ndarray`
        Array of membership funcions with shape (hm-classes, observables,
        indep-ranges, 5)
    """
    gzip = util.import_optional("gzip")

    with gzip.open(filename, "rb") as f:
        nclass = int(f.readline().decode().split(":")[1].strip())
        nobs = int(f.readline().decode().split(":")[1].strip())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            data = np.genfromtxt(f, skip_header=10, autostrip=True, invalid_raise=False)

    data = np.reshape(data, (nobs, int(data.shape[0] / nobs), data.shape[1]))
    msf = np.reshape(
        data, (data.shape[0], nclass, int(data.shape[1] / nclass), data.shape[2])
    )
    msf = np.swapaxes(msf, 0, 1)

    return msf
