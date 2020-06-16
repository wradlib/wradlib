#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Digital Elevation Model I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide surface/terrain elevation information from SRTM data

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["download_srtm", "get_srtm"]
__doc__ = __doc__.format("\n   ".join(__all__))

import os

import numpy as np
import requests
from osgeo import gdal

from wradlib import util


def download_srtm(
    filename, destination=None, version=2, resolution=3, region="Eurasia", token=None
):
    """
    Download NASA SRTM elevation data
    Version 3 is only available with a login and a token

    Parameters
    ----------
    filename : str
        srtm file to download
    destination : str
        output filename
    version : int
        srtm version (2 or 3)
    resolution : int
        resolution of SRTM data (1, 3 or 30)
    region : str
        name of the region for SRTM version 2 only:
        Africa, Australia, Eurasia, Islands, North America, South America
    token : str
        filename with authorization token (required for version 3)

    """
    if version == 2:
        website = "https://dds.cr.usgs.gov/srtm/version2_1"
        resolution = "SRTM%s" % (resolution)
        source = os.path.join(website, resolution, region)
    if version == 3:
        website = "https://e4ftl01.cr.usgs.gov/MEASURES"
        resolution = "SRTMGL%d.003" % (resolution)
        source = os.path.join(website, resolution, "2000.02.11")
    url = os.path.join(source, filename)

    headers = None
    if token is not None:
        headers = {"Authorization": "Bearer %s" % (token)}
    try:
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        status_code = err.response.status_code
        if status_code == 404:
            return
        else:
            raise err

    if destination is None:
        destination = filename
    with open(destination, "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def get_srtm(extent, version=2, resolution=3, merge=True, download=None):
    """
    Get NASA SRTM elevation data

    Parameters
    ----------
    extent : str
        lonmin, lonmax, latmin, latmax
    version : int
        srtm version (2 or 3)
    resolution : int
        resolution of SRTM data (1, 3 or 30)
    merge : bool
        True to merge the tiles in one dataset
    download : dict
        download options (see download_srtm)

    Returns
    -------
    dataset : gdal.Dataset
        Raster dataset containing elevation information
    """

    extent = [int(np.floor(x)) for x in extent]
    lonmin, lonmax, latmin, latmax = extent

    filelist = []
    for latitude in range(latmin, min(latmax, 0)):
        for longitude in range(lonmin, min(lonmax, 0)):
            georef = "S%02gW%03g" % (-latitude, -longitude)
            filelist.append(georef)
        for longitude in range(max(lonmin, 0), lonmax + 1):
            georef = "S%02gE%03g" % (-latitude, longitude)
            filelist.append(georef)
    for latitude in range(max(0, latmin), latmax + 1):
        for longitude in range(lonmin, min(lonmax, 0)):
            georef = "N%02gW%03g" % (latitude, -longitude)
            filelist.append(georef)
        for longitude in range(max(lonmin, 0), lonmax + 1):
            georef = "N%02gE%03g" % (latitude, longitude)
            filelist.append(georef)
    if version == 3:
        filelist = ["%s.SRTMGL%s" % (f, resolution) for f in filelist]

    filelist = ["%s.hgt.zip" % (f) for f in filelist]

    wrl_data_path = util.get_wradlib_data_path()
    srtm_path = os.path.join(wrl_data_path, "geo")
    if not os.path.exists(srtm_path) and download is not None:
        os.makedirs(srtm_path)
    demlist = []
    for filename in filelist:
        path = os.path.join(srtm_path, filename)
        if os.path.exists(path):
            demlist.append(path)
            continue
        if download is not None:
            download_srtm(filename, path, version, resolution, **download)
            if os.path.exists(path):
                demlist.append(path)

    demlist = [gdal.Open(d) for d in demlist]
    if not merge:
        return demlist
    dem = gdal.Warp("", demlist, format="MEM")

    return dem
