#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Digital Elevation Model Data I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide surface/terrain elevation information from SRTM data

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["download_srtm", "get_srtm", "get_srtm_tile_names"]
__doc__ = __doc__.format("\n   ".join(__all__))

import os

import numpy as np

from wradlib import util

gdal = util.import_optional("osgeo.gdal")
requests = util.import_optional("requests")


def init_header_redirect_session(token):
    class HeaderRedirection(requests.Session):
        AUTH_HOST = "urs.earthdata.nasa.gov"

        def __init__(self, token):
            super().__init__()
            self.headers.update({"Authorization": f"Bearer {token}"})

        def rebuild_auth(self, request, response):
            headers = request.headers
            url = request.url
            if "Authorization" in headers:
                original = requests.utils.urlparse(response.request.url).hostname
                redirect = requests.utils.urlparse(url).hostname
                if (
                    original != redirect
                    and redirect != self.AUTH_HOST
                    and original != self.AUTH_HOST
                ):
                    del headers["Authorization"]
            return

    return HeaderRedirection(token)


def download_srtm(filename, destination, resolution=3):
    """
    Download NASA SRTM elevation data
    Only available with login/password

    Parameters
    ----------
    filename : str
        srtm file to download
    destination : str
        output filename
    resolution : int
        resolution of SRTM data (1, 3 or 30)
    """

    website = "https://e4ftl01.cr.usgs.gov/MEASURES"
    subres = 3
    if resolution == 30:
        subres = 2
    resolution = f"SRTMGL{resolution}.00{subres}"
    source = "/".join([website, resolution, "2000.02.11"])
    url = "/".join([source, filename])
    token = os.environ.get("WRADLIB_EARTHDATA_BEARER_TOKEN", None)

    if token is None:
        raise ValueError(
            "WRADLIB_EARTHDATA_BEARER_TOKEN environment variable missing. "
            "Downloading SRTM data requires a NASA Earthdata Account and Bearer Token. "
            "To obtain a NASA Earthdata Login account, "
            "please visit https://urs.earthdata.nasa.gov/users/new/."
        )
    session = init_header_redirect_session(token)
    status_code = 0
    try:
        r = session.get(url, stream=True, timeout=5)
        r.raise_for_status()
        if destination is None:
            destination = filename
        with open(destination, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024 * 1014):
                fd.write(chunk)
    except requests.exceptions.HTTPError as err:
        status_code = err.response.status_code
        if status_code != 404:
            raise err
    except requests.exceptions.Timeout as err:
        raise err
    return status_code


def get_srtm_tile_names(extent):
    """
    Get NASA SRTM elevation data tile names

    Parameters
    ----------
    extent : list
        list containing lonmin, lonmax, latmin, latmax

    Returns
    -------
    out : list
        list of tile names
    """
    extent = [int(np.floor(x)) for x in extent]
    lonmin, lonmax, latmin, latmax = extent

    filelist = []
    for latitude in range(latmin, min(latmax, 0)):
        for longitude in range(lonmin, min(lonmax, 0)):
            tilename = f"S{-latitude:02g}W{-longitude:03g}"
            filelist.append(tilename)
        for longitude in range(max(lonmin, 0), lonmax + 1):
            tilename = f"S{-latitude:02g}E{longitude:03g}"
            filelist.append(tilename)
    for latitude in range(max(0, latmin), latmax + 1):
        for longitude in range(lonmin, min(lonmax + 1, 0)):
            tilename = f"N{latitude:02g}W{-longitude:03g}"
            filelist.append(tilename)
        for longitude in range(max(lonmin, 0), lonmax + 1):
            tilename = f"N{latitude:02g}E{longitude:03g}"
            filelist.append(tilename)
    return filelist


def get_srtm(extent, resolution=3, merge=True):
    """
    Get NASA SRTM elevation data

    Parameters
    ----------
    extent : list
        list containing lonmin, lonmax, latmin, latmax
    resolution : int
        resolution of SRTM data (1, 3 or 30)
    merge : bool
        True to merge the tiles in one dataset

    Returns
    -------
    dataset : :py:class:`gdal:osgeo.gdal.Dataset`
        gdal.Dataset Raster dataset containing elevation information
    """
    filelist = get_srtm_tile_names(extent)
    filelist = [f"{f}.SRTMGL{resolution}.hgt.zip" for f in filelist]
    wrl_data_path = util.get_wradlib_data_path()
    srtm_path = os.path.join(wrl_data_path, "geo")
    if not os.path.exists(srtm_path):
        os.makedirs(srtm_path)
    demlist = []
    for filename in filelist:
        path = os.path.join(srtm_path, filename)
        status_code = 0
        if not os.path.exists(path):
            status_code = download_srtm(filename, path, resolution)
        if status_code == 0:
            demlist.append(path)

    demlist = [gdal.Open(d) for d in demlist]
    if not merge:
        return demlist
    dem = gdal.Warp("", demlist, format="MEM")

    return dem
