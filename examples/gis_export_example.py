# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import wradlib


def ex_grid_to_gis():
    # We will export this RADOLAN dataset to a GIS compatible format
    wdir = wradlib.util.get_wradlib_data_path() + '/radolan/grid/'
    filename = wradlib.util.get_wradlib_data_file('radolan/misc/raa01-sf_10000-1408102050-dwd---bin.gz')
    data, meta = wradlib.io.read_RADOLAN_composite(filename)

    # This is the RADOLAN projection
    proj_osr = wradlib.georef.create_osr("dwd-radolan")

    # Get projected RADOLAN coordinates for corner definition
    xy = wradlib.georef.get_radolan_grid(900, 900)

    # Export to Arc/Info ASCII Grid format (aka ESRI grid)
    #     It should be possible to import this to most conventional GIS software.
    wradlib.io.to_AAIGrid(wdir + "aaigrid.asc", data, xy[0, 0, 0], xy[0, 0, 1], 1., proj=proj_osr, to_esri=False)

    # Export to GeoTIFF format
    #     For RADOLAN grids, this projection will probably not be recognized by ESRI ArcGIS.
    #     Please note that that the geotransform for creating GeoTIFF files requires the
    #     top-left corner of the bounding box. See help for io.to_GeoTIFF for further
    #     instructions, particularly on how to define the geotransform.
    geotransform = [xy[0, 0, 0], 1., 0, xy[-1, -1, 1] + 1., 0, -1.]
    wradlib.io.to_GeoTIFF(wdir + "geotiff.tif", data, geotransform, proj=proj_osr)


# =======================================================
if __name__ == '__main__':
    ex_grid_to_gis()
