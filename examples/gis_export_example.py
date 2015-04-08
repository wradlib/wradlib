#-------------------------------------------------------------------------------
# Name:         raster_to_gis_example
# Author:       Maik Heistermann
# Licence:      The MIT License
#-------------------------------------------------------------------------------

import wradlib

def ex_grid_to_gis():
    
    # We will export this RADOLAN dataset to a GIS compatible format
    data, meta = wradlib.io.read_RADOLAN_composite("data/radolan/raa01-sf_10000-1408102050-dwd---bin.gz")
    
    # This is the RADOLAN projection
    proj_osr = wradlib.georef.create_osr( "dwd-radolan" )
    
    # Get projected RADOLAN coordinates for corner definition
    radolan_grid_xy = wradlib.georef.get_radolan_grid(900,900)
    x = radolan_grid_xy[:,:,0]
    y = radolan_grid_xy[:,:,1]
    
    # Export to Arc/Info ASCII Grid format (aka ESRI grid)
    # It should be possible to import this to most conventional GIS software.
    wradlib.io.to_AAIGrid("data/out.asc", data, x[0,0], y[0,0], 1., proj=proj_osr)


# =======================================================
if __name__ == '__main__':
    ex_grid_to_gis()