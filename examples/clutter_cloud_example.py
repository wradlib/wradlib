#-------------------------------------------------------------------------------
# Name:        clutter_cloud_example
# Purpose:
#
# Author:      Edouard Goudenhoofdt
#
# Created:     13.02.2014
# Copyright:   (c) Edouard Goudenhoofdt 2014
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import math
import os
import sys

import numpy as np

import wradlib.vis as vis
import wradlib.clutter as cl
import wradlib.georef as georef
import wradlib.ipol as ipol
import wradlib.io as io
import matplotlib.pyplot as plt

def ex_clutter_cloud():

    # read the radar volume scan
    path = os.path.dirname(__file__) + '/'
    pvol = io.read_OPERA_hdf5(path + 'data/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf')

    # Count the number of dataset

    ntilt=1
    for i in range(100):
        try:
            pvol["dataset%d/what" %(ntilt)]
            ntilt += 1
        except:
            ntilt -= 1
            break

    # Construct radar values

    nrays = int(pvol["dataset1/where"]["nrays"])
    nbins = int(pvol["dataset1/where"]["nbins"])
    val = np.empty((ntilt,nrays,nbins))
    for t in range(ntilt):
        val[t,...] = pvol["dataset%d/data1/data" %(t+1)]
    gain = float(pvol["dataset1/data1/what"]["gain"])
    offset = float(pvol["dataset1/data1/what"]["offset"])
    val = val*gain + offset

    # Construct radar coordinates

    rscale = int(pvol["dataset1/where"]["rscale"])
    coord = np.empty((ntilt,nrays,nbins,3))
    for t in range(ntilt):
        elangle = pvol["dataset%d/where" %(t+1)]["elangle"]
        coord[t,...] = georef.sweep_centroids(nrays,rscale,nbins,elangle)
    ascale = math.pi/nrays
    sitecoords = (pvol["where"]["lon"],pvol["where"]["lat"],pvol["where"]["height"])
    proj_radar = georef.proj4_to_osr(georef.create_projstr("aeqd",lat_0=pvol["where"]["lat"],lon_0=pvol["where"]["lon"]))
    coord[...,0], coord[...,1], coord[...,2] = georef.polar2lonlatalt_n(coord[...,0], np.degrees(coord[...,1]), coord[...,2], sitecoords, re=6370040., ke=4./3.)
    #proj4str = "+proj=aeqd  +lat_0=%f +lon_0=%f" %(pvol["where"]["lat"],pvol["where"]["lon"])
    coord = georef.reproject(coord, projection_target=proj_radar)

    # Construct collocated satellite data

    sat_gdal = io.read_safnwc(path + 'data/SAFNWC_MSG3_CT___201304290415_BEL_________.h5')
    val_sat = georef.read_gdal_values(sat_gdal)
    coord_sat = georef.read_gdal_coordinates(sat_gdal)
    proj_sat = georef.read_gdal_projection(sat_gdal)
    coord_sat = georef.reproject(coord_sat, projection_source=proj_sat, projection_target=proj_radar)
    coord_radar = coord
    interp = ipol.Nearest(coord_sat[...,0:2].reshape(-1,2),coord_radar[...,0:2].reshape(-1,2))
    val_sat = interp(val_sat.ravel()).reshape(val.shape)

    # Estimate localisation errors

    timelag = 9*60
    wind = 10
    error=np.absolute(timelag)*wind

    # Identify clutter based on collocated cloudtype
    clutter = cl.filter_cloudtype(val[0,...],val_sat[0,...],scale=rscale,smoothing=error)

    # visualize the result
    vis.plot_ppi(clutter)
    plt.suptitle('clutter')
    plt.savefig('clutter_cloud_example_1.png')
    plt.close()
    vis.plot_ppi(val_sat[0,...])
    plt.suptitle('satellite')
    plt.savefig('clutter_cloud_example_2.png')

if __name__ == '__main__':
    ex_clutter_cloud()
