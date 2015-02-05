#-------------------------------------------------------------------------------
# Name:        test_coord
# Purpose:
#
# Author:      Edouard Goudenhoofdt
#
# Created:     07.03.2014
# Copyright:   (c) Edouard Goudenhoofdt 2014
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import math
import os

import numpy as np

import wradlib.vis as vis
import wradlib.georef as georef
import wradlib.io as io
import matplotlib.pyplot as plt

def ex_coord():
    ### Using wradlib ###

    pvol = io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + 'data/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf')
    
    # Count the number of dataset

    ntilt=1
    for i in range(100):
        try:
            pvol["dataset%d/what" %(ntilt)]
            ntilt += 1
        except:
            ntilt -= 1
            break

    nrays = int(pvol["dataset1/where"]["nrays"])
    nbins = int(pvol["dataset1/where"]["nbins"])
    rscale = int(pvol["dataset1/where"]["rscale"])
    coord = np.empty((ntilt,nrays,nbins,3))
    for t in range(ntilt):
        elangle = pvol["dataset%d/where" %(t+1)]["elangle"]
        coord[t,...] = georef.sweep_centroids(nrays,rscale,nbins,elangle)
    ascale = math.pi/nrays
    sitecoords = (pvol["where"]["lon"],pvol["where"]["lat"],pvol["where"]["height"])
    proj_radar = georef.proj4_to_osr(georef.create_projstr("aeqd",lat_0=pvol["where"]["lat"],lon_0=pvol["where"]["lon"]))
    radius = georef.get_earth_radius(pvol["where"]["lat"],proj_radar)
    
    lon, lat, height = georef.polar2lonlatalt_n(coord[...,0], np.degrees(coord[...,1]), coord[...,2], sitecoords, re=radius, ke=4./3.)
    
    #proj4str = "+proj=aeqd  +lat_0=%f +lon_0=%f" %(pvol["where"]["lat"],pvol["where"]["lon"])
    x, y = georef.reproject(lon, lat, projection_target=proj_radar)

    test = x[0,90,0:960:60]
    print(test)

if __name__ == '__main__':
    import sys
    print("sysargv",sys.argv)
    ex_coord()