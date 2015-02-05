#-------------------------------------------------------------------------------
# Name:        Reading polar volume data
# Purpose:
#
# Author:      heistermann
#
# Created:     14.01.2013
# Copyright:   (c) heistermann 2013
# Licence:     MIT
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib
import numpy as np
import pylab as pl
# just making sure that the plots immediately pop up
pl.interactive(True)
import datetime as dt
import os

def recipe_polar_volume_example():
    # read the data (sample file in wradlib/examples/data)
    raw = wradlib.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    # this is the radar position tuple (longitude, latitude, altitude)
    sitecoords = (raw["where"]["lon"][0], raw["where"]["lat"][0],raw["where"]["height"][0])
    # define your cartesian reference system
    projstr = wradlib.georef.create_projstr("utm",zone=32, hemisphere="north")
    # containers to hold Cartesian bin coordinates and data
    xyz, data = np.array([]).reshape((-1,3)), np.array([])
    # iterate over 14 elevation angles
    for i in range(14):
        # get the scan metadata for each elevation
        where = raw["dataset%d/where"%(i+1)]
        what  = raw["dataset%d/data1/what"%(i+1)]
        # define arrays of polar coordinate arrays (azimuth and range)
        az = np.arange(0.,360.,360./where["nrays"])
        r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
        # derive 3-D Cartesian coordinate tuples
        xyz_ = wradlib.vpr.volcoords_from_polar(sitecoords, where["elangle"], az, r, projstr)
        # get the scan data for this elevation
        #   here, you can do all the processing on the 2-D polar level
        #   e.g. clutter elimination, attenuation correction, ...
        data_ = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(i+1)]
        # transfer to containers
        xyz, data = np.vstack( (xyz, xyz_) ), np.append(data, data_.ravel())

    # generate 3-D Cartesian target grid coordinates
    maxrange  = 200000.
    minelev   = 0.1
    maxelev   = 25.
    maxalt    = 5000.
    horiz_res = 1000.
    vert_res  = 250.
    trgxyz, trgshape = wradlib.vpr.make_3D_grid(sitecoords, projstr, maxrange, maxalt, horiz_res, vert_res)

    # interpolate to Cartesian 3-D volume grid
    tstart = dt.datetime.now()
    gridder = wradlib.vpr.CAPPI(xyz, trgxyz, trgshape, maxrange, minelev, maxelev)
    vol = np.ma.masked_invalid( gridder(data).reshape(trgshape) )
    print "3-D interpolation took:", dt.datetime.now() - tstart

    # diagnostic plot
    trgx = trgxyz[:,0].reshape(trgshape)[0,0,:]
    trgy = trgxyz[:,1].reshape(trgshape)[0,:,0]
    trgz = trgxyz[:,2].reshape(trgshape)[:,0,0]
    wradlib.vis.plot_max_plan_and_vert(trgx, trgy, trgz, vol, unit="dBZH", levels=range(-32,60))

if __name__ == '__main__':

    recipe_polar_volume_example()


