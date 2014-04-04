# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Thomas Pfaff
#
# Created:     28.10.2011
# Copyright:   (c) Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.ipol as ipol
import wradlib.qual as qual
import wradlib.comp as comp
import os

def ex_compose_by_height():

    import numpy as np
    import pylab as pl
    # just making sure that the plots immediately pop up
    pl.interactive(True)

    #---------------------------------------------------------------------------
    # load the data for the first radar
    #---------------------------------------------------------------------------
    path = os.path.dirname(__file__) + '/'
    rad1 = np.loadtxt(path + 'data/polar_dBZ_tur.gz').ravel()
    rad1coords = np.loadtxt(path + 'data/bin_coords_tur.gz')
    center1 = rad1coords.mean(axis=0)
    radius1 = 128000.

    #---------------------------------------------------------------------------
    # calculate height field for the first radar
    #---------------------------------------------------------------------------
    ranges1 = np.arange(128)[None,:]*1000.
    # for the time being assuming a sinusoidal elevation pattern
    elevs1 = np.sin(np.deg2rad(np.arange(360)[:,None]))*0.2 + 0.3
    heights1 = qual.beam_height_ft(ranges1, elevs1).ravel()

    #---------------------------------------------------------------------------
    # load the data for the second radar
    #---------------------------------------------------------------------------
    rad2 = np.loadtxt(path + 'data/polar_dBZ_fbg.gz').ravel()
    rad2coords = np.loadtxt(path + 'data/bin_coords_fbg.gz')
    center2 = rad2coords.mean(axis=0)
    radius2 = 128000.

    #---------------------------------------------------------------------------
    # calculate the height field for the second radar
    #---------------------------------------------------------------------------
    # here assuming it to be the same as for radar 1
    heights2 = qual.beam_height_ft(ranges1, elevs1).ravel()

    #---------------------------------------------------------------------------
    # set up the common grid
    #---------------------------------------------------------------------------
    xc = np.concatenate((rad1coords[:,0],rad2coords[:,0]))
    yc = np.concatenate((rad1coords[:,1],rad2coords[:,1]))
    xmin = np.min(xc)-5000
    xmax = np.max(xc)+5000
    ymin = np.min(yc)-5000
    ymax = np.max(yc)+5000

    gridshape=(100,100)
    coords = np.meshgrid(np.linspace(xmin,xmax,gridshape[0]), np.linspace(ymin,ymax,gridshape[1]))
    coords = np.vstack((coords[0].ravel(), coords[1].ravel())).transpose()

    #---------------------------------------------------------------------------
    # transfer the first radar data to the grid
    #---------------------------------------------------------------------------
    rad1_gridded = comp.togrid(rad1coords, coords, radius1, center1, rad1, ipol.Nearest)
    heights1_gridded = comp.togrid(rad1coords, coords, radius1, center1, heights1, ipol.Nearest)

    #---------------------------------------------------------------------------
    # transfer the second radar data to the grid
    #---------------------------------------------------------------------------
    rad2_gridded = comp.togrid(rad2coords, coords, radius2, center2, rad2, ipol.Nearest)
    heights2_gridded = comp.togrid(rad2coords, coords, radius2, center2, heights2, ipol.Nearest)

    #---------------------------------------------------------------------------
    # combine radar data according to height (lower bin wins)
    #---------------------------------------------------------------------------
##    radinfo = np.hstack((np.empty_like(rad1_gridded)*np.nan, rad1_gridded, rad2_gridded))
##    heightinfo = np.hstack((np.ones_like(heights1_gridded)*1e10, heights1_gridded, heights2_gridded))
##    select = np.nanargmin(heightinfo, axis=1)
##    composite = radinfo[np.arange(select.shape[0]),select]
    # second approach using the function
##    composite = comp.compose_ko([rad1_gridded, rad2_gridded],[1./(heights1_gridded+0.001), 1./(heights2_gridded+0.001)])
    # third approach using weighted averaging
    composite = comp.compose_weighted([rad1_gridded, rad2_gridded],[1./(heights1_gridded+0.001), 1./(heights2_gridded+0.001)])

    #---------------------------------------------------------------------------
    # visualize the results
    #---------------------------------------------------------------------------
    pl.figure()
    pl.imshow(rad1_gridded.reshape(gridshape), interpolation='nearest', origin='lower')
    pl.figure()
    pl.imshow(rad2_gridded.reshape(gridshape), interpolation='nearest', origin='lower')
    pl.figure()
    pl.imshow(composite.reshape(gridshape), interpolation='nearest', origin='lower')
    pl.figure()
    pl.imshow(heights1_gridded.reshape(gridshape), interpolation='nearest', origin='lower')
    pl.figure()
    pl.imshow(heights2_gridded.reshape(gridshape), interpolation='nearest', origin='lower')
    pl.show()

if __name__ == '__main__':
    ex_compose_by_height()