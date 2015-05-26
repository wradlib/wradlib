# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_qual.py
# Purpose:     unit tests for the wrdalib.qual module
#
# Author:      Kai Muehlbauer
#
# Created:     26.05.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import numpy as np
import wradlib.qual as qual
import wradlib.georef as georef
import wradlib.util as util
import wradlib.vis as vis
import unittest
import matplotlib.pyplot as plt

class BeamBlockFracTest(unittest.TestCase):

    def setUp(self):

        NBINS = 1000
        NRAYS = 360
        #vrays = np.arange(0, NRAYS)
        vrays = np.linspace(0, 100000, num=NBINS)
        vbins = np.linspace(0, 2*np.pi, num=NRAYS)
        vbins = np.abs(np.sin(vbins) * 500)
        #vbins = np.zeros(NBINS)
        #vbins[NBINS/2::] = 2000
        #print(vx)

        self.range = np.repeat(vrays[np.newaxis, :], NRAYS, 0)

        #self.beamheight = np.repeat(vrays[:, np.newaxis], NBINS, 1)
        self.beamheight = np.repeat(vrays[np.newaxis, :], NRAYS, 0)/50
        print(self.beamheight.shape, self.beamheight)
        vis.plot_ppi(self.beamheight)
        plt.show()

        self.beamradius = util.half_power_radius(self.range, 1.0)
        #self.beamradius = np.ma.masked_invalid(self.beamradius)
        print(self.beamradius)
        vis.plot_ppi(np.ma.masked_invalid(self.beamradius))
        plt.show()

        self.terrainheight = np.repeat(vbins[:, np.newaxis], NBINS, 1)
        self.terrainheight[:,0:NBINS/4] = 0.

        print(self.beamheight.shape, self.terrainheight.shape)
        print(self.terrainheight)
        #plt.pcolormesh(self.terrainheight)
        vis.plot_ppi(self.terrainheight)
        plt.show()

        coord = georef.sweep_centroids(360,1,NBINS,0.)
        xx = coord[...,0]
        yy = np.degrees(coord[...,1])

        xxx = xx * np.cos(np.radians(90.-yy))
        x = xx * np.sin(np.radians(90.-yy))
        y = xxx

        self.newgrid = np.dstack((x, y))


    def test_beam_block_frac(self):

        sitecoords = (7.071663,50.73052,99.5)
        nrays = 360
        nbins = 1000
        el = 1.0
        bw = 1.0
        range_res = 100

        # create range and beamradius arrays
        r = np.arange(nbins)*range_res
        beamradius = util.half_power_radius(r, bw)

        # calculate radar bin centroids and lat, lon, alt of radar bins
        coord = georef.sweep_centroids(nrays,range_res,nbins,el)
        lon, lat, alt = np.array(georef.polar2lonlatalt_n(coord[...,0], np.degrees(coord[...,1]),
                                                              coord[...,2], sitecoords))
        terrain_height = np.arange(0,20)
        #beam_height = np.ones_like(terrain_height) * 10
        beamradius = 10.
        PBB = qual.beam_block_frac(self.terrainheight, self.beamheight, self.beamradius)
        print(PBB.shape, PBB)

        PBB = np.ma.masked_invalid(PBB)
        vis.plot_ppi(PBB)
        plt.show()
        ind = np.nanargmax(PBB, axis=1)
        print(ind)

        CBB = np.zeros_like(PBB)
        for ii, index in enumerate(ind):
            CBB[ii,0:index] = PBB[ii,0:index]
            CBB[ii,index:] = PBB[ii,index]
        vis.plot_ppi(CBB)
        #plt.pcolormesh(BBB)
        plt.show()



