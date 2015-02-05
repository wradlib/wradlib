# -*- coding: iso-8859-1 -*-
#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
# Author:      jacobi
# Created:     05.04.2011
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.clutter as clutter
import numpy as np
import pylab as pl
# just making sure that the plots immediately pop up
pl.interactive(True)
import os


def ex_histo_cut():

    # load annual rainfall radar array
    yearsum = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/annual_rainfall_fbg.gz')

    # get boolean array for clutter and shading
    mask = clutter.histo_cut(yearsum)

    # substitute cluttered/shaded pixels in the annual rainfall image by NaNs based on the boolean mask
    yearsum_masked = np.where(mask,np.nan,yearsum)

    # Requirements for the plots
    R = 1.
    n_theta, n_r = 360, 128
    theta = np.linspace(0, 2*np.pi, n_theta + 1)
    r = np.linspace(0., R, n_r + 1)

    # radar plots before and after masking
    pl.figure(figsize = (14, 8))
    palette = pl.cm.get_cmap()
    palette.set_bad(alpha = 0)
    pl.rc('xtick', labelsize=0)
    pl.rc('ytick', labelsize=0)
    pl.subplot(1, 2, 1, projection = "polar", aspect = 1.)
    plotarr = np.fliplr(np.transpose(yearsum[0:360, 0:128]))
    pl.pcolormesh(theta + np.pi / 2, r, plotarr, cmap = palette, rasterized = True, vmin = 0, vmax = yearsum_masked[np.isfinite(yearsum_masked)].max())
    pl.rc('xtick', labelsize=8)
    pl.rc('ytick', labelsize=8)
    pl.colorbar(shrink = 0.7)
    pl.title('Annual rainfall Yearsum raw [mm]')
    pl.rc('xtick', labelsize=0)
    pl.rc('ytick', labelsize=0)
    pl.subplot(1, 2, 2, projection = "polar", aspect = 1.)
    plotarr = np.fliplr(np.transpose(yearsum_masked[0:360, 0:128]))
    pl.pcolormesh(theta + np.pi / 2, r, plotarr, cmap = palette, rasterized = True, vmin = 0, vmax = yearsum_masked[np.isfinite(yearsum_masked)].max())
    pl.rc('xtick', labelsize=8)
    pl.rc('ytick', labelsize=8)
    pl.colorbar(shrink = 0.7)
    pl.title('Annual rainfall clutter masked [mm]')
    pl.show()

if __name__ == '__main__':
    ex_histo_cut()