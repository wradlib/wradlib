#------------------------------------------------------------------------------
# Name:        beamblock_example.py
# Purpose:     example file for porting PyRadarMet  functionality
#
# Author:      Kai Muehlbauer
#
# Created:     17.04.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wradlib as wrl

def _check_file(filename):
    geo_src = 'https://bitbucket.org/kaimuehlbauer/wradlib_miub/downloads/geo.tar.gz'
    if not os.path.exists(filename):
        warnings.warn("File does not exist: {0}\nGet data from {1} and extract archive to wradlib/example/data folder".format(filename, geo_src))
        exit(0)

# example function for calculation and visualisation of beam blockage
# mimics PyRadarMets output
def nex_beamblock(rasterfile, **kwargs):
    """
    Function to calculate and visualize beamblock fraction
    """

    # check if raster file is available, graceful exit
    _check_file(rasterfile)

    # setup radar specs (Bonn Radar)
    sitecoords = (7.071663,50.73052,99.5)
    nrays = 360
    nbins = 1000
    el = 1.0
    bw = 1.0
    range_res = 100

    # create range and beamradius arrays
    r = np.arange(nbins)*range_res
    beamradius = wrl.util.half_power_radius(r, bw)

    # calculate radar bin centroids and lat, lon, alt of radar bins
    coord = wrl.georef.sweep_centroids(nrays,range_res,nbins,el)
    lon, lat, alt = np.array(wrl.georef.polar2lonlatalt_n(coord[...,0], np.degrees(coord[...,1]),
                                                          coord[...,2], sitecoords))
    polcoords = np.dstack((lon,lat))
    print("lon,lat,alt:", lon.shape, lat.shape, alt.shape)

    # get radar bounding box lonlat
    lonmin = np.min(lon)
    lonmax = np.max(lon)
    latmin = np.min(lat)
    latmax = np.max(lat)
    rlimits = [lonmin, latmin, lonmax, latmax]
    print("radar bounding box:", rlimits)

    # read raster data
    rastercoords,rastervalues = wrl.io.read_raster_data(rasterfile, **kwargs)

    # apply radar bounding box to raster data
    # this actually cuts out the interesting box from rasterdata
    #rastercoords, rastervalues = wrl.util.clip_array_by_value(rastercoords, rastervalues, rlimits)
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords = rastercoords[ind[1]:ind[3],ind[0]:ind[2],...]
    rastervalues = rastervalues[ind[1]:ind[3],ind[0]:ind[2]]

    # map rastervalues to polar grid points
    polarvalues = wrl.ipol.cart2irregular_spline(rastercoords, rastervalues, polcoords, order=3, prefilter=False)

    # calculate partial beam blockage PBB
    PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
    PBB = np.ma.masked_invalid(PBB)

    # calculate cumulative beam blockage CBB
    ind = np.nanargmax(PBB, axis=1)
    CBB = np.copy(PBB)
    for ii, index in enumerate(ind):
        CBB[ii,0:index] = PBB[ii,0:index]
        CBB[ii,index:] = PBB[ii,index]

    # plotting the stuff
    fig = plt.figure(figsize=(10,8))

    # create subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)

    # azimuth angle
    angle = 225

    # plot terrain
    dem = ax1.pcolormesh(lon, lat, polarvalues/1000., cmap=mpl.cm.terrain, vmin=-0.3, vmax=0.8)
    ax1.plot(sitecoords[0], sitecoords[1], 'rD')
    ax1.set_title('Terrain within {0} km range of Radar'.format(np.max(r/1000.) + 0.1))
    # colorbar
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(dem, cax=cax1)
    # limits

    ax1.set_xlim(lonmin, lonmax)
    ax1.set_ylim(latmin, latmax)
    ax1.set_aspect('auto')

    # plot CBB on ax2
    cbb = ax2.pcolormesh(lon, lat, CBB, cmap=mpl.cm.PuRd, vmin=0, vmax=1)
    ax2.set_title('Beam-Blockage Fraction')
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    # colorbar
    cb = fig.colorbar(cbb, cax=cax2)
    # limits
    ax2.set_xlim(lonmin, lonmax)
    ax2.set_ylim(latmin, latmax)
    ax2.set_aspect('auto')

    # plot single ray terrain profile on ax3
    bc, = ax3.plot(r / 1000., alt[angle,:] / 1000., '-b',
                     linewidth=3, label='Beam Center')
    b3db, = ax3.plot(r / 1000., (alt[angle,:] + beamradius) / 1000., ':b',
                   linewidth=1.5, label='3 dB Beam width')
    ax3.plot(r / 1000., (alt[angle,:] - beamradius) / 1000., ':b')
    tf = ax3.fill_between(r / 1000., 0.,
                    polarvalues[angle,:] / 1000.,
                    color='0.75')
    ax3.set_xlim(0., np.max(r/1000.) + 0.1)
    ax3.set_ylim(0., 5)
    ax3.set_xlabel('Range (km)')
    ax3.set_ylabel('Height (km)')

    axb = ax3.twinx()
    bbf, = axb.plot(r / 1000., CBB[angle,:], '-k',
                   label='BBF')
    axb.set_ylabel('Beam-blockage fraction')
    axb.set_ylim(0., 1.)
    axb.set_xlim(0., np.max(r/1000.) + 0.1)

    ax3.legend((bc, b3db, bbf), ('Beam Center', '3 dB Beam width', 'BBF'),
                               loc='upper left', fontsize=10)

# =======================================================
if __name__ == '__main__':

    # set filepath
    filepath = os.path.join(os.path.dirname(__file__), 'data/geo')
    filename = os.path.join(filepath, 'bonn_gtopo.tif')
    print("DataSource: GTOPO30")
    nex_beamblock(filename)#, spacing=0.083333333333, resample=gdal.GRA_Bilinear)

    print("DataSource: SRTM")
    filename = os.path.join(filepath, 'bonn_new.tif')
    nex_beamblock(filename)#, spacing=0.008333333333333333, resample=gdal.GRA_Lanczos)#gdal.GRA_Bilinear)

    plt.show()
