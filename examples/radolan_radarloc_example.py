#------------------------------------------------------------------------------
# Name:        radolan_radarloc_example.py
# Purpose:     showing radar-locations and range-rings
#              for radolan composites
#
# Author:      Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------

import wradlib as wrl
import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
import os
from osgeo import osr


def get_radar_locations():

    radars = {}
    radar = {}
    radar['name'] = 'ASR Dresden'
    radar['wmo'] = 10487
    radar['lon'] = 13.76347
    radar['lat'] = 51.12404
    radar['alt'] = 261
    radars['ASD'] = radar

    radar = {}
    radar['name'] = 'Boostedt'
    radar['wmo'] = 10132
    radar['lon'] = 10.04687
    radar['lat'] = 54.00438
    radar['alt'] = 124.56
    radars['BOO'] = radar

    radar = {}
    radar['name'] = 'Dresden'
    radar['wmo'] = 10488
    radar['lon'] = 13.76865
    radar['lat'] = 51.12465
    radar['alt'] = 263.36
    radars['DRS'] = radar

    radar = {}
    radar['name'] = 'Eisberg'
    radar['wmo'] = 10780
    radar['lon'] = 12.40278
    radar['lat'] = 49.54066
    radar['alt'] = 798.79
    radars['EIS'] = radar

    radar = {}
    radar['name'] = 'Emden'
    radar['wmo'] = 10204
    radar['lon'] = 7.02377
    radar['lat'] = 53.33872
    radar['alt'] = 58
    radars['EMD'] = radar

    radar = {}
    radar['name'] = 'Essen'
    radar['wmo'] = 10410
    radar['lon'] = 6.96712
    radar['lat'] = 51.40563
    radar['alt'] = 185.10
    radars['ESS'] = radar

    radar = {}
    radar['name'] = 'Feldberg'
    radar['wmo'] = 10908
    radar['lon'] = 8.00361
    radar['lat'] = 47.87361
    radar['alt'] = 1516.10
    radars['FBG'] = radar

    radar = {}
    radar['name'] = 'Flechtdorf'
    radar['wmo'] = 10440
    radar['lon'] = 8.802
    radar['lat'] = 51.3112
    radar['alt'] = 627.88
    radars['FLD'] = radar

    radar = {}
    radar['name'] = 'Hannover'
    radar['wmo'] = 10339
    radar['lon'] = 9.69452
    radar['lat'] = 52.46008
    radar['alt'] = 97.66
    radars['HNR'] = radar

    radar = {}
    radar['name'] = 'Neuhaus'
    radar['wmo'] = 10557
    radar['lon'] = 11.13504
    radar['lat'] = 50.50012
    radar['alt'] = 878.04
    radars['NEU'] = radar

    radar = {}
    radar['name'] = 'Neuheilenbach'
    radar['wmo'] = 10605
    radar['lon'] = 6.54853
    radar['lat'] = 50.10965
    radar['alt'] = 585.84
    radars['NHB'] = radar

    radar = {}
    radar['name'] = 'Offenthal'
    radar['wmo'] = 10629
    radar['lon'] = 8.71293
    radar['lat'] = 49.9847
    radar['alt'] = 245.80
    radars['OFT'] = radar

    radar = {}
    radar['name'] = 'Proetzel'
    radar['wmo'] = 10392
    radar['lon'] = 13.85821
    radar['lat'] = 52.64867
    radar['alt'] = 193.92
    radars['PRO'] = radar

    radar = {}
    radar['name'] = 'Memmingen'
    radar['wmo'] = 10950
    radar['lon'] = 10.21924
    radar['lat'] = 48.04214
    radar['alt'] = 724.40
    radars['MEM'] = radar

    radar = {}
    radar['name'] = 'Rostock'
    radar['wmo'] = 10169
    radar['lon'] = 12.05808
    radar['lat'] = 54.17566
    radar['alt'] = 37
    radars['ROS'] = radar

    radar = {}
    radar['name'] = 'Isen'
    radar['wmo'] = 10873
    radar['lon'] = 12.10177
    radar['lat'] = 48.1747
    radar['alt'] = 677.77
    radars['ISN'] = radar

    radar = {}
    radar['name'] = 'Tuerkheim'
    radar['wmo'] = 10832
    radar['lon'] = 9.78278
    radar['lat'] = 48.58528
    radar['alt'] = 767.62
    radars['TUR'] = radar

    radar = {}
    radar['name'] = 'Ummendorf'
    radar['wmo'] = 10356
    radar['lon'] = 11.17609
    radar['lat'] = 52.16009
    radar['alt'] = 183
    radars['UMM'] = radar

    return radars


def ex_radolan_radarloc():

    # load radolan file
    rw_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

    # print the available attributes
    print("RW Attributes:", rwattrs)

    # mask data
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    rwdata = np.ma.masked_equal(rwdata, -9999)

    # create radolan projection object
    dwd_string = wrl.georef.create_projstr("dwd-radolan")
    proj_stereo = wrl.georef.proj4_to_osr(dwd_string)

    # create wgs84 projection object
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)

    # get radolan grid
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
    x1 = radolan_grid_xy[:, :, 0]
    y1 = radolan_grid_xy[:, :, 1]

    # convert to lonlat
    radolan_grid_ll = wrl.georef.reproject(radolan_grid_xy, projection_source=proj_stereo, projection_target=proj_wgs)
    lon1 = radolan_grid_ll[:, :, 0]
    lat1 = radolan_grid_ll[:, :, 1]

    # plot two projections side by side
    fig1 = pl.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    pm = ax1.pcolormesh(lon1, lat1, rwdata, cmap='spectral')
    cb = fig1.colorbar(pm, shrink=0.75)
    cb.set_label("mm/h")
    pl.xlabel("Longitude ")
    pl.ylabel("Latitude")
    pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat() + '\n WGS84')
    pl.xlim((lon1[0, 0],lon1[-1, -1]))
    pl.ylim((lat1[0, 0],lat1[-1, -1]))
    pl.grid(color='r')

    fig2 = pl.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    pm = ax2.pcolormesh(x1, y1, rwdata, cmap='spectral')
    cb = fig2.colorbar(pm, shrink=0.75)
    cb.set_label("mm/h")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat() + '\n Polar Stereographic Projection')
    pl.xlim((x1[0, 0],x1[-1, -1]))
    pl.ylim((y1[0, 0],y1[-1, -1]))
    pl.grid(color='r')

    # range array 150 km
    print("Max Range: ", rwattrs['maxrange'])
    r = np.arange(1, 151)*1000
    # azimuth array 1 degree spacing
    az = np.linspace(0, 360, 361)[0:-1]

    # get radar dict
    radars = get_radar_locations()

    # iterate over all radars in rwattrs
    # plot range rings and radar location for the two projections
    for radar_id in rwattrs['radarlocations']:

        # get radar coords etc from dict
        # repair Ummendorf ID
        if radar_id == 'umd':
            radar_id = 'umm'
        radar = radars[radar_id.upper()]

        # build polygons for maxrange rangering
        polygons = wrl.georef.polar2polyvert(r, az, (radar['lon'], radar['lat']))
        polygons.shape = (len(az), len(r), 5, 2)
        polygons_ll = polygons[:, -1, :, :]

        # reproject to radolan polar stereographic projection
        polygons_xy = wrl.georef.reproject(polygons_ll, projection_source=proj_wgs, projection_target=proj_stereo)

        # create PolyCollections and add to respective axes
        polycoll = mpl.collections.PolyCollection(polygons_ll, closed=True, edgecolors='r', facecolors='r')
        ax1.add_collection(polycoll, autolim=True)
        polycoll = mpl.collections.PolyCollection(polygons_xy, closed=True, edgecolors='r', facecolors='r')
        ax2.add_collection(polycoll, autolim=True)

        # plot radar location and information text
        ax1.plot(radar['lon'], radar['lat'], 'r+')
        ax1.text(radar['lon'], radar['lat'], radar_id, color='r')

        # reproject lonlat radar location coordinates to polar stereographic projection
        x_loc, y_loc = wrl.georef.reproject(radar['lon'], radar['lat'], projection_source=proj_wgs,
                                            projection_target=proj_stereo)
        # plot radar location and information text
        ax2.plot(x_loc, y_loc, 'r+')
        ax2.text(x_loc, y_loc, radar_id, color='r')

    pl.tight_layout()
    pl.show()

# =======================================================
if __name__ == '__main__':
    ex_radolan_radarloc()