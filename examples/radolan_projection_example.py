
#------------------------------------------------------------------------------
# Name:        radolan_projection_example.py
# Purpose:     example for calculation of radolan grid coordinates
#
# Author:      Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import wradlib as wrl
from osgeo import osr


def ex_radolan_projection():

    # create radolan grid coordinates
    # add 1 to each dimension to get upper left corner coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)

    # create radolan projection osr object
    dwd_string = wrl.georef.create_projstr("dwd-radolan")
    proj_stereo = wrl.georef.proj4_to_osr(dwd_string)

    # create wgs84 projection osr object
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)

    # create Gauss Krueger zone 3 projection osr object
    proj_gk3 = osr.SpatialReference()
    proj_gk3.ImportFromEPSG(31467)

    # transform radolan polar stereographic projection to wgs84 and then to gk3
    radolan_grid_ll = wrl.georef.reproject(radolan_grid_xy, projection_source=proj_stereo, projection_target=proj_wgs)
    radolan_grid_gk = wrl.georef.reproject(radolan_grid_ll, projection_source=proj_wgs, projection_target=proj_gk3)

    lon_wgs0 = radolan_grid_ll[:, :, 0]
    lat_wgs0 = radolan_grid_ll[:, :, 1]

    x_gk3 = radolan_grid_gk[:, :, 0]
    y_gk3 = radolan_grid_gk[:, :, 1]

    x_rad = radolan_grid_xy[:, :, 0]
    y_rad = radolan_grid_xy[:, :, 1]

    print("\n------------------------------")
    print("source radolan x,y-coordinates")
    print(u"       {0}      {1} ".format('x [km]', 'y [km]'))
    print("ll: {:10.4f} {:10.3f} ".format(x_rad[0, 0], y_rad[0, 0]))
    print("lr: {:10.4f} {:10.3f} ".format(x_rad[0, -1], y_rad[0, -1]))
    print("ur: {:10.4f} {:10.3f} ".format(x_rad[-1, -1], y_rad[-1, -1]))
    print("ul: {:10.4f} {:10.3f} ".format(x_rad[-1, 0], y_rad[-1, 0]))
    print("\n--------------------------------------")
    print("transformed radolan lonlat-coordinates")
    print(u"      {0}  {1} ".format('lon [degE]', 'lat [degN]'))
    print("ll: {:10.4f}  {:10.4f} ".format(lon_wgs0[0, 0], lat_wgs0[0, 0]))
    print("lr: {:10.4f}  {:10.4f} ".format(lon_wgs0[0, -1], lat_wgs0[0, -1]))
    print("ur: {:10.4f}  {:10.4f} ".format(lon_wgs0[-1, -1], lat_wgs0[-1, -1]))
    print("ul: {:10.4f}  {:10.4f} ".format(lon_wgs0[-1, 0], lat_wgs0[-1, 0]))
    print("\n-----------------------------------")
    print("transformed radolan gk3-coordinates")
    print(u"     {0}   {1} ".format('easting [m]', 'northing [m]'))
    print("ll: {:10.0f}   {:10.0f} ".format(x_gk3[0, 0], y_gk3[0, 0]))
    print("lr: {:10.0f}   {:10.0f} ".format(x_gk3[0, -1], y_gk3[0, -1]))
    print("ur: {:10.0f}   {:10.0f} ".format(x_gk3[-1, -1], y_gk3[-1, -1]))
    print("ul: {:10.0f}   {:10.0f} ".format(x_gk3[-1, 0], y_gk3[-1, 0]))

# =======================================================
if __name__ == '__main__':
    ex_radolan_projection()