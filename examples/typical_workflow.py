#-------------------------------------------------------------------------------
# Name:        A typical workflow for radar-based rainfall estimation
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2012
# Copyright:   (c) heistermann 2012
# Licence:     MIT
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib
import numpy as np
import pylab as pl

if __name__ == '__main__':

    # read the data
    data, metadata = wradlib.io.readDX("data/sample.dx")
    wradlib.vis.polar_plot(data)
    # identify and visualise clutters
    clutter = wradlib.clutter.filter_gabella(data, tr1=12, n_p=6, tr2=1.1)
    wradlib.vis.polar_plot(clutter,title='Clutter Map',colormap=pl.cm.gray)
    # Remove and fill clutter
    data_no_clutter = wradlib.ipol.interpolate_polar(data, clutter)
    # Attenuation correction according to Kraemer
    pia = wradlib.atten.correctAttenuationKraemer(data_no_clutter)
    data_attcorr = data_no_clutter + pia
    # compare reflectivity with and without attenuation correction for one beam
    pl.plot(data_attcorr[65], label="attcorr")
    pl.plot(data_no_clutter[65], label="no attcorr")
    pl.xlabel("km")
    pl.ylabel("dBZ")
    pl.legend()
    pl.show()
    # converting to rainfall intensity
    R = wradlib.zr.z2r( wradlib.trafo.idecibel(data_attcorr) )
    # and then to rainfall depth over 5 minutes
    depth = wradlib.trafo.r2depth(R, 300)
    # example for rainfall accumulation in case we have a series of sweeps (here: random numbers)
    import numpy as np
    sweep_times  = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 300)
    depths_5min  = np.random.uniform(size=(len(sweep_times)-1, 360, 128))
    hours        = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 3600)
    depths_hourly= wradlib.util.aggregate_in_time(depths_5min, sweep_times, hours, func='sum')
    # Georeferencing
    radar_location = (47.8744, 8.005, 1517) # (lat, lon, alt) in decimal degree and meters
    elevation = 0.5 # in degree
    azimuths = np.arange(0,360) # in degrees
    ranges = np.arange(0, 128000., 1000.) # in meters
    lat, lon, alt = wradlib.georef.polar2latlonalt(ranges, azimuths, elevation, radar_location)
    # projection to Gauss Krueger zone 3
    gk3 = wradlib.georef.create_projstr("gk", zone=3)
    x, y = wradlib.georef.project(lat, lon, gk3)
    xy = np.vstack((x, y)).transpose()
    # transfer the north-east sector to a 1kmx1km grid
    xgrid = np.arange(x.mean(), x.max(), 1000.)
    ygrid = np.arange(y.mean(), y.max(), 1000.)
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_coords[0].ravel(), grid_coords[1].ravel())).transpose()
    gridded = wradlib.comp.togrid(xy, grid_xy, 128000., [x.mean(), y.mean()],depth, wradlib.ipol.Nearest)
    wradlib.vis.cartesian_plot(gridded)








