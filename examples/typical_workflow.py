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


def ex_typical_workflow():

    import wradlib
    import numpy as np
    import pylab as pl
    import os
    pl.interactive(True)
    # read the data
    data, metadata = wradlib.io.readDX(os.path.join( os.path.dirname(__file__), "data/sample.dx") )
    fig = pl.figure()
    ax = pl.subplot(111)
    ax, pm = wradlib.vis.plot_ppi(data, ax=ax)
    cmap = pl.colorbar(pm, shrink=0.75)

    # identify and visualise clutters
    clutter = wradlib.clutter.filter_gabella(data, tr1=12, n_p=6, tr2=1.1)
    fig = pl.figure()
    ax = pl.subplot(111)
    ax, pm = wradlib.vis.plot_ppi(clutter, ax=ax, cmap=pl.cm.gray)
    pl.title('Clutter Map')
    cmap = pl.colorbar(pm, shrink=0.75)

    # Remove and fill clutter
    data_no_clutter = wradlib.ipol.interpolate_polar(data, clutter)

    # Attenuation correction according to Kraemer
    pia = wradlib.atten.correctAttenuationKraemer(data_no_clutter)
    data_attcorr = data_no_clutter + pia
    # compare reflectivity with and without attenuation correction for one beam
    fig = pl.figure()
    ax = pl.subplot(111)
    pl.plot(data_attcorr[240], label="attcorr")
    pl.plot(data_no_clutter[240], label="no attcorr")
    pl.xlabel("km")
    pl.ylabel("dBZ")
    pl.legend()
#    pl.savefig("_test_ppi_attcorr.png")

    # converting to rainfall intensity
    R = wradlib.zr.z2r( wradlib.trafo.idecibel(data_attcorr) )
    # and then to rainfall depth over 5 minutes
    depth = wradlib.trafo.r2depth(R, 300)

    # example for rainfall accumulation in case we have a series of sweeps (here: random numbers)
    sweep_times  = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-26 02:00:00", 300)
    depths_5min  = np.random.uniform(size=(len(sweep_times)-1, 360, 128))
    hours        = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-26 02:00:00", 3600)
    depths_hourly= wradlib.util.aggregate_in_time(depths_5min, sweep_times, hours, func='sum')

    # Georeferencing
    radar_location = (8.005, 47.8744, 1517) # (lon, lat, alt) in decimal degree and meters
    elevation = 0.5 # in degree
    azimuths = np.arange(0,360) # in degrees
    ranges = np.arange(0, 128000., 1000.) # in meters
    polargrid = np.meshgrid(ranges, azimuths)
    lon, lat, alt = wradlib.georef.polar2lonlatalt_n(polargrid[0], polargrid[1], elevation, radar_location)

    # projection to Gauss Krueger zone 3
    gk3 = wradlib.georef.create_projstr("gk", zone=3)
    proj_gk3 = wradlib.georef.proj4_to_osr(gk3)
    x, y = wradlib.georef.reproject(lon, lat, projection_target=proj_gk3)
    xy = np.vstack((x.ravel(), y.ravel())).transpose()

    # transfer the north-east sector to a 1kmx1km grid
    xgrid = np.linspace(x.min(), x.mean(), 100)
    ygrid = np.linspace(y.min(), y.mean(), 100)
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
    gridded = wradlib.comp.togrid(xy, grid_xy, 128000., np.array([x.mean(), y.mean()]), data.ravel(), wradlib.ipol.Idw)
    gridded = np.ma.masked_invalid(gridded).reshape((len(xgrid), len(ygrid)))

    fig = pl.figure(figsize=(10,8))
    ax = pl.subplot(111, aspect="equal")
    pm = pl.pcolormesh(xgrid, ygrid, gridded)
    pl.colorbar(pm, shrink=0.75)
    pl.xlabel("Easting (m)")
    pl.ylabel("Northing (m)")

    # Adjustment example
    radar_coords = np.arange(0,101)
    truth = np.abs(1.5+np.sin(0.075*radar_coords)) + np.random.uniform(-0.1,0.1,len(radar_coords))
    #The radar rainfall estimate ``radar`` is then computed by imprinting a multiplicative ``error`` on ``truth`` and adding some noise.
    error = 0.75 + 0.015*radar_coords
    radar = error * truth + np.random.uniform(-0.1,0.1,len(radar_coords))
    #Synthetic gage observations ``obs`` are then created by selecting arbitrary "true" values.
    obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    obs = truth[obs_coords]
    #Now we adjust the ``radar`` rainfall estimate by using the gage observations. First, you create an "adjustment object" from the approach you
    #want to use for adjustment. After that, you can call the object with the actual data that is to be adjusted. Here, we use a multiplicative error model with spatially heterogenous error (see :doc:`wradlib.adjust.AdjustMultiply`).
    adjuster = wradlib.adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=3)
    adjusted = adjuster(obs, radar)
    #Let's compare the ``truth``, the ``radar`` rainfall estimate and the ``adjusted`` product:
    fig = pl.figure()
    ax = pl.subplot(111)
    pl.plot(radar_coords, truth, 'k-', label="True rainfall", linewidth=2.)
    pl.xlabel("Distance (km)")
    pl.ylabel("Rainfall intensity (mm/h)")
    pl.plot(radar_coords, radar, 'k-', label="Raw radar rainfall", linewidth=2., linestyle="dashed")
    pl.plot(obs_coords, obs, 'o', label="Gage observation", markersize=10.0, markerfacecolor="grey")
    pl.plot(radar_coords, adjusted, '-', color="green", label="Multiplicative adjustment", linewidth=2., )
    pl.legend(prop={'size':12})

    # Verification
    raw_error  = wradlib.verify.ErrorMetrics(truth, radar)
    adj_error  = wradlib.verify.ErrorMetrics(truth, adjusted)

    raw_error.report()
    adj_error.report()

    # Export
    #Export your data array as a text file:
    np.savetxt("mydata.txt", data)
    #Or as a gzip-compressed text file:
    np.savetxt("mydata.gz", data)
    #Or as a NetCDF file:
    import netCDF4
    rootgrp = netCDF4.Dataset('test.nc', 'w', format='NETCDF4')
    sweep_xy = rootgrp.createGroup('sweep_xy')
    dim_azimuth = sweep_xy.createDimension('azimuth', None)
    dim_range = sweep_xy.createDimension('range', None)
    azimuths_var = sweep_xy.createVariable('azimuths','i4',('azimuth',))
    ranges_var = sweep_xy.createVariable('ranges','f4',('range',))
    dBZ_var = sweep_xy.createVariable('dBZ','f4',('azimuth','range',))
    azimuths_var[:] = np.arange(0,360)
    ranges_var[:] = np.arange(0, 128000., 1000.)
    dBZ_var[:] = data
    rootgrp.bandwith = "C-Band"
    sweep_xy.datetime = "2012-11-02 10:15:00"

    rootgrp.close()

if __name__ == '__main__':

    ex_typical_workflow()












