#-------------------------------------------------------------------------------
# Name:        recipe1_clutter_attenuation_composition.py
# Purpose:
#
# Author:      heistermann
#
# Created:     15.01.2013
# Copyright:   (c) heistermann 2013
# Licence:     MIT
#-------------------------------------------------------------------------------
#!/usr/bin/env python



import wradlib
import numpy as np
import pylab as pl
import glob
import os

if __name__ == '__main__':

    # set working directory
    os.chdir("recipe1_data")
    # preparations for loading a couple of files
    tur_files = glob.glob('raa*tur*bin')
    tur_data = np.empty((25,360,128))
    # loading the data (1h of 5-minute images) and correcting for clutter
    for i in range(25):
        tur_dx, tur_attrs = wradlib.io.readDX(tur_files[i])
        tur_clmap = wradlib.clutter.filter_gabella(tur_dx, tr1=12, n_p=6, tr2=1.1)
        tur_data[i] = wradlib.ipol.interpolate_polar(tur_dx, tur_clmap)
    # correcting for attenuation
    tur_k = wradlib.atten.correctAttenuationHJ(tur_data)
    tur_acorr = tur_data + tur_k
    # converting to precipitation depth
    tur_r = wradlib.zr.z2r(wradlib.trafo.idecibel(tur_acorr), a=256, b=1.4)
    tur_r_depth = wradlib.trafo.r2depth(tur_r, 300.)
    # calculate hourly accumulation
    tur_accum = tur_r_depth.sum(axis=0)

    fbg_files = glob.glob('raa*fbg*bin')
    fbg_data = np.empty((25,360,128))
    # loading the data (1h of 5-minute images) and correcting for clutter
    for i in range(25):
        fbg_dx, fbg_attrs = wradlib.io.readDX(fbg_files[i])
        fbg_clmap = wradlib.clutter.filter_gabella(fbg_dx, tr1=12, n_p=6, tr2=1.1)
        fbg_data[i] = wradlib.ipol.interpolate_polar(fbg_dx, fbg_clmap)
    # correcting for attenuation
    fbg_k = wradlib.atten.correctAttenuationHJ(fbg_data)
    fbg_acorr = fbg_data + fbg_k
    # converting to precipitation depth
    fbg_r = wradlib.zr.z2r(wradlib.trafo.idecibel(fbg_acorr), a=256, b=1.4)
    fbg_r_depth = wradlib.trafo.r2depth(fbg_r, 300.)
    # calculate hourly accumulation
    fbg_accum = fbg_r_depth.sum(axis=0)


    # arrays with scan geometry
    r = np.arange(1,129)*1000
    az = np.linspace(0,360,361)[0:-1]
    # PROJ.4 style projection string
    gk3 = '+proj=tmerc +lat_0=0 +lon_0=9 +k=1 ' + \
    '+x_0=3500000 +y_0=0 +ellps=bessel ' + \
    '+towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 ' + \
    '+units=m +no_defs'
    # derive Gauss-Krueger Zone 3 coordinates for Tuerkheim range-bin centroids
    tur_sitecoords = (48.5861, 9.7839)
    tur_cent_lon, tur_cent_lat = wradlib.georef.polar2centroids(r, az, tur_sitecoords)
    tur_x, tur_y = wradlib.georef.project(tur_cent_lat, tur_cent_lon, gk3)
    tur_coord = np.array([tur_x.ravel(),tur_y.ravel()]).transpose()
    # do the same for the Feldberg radar
    fbg_sitecoords = (47.8744, 8.005)
    fbg_cent_lon, fbg_cent_lat = wradlib.georef.polar2centroids(r, az, fbg_sitecoords)
    fbg_x, fbg_y = wradlib.georef.project(fbg_cent_lat, fbg_cent_lon, gk3)
    fbg_coord = np.array([fbg_x.ravel(),fbg_y.ravel()]).transpose()


    xc = np.concatenate((tur_coord[:,0],fbg_coord[:,0]))
    yc = np.concatenate((tur_coord[:,1],fbg_coord[:,1]))
    xmin = np.min(xc)-5000
    xmax = np.max(xc)+5000
    ymin = np.min(yc)-5000
    ymax = np.max(yc)+5000

    gridshape=(500,500)
    grid_coords = np.meshgrid(np.linspace(xmin,xmax,gridshape[0]), np.linspace(ymin,ymax,gridshape[1]))
    grid_coords = np.vstack((grid_coords[0].ravel(), grid_coords[1].ravel())).transpose()



    # derive quality information - in this case, the pulse volume
    ranges = np.arange(500., 128000., 1000.)
    pulse_volumes = np.tile(wradlib.qual.pulse_volume(ranges, 1000., 1.),360)
    # interpolate polar radar-data and quality data to the grid
    radius = 128000.
    tur_center = tur_coord.mean(axis=0)
    tur_quality_gridded = wradlib.comp.togrid(tur_coord, grid_coords, radius, tur_center,
    pulse_volumes, wradlib.ipol.Nearest)
    tur_gridded = wradlib.comp.togrid(tur_coord, grid_coords, radius, tur_center,
    tur_accum.ravel(), wradlib.ipol.Nearest)

    fbg_center = fbg_coord.mean(axis=0)
    fbg_quality_gridded = wradlib.comp.togrid(fbg_coord, grid_coords, radius, fbg_center,
    pulse_volumes, wradlib.ipol.Nearest)
    fbg_gridded = wradlib.comp.togrid(fbg_coord, grid_coords, radius, fbg_center,
    fbg_accum.ravel(), wradlib.ipol.Nearest)


    # compose the both radar-data based on the quality information calculated above
    composite = wradlib.comp.compose_weighted([tur_gridded, fbg_gridded],
    [1./(tur_quality_gridded+0.001),
    1./(fbg_quality_gridded+0.001)])

    import wradlib.vis as vis

    composite = np.ma.masked_invalid(composite)

    classes = [0,5,10,20,30,40,50,75,100,125,150,200]

    vis.cartesian_plot(composite.reshape(gridshape), x=np.linspace(xmin,xmax,gridshape[0]), y=np.linspace(ymin,ymax,gridshape[1]), unit="mm", colormap="spectral", classes=classes)
