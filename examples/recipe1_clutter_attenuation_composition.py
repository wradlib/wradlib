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
# just making sure that the plots immediately pop up
pl.interactive(True)
import glob
import os
import datetime as dt
import zipfile
import shutil



def process_polar_level_data(radarname):
    """Reading and processing polar level data (DX) for radar <radarname>
    """
    print "Polar level processing for radar %s..." % radarname
    # preparations for loading sample data in source directory
    #files = glob.glob(os.path.dirname(__file__) + '/' + 'data/raa*%s*bin'%radarname)
    files = glob.glob(os.path.dirname(__file__) + '/' + 'data/recipe1_data/raa*%s*bin'%radarname)

    if len(files)==0:
        print "WARNING: No data files found - maybe you did not extract the data from data/recipe1_data.zip?"
    data  = np.empty((len(files),360,128))
    # loading the data (two hours of 5-minute images)
    for i, f in enumerate(files):
        data[i], attrs = wradlib.io.readDX(f)
    # Clutter filter on an event base
    clmap = wradlib.clutter.filter_gabella(data.mean(axis=0), tr1=12, n_p=6, tr2=1.1)
    for i, scan in enumerate(data):
        data[i] = wradlib.ipol.interpolate_polar(scan, clmap)
    # correcting for attenuation
    k = wradlib.atten.correctAttenuationHJ(data)
    data = data + k
    # converting to precipitation depth
    R = wradlib.zr.z2r(wradlib.trafo.idecibel(data), a=256, b=1.4)
    depth = wradlib.trafo.r2depth(R, 300.)
    # calculate hourly accumulation
    accum = depth.sum(axis=0)

    return accum


def bbox(*args):
    """Get bounding box from a set of radar bin coordinates
    """
    x = np.array([])
    y = np.array([])
    for arg in args:
        x = np.append(x, arg[:,0])
        y = np.append(y, arg[:,1])
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    return xmin, xmax, ymin, ymax

def recipe_clutter_attenuation():

    # set timer
    start = dt.datetime.now()
    # unzip data
    filename = os.path.dirname(__file__) + '/' + 'data/recipe1_data.zip'
    targetdir = os.path.dirname(__file__) + '/' + 'data/recipe1_data'
    with zipfile.ZipFile(filename, 'r') as z:
        z.extractall(targetdir)

    # set scan geometry and radar coordinates
    r               = np.arange(500.,128500.,1000.)
    az              = np.arange(0,360)
    tur_sitecoords  = (9.7839, 48.5861)
    fbg_sitecoords  = (8.005, 47.8744)

    # PROJ.4 style projection string for target reference system
    gk3 = wradlib.georef.create_projstr("gk", zone=3)

    # processing polar level radar data
    #   Tuerkheim
    tur_accum = process_polar_level_data("tur")
    #   Feldberg
    fbg_accum = process_polar_level_data("fbg")

    # remove unzipped files
    if os.path.exists(targetdir):
        try:
            shutil.rmtree(targetdir)
        except:
            print "WARNING: Could not remove directory data/recipe1_data"

    # derive Gauss-Krueger Zone 3 coordinates of range-bin centroids
    #   for Tuerkheim radar
    proj_gk3 = wradlib.georef.proj4_to_osr(gk3)
    tur_cent_lon, tur_cent_lat = wradlib.georef.polar2centroids(r, az, tur_sitecoords)
    tur_x, tur_y = wradlib.georef.reproject(tur_cent_lon, tur_cent_lat, projection_target=proj_gk3)
    tur_coord = np.array([tur_x.ravel(),tur_y.ravel()]).transpose()
    #    for Feldberg radar
    fbg_cent_lon, fbg_cent_lat = wradlib.georef.polar2centroids(r, az, fbg_sitecoords)
    fbg_x, fbg_y = wradlib.georef.reproject(fbg_cent_lon, fbg_cent_lat, projection_target=proj_gk3)
    fbg_coord = np.array([fbg_x.ravel(),fbg_y.ravel()]).transpose()

    # define target grid for composition
    xmin, xmax, ymin, ymax = bbox(tur_coord, fbg_coord)
    x = np.linspace(xmin,xmax+1000.,1000.)
    y = np.linspace(ymin,ymax+1000.,1000.)
    grid_coords = wradlib.util.gridaspoints(y,x)

    # derive quality information - in this case, the pulse volume
    pulse_volumes = np.tile(wradlib.qual.pulse_volume(r, 1000., 1.),360)
    # interpolate polar radar-data and quality data to the grid
    print "Gridding Tuerkheim data..."
    tur_quality_gridded = wradlib.comp.togrid(tur_coord, grid_coords, r.max()+500., tur_coord.mean(axis=0), pulse_volumes, wradlib.ipol.Nearest)
    tur_gridded = wradlib.comp.togrid(tur_coord, grid_coords, r.max()+500., tur_coord.mean(axis=0), tur_accum.ravel(), wradlib.ipol.Nearest)

    print "Gridding Feldberg data..."
    fbg_quality_gridded = wradlib.comp.togrid(fbg_coord, grid_coords, r.max()+500., fbg_coord.mean(axis=0), pulse_volumes, wradlib.ipol.Nearest)
    fbg_gridded = wradlib.comp.togrid(fbg_coord, grid_coords, r.max()+500., fbg_coord.mean(axis=0), fbg_accum.ravel(), wradlib.ipol.Nearest)

    # compose the both radar-data based on the quality information calculated above
    print "Composing Tuerkheim and Feldbarg data on a common grid..."
    composite = wradlib.comp.compose_weighted([tur_gridded, fbg_gridded],[1./(tur_quality_gridded+0.001),1./(fbg_quality_gridded+0.001)])
    composite = np.ma.masked_invalid(composite)

    print "Processing took:", dt.datetime.now()-start

    # Plotting rainfall map
    ax = pl.subplot(111, aspect="equal")
    pm = pl.pcolormesh(x, y, composite.reshape((len(x),len(y))), cmap="spectral")
    pl.grid()
    pl.colorbar(pm)

if __name__ == '__main__':
    recipe_clutter_attenuation()