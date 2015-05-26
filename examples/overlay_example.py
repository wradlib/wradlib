#------------------------------------------------------------------------------
# Name:        overlay_example.py
# Purpose:     example file for creating image overlays from shapefiles
#              and underlays of terrain maps from raster files
#
# Author:      Kai Muehlbauer
#
# Created:     21.04.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------
# coding: utf-8

# some imports needed
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# convenient dataset and projection functions
import wradlib as wrl

# georef, gis
from osgeo import osr


def _check_file(filename):
    geo_src = 'https://bitbucket.org/kaimuehlbauer/wradlib_miub/downloads/geo.tar.gz'
    if not os.path.exists(filename):
        warnings.warn("File does not exist: {0}\nGet data from {1} and extract archive to wradlib/example/data folder".format(filename, geo_src))
        exit(0)

def nex_overlay():

    # set filepath
    filepath = os.path.join(os.path.dirname(__file__), 'data/geo')

    # setup figure
    fig1 = plt.figure(figsize=(10,8))

    # create 4 subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # we use this special prepared geotiff
    # created from srtm data via the following shell command:
    # gdalwarp -te 88. 20. 93. 27. srtm_54_07.tif srtm_55_07.tif srtm_54_08.tif srtm_55_08.tif bangladesh.tif

    filename = os.path.join(filepath,'bangladesh.tif')
    _check_file(filename)
    # pixel_spacing is in output units (lonlat)
    rastercoords, rastervalues = wrl.io.read_raster_data(filename, spacing=0.005)
    # specify kwargs for plotting, using terrain colormap and LogNorm
    dem = ax1.pcolormesh(rastercoords[...,0], rastercoords[...,1], rastervalues,  cmap=mpl.cm.terrain, norm=LogNorm(), vmin=1, vmax=3000)

    # make some space on the right for colorbar axis
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    # add colorbar and title
    # we use LogLocator for colorbar
    cb = fig1.colorbar(dem, cax=cax1, ticks = ticker.LogLocator(subs=range(10)))
    cb.set_label('terrain height [m]')

    # plot country borders from esri vector shape, filter by attribute
    # create wgs84 and india osr objects (spatial reference system)
    wgs84 = osr.SpatialReference ()
    wgs84.ImportFromEPSG(4326)
    india = osr.SpatialReference()
    # asia south albers equal area conic
    india.ImportFromEPSG(102028)

    # country list
    countries = ['India', 'Nepal', 'Bhutan', 'Myanmar']
    # open the input data source and get the layer
    filename = os.path.join(filepath, 'ne_10m_admin_0_boundary_lines_land.shp')
    _check_file(filename)
    dataset, inLayer = wrl.io.open_shape(filename)
    # iterate over countries, filter accordingly, get coordinates and plot
    for item in countries:
        # SQL-like selection syntax
        fattr = "(adm0_left = '"+item+"' or adm0_right = '"+item+"')"
        inLayer.SetAttributeFilter(fattr)
        # get borders and names
        borders, keys = wrl.georef.get_shape_coordinates(inLayer, key='name')
        wrl.vis.add_lines(ax1, borders, color='black', lw=2, zorder=4)

    # some testing on additional axes
    # add Bangladesh to countries
    countries.append('Bangladesh')
    # create colors for country-patches
    cm = mpl.cm.jet
    colors = []
    for i in range(len(countries)):
        colors.append(cm(1.*i/len(countries)))

    # open the input data source and get the layer
    filename = os.path.join(filepath, 'ne_10m_admin_0_countries.shp')
    _check_file(filename)
    dataset, layer = wrl.io.open_shape(filename)
    # iterate over countries, filter by attribute and plot single patches on ax2
    for i, item in enumerate(countries):
        fattr = "name = '"+item+"'"
        layer.SetAttributeFilter(fattr)
        # get country patches and geotransform to destination srs
        patches, keys = wrl.georef.get_shape_coordinates(layer, dest_srs=india, key='name')
        wrl.vis.add_patches(ax2, patches, facecolor=colors[i])

    ax2.autoscale(True)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X - Coordinate')
    ax2.set_ylabel('Y - Coordinate')
    ax2.ticklabel_format(style='sci', scilimits=(0,0))
    ax2.set_title('South Asia - Albers Equal Area Conic ')

    # reset Layer filter
    layer.SetAttributeFilter(None)
    layer.SetSpatialFilter(None)

    # filter spatially and plot as PatchCollection on ax3
    layer.SetSpatialFilterRect(88,20,93,27)
    patches, keys = wrl.georef.get_shape_coordinates(layer, dest_srs=wgs84, key='name')
    i = 0
    for name, patch in zip(keys, patches):
        # why comes the US in here?
        if name in countries:
            wrl.vis.add_patches(ax3, patch, facecolor=colors[i], cmap=mpl.cm.jet, alpha=0.4)
            i = i + 1
    ax3.autoscale(True)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('South Asia - WGS 84')

    # plot rivers from esri vector shape, filter spatially
    # http://www.fao.org/geonetwork/srv/en/metadata.show?id=37331

    # open the input data source and get the layer
    filename = os.path.join(filepath, 'rivers_asia_37331.shp')
    _check_file(filename)
    dataset, inLayer = wrl.io.open_shape(filename)

    # do spatial filtering to get only geometries inside bounding box
    inLayer.SetSpatialFilterRect(88,20,93,27)
    rivers, keys = wrl.georef.get_shape_coordinates(inLayer, key='MAJ_NAME')

    # plot on ax1, and ax4
    wrl.vis.add_lines(ax1, rivers, color=mpl.cm.terrain(0.), lw=0.5, zorder=3)
    wrl.vis.add_lines(ax4, rivers, color=mpl.cm.terrain(0.), lw=0.5, zorder=3)
    ax4.autoscale(True)
    ax4.set_aspect('equal')
    ax4.set_xlim((88,93))
    ax4.set_ylim((20,27))
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title('Bangladesh - Rivers')


    # plot rivers from esri vector shape, filter spatially
    # plot rivers from NED
    # open the input data source and get the layer
    filename = os.path.join(filepath, 'ne_10m_rivers_lake_centerlines.shp')
    _check_file(filename)
    dataset, inLayer = wrl.io.open_shape(filename)
    inLayer.SetSpatialFilterRect(88,20,93,27)
    rivers, keys = wrl.georef.get_shape_coordinates(inLayer)
    wrl.vis.add_lines(ax1, rivers, color=mpl.cm.terrain(0.), lw=0.5, zorder=3)
    ax1.autoscale(True)

    # ### plot city dots with annotation, finalize plot
    # lat/lon coordinates of five cities in Bangladesh
    lats = [23.73, 22.32, 22.83, 24.37, 24.90]
    lons = [90.40, 91.82, 89.55, 88.60, 91.87]
    cities=['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet']
    for lon, lat, city in zip(lons,lats,cities):
        ax1.plot(lon,lat,'ro', zorder=5)
        ax1.text(lon+0.01,lat+0.01,city)

    # set axes limits and equal aspect
    ax1.set_xlim((88,93))
    ax1.set_ylim((20,27))
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')
    ax1.set_title('Bangladesh')

    plt.tight_layout(w_pad=0.1)
    plt.show()


# =======================================================
if __name__ == '__main__':
    nex_overlay()