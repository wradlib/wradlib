#-------------------------------------------------------------------------------
# Name:        vis
# Purpose:
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     26.10.2011
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Visualisation
^^^^^^^^^^^^^

Standard plotting and mapping procedures

.. autosummary::
   :nosignatures:
   :toctree: generated/

   polar_plot


"""

import os.path as path
import numpy as np
import pylab as pl
from matplotlib import mpl
from mpl_toolkits.basemap import Basemap

import wradlib.georef as georef

from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform

class NorthPolarAxes(PolarAxes):
    '''
    A variant of PolarAxes where theta starts pointing north and goes
    clockwise.
    '''
    name = 'northpolar'

    class NorthPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy   = np.zeros(tr.shape, np.float_)
            t    = tr[:, 0:1]
            r    = tr[:, 1:2]
            x    = xy[:, 0:1]
            y    = xy[:, 1:2]
            x[:] = r * np.sin(t)
            y[:] = r * np.cos(t)
            return xy

        transform_non_affine = transform

        def inverted(self):
            return NorthPolarAxes.InvertedNorthPolarTransform()

    class InvertedNorthPolarTransform(PolarAxes.InvertedPolarTransform):
        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            return np.concatenate((theta, r), 1)

        def inverted(self):
            return NorthPolarAxes.NorthPolarTransform()

    def _set_lim_and_transforms(self):
        PolarAxes._set_lim_and_transforms(self)
        self.transProjection = self.NorthPolarTransform()
        self.transData = (
            self.transScale +
            self.transProjection +
            (self.transProjectionAffine + self.transAxes))
        self._xaxis_transform = (
            self.transProjection +
            self.PolarAffine(IdentityTransform(), Bbox.unit()) +
            self.transAxes)
        self._xaxis_text1_transform = (
            self._theta_label1_position +
            self._xaxis_transform)
        self._yaxis_transform = (
            Affine2D().scale(np.pi * 2.0, 1.0) +
            self.transData)
        self._yaxis_text1_transform = (
            self._r_label1_position +
            Affine2D().scale(1.0 / 360.0, 1.0) +
            self._yaxis_transform)

register_projection(NorthPolarAxes)


def polar_plot(data, title='', unit='', saveto='', fig=None, axpos=111, R=1., theta0=0, colormap='jet', classes=None, extend='neither', **kwargs):
    """Plots data from a polar grid.

    The data must be an array of shape (number of azimuth angles, number of range bins).
    The azimuth angle of zero corresponds to the north, the angles are counted clock-wise forward.

    additional `kwargs` will be passed to the pcolormesh routine displaying
    the data.

    Parameters
    ----------
    data : 2-d array
        polar grid data to be plotted
        1st dimension must be azimuth angles, 2nd must be ranges!
    title : string
        a title of the plot
    unit : string
        the unit of the data which is plotted
    saveto : string - path of the file in which the figure should be saved
        if string is empty, no figure will be saved and the plot will be
        sent to screen
    fig : matplotlib axis object
        if None, a new matplotlib figure will be created, otherwise we plot on ax
    axpos : an integer or a string
        correponds to the positional argument of matplotlib.figure.add_subplot
    R : float
        maximum range
    theta0 : integer
        azimuth angle which corresponds to the first slice of the dataset
        (normally corresponds to 0)
    colormap : string
        choose between the colormaps "jet" (per default) and "spectral"
    classes : sequence of numerical values
        class boundaries for plotting
    extend : string
        determines the behaviour of the colorbar: default value 'neither' produces
        a standard colorbar, 'min' and 'max' produces an arrow at the minimum or
        maximum end, respectively, and 'both' produces an arrow at both ends. If
        you use class boundaries for plotting, you should typically use 'both'.

    """
    n_theta, n_r = data.shape
    theta = np.linspace(0, 2*np.pi, n_theta+1)
    r = np.linspace(0., R, n_r+1)

    data = np.transpose(data)
    data = np.roll(data, theta0, axis=1)

    # plot as pcolormesh
    if fig==None:
        # crate a new figure object
        fig = pl.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection="northpolar", aspect=1.)
    else:
        # plot on the axes object which was passed to this function
        ax = fig.add_subplot(axpos, projection="northpolar", aspect=1.)
    if classes==None:
        # automatic color normalization by vmin and vmax (not recommended)
        circle = ax.pcolormesh(theta, r, data,rasterized=True, cmap=colormap, **kwargs)
    else:
        # colors are assigned according to class boundaries and colormap argument
        mycmap = pl.get_cmap(colormap, lut=len(classes))
        mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(classes)-1) ))
        norm   = mpl.colors.BoundaryNorm(classes, mycmap.N)
        circle = ax.pcolormesh(theta, r, data,rasterized=True, cmap=mycmap, norm=norm, **kwargs)
    pl.grid(True)
    cbar = pl.colorbar(circle, shrink=0.75, extend=extend)
    cbar.set_label('('+unit+')')
    pl.title(title)
    if saveto=='':
        # show plot
        pl.show()
        if not pl.isinteractive():
            # close figure eplicitely if pylab is not in interactive mode
            pl.close()
    else:
        # save plot to file
        if ( path.exists(path.dirname(saveto)) ) or ( path.dirname(saveto)=='' ):
            pl.savefig(saveto)


class PolarBasemap():
    '''
    Plot a spatial points dataset as a map (or a time series of maps)

    Parameters
    ----------
    data    : Dataset which should be plotted
                if <dset> contains different time steps, one map will be generated for each time step
    conf    : a config object
    title   : a base title - other elements will be appended to this base title
    bbox    : the bounding box of the entire map in lat/lon; if None, the specs will be read from the config file key 'bbox_map'
    ncolors : number of colors in the colomap lookup table - will be overridden by the classes argument
    classes : classes of the plotting variable for which colors should be homogenoeous - overrides ncolors!
    cmap    : name of the default colormap in case no colormap is provided in the config file
    ensstat : in case dset contains an ensemble Dimension, the statistic function with name <ensstat> will be used to remove the ensemble Dimension by applying ensstat along the ens Dimension
                <ensstat> should be contained in numpy and be retrived by getattr(numpy, ensstat) and it should have an axis argument
    saveto  : if None, the plots are shown on the screen - otherwise the figures are saved to directory <saveto>
    '''
    def __init__(self, polygons, sitecoords, r, az, title='', bbox=None, ncolors=10, classes=None, cmap='jet'):

        # Georeferencing the radar data
        polygons = georef.polar2polyvert(r, az, sitecoords)

        # define spatial bounding box of the Basemap
        if bbox==None:
            self.bbox={'llcrnrlon':np.min(polygons[:,:,0]),
                  'llcrnrlat':np.min(polygons[:,:,1]),
                  'urcrnrlon':np.max(polygons[:,:,0]),
                  'urcrnrlat':np.max(polygons[:,:,1])}
        else:
            self.bbox = bbox

        # define class boundaries for plotting
        if classes!=None:
            self.classes = np.array(classes)
        else:
            self.classes = np.array([-100, 10, 20, 30, 40, 50, 60, 70])
        self.ncolors = len(self.classes)

        # define map center
        lon0=sitecoords[1]
        lat0=sitecoords[0]

        # plot the Basemap
        self.m = Basemap(llcrnrlon=self.bbox['llcrnrlon'],llcrnrlat=self.bbox['llcrnrlat'],
                        urcrnrlon=self.bbox['urcrnrlon'],urcrnrlat=self.bbox['urcrnrlat'],
                    resolution='i',projection='tmerc',lat_0=lat0, lon_0=lon0)

        # draw parallels and meridians
##        self.m.drawmapboundary(fill_color='aqua')
        # fill continents, set lake color same as ocean color.
##        self.m.fillcontinents(color='coral',lake_color='aqua')
        self.m.drawcoastlines(color='white')
        self.m.drawparallels(np.linspace(start=np.round(self.bbox['llcrnrlat']), stop=np.round(self.bbox['urcrnrlat']), num=3), labels=[1,0,0,0])
        self.m.drawmeridians(np.linspace(start=np.round(self.bbox['llcrnrlon']), stop=np.round(self.bbox['urcrnrlon']), num=3), labels=[0,0,0,1])
        # draw map scale
        self.m.drawmapscale(lon=self.bbox['urcrnrlon']-0.2*(self.bbox['urcrnrlon']-self.bbox['llcrnrlon']), lat=self.bbox['llcrnrlat']+0.1*(self.bbox['urcrnrlat']-self.bbox['llcrnrlat']), lon0=lon0, lat0=lat0, length=50., units='km', barstyle='fancy')

        polygons[:,:,0], polygons[:,:,1] = self.m(polygons[:,:,0], polygons[:,:,1])
        self.polygons = polygons
    ##    # read shapefile which defines the plotting locations as polygons
    ##    s = m.readshapefile(conf['shapefile_locations'], 'datashp', drawbounds=False)
    ##
    ##    # read the other shapefiles (which are only plotted as lines)
    ##    if conf.has_key('shapefiles_extra'):
    ##        oshps = {}
    ##        for key in conf['shapefiles_extra'].keys():
    ##            oshps[key] = m.readshapefile(conf['shapefiles_extra'][key], key, linewidth=conf['shapefiles_lwds'][key], color=conf['shapefiles_colors'][key])

        # define plotting colormap and normalization

        #   the color map needs one entry less than class boundaries!
##        if unit=='p':
##            mycmap = pl.get_cmap(cmap, lut=len(classes)-2)
##            myclist= mycmap( np.arange(mycmap.N) ).tolist()
##            myclist.insert(0,(0,0,0))
##            self.mycmap = mpl.colors.ListedColormap(myclist)
##        else:
##            mycmap = pl.get_cmap(cmap, lut=len(classes))
##            self.mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(classes)-1) ))
        self.mycmap = pl.get_cmap(cmap, lut=len(self.classes))
        self.mycmap = mpl.colors.ListedColormap(self.mycmap( np.arange(len(self.classes)-1) ))

        norm   = mpl.colors.BoundaryNorm(self.classes, self.mycmap.N)

        # define colorbar (we use a dummy mappable object via imshow)
        self.cbar = pl.colorbar(mappable=pl.imshow(np.repeat(self.classes,2).reshape((2,-1)),
                    cmap=self.mycmap, norm = norm), orientation='vertical', shrink=0.8, extend='max')
##        self.cbar.set_label('('+unit+')')

        # get current axes instance
        self.ax = pl.gca()


##        plot_data_on_map(ax=ax, data=data.ravel(), dtime='', mycmap=mycmap,
##                    polygons=polygons, classes=classes, bbox=bbox, name=var, saveto=None)
##
##        pl.close()

    def __call__(self, data, dtime='', varname='', varunit='', saveto=None):
        '''
        Takes care of the actual data plot for each time step (plotting coloured polygons)
        ---
        ax      : matplotlib axes instance on which to plot the polygons
        data    : a data array which must be consistent with the number of polygons as given by polygons
        dtime   : the datetime which defines the end of the period represented by data
        mycmap  : a colormap as defined in the calling function
        polygons: a numpay ndarray of shape (number of polygons, number of polygon corners)
        bbox    : the map's bounding box
        name    : the name of the dataset (normally a parameter such as <p> or <wc>)
        dsettype: the dsettype of the Dataset the data comes from
        saveto  : if None, the map will be pplotted to the screen, otherwise it will be saved to directory <saveto>
        '''
        # give each polygon of the shapefile <datashp> a fillcolor based on its value
        facecolors = np.repeat(self.mycmap(0)[0:3], len(self.polygons) ).reshape((-1,3),order='F')

        for i,classval in enumerate(self.classes[1:]):
            colidx = np.where(data.ravel()>=classval)[0]
            facecolors[colidx,:] = np.array(self.mycmap(i+1)[0:3])

        # plot polygons using matplotlib PolyCollection
        polycoll = mpl.collections.PolyCollection(self.polygons,closed=True, facecolors=facecolors,edgecolors=facecolors)
        mainplot = self.ax.add_collection(polycoll, autolim=True)

        # add title to plot
    ##    pl.title( get_map_title(name, dsettype, dtime) )

        # if no save directory is given, show plot on screen
        if saveto==None:
            pl.show()
        else:
            fname    = name + '_' + dtime.strftime('%Y%m%d%H%M%S') + '.png'
            savepath = path.join(saveto, fname)
            pl.savefig(savepath)
        # remove the PolygonCollection from the axis (otherwise the axis object becomes successively overcrowded)
        self.ax.collections.remove(polycoll)


class Grid2Basemap():
    '''
    Plot a grid a map (or a time series of maps)

    Parameters
    ----------
    data    : Dataset which should be plotted
                if <dset> contains different time steps, one map will be generated for each time step
    conf    : a config object
    title   : a base title - other elements will be appended to this base title
    bbox    : the bounding box of the entire map in lat/lon; if None, the specs will be read from the config file key 'bbox_map'
    ncolors : number of colors in the colomap lookup table - will be overridden by the classes argument
    classes : classes of the plotting variable for which colors should be homogenoeous - overrides ncolors!
    cmap    : name of the default colormap in case no colormap is provided in the config file
    ensstat : in case dset contains an ensemble Dimension, the statistic function with name <ensstat> will be used to remove the ensemble Dimension by applying ensstat along the ens Dimension
                <ensstat> should be contained in numpy and be retrived by getattr(numpy, ensstat) and it should have an axis argument
    saveto  : if None, the plots are shown on the screen - otherwise the figures are saved to directory <saveto>
    '''
    def __init__(self, bbox, title='', ncolors=10, classes=None, cmap='jet'):

        self.bbox = bbox

        # define class boundaries for plotting
        if classes!=None:
            self.classes = np.array(classes)
        else:
            self.classes = np.array([-100, 10, 20, 30, 40, 50, 60, 70])
        self.ncolors = len(self.classes)

        # define map center
        lon0=(bbox['llx']+bbox['urx'])/2
        lat0=(bbox['lly']+bbox['ury'])/2

        # plot the Basemap
        self.m = Basemap(llcrnrlon=self.bbox['llx'],llcrnrlat=self.bbox['lly'],
                        urcrnrlon=self.bbox['urx'],urcrnrlat=self.bbox['ury'],
                    resolution='i',projection='tmerc',lat_0=lat0, lon_0=lon0)

        # draw parallels and meridians
##        self.m.drawmapboundary(fill_color='aqua')
        # fill continents, set lake color same as ocean color.
##        self.m.fillcontinents(color='coral',lake_color='aqua')
        self.m.drawcoastlines(color='white')
        self.m.drawparallels(np.linspace(start=np.round(self.bbox['lly']), stop=np.round(self.bbox['ury']), num=3), labels=[1,0,0,0])
        self.m.drawmeridians(np.linspace(start=np.round(self.bbox['llx']), stop=np.round(self.bbox['urx']), num=3), labels=[0,0,0,1])
        # draw map scale
        self.m.drawmapscale(lon=self.bbox['urx']-0.2*(self.bbox['urx']-self.bbox['llx']), lat=self.bbox['lly']+0.1*(self.bbox['ury']-self.bbox['lly']), lon0=lon0, lat0=lat0, length=50., units='km', barstyle='fancy')

        self.mycmap = pl.get_cmap(cmap, lut=len(self.classes))
        self.mycmap = mpl.colors.ListedColormap(self.mycmap( np.arange(len(self.classes)-1) ))

        norm   = mpl.colors.BoundaryNorm(self.classes, self.mycmap.N)

        # define colorbar (we use a dummy mappable object via imshow)
        self.cbar = pl.colorbar(mappable=pl.imshow(np.repeat(self.classes,2).reshape((2,-1)),
                    cmap=self.mycmap, norm = norm), orientation='vertical', shrink=0.8, extend='max')
##        self.cbar.set_label('('+unit+')')

        # get current axes instance
        self.ax = pl.gca()


##        plot_data_on_map(ax=ax, data=data.ravel(), dtime='', mycmap=mycmap,
##                    polygons=polygons, classes=classes, bbox=bbox, name=var, saveto=None)
##
##        pl.close()

    def __call__(self, x, y, data, dtime='', varname='', varunit='', saveto=None):
        '''
        Takes care of the actual data plot for each time step (plotting coloured polygons)
        ---
        ax      : matplotlib axes instance on which to plot the polygons
        data    : a data array which must be consistent with the number of polygons as given by polygons
        dtime   : the datetime which defines the end of the period represented by data
        mycmap  : a colormap as defined in the calling function
        polygons: a numpay ndarray of shape (number of polygons, number of polygon corners)
        bbox    : the map's bounding box
        name    : the name of the dataset (normally a parameter such as <p> or <wc>)
        dsettype: the dsettype of the Dataset the data comes from
        saveto  : if None, the map will be pplotted to the screen, otherwise it will be saved to directory <saveto>
        '''
        # add title to plot
    ##    pl.title( get_map_title(name, dsettype, dtime) )

        # if no save directory is given, show plot on screen
        if saveto==None:
            pl.show()
        else:
            fname    = name + '_' + dtime.strftime('%Y%m%d%H%M%S') + '.png'
            savepath = path.join(saveto, fname)
            pl.savefig(savepath)
        # remove the PolygonCollection from the axis (otherwise the axis object becomes successively overcrowded)
        self.ax.collections.remove(polycoll)



if __name__ == '__main__':
    print 'wradlib: Calling module <vis> as main...'



