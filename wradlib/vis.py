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
   Grid2Basemap

"""

import os.path as path
import numpy as np
import pylab as pl
from matplotlib import mpl
from mpl_toolkits.basemap import Basemap, cm

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


class PolarPlot(object):
    def __init__(self, ax=None, fig=None, axpos=111, **kwargs):
        if ax is None:
            if fig is None:
                # crate a new figure object
                fig = pl.figure(**kwargs)
            # plot on the axes object which was passed to this function
            ax = fig.add_subplot(axpos, projection="northpolar", aspect=1.)

        self.fig = fig
        self.ax = ax
        self.cmap = 'jet'
        self.norm = None

    def set_cmap(self, cmap, classes=None):
        if classes is None:
            self.cmap = cmap
        else:
            mycmap = pl.get_cmap(cmap, lut=len(classes))
            mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(classes)-1) ))
            norm   = mpl.colors.BoundaryNorm(classes, mycmap.N)
            self.cmap = mycmap
            self.norm = norm


    def plot(self, data, R=1., theta0=0, **kwargs):
        n_theta, n_r = data.shape
        theta = np.linspace(0, 2*np.pi, n_theta+1)
        r = np.linspace(0., R, n_r+1)

        data = np.transpose(data)
        data = np.roll(data, theta0, axis=1)

        circle = self.ax.pcolormesh(theta, r, data, rasterized=True, cmap=self.cmap,
                           norm=self.norm, **kwargs)
        return circle


    def colorbar(self, *args, **kwargs):
        #if not kwargs.has_key('shrink'):
        #    kwargs['shrink'] = 0.75
        cbar = pl.colorbar(*args, **kwargs)
        return cbar


    def title(self, s, *args, **kwargs):
        l = self.ax.set_title(s, *args, **kwargs)
        pl.draw_if_interactive()
        return l


    def grid(self, b=None, which='major', **kwargs):
        ret =  self.ax.grid(b, which, **kwargs)
        pl.draw_if_interactive()
        return ret


def polar_plot2(data, title='', unit='', saveto='', fig=None, axpos=111, R=1., theta0=0, colormap='jet', classes=None, extend='neither', **kwargs):
    pp = PolarPlot(fig=fig, axpos=axpos, figsize=(8,8))
    pp.set_cmap(colormap, classes=classes)
    circle = pp.plot(data, R=R, theta0=theta0, **kwargs)
    pp.grid(True)
    cbar = pp.colorbar(circle, shrink=0.75, extend=extend)
    cbar.set_label('('+unit+')')
    pp.title(title)
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
            pl.close()


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
            pl.close()


class CartesianPlot(object):
    def __init__(self, ax=None, fig=None, axpos=111, **kwargs):
        if ax is None:
            if fig is None:
                # create a new figure object
                fig = pl.figure(**kwargs)
            # plot on the axes object which was passed to this function
            ax = fig.add_subplot(axpos, aspect=1.)

        self.fig = fig
        self.ax = ax
        self.cmap = 'jet'
        self.norm = None

    def set_cmap(self, cmap, classes=None):
        if classes is None:
            self.cmap = cmap
        else:
            mycmap = pl.get_cmap(cmap, lut=len(classes))
            mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(classes)-1) ))
            norm   = mpl.colors.BoundaryNorm(classes, mycmap.N)
            self.cmap = mycmap
            self.norm = norm


    def plot(self, data, x, y, **kwargs):

        grd = self.ax.pcolormesh(x,y,data,rasterized=True, cmap=self.cmap,
                          norm=self.norm, **kwargs)
        return grd


    def colorbar(self, *args, **kwargs):
        #if not kwargs.has_key('shrink'):
        #    kwargs['shrink'] = 0.75
        cbar = pl.colorbar(*args, **kwargs)
        return cbar


    def title(self, s, *args, **kwargs):
        l = self.ax.set_title(s, *args, **kwargs)
        pl.draw_if_interactive()
        return l


    def grid(self, b=None, which='major', **kwargs):
        ret =  self.ax.grid(b, which, **kwargs)
        pl.draw_if_interactive()
        return ret



def cartesian_plot(data, x=None, y=None, title='', unit='', saveto='', fig=None, axpos=111, colormap='jet', classes=None, extend='neither', **kwargs):
    """Plots data from a cartesian grid.

    The data must be an array of shape (number of rows, number of columns).

    additional `kwargs` will be passed to the pcolormesh routine displaying
    the data.

    Parameters
    ----------
    data : 2-d array
        regular cartesian grid data to be plotted
        1st dimension must be number of rows, 2nd must be number of columns!
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
    pp = CartesianPlot(fig=fig, axpos=axpos, figsize=(8,8))
    pp.set_cmap(colormap, classes=classes)
    if (x==None) and (y==None):
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
    grd = pp.plot(data, x, y, **kwargs)
    pp.grid(True)
    cbar = pp.colorbar(grd, shrink=0.75, extend=extend)
    cbar.set_label('('+unit+')')
    pp.title(title)
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
            pl.close()


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
    """Plot gridded data on a background map

    This class allows to plot gridded data (e.g. PPIs or composites) on a background.
    The background map (Basemap) can include country borders, coastlines, meridians
    as well as user-defined shapefiles.

    In order to plot the data over user-defined shapefiles, these have to be projected
    to "geographical projection", i.e. in lat/lon coordinates based on WGS84. You can
    use any GIS for this task. The shapefiles are then passed to the constructor
    by providing a list of file paths in the argument *shapefiles* (see `Parameters`).

    Using Grid2Basemap(...), the background map is plotted. The actual data is plotted
    by using the ``plot`` method. This procedure allows to repeatedly plot data on
    a map (e.g. a time series) without each time plotting the background again.

    Parameters
    ----------
    bbox    : the bounding box of the entire map in lat/lon
    ncolors : number of colors in the colormap lookup table - will be overridden by the classes argument
    classes : classes of the plotting variable for which colors should be homogenoeous - overrides ncolors!
    shapefiles : list of paths to shapefiles which will be plotted as map background
    cmap    : name of the default colormap in case no colormap is provided in the config file
    saveto  : if None, the plots are shown on the screen - otherwise the figures are saved to directory *saveto*

    Examples
    --------
    >>> import wradlib.vis as vis


    """
    def __init__(self, bbox, unit='', ncolors=10, classes=None, cmap=cm.s3pcpn):

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

        # draw nice stuff
        ##self.m.bluemarble()
        self.m.fillcontinents(color='grey', zorder=0)#,lake_color='aqua')
        self.m.drawcoastlines()
        self.m.drawparallels(np.linspace(start=np.round(self.bbox['lly']), stop=np.round(self.bbox['ury']), num=3), labels=[1,0,0,0])
        self.m.drawmeridians(np.linspace(start=np.round(self.bbox['llx']), stop=np.round(self.bbox['urx']), num=3), labels=[0,0,0,1])
        # draw map scale
        self.m.drawmapscale(lon=self.bbox['urx']-0.2*(self.bbox['urx']-self.bbox['llx']), lat=self.bbox['lly']+0.1*(self.bbox['ury']-self.bbox['lly']), lon0=lon0, lat0=lat0, length=50., units='km', barstyle='fancy')

        # remember cmap arg
        self.mycmap = cmap

        # define colorbar (we use a dummy mappable object via imshow)
##        self.cbar = pl.colorbar(mappable=pl.imshow(np.repeat(self.classes,2).reshape((2,-1)),
##                    cmap=self.mycmap), orientation='vertical', shrink=0.8, extend='max')
        self.cbar = pl.colorbar(mappable=pl.contourf(np.repeat(self.classes,2).reshape((2,-1)), self.classes, cmap=self.mycmap),
                    orientation='vertical', shrink=0.8, extend='max')
        self.cbar.set_label('('+unit+')')

    def plot(self, lon, lat, data, title='', saveto=None):
        """Plot the data on the map background

        Parameters
        ----------
        lon : array of longitudes
        lat : array of latitudes
        data : data array of shape (number of longitudes, number of latitudes)
        title : figure title
        saveto : string to a directory where figurs should be stored

        """
        # add title plot
        pl.title( title)
        # get map coordinates
        x, y =self.m(lon, lat)
        # plot data
        cs = self.m.contourf(x,y,data,self.classes, cmap=self.mycmap)

        # if no save directory is given, show plot on screen
        if saveto==None:
            pl.show()
        else:
            if title=='':
                fname = "radarfig.png"
            else:
                fname    = title.replace(" ", "").replace("\n", "").replace(":","").strip() + '.png'
            savepath = path.join(saveto, fname)
            pl.savefig(savepath)
        # remove data plot from the axis (otherwise the axis object becomes successively overcrowded)
        for coll in cs.collections:
            pl.gca().collections.remove(coll)



if __name__ == '__main__':
    print 'wradlib: Calling module <vis> as main...'



