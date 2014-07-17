# -*- coding: utf-8 -*-
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

   plot_ppi
   plot_ppi_crosshair
   plot_rhi
   plot_cg_ppi
   plot_cg_rhi
   rhi_plot
   cg_plot
   plot_scan_strategy
   plot_plan_and_vert
   plot_tseries

"""

# standard libraries
import os.path as path
import math

# site packages
import numpy as np
import pylab as pl
import matplotlib
import matplotlib as mpl
#from mpl_toolkits.basemap import Basemap, cm
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform
from mpl_toolkits.axisartist import SubplotHost, ParasiteAxesAuxTrans, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size as Size
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.ticker import NullFormatter, FuncFormatter
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# wradlib modules
import wradlib.georef as georef
import wradlib.util as util

from util import deprecated


class NorthPolarAxes(PolarAxes):
    '''
    A variant of PolarAxes where theta starts pointing north and goes
    clockwise.
    Obsolete since matplotlib version 1.1.0, where the same behaviour may
    be achieved with a reconfigured standard PolarAxes object.
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


@deprecated("plot_ppi")
class PolarPlot(object):
    def __init__(self, ax=None, fig=None, axpos=111, **kwargs):
        if ax is None:
            if fig is None:
                # crate a new figure object
                fig = pl.figure(**kwargs)
            # plot on the axes object which was passed to this function
            try: # version working before matplotlib 1.1.0. may be removed some time
                ax = fig.add_subplot(axpos, projection="northpolar", aspect=1.)
            except AttributeError: # happens in new versions of matplotlib (v.1.1 and newer due to changes to the transforms api)
                # but then again, we have new functionality obsolescing the old
                # northpolar axes object
                ax = fig.add_subplot(axpos, projection="polar", aspect=1.)
                ax.set_theta_direction(-1)
                ax.set_theta_zero_location("N")

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


@deprecated("plot_ppi")
def polar_plot2(data, title='', unit='', saveto='', fig=None, axpos=111, R=1., theta0=0, colormap='jet', classes=None, extend='neither', show=True, **kwargs):
    pp = PolarPlot(fig=fig, axpos=axpos, figsize=(8,8))
    pp.set_cmap(colormap, classes=classes)
    circle = pp.plot(data, R=R, theta0=theta0, **kwargs)
    pp.grid(True)
    cbar = pp.colorbar(circle, shrink=0.75, extend=extend)
    cbar.set_label('('+unit+')')
    pp.title(title)
    if saveto=='':
        # show plot
        if show:
            pl.show()
        if (not pl.isinteractive() ) and show:
            # close figure eplicitely if pylab is not in interactive mode
            pl.close()
    else:
        # save plot to file
        if ( path.exists(path.dirname(saveto)) ) or ( path.dirname(saveto)=='' ):
            pl.savefig(saveto)
            pl.close()


@deprecated("plot_ppi")
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
        try: # version working before matplotlib 1.1.0. may be removed some time
            ax = fig.add_subplot(111, projection="northpolar", aspect=1.)
        except AttributeError: # happens in new versions of matplotlib (v.1.1 and newer due to changes to the transforms api)
            # but then again, we have new functionality obsolescing the old
            # northpolar axes object
            ax = fig.add_subplot(111, projection="polar", aspect=1.)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
    else:
        # plot on the axes object which was passed to this function
        try: # version working before matplotlib 1.1.0. may be removed some time
            ax = fig.add_subplot(axpos, projection="northpolar", aspect=1.)
        except AttributeError: # happens in new versions of matplotlib
            # but then again, we have new functionality obsolescing the old
            # northpolar axes object
            ax = fig.add_subplot(axpos, projection="polar", aspect=1.)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
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


def plot_ppi(data, r=None, az=None, autoext=True,
             site=(0,0), proj=None, elev=0.,
             ax=None,
             **kwargs):
    """Plots a Plan Position Indicator (PPI).

    The implementation of this plot routine is in cartesian axes and does all
    coordinate transforms beforehand. This allows zooming into the data as well
    as making it easier to plot additional data (like gauge locations) without
    having to convert them to the radar's polar coordinate system.

    `**kwargs` may be used to try to influence the matplotlib.pcolormesh and
    wradlib.georef.polar2lonlatalt_n routines under the hood.

    There is one major caveat concerning the values of `r` and `az`.
    Due to the way matplotlib.pcolormesh works, `r` should give the location
    of the start of each range bin, while `az` should give the angle also at
    the begin (i.e. 'leftmost') of the beam. This might be in contrast to
    other conventions, which might define ranges and angles at the center of
    bin and beam.
    This affects especially the default values set for `r` and `az`, but ìt
    should be possible to accommodate all other conventions by setting `r` and
    `az` properly.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily, unless proj is set. In that
        case the units must be meters. If None, a default is
        calculated from the dimensions of `data`.
    az : np.array
        The azimuth angles in degrees. If None, a default is
        calculated from the dimensions of `data`.
    autoext : True | False
        This routine uses matplotlib.pyplot.pcolormesh to draw the bins.
        As this function needs one set of coordinates more than would usually
        be provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    site : tuple
        Tuple of coordinates of the radar site.
        If `proj` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If `proj` is used, values must be given as (longitude, latitude)
        tuple of geographical coordinates.
    proj : str
        PROJ.4 compatible projection string
        If this parameter is not None, `site` must be set. Then the function
        will attempt to georeference the radar bins and display the PPI in the
        coordinate system defined by the projection string.
    elev : float or array of same shape as az
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
    ax : matplotlib Axes object
        If given, the PPI will be plotted into this axes object. If None a
        new axes object will be created

    See also
    --------
    wradlib.georef.reproject - for information on projection strings
    wradlib.georef.create_projstr - routine to generate pre-defined projection
        strings


    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the PPI was plotted
    pm : matplotlib QuadMesh object
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    """
    # kwargs handling
    kw_polar2lonlatalt_n = {}
    if 're' in kwargs:
        kw_polar2lonlatalt_n.append(kwargs.pop('re'))
    if 'ke' in kwargs:
        kw_polar2lonlatalt_n.append(kwargs.pop('ke'))

    # this may seem odd at first, but d1 and d2 are also used in plot_rhi
    # and thus it may be easier to compare the two functions
    d1 = r
    d2 = az

    # providing 'reasonable defaults', based on the data's shape
    if d1 is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    if d2 is None:
        d2 = np.arange(data.shape[0], dtype=np.float)

    if autoext:
        # the ranges need to go 'one bin further', assuming some regularity
        # we extend by the distance between the preceding bins.
        x = np.append(d1, d1[-1]+(d1[-1]-d1[-2]))
        # the angular dimension is supposed to be cyclic, so we just add the
        # first element
        y = np.append(d2, d2[0])
    else:
        # no autoext basically is only useful, if the user supplied the correct
        # dimensions himself.
        x = d1
        y = d2

    # coordinates for all vertices
    xx, yy = np.meshgrid(x, y)

    if proj is None:
        # no georeferencing -> simple trigonometry
        xxx = xx * np.cos(np.radians(90.-yy)) + site[0]
        yy = xx * np.sin(np.radians(90.-yy))  + site[1]
        xx = xxx
    else:
        # with georeferencing
        if r is None:
            # if we produced a default, this one is still in 'kilometers'
            # therefore we need to get from km to m
            xx *= 1000
        # latitude longitudes from the polar data still stored in xx and yy
        lon, lat, alt = georef.polar2lonlatalt_n(xx, yy, elev, site, **kw_polar2lonlatalt_n)
        # projected to the final coordinate system
        osr_proj = georef.proj4_to_osr(proj)
        xx, yy = georef.reproject(lon, lat, projection_target=osr_proj)

    # get the current axes.
    # this creates one, if there is none
    if ax is None:
        ax = pl.gca()

    # plot the colormesh
    pm = ax.pcolormesh(xx, yy, data, **kwargs)

    ax.set_aspect('equal')

    # return the axes and the colormesh object
    # so that the user may add colorbars etc.
    return ax, pm


def plot_ppi_crosshair(site, ranges, angles=[0,90,180,270],
                       proj=None, elev=0., ax=None, kwds={}):
    """Plots a Crosshair for a Plan Position Indicator (PPI).

    Parameters
    ----------
    site : tuple
        Tuple of coordinates of the radar site.
        If `proj` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If `proj` is used, values must be given as (longitude, latitude)
        tuple of geographical coordinates.
    ranges : list
        List of ranges, for which range circles should be drawn.
        If `proj` is None arbitrary units may be used (such that they fit with
        the underlying PPI plot. Otherwise the ranges must be given in meters.
    angles : list
        List of angles (in degrees) for which straight lines should be drawn.
        These lines will be drawn starting from the center and until the largest
        range.
    proj : str
        PROJ.4 compatible projection string
        The function will calculate lines and circles according to
        georeferenced coordinates taking beam propagation, earth's curvature
        and scale effects due to projection into account.
        Depending on the projection, crosshair lines might not be straight and
        range circles might appear elliptical (also check if the aspect of the
        axes might not also be responsible for this).
    elev : float or array of same shape as az
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
    ax : matplotlib Axes object
        If given, the crosshair will be plotted into this axes object. If None
        matplotlib's current axes (function gca()) concept will be used to
        determine the axes.
    kwds : dictionary
        Dictionary of settings to alter the appearance of lines and range
        circles. With the key 'line', you may pass a dictionary, which will be
        passed to the line objects using the standard keyword inheritance
        mechanism.
        With the key 'circle' you may do the same for the range circles.
        If not given defaults will be used.
        See the file plot_ppi_example.py in the examples folder for examples on
        how this works.

    See also
    --------
    wradlib.vis.plot_ppi - plotting a PPI in cartesian coordinates
    wradlib.georef.create_projstr - routine to generate pre-defined projection
        strings


    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the PPI was plotted

    """
    # if we didn't get an axes object, find the current one
    if ax is None:
        ax = pl.gca()

    # set default line keywords
    linekw = dict(color='gray', linestyle='dashed')
    # update with user settings
    linekw.update(kwds.get('line', {}))

    # set default circle keywords
    circkw = dict(edgecolor='gray', linestyle='dashed', facecolor='none')
    # update with user settings
    circkw.update(kwds.get('circle', {}))

    # determine coordinates for 'straight' lines
    if proj:
        # projected
        # reproject the site coordinates
        osr_proj = georef.proj4_to_osr(proj)
        psite = georef.reproject(*site, projection_target=osr_proj)
        # these lines might not be straigt so we approximate them with 10
        # segments. Produce polar coordinates
        rr, az = np.meshgrid(np.linspace(0,ranges[-1],10), angles)
        # and reproject using polar2lonlatalt to convert from polar to geographic
        nsewx, nsewy = georef.reproject(*georef.polar2lonlatalt_n(rr, az, elev,
                                                                site)[:2],
                                      projection_target=osr_proj)
    else:
        # no projection
        psite = site
        rr, az = np.meshgrid(np.linspace(0,ranges[-1],2), angles)
        # use simple trigonometry to calculate coordinates
        nsewx, nsewy = (psite[0]+rr*np.cos(np.radians(az)),
                        psite[1]+rr*np.sin(np.radians(az)))

    # mark the site, just in case nothing else would be drawn
    ax.plot(*psite, marker='+', **linekw)

    # draw the lines
    for i in range(len(angles)):
        ax.add_line(mpl.lines.Line2D(nsewx[i,:], nsewy[i,:], **linekw))

    # draw the range circles
    for r in ranges:
        if proj:
            # produce an approximation of the circle
            x, y = georef.reproject(*georef.polar2lonlatalt_n(r,
                                                            np.arange(360),
                                                            elev,
                                                            site)[:2],
                                  projection_target=osr_proj)
            ax.add_patch(mpl.patches.Polygon(np.concatenate([x[:,None],
                                                             y[:,None]],
                                                             axis=1),
                                             **circkw))
        else:
            # in the unprojected case, we may use 'true' circles.
            ax.add_patch(mpl.patches.Circle(psite, r, **circkw))

    # there should be not much wrong, setting the axes aspect to equal by default
    ax.set_aspect('equal')

    # return the axes object for later use
    return ax


def plot_rhi(data, r=None, th=None, th_res=None, autoext=True, refrac=True,
             ax=None, **kwargs):
    """Plots a Range Height Indicator (RHI).

    The implementation of this plot routine is in cartesian axes and does all
    coordinate transforms beforehand. This allows zooming into the data as well
    as making it easier to plot additional data (like gauge locations) without
    having to convert them to the radar's polar coordinate system.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the elevation angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily. If None, a default is
        calculated from the dimensions of `data`.
    th : np.array
        The elevation angles in degrees. If None, a default is
        calculated from the dimensions of `data`.
    th_res : float or np.array of same shape as `th`
        In RHI's it happens that the elevation angles are spaced wider than
        the beam width. If this beam width (in degrees) is given in `th_res`,
        plot_rhi will plot the beams accordingly. Otherwise the behavior of
        matplotlib.pyplot.pcolormesh assumes all beams to be adjacent to each
        other, which might lead to unexpected results.
    autoext : True | False
        This routine uses matplotlib.pyplot.pcolormesh to draw the bins.
        As this function needs one set of coordinates more than would usually
        provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    refrac : True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account. If False, simple
        trigonometry will be used to calculate beam propagation.
        Functionality for this will be provided by functions
        wradlib.georef.arc_distance_n and wradlib.georef.beam_height_n, which
        assume distances to be given in meters. Therefore, if `refrac` is True,
        `r` must be given in meters.
    ax : matplotlib Axes object
        If given, the RHI will be plotted into this axes object. If None a
        new axes object will be created.

    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the RHI was plotted
    pm : matplotlib QuadMesh object
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    """
    # autogenerate axis dimensions
    if r is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    else:
        d1 = np.asanyarray(r)

    if th is None:
        d2 = np.arange(data.shape[0], dtype=np.float)
    else:
        d2 = np.asanyarray(th)

    if autoext:
        # extend the range by the delta of the two last bins
        x = np.append(d1, d1[-1]+d1[-1]-d1[-2])
        # RHIs usually aren't cyclic, so we best guess a regular extension
        # here as well
        y = np.append(d2, d2[-1]+d2[-1]-d2[-2])
    else:
        # hopefully, the user supplied everything correctly...
        x = d1
        y = d2

    if th_res is not None:
        # we are given a beam resolution and thus may not just glue each
        # beam to its neighbor
        # solving this still with the efficient pcolormesh but interlacing
        # the data with masked values, simulating the gap between beams
        # make a temporary data array with one dimension twice the size of
        # the original
        img = np.ma.empty((data.shape[0], data.shape[1]*2))
        # mask everything
        img.mask = np.ma.masked
        # set the data in the first half of the temporary array
        # this automatically unsets the mask
        img[:,:data.shape[1]] = data
        # reshape so that data and masked lines interlace each other
        img = img.reshape((-1, data.shape[1]))
        # produce lower and upper y coordinates for the actual data
        yl = d2-th_res*0.5
        yu = d2+th_res*0.5
        # glue them together to achieve the proper dimensions for the
        # interlaced array
        y = np.concatenate([yl[None,:], yu[None,:]], axis=0).T.ravel()
    else:
        img=data

    # coordinates for all vertices
    xx, yy = np.meshgrid(x, y)

    if refrac:
        # observing air refractivity, so ground distances and beam height
        # must be calculated specially
        xxx = georef.arc_distance_n(xx, yy)
        yy = georef.beam_height_n(xx, yy)
    else:
        # otherwise plane trigonometry will do
        xxx = xx * np.cos(np.radians(yy))
        yy = xx * np.sin(np.radians(yy))
    xx = xxx

    # get current axes if not given
    if ax is None:
        ax = pl.gca()

    # plot the stuff
    pm = ax.pcolormesh(xx, yy, img)

    # return references to important and eventually new objects
    return ax, pm


def create_cg(st, fig=None, subplot=111):
    """ Helper function to create curvelinear grid

    The function makes use of the Matplotlib AXISARTIST namespace
    http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib AxesGrid Toolkit
    http://matplotlib.org/mpl_toolkits/axes_grid/index.html
    most of the limitations can be overcome.
    See http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html.

    Parameters
    ----------
    st : scan type, 'PPI' or 'RHI'
    fig : matplotlib Figure object
        If given, the PPI will be plotted into this figure object. Axes are
        created as needed. If None a new figure object will be created or
        current figure will be used, depending on "subplot".
    subplot : matplotlib grid definition, gridspec definition
        nrows/ncols/plotnumber, see examples section
        defaults to '111', only one subplot

    Returns
    -------
    cgax : matplotlib toolkit axisartist Axes object
        curvelinear Axes (r-theta-grid)
    caax : matplotlib Axes object (twin to cgax)
        Cartesian Axes (x-y-grid) for plotting cartesian data
    paax : matplotlib Axes object (parasite to cgax)
        The parasite axes object for plotting polar data
    """

    if st == 'RHI':
        # create transformation
        tr = Affine2D().scale(np.pi / 180, 1.) + PolarAxes.PolarTransform()

        # build up curvelinear grid
        extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=100,
                                                     lat_cycle=None,
                                                     lon_minmax=(0, np.inf),
                                                     lat_minmax=(0, np.inf),
                                                     )

        # locator and formatter for angular annotation
        grid_locator1 = angle_helper.LocatorD(10.)
        tick_formatter1 = angle_helper.FormatterDMS()

        # grid_helper for curvelinear grid
        grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        grid_locator2=None,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

        # try to set nice locations for range gridlines
        grid_helper.grid_finder.grid_locator2._nbins = 30.0
        grid_helper.grid_finder.grid_locator2._steps = [0, 1, 1.5,
                                                        2, 2.5, 5, 10]

    if st == 'PPI':
        # create transformation
        tr = (Affine2D().scale(np.pi / 180, 1.) +
              NorthPolarAxes.NorthPolarTransform())

        # build up curvelinear grid
        extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=360.,
                                                     lat_cycle=None,
                                                     lon_minmax=(360., 0.),
                                                     lat_minmax=(0, np.inf),
                                                     )

        # locator and formatter for angle annotation
        grid_locator1 = FixedLocator([i for i in np.arange(0, 359, 10)])
        tick_formatter1 = angle_helper.FormatterDMS()

        # grid_helper for curvelinear grid
        grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        grid_locator2=None,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    # if there is no figure object given
    if fig is None:
        # create new figure if there is only one subplot
        if subplot is 111:
            fig = pl.figure()
        # otherwise get current figure or create new figure
        else:
            fig = pl.gcf()

    # generate Axis
    cgax = SubplotHost(fig, subplot, grid_helper=grid_helper)

    fig.add_axes(cgax)

    # PPIs always plottetd with equal aspect
    if st == 'PPI':
        cgax.set_aspect('equal', adjustable='box-forced')

    # get twin axis for cartesian grid
    caax = cgax.twin()
    # move axis annotation from right to left and top to bottom
    caax.toggle_axisline()

    # make ticklabels of right and top axis visible,
    cgax.axis["right"].major_ticklabels.set_visible(True)
    cgax.axis["top"].major_ticklabels.set_visible(True)
    cgax.axis["right"].get_helper().nth_coord_ticks = 0
    cgax.axis["top"].get_helper().nth_coord_ticks = 0

    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)

    # make ticklabels of left and bottom axis unvisible,
    # because we are drawing them
    cgax.axis["left"].major_ticklabels.set_visible(False)
    cgax.axis["bottom"].major_ticklabels.set_visible(False)

    # and also set tickmarklength to zero for better presentation
    cgax.axis["left"].major_ticks.set_ticksize(0)
    cgax.axis["bottom"].major_ticks.set_ticksize(0)

    # generate and add parasite axes with given transform
    paax = ParasiteAxesAuxTrans(cgax, tr, "equal")
    # note that paax.transData == tr + cgax.transData
    # Anthing you draw in paax will match the ticks and grids of cgax.
    cgax.parasites.append(paax)

    return cgax, caax, paax


def plot_cg_ppi(data, r=None, az=None, rf=1.0, autoext=True,
             refrac=True, elev=0., fig=None, subplot=111,
             **kwargs):
    """Plots a Plan Position Indicator (PPI) on a curvelinear grid.

    The implementation of this plot routine is in curvelinear grid axes and
    does all coordinate transforms beforehand. This allows zooming into the
    data as well as making it easier to plot additional data (like gauge
    locations).

    Additional data can be plottet in polar coordinates or cartesian
    coordinates depending which axes object is used.

    The function uses create_cg wich uses the Matplotlib AXISARTIST namespace
    http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib AxesGrid Toolkit
    http://matplotlib.org/mpl_toolkits/axes_grid/index.html
    most of the limitations can be overcome.
    See http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html.

    `**kwargs` may be used to try to influence the matplotlib.pcolormesh
    routine under the hood.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily. If None, a default is
        calculated from the dimensions of `data`.
    rf: float
        If present, factor for scaling range axis.
        defaults to 1.
    az : np.array
        The azimuth angles in degrees. If None, a default is
        calculated from the dimensions of `data`.
    autoext : True | False
        This routine uses matplotlib.pyplot.pcolormesh to draw the bins.
        As this function needs one set of coordinates more than would usually
        be provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    elev : float or array of same shape as az
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
    fig : matplotlib Figure object
        If given, the PPI will be plotted into this figure object. Axes are
        created as needed. If None a new figure object will be created or
        current figure will be used, depending on "subplot".
   subplot : matplotlib grid definition, gridspec definition
        nrows/ncols/plotnumber, see examples section
        defaults to '111', only one subplot

    See also
    --------
    create_cg : creation of curvelinear grid axes objects

    Returns
    -------
    cgax : matplotlib toolkit axisartist Axes object
        Curvelinear Axes (r-theta-grid)
    caax : matplotlib Axes object (twin to cgax)
        Cartesian Axes (x-y-grid) for plotting cartesian data
    paax : matplotlib Axes object (parasite to cgax)
        The parasite axes object for plotting polar data
        all data in polar format must be plottet to this axis

    pm : matplotlib QuadMesh object
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    """
    # this may seem odd at first, but d1 and d2 are also used in plot_rhi
    # and thus it may be easier to compare the two functions
    d1 = r
    d2 = az

    # providing 'reasonable defaults', based on the data's shape
    if d1 is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    if d2 is None:
        d2 = np.arange(data.shape[0], dtype=np.float)

    if autoext:
        # the ranges need to go 'one bin further', assuming some regularity
        # we extend by the distance between the preceding bins.
        x = np.append(d1, d1[-1] + (d1[-1] - d1[-2]))
        # the angular dimension is supposed to be cyclic, so we just add the
        # first element
        y = np.append(d2, d2[0])
    else:
        # no autoext basically is only useful, if the user supplied the correct
        # dimensions himself.
        x = d1
        y = d2
    print(x.shape, y.shape)

    if refrac:
        # with refraction correction, significant at higher elevations
        # calculate new range values
        x = georef.arc_distance_n(x, elev)

    # create curvelinear axes
    cgax, caax, paax = create_cg('PPI', fig, subplot)

    # this is in fact the outermost thick "ring"
    cgax.axis["lon"] = cgax.new_floating_axis(1, np.max(x) / rf)
    cgax.axis["lon"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    cgax.axis["lon"].major_ticks.set_ticksize(0)

    xx, yy = np.meshgrid(y, x)
    # set bounds to min/max
    xa = yy * np.sin(np.radians(xx))
    ya = yy * np.cos(np.radians(xx))
    cgax.set_ylim(np.min(ya), np.max(ya))
    cgax.set_xlim(np.min(xa), np.max(xa))
    yy = yy / rf
    data = data.transpose()

    # plot the stuff
    pm = paax.pcolormesh(xx, yy, data, **kwargs)

    # show curvelinear and cartesian grids
    # makes no sense not to plot, if we made such a fuss to get that handled
    cgax.grid(True)
    caax.grid(True)

    # return the axes and the colormesh object
    # so that the user may add colorbars etc.
    return cgax, caax, paax, pm


def plot_cg_rhi(data, r=None, th=None, th_res=None, autoext=True, refrac=True,
             rf=1., fig=None, subplot=111, **kwargs):
    """Plots a Range Height Indicator (RHI) on a curvelinear grid.

    The implementation of this plot routine is in a curvelinear grid axes and
    does all coordinate transforms beforehand.

    This allows zooming into the data as well as making it easier to plot
    additional cartesian data (like certain special points) without having to
    convert them to the radar's polar coordinate system.

    Plotting in the radar's polar coordinate system is possible as well.

    The function uses create_cg wich uses the Matplotlib AXISARTIST namespace
    http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib AxesGrid Toolkit
    http://matplotlib.org/mpl_toolkits/axes_grid/index.html
    most of the limitations can be overcome.
    See http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html.

    `**kwargs` may be used to try to influence the matplotlib.pcolormesh
    routine under the hood.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the elevation angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily. If None, a default is
        calculated from the dimensions of `data`.
    rf: float
        if present, factor for scaling range axis
        defaults to 1.
    th : np.array
        The elevation angles in degrees. If None, a default is
        calculated from the dimensions of `data` and span from 0 to 90 degrees
        is assumed
    th_res : float or np.array of same shape as `th`
        In RHI's it happens that the elevation angles are spaced wider than
        the beam width. If this beam width (in degrees) is given in `th_res`,
        plot_rhi will plot the beams accordingly. Otherwise the behavior of
        matplotlib.pyplot.pcolormesh assumes all beams to be adjacent to each
        other, which might lead to unexpected results.
    autoext : True | False
        This routine uses matplotlib.pyplot.pcolormesh to draw the bins.
        As this function needs one set of coordinates more than would usually
        provided by `r` and `th`, setting ´autoext´ to True automatically
        extends r and th so that all of `data` will be plotted.
    refrac : True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account.
        Functionality for this will be provided by functions
        wradlib.georef.arc_distance_n and wradlib.georef.beam_height_n, which
        assume distances to be given in meters. Therefore, if `refrac` is True,
        `r` must be given in meters. Cartesian Axis caax ist used for plotting.
        If False, PolarAxes.PolarTransform will be used to calculate
        beam propagation.
    fig : matplotlib Figure object
        If given, the RHI will be plotted into this figure object. Axes are
        created as needed. If None a new figure object will be created or
        current figure will be used, depending on "subplot".
    subplot : matplotlib grid definition
        nrows/ncols/plotnumber
        defaults to '111', only one subplot

    See also
    --------
    create_cg : creation of curvelinear grid axes objects

    Returns
    -------
    cgax : matplotlib Axes object
        curvelinear Axes (r-theta-grid)
    caax : matplotlib Axes object
        Cartesian Axes (x-y-grid) for plotting cartesian data
    paax : matplotlib Axes object
        The parasite axes object for plotting polar data
        All data in polar format must be plottet to this axis
    pm : matplotlib QuadMesh object
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    """
    # autogenerate axis dimensions
    if r is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    else:
        d1 = np.asanyarray(r)

    if th is None:
        # assume, data is evenly spaced between 0 and 90 degree
        d2 = np.linspace(0., 90., num=data.shape[0], endpoint=True)
        #d2 = np.arange(data.shape[0], dtype=np.float)
    else:
        d2 = np.asanyarray(th)

    if autoext:
        # extend the range by the delta of the two last bins
        x = np.append(d1, d1[-1] + d1[-1] - d1[-2])
        # RHIs usually aren't cyclic, so we best guess a regular extension
        # here as well
        y = np.append(d2, d2[-1] + d2[-1] - d2[-2])
    else:
        # hopefully, the user supplied everything correctly...
        x = d1
        y = d2

    if th_res is not None:
        # we are given a beam resolution and thus may not just glue each
        # beam to its neighbor
        # solving this still with the efficient pcolormesh but interlacing
        # the data with masked values, simulating the gap between beams
        # make a temporary data array with one dimension twice the size of
        # the original
        img = np.ma.empty((data.shape[0], data.shape[1] * 2))
        # mask everything
        img.mask = np.ma.masked
        # set the data in the first half of the temporary array
        # this automatically unsets the mask
        img[:, :data.shape[1]] = data
        # reshape so that data and masked lines interlace each other
        img = img.reshape((-1, data.shape[1]))
        # produce lower and upper y coordinates for the actual data
        yl = d2 - th_res * 0.5
        yu = d2 + th_res * 0.5
        # glue them together to achieve the proper dimensions for the
        # interlaced array
        y = np.concatenate([yl[None, :], yu[None, :]], axis=0).T.ravel()
    else:
        img = data

    # create curvelinear axes
    cgax, caax, paax = create_cg('RHI', fig, subplot)

    # this is in fact the outermost thick "ring" aka max_range
    cgax.axis["lon"] = cgax.new_floating_axis(1, np.max(x) / rf)
    cgax.axis["lon"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    cgax.axis["lon"].major_ticks.set_ticksize(0)

    if refrac:
        # observing air refractivity, so ground distances and beam height
        # must be calculated specially
        # create coordinates for all vertices
        xx, yy = np.meshgrid(x, y)
        xxx = georef.arc_distance_n(xx, yy) / rf
        yyy = georef.beam_height_n(xx, yy) / rf
        # assign twin-axis/cartesian-axis as plotting axis
        plax = caax
    else:
        # otherwise plotting to parasite axis will do
        # create meshgrid for polar data
        # please note that the data is plottet within a polar grid
        # with 0 degree at 3 o'clock, hence the slightly other data handling
        xxx, yyy = np.meshgrid(y, x)
        yyy = yyy / rf
        img = img.transpose()
        # assign parasite axis as plotting axis
        plax = paax

    # plot the stuff
    pm = plax.pcolormesh(xxx, yyy, img, **kwargs)

    # set bounds to maximum
    cgax.set_ylim(0, np.max(x) / rf)
    cgax.set_xlim(0, np.max(x) / rf)

    # show curvelinear and cartesian grids
    # makes no sense not to plot, if we made such a fuss to get that handled
    cgax.grid(True)
    caax.grid(True)

    # return references to important and eventually new objects
    return cgax, caax, paax, pm


@deprecated()
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


@deprecated()
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

@deprecated()
def get_tick_vector(vrange,vres):
    """Calculates Vector for tickmarks for function create_curvilinear_axes.

    Calculates tickmarks according to value range and wanted resolution. If no resolution is given,
    standard values [100., 50., 25., 20., 10., 5., 2.5, 2., 1., 0.5, 0.25, 0.2] are used.
    The number of tickmarks is normally between 5 and 10.

    Parameters
    ----------
    vrange : value range (first and last tickmark)
    vres : array of tick resolution (empty list, single value, multiple values)

    Returns
    ----------
    output : array of tickmarks

    """

    x = vrange[1]-vrange[0]

    if not vres:
        for div in [100.,50.,20.,10.,5.,2.5,2.,1.,0.5,0.25,0.2]:
            cnt = x/div
            if cnt >= 5:
                rem = np.mod(x,div)
                break
    else:
        if vres.size > 1:
            for div in vres:
                cnt = x/div
                if cnt >= 5:
                    rem = np.mod(x,div)
                    break
        elif vres.size == 1:
            cnt = x/vres
            rem = np.mod(x,vres)

    return np.linspace(vrange[0],vrange[1]-rem,num=cnt+1)

@deprecated("create_cg")
def create_curvilinear_axes(fig, **kwargs):
    """Creates Axis with Parasite Axis for curvilinear grid.

    Parameters
    ----------
    fig : figure object where to create axes
    **kwargs : some axis properties
       ['R','H','r_res','h_res', 'a_res', 'xtitle',  xunit, yunit, 'ytitle', 'atitle', 'title']

    """

    # get and process arguments
    x_range = kwargs.get('R')
    xd = x_range[1]-x_range[0]
    y_range = kwargs.get('H')
    yd = y_range[1]-y_range[0]
    axpos = kwargs.get('axpos')

    x_res = np.array(kwargs.get('r_res', 0.))
    y_res = np.array(kwargs.get('h_res', 0.))
    print(x_res, y_res)
    a_res = kwargs.get('a_res', 10.)
    xunit = kwargs.get('xunit', '')
    yunit = kwargs.get('yunit', '')
    xtitle = kwargs.get('xtitle','Range') + ' ('+xunit+')'
    ytitle = kwargs.get('ytitle','Height') + ' ('+yunit+')'
    atitle = kwargs.get('atitle','$Angle$')# ($^{\circ}$)')
    title = kwargs.get('title','Range Height Indicator')

    # get tickmark vectors for x and y
    rad = get_tick_vector(x_range[0:2], x_res)
    hgt = get_tick_vector(y_range, y_res)

    # construct transform
    tr = Affine2D().scale(np.pi/180, 1.) + PolarAxes.PolarTransform()

    # build up curvilinear grid
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 100,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )
    grid_locator1 = angle_helper.LocatorD(a_res)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_locator2 = FixedLocator([i for i in rad])
    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    # generate Axis
    ax1 = SubplotHost(fig, axpos , grid_helper=grid_helper)

    # make ticklabels of right and top axis visible.
    ax1.axis["right"].major_ticklabels.set_visible(True)
    ax1.axis["top"].major_ticklabels.set_visible(True)
    # but set tickmarklength to zero for better presentation
    ax1.axis["right"].major_ticks.set_ticksize(0)
    ax1.axis["top"].major_ticks.set_ticksize(0)

    # make ticklabels of right and top axis unvisible,
    # because we are drawing them
    ax1.axis["left"].major_ticklabels.set_visible(False)
    ax1.axis["bottom"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    ax1.axis["left"].major_ticks.set_ticksize(0)
    ax1.axis["bottom"].major_ticks.set_ticksize(0)

    # let right and top axis shows ticklabels for 1st coordinate (angle)
    ax1.axis["right"].get_helper().nth_coord_ticks=0
    ax1.axis["top"].get_helper().nth_coord_ticks=0

    # draw grid, tickmarks and ticklabes for left (y) and bottom (x) axis
    for xmaj in rad:
        ax1.axvline(x=xmaj,color='k', ls=':')
        if np.equal(np.mod(xmaj, 1), 0):
            xmaj = np.int(xmaj)
        ax1.text(xmaj,-yd/50.+y_range[0],str(xmaj), va='top', ha='center')
        line = mpl.lines.Line2D([xmaj,xmaj],[-yd/80.+y_range[0], y_range[0]], color='k')
        line.set_clip_on(False)
        ax1.add_line(line)

    for ymaj in hgt:
        ax1.axhline(y=ymaj,color='k', ls=':')
        if np.equal(np.mod(ymaj, 1), 0):
            ymaj = np.int(ymaj)
        ax1.text(-xd/80.+x_range[0],ymaj,str(ymaj).rjust(4), va='center', ha='right')
        line = mpl.lines.Line2D([-xd/130.+x_range[0],x_range[0]],[ymaj, ymaj], color='k')
        line.set_clip_on(False)
        ax1.add_line(line)

    # add axis to figure
    fig.add_subplot(ax1)

    # plot xy-axis labels and title
    ax1.text(-xd/15.+x_range[0],yd/2.0+y_range[0],ytitle, va='center', ha='right', rotation='vertical')
    ax1.text(xd/2.0+x_range[0],-yd/15.+y_range[0],xtitle, va='top', ha='center')
    # there is no "convenient" position for the "angle" label, maybe we dont need it at all
    # ax1.text(x_range[1],y_range[1] + yd/21.,atitle, va='top', ha='right')

    # plot axis title
    ax1.set_title(title)
    (tx,ty) = ax1.title.get_position()
    ax1.title.set_y(1.05 * ty)

    # generate and add parasite axes with given transform
    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    # note that ax2.transData == tr + ax1.transData
    # Anthing you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)

    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.grid(True)

    return ax1, ax2

@deprecated("plot_cg_rhi")
def rhi_plot(data, **kwargs):
    """Returns figure and pylab object of plotted data from a polar grid as an RHI (Range Height Indicator).

    Plotting need to be done outside wradlib

    The data must be an array of shape (number of azimuth angles, number of range bins).
    The azimuth angle of 0 degrees corresponds to y-axis = 0 (east direction)
    The azimuth angle of 90 degrees corresponds to y-axis = 0 (north direction)
    The azimuth the angles are counted counter-clock-wise forward.

    Additional `myargs` are extracted from `kwargs`, processed and/or passed
    to the create_curvilinear_axes routine

    Additional remaining `kwargs` will be passed to the pcolormesh routine displaying
    the data. Be careful!

    Parameters
    ----------
    data : 2-d array
        polar grid data to be plotted
        1st dimension must be azimuth angles, 2nd must be ranges!

    Keyword arguments:

    R : tuple of array of float and unit string
        [display min range, display max range, data max range}, unit string
        defaults to [0, data.shape range, data.shape range], empty string
    H : array of array float and unit string
        [display min height, display max height], unit string
        defaults to [0,data.shape range ], empty string
    theta_range: float array
        theta range (min, max) used to display data
    rad_range: float array
        radial range (min, max) used to display data
    r_res : float array of range (x) tick resolution (empty, single value, multiple values)
    h_res : float array of height (y) tick resolution (empty, single value, multiple values)
    a_res : float
         sets # of angle gridlines and labels, defaults to 8, wich means 10 deg resolution

    title : string
        a title of the plot, defaults to 'Range Height Indicator'
    xtitle : string
        x-axis label
        defaults to 'Range' or 'Range (km)' if R is given (mostly km)
    ytitle : string
        y-axis label
        defaults to 'Height' or 'Height (km)' if H is given (mostly km)
    atitle : string
        angle-axis label, not used at the moment, due to inconvenient placing
        defaults to '$Angle$')# ($^{\circ}$)'
    saveto : string - path of the file in which the figure should be saved
        if string is empty, no figure will be saved and the plot will be
        sent to screen
    fig : matplotlib axis object
        if None, a new matplotlib figure will be created, otherwise we plot
        on given figure
    figsize : width , hight tuple in inches
         defaults to (10,6)
    axpos : an integer or a string
        correponds to the positional argument of mpl_toolkits.axisartist.SubplotHost
        defaults to '111'
        TODO: if multiple plots are used, position and size of labels have to be corrected
        in source code
    colormap :  string
        choose the colormap ("Paired" per default)
    classes :   sequence of numerical values
        class boundaries for plotting
    unit : string
        the unit of the data which is plotted
    extend :    string
        determines the behaviour of the colorbar: default value 'neither' produces
        a standard colorbar, 'min' and 'max' produces an arrow at the minimum or
        maximum end, respectively, and 'both' produces an arrow at both ends. If
        you use class boundaries for plotting, you should typically use 'both'.

    Returns
    ----------
    fig : figure object, just for testing and in the case of multiplot
    pl : pylab object, just for testing and in the case of multiplot

    """

    n_theta, n_r = data.shape

    # process kwargs
    if kwargs:
        key = kwargs.keys()
        value = kwargs.values()
        myargs = dict(zip(key, value))
    else:
        myargs = {}

    myargs['R'] = R = myargs.pop('R',([0, n_r, n_r], ''))
    myargs['R'] = R[0]
    myargs['xunit'] = R[1]
    R = R[0]

    H = myargs.pop('H',([0, n_r], ''))
    myargs['H'] = H[0]
    myargs['yunit'] = H[1]
    H = H[0]

    axpos = myargs.pop('axpos', '111')
    myargs['axpos'] = axpos

    extend = myargs.pop('extend', 'neither')
    classes = myargs.pop('classes', None)
    figsize = myargs.pop('figsize', (10,6))
    fig = myargs.pop('fig', None)
    dunit = myargs.pop('dunit','')
    saveto = myargs.pop('saveto','')
    colormap = myargs.pop('colormap','Paired')

    theta_range = myargs.pop('theta_range', [0,90])
    rad_range = myargs.pop('rad_range', [0,R[2]])

    # remove existing myargs from kwargs
    # remaining kwargs are for pccolormesh routine
    key = ['R','H','r_res','h_res', 'a_res', 'xtitle', \
           'ytitle', 'atitle', 'title', 'figsize', \
           'theta_range','rad_range', 'fig', 'dunit', \
           'saveto','colormap', 'axpos', 'xunit', 'yunit']
    if kwargs:
        value = myargs.values()
        for k in key:
            if k in kwargs:
                kwargs.pop(k)

    # setup vectors
    theta = np.linspace( 0, np.pi/2 , n_theta) # for RHI
    r = np.linspace(0., R[2], n_r)
    theta = theta * 180. / np.pi
    data = np.transpose(data)

    #calculate indices for data range to be plotted
    ind_start = np.where(theta >= theta_range[0])
    ind_stop = np.where(theta <= theta_range[1])
    ind_start1 = ind_start[0][0]
    ind_stop1 = ind_stop[0][-1]
    ind_start = np.where(r >= rad_range[0])
    ind_stop = np.where(r <= rad_range[1])
    ind_start2 = ind_start[0][0]
    ind_stop2 = ind_stop[0][-1]

    # apply data ranges to arrays
    theta = theta[ind_start1:ind_stop1]
    r = r[ind_start2:ind_stop2]
    data = data[ind_start2:ind_stop2,ind_start1:ind_stop1]

    # create figure, grids etc
    if fig==None:
        # create a new figure object
        fig = pl.figure(figsize=figsize)
        ax, ax2 = create_curvilinear_axes(fig, **myargs)
    else:
        # plot on the axes object which was passed to this function
        ax, ax2 = create_curvilinear_axes(fig, **myargs)

    # create rectangular meshgrid of polar data
    X,Y = np.meshgrid(theta,r)

    # plot data to parasite axis
    if classes==None:
        # automatic color normalization by vmin and vmax (not recommended)
        circle = ax2.pcolormesh(X, Y, data, rasterized=True, cmap=colormap, **kwargs)#, vmin=-32., vmax=95.5)
    else:
        # colors are assigned according to class boundaries and colormap argument
        mycmap = pl.get_cmap(colormap, lut=len(classes))
        mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(classes)-1) ))
        norm   = mpl.colors.BoundaryNorm(classes, mycmap.N)
        circle = ax2.pcolormesh(X, Y, data, rasterized=True, cmap=mycmap, norm=norm, **kwargs)

    # plot colorbar
    cbar = fig.colorbar(circle, extend=extend)
    cbar.set_label('('+dunit+')')

    if saveto!='':
        # save plot to file
        if ( path.exists(path.dirname(saveto)) ) or ( path.dirname(saveto)=='' ):
            pl.savefig(saveto)
            pl.close()

    return fig, pl

@deprecated("plot_cg_ppi, plot_cg_rhi")
class cg_plot(object):
    """Class for plotting curvilinear axes.

    PPI (Plan Position Indicator) and RHI (Range Height Indicator) are supported.

    For RHI:
        The data must be an array of shape (number of azimuth angles, number of range bins).
        The azimuth angle of 0 degrees corresponds to y-axis = 0 (east direction)
        The azimuth angle of 90 degrees corresponds to x-axis = 0 (north direction)
        The azimuth angles are counted counter-clock-wise forward.

    For PPI:
        The data must be an array of shape (number of azimuth angles, number of range bins).
        The azimuth angle of 0 degrees corresponds to x-axis = 0 (north direction)
        The azimuth angle of 90 degrees corresponds to y-axis = 0 (east direction)
        The azimuth angles are counted clock-wise forward.

    Additional `myargs` are extracted from `kwargs`, processed and/or passed
    to the create_curvilinear_axes routine

    Additional remaining `kwargs` will be passed to the pcolormesh routine displaying
    the data. Be careful!

    Parameters
    ----------
    ind : string
            RHI or PPI indicating wanted product

    ax : actual axes

    fig : figure to plot on

    x_range :   tuple of an array of floats and a unit string
                    [display min range, display max range, data max range}, unit string
                    defaults to [0, data.shape range, data.shape range], empty string

    y_range :   tuple of an array floats and a unit string
                    [display min height, display max height], unit string
                    defaults to [0,data.shape range ], empty string

    theta_range: float array
                    theta range (min, max) used to display data

    radial_range: float array
                    radial range (min, max) used to display data

    data_range: float array
                    radial range (min, max) of the raw data array

    x_res : float array of range (x) tick resolution (empty, single value, multiple values)

    y_res : float array of height (y) tick resolution (empty, single value, multiple values)

    z_res : float array of colorbar (z) tick resolution (empty, single value, multiple values)

    a_res : float array of angle gridlines and labels, defaults to 8, wich means 10 deg resolution

    faxis : float
        if polar grid, angle where the first floating axis points to

    ftitle : string
        a title of the plot, defaults to None

    xtitle : string
        x-axis label
        defaults to None

    ytitle : string
        y-axis label
        defaults to None

    atitle : string
        angle-axis label, not used at the moment, due to inconvenient placing
        defaults to 'Angle$^{\circ}$'

    saveto : string - path of the file in which the figure should be saved
        if string is empty, no figure will be saved and the plot will be
        sent to screen

    fig : matplotlib axis object
        if None, a new matplotlib figure will be created, otherwise we plot
        on given figure

    figsize : width , height tuple in inches
        defaults to (10,6)

    axpos : an integer or a string
        correponds to the positional argument of mpl_toolkits.axisartist.SubplotHost
        defaults to '111'
        TODO: if multiple plots are used, position and size of labels have to be corrected
        in source code

    colormap :  string
        choose the colormap ("Paired" per default)

    classes :   sequence of numerical values
        class boundaries for plotting

    [x,y,z]unit : string
        the unit of the data which is plotted

    extend :    string
        determines the behaviour of the colorbar: default value 'neither' produces
        a standard colorbar, 'min' and 'max' produces an arrow at the minimum or
        maximum end, respectively, and 'both' produces an arrow at both ends. If
        you use class boundaries for plotting, you should typically use 'both'.

    Returns
    ----------
    class object

    """
    def __init__(self, ind='PPI', ax=None, fig=None, **kwargs):
        self.ind = ind
        self.ax = ax
        self.fig = fig
        self.mdpi = 80.0

        # process myargs
        self.x_range = kwargs.get('x_range',None)
        self.y_range = kwargs.get('y_range',None)
        self.theta_range = kwargs.get('theta_range', None)
        self.radial_range = kwargs.get('radial_range',None)
        self.data_range = kwargs.get('data_range', None)

        self.float_axis = kwargs.get('faxis',45)

        self.xunit = kwargs.get('xunit',None)
        self.yunit = kwargs.get('yunit',None)
        self.zunit = kwargs.get('zunit',None)
        self.xtitle = kwargs.get('xtitle',None)
        self.ytitle = kwargs.get('ytitle',None)
        self.ztitle = kwargs.get('ztitle',None)
        self.ftitle = kwargs.pop('ftitle', None)

        self.fsize = "5%"

        self.axpos = kwargs.get('axpos', '111')
        self.extend = kwargs.get('extend', None)
        self.classes = kwargs.get('classes', None)

        self.saveto = kwargs.get('saveto',None)
        self.colormap = kwargs.get('colormap','jet')


        if ind == 'PPI':
            self.ndeg = 360.
            self.theta_range = [0, 360]
            self.figsize = (8,8)
            self.aspect = 1.
            self.cbp = "5%"
            self.cbw = "5%"
        if ind == 'RHI':
            self.ndeg = 90.
            self.theta_range = [0, 90]
            self.figsize = (10,6)
            self.aspect = 0.
            self.cbp = "5%"
            self.cbw = "3%"

        self.x_res = np.array(kwargs.get('x_res', None))
        self.y_res = np.array(kwargs.get('y_res', None))
        self.z_res = np.array(kwargs.get('z_res', None))
        self.a_res = np.array(kwargs.get('a_res', None))


    def get_tick_vector(self, vrange, vres):
        """Calculates Vector for tickmarks.

        Calculates tickmarks according to value range and wanted resolution. If no resolution is given,
        standard values [100., 50., 25., 20., 10., 5., 2.5, 2., 1., 0.5, 0.25, 0.2] are used.
        The number of tickmarks is normally between 5 and 10.

        Parameters
        ----------
        vrange : value range (first and last tickmark)
        vres : array of tick resolution (empty list, single value, multiple values)

        Returns
        ----------
        output : array of tickmarks

        """

        x = vrange[1]- vrange[0]

        if not vres:
            for div in [200.,100.,50.,20.,10.,5.,2.5,2.,1.,0.5,0.25,0.2]:
                cnt = x/div
                if cnt >= 5:
                    rem = np.mod(x,div)
                    break
        else:
            if vres.size > 1:
                for div in vres:
                    cnt = x/div
                    if cnt >= 5:
                        rem = np.mod(x,div)
                        break
            elif vres.size == 1:
                cnt = x/vres
                rem = np.mod(x,vres)

        return np.linspace(vrange[0],vrange[1]-rem,num=cnt+1)

    def create_curvilinear_axes(self):
        """Creates Curvilinear Axes.

        All needed parameters are calculated in the init() and plot() routines. Normally called from plot().

        RHI - uses PolarAxes.PolarTransform
        PPI - uses NorthPolarAxes.NorthPolarTransform

        Parameters
        ----------
        None


        Returns
        ----------
        ax1 : axes object,
        ax2 : axes object, axes object, where polar data is plotted

        """

        if self.ind == 'RHI':
            tr = Affine2D().scale(np.pi/180, 1.) + PolarAxes.PolarTransform()
            # build up curvilinear grid
            extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                         lon_cycle = 100,
                                                         lat_cycle = None,
                                                         lon_minmax = None,
                                                         lat_minmax = (0, np.inf),
                                                         )
            #grid_locator1 = angle_helper.LocatorD(self.a_res)
            if isinstance(self.a_res, int):
                grid_locator1 = FixedLocator([i for i in np.arange(0,91,self.a_res)])
            else:
                grid_locator1 = FixedLocator(self.a_res)
            tick_formatter1 = angle_helper.FormatterDMS()
            grid_locator2 = FixedLocator([i for i in self.rad])
            grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=None,
                                            )

            # generate Axis
            ax1 = SubplotHost(self.fig, self.axpos , grid_helper=grid_helper)
            # add axis to figure
            self.fig.add_subplot(ax1, aspect=self.aspect)
            ax1.set_aspect(self.aspect, adjustable='box-forced')

            # make ticklabels of right and top axis visible.
            ax1.axis["right"].major_ticklabels.set_visible(True)
            ax1.axis["top"].major_ticklabels.set_visible(True)
            # but set tickmarklength to zero for better presentation
            ax1.axis["right"].major_ticks.set_ticksize(0)
            ax1.axis["top"].major_ticks.set_ticksize(0)
            # let right and top axis shows ticklabels for 1st coordinate (angle)
            ax1.axis["right"].get_helper().nth_coord_ticks=0
            ax1.axis["top"].get_helper().nth_coord_ticks=0

        elif self.ind == 'PPI':

            tr = Affine2D().scale(np.pi/180, 1.) + NorthPolarAxes.NorthPolarTransform()
            # build up curvilinear grid
            extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                         lon_cycle = 360.,
                                                         lat_cycle = None,
                                                         lon_minmax = (360.,0.),
                                                         lat_minmax = (0,self.radial_range[1]),
                                                         )
            if isinstance(self.a_res, int):
                grid_locator1 = FixedLocator([i for i in np.arange(0,359,self.a_res)])
            else:
                grid_locator1 = FixedLocator(self.a_res)
            tick_formatter1 = angle_helper.FormatterDMS()
            grid_locator2 = FixedLocator([i for i in self.rad])
            grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=None,
                                            )

            # generate Axis
            ax1 = SubplotHost(self.fig, self.axpos , grid_helper=grid_helper)
            # add axis to figure
            self.fig.add_subplot(ax1, aspect=self.aspect)
            ax1.set_aspect(self.aspect, adjustable='box-forced')
            #create floating axis,
            if self.float_axis:
                ax1.axis["lon"] = axis = ax1.new_floating_axis(0, self.float_axis)
                ax1.axis["lon"].set_visible(False)
                ax1.axis["lon"].major_ticklabels.set_visible(False)
                # and also set tickmarklength to zero for better presentation
                ax1.axis["lon"].major_ticks.set_ticksize(0)

##            # this is only for special plots with an "annulus"
##            ax1.axis["lon1"] = axis2 = ax1.new_floating_axis(1, self.data_range[0])
##            ax1.axis["lon1"].major_ticklabels.set_visible(False)
##            # and also set tickmarklength to zero for better presentation
##            ax1.axis["lon1"].major_ticks.set_ticksize(0)

            # this is in fact the outermost thick "ring"
            ax1.axis["lon2"] = axis = ax1.new_floating_axis(1, self.radial_range[1])
            ax1.axis["lon2"].major_ticklabels.set_visible(False)
            # and also set tickmarklength to zero for better presentation
            ax1.axis["lon2"].major_ticks.set_ticksize(0)

            # make ticklabels of right and bottom axis unvisible,
            # because we are drawing them
            ax1.axis["right"].major_ticklabels.set_visible(False)
            ax1.axis["top"].major_ticklabels.set_visible(False)

            # and also set tickmarklength to zero for better presentation
            ax1.axis["right"].major_ticks.set_ticksize(0)
            ax1.axis["top"].major_ticks.set_ticksize(0)

        # make ticklabels of left and bottom axis unvisible,
        # because we are drawing them
        ax1.axis["left"].major_ticklabels.set_visible(False)
        ax1.axis["bottom"].major_ticklabels.set_visible(False)

        # and also set tickmarklength to zero for better presentation
        ax1.axis["left"].major_ticks.set_ticksize(0)
        ax1.axis["bottom"].major_ticks.set_ticksize(0)

        # generate and add parasite axes with given transform
        ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
        # note that ax2.transData == tr + ax1.transData
        # Anthing you draw in ax2 will match the ticks and grids of ax1.
        ax1.parasites.append(ax2)

        if self.ind == 'RHI':
            ax1.grid(True)

        return ax1, ax2

    def plot(self, data, **kwargs):
        """ plot data

        Parameters
        ----------
        data : 2-d array
            polar grid data to be plotted
            1st dimension must be azimuth angles, 2nd must be ranges!


        Returns
        ----------
        circle : plot object

        """
        n_theta, n_r = data.shape

        if self.ind == 'PPI':
            self.x_range = kwargs.pop('x_range',[-n_r, n_r])

        if self.ind == 'RHI':
            self.x_range = kwargs.pop('x_range',[0, n_r])


        self.y_range = kwargs.pop('y_range', self.x_range)

        self.xunit = kwargs.pop('xunit', self.xunit)
        self.yunit = kwargs.pop('yunit', self.yunit)
        self.x_res = np.array(kwargs.pop('x_res', self.x_res))
        self.y_res = np.array(kwargs.pop('y_res', self.y_res))
        self.a_res = kwargs.pop('a_res', self.a_res)
        self.float_axis = kwargs.pop('faxis', self.float_axis)

        self.xtitle = kwargs.pop('xtitle', self.xtitle)
        self.ytitle = kwargs.pop('ytitle', self.ytitle)
        self.ftitle = kwargs.pop('ftitle', self.ftitle)
        self.data_range = kwargs.pop('data_range',[0,self.x_range[1]])
        self.radial_range = kwargs.pop('radial_range',[0,self.x_range[1]])
        self.theta_range = kwargs.pop('theta_range',self.theta_range)


        self.aspect = kwargs.pop('aspect', self.aspect)

        #print('Data-Shape:',data.shape)

        # remove existing myargs from kwargs
        # remaining kwargs are for pccolormesh routine
        key = ['x_range','y_range','x_res','y_res', 'a_res', 'z_res', 'xtitle', \
            'ytitle', 'atitle', 'title', 'ztitle', 'figsize', \
            'theta_range','data_range', 'fig', 'zunit', \
            'saveto','colormap', 'axpos', 'xunit', 'yunit', 'extend']

        if kwargs:
            for k in key:
                if k in kwargs:
                    kwargs.pop(k)

        # setup theta and range vectors
        theta = np.linspace( 0, np.pi/180 * self.ndeg , n_theta)
        r = np.linspace(0., self.data_range[1], n_r)
        theta = theta * 180. / np.pi
        data = np.transpose(data)

        #calculate indices for data range to be plotted
        ind_start = np.where(theta >= self.theta_range[0])
        ind_stop = np.where(theta <= self.theta_range[1])
        ind_start1 = ind_start[0][0]
        ind_stop1 = ind_stop[0][-1]
        ind_start = np.where(r >= self.radial_range[0])
        ind_stop = np.where(r <= self.radial_range[1])
        ind_start2 = ind_start[0][0]
        ind_stop2 = ind_stop[0][-1]

        # apply data ranges to arrays
        theta = theta[ind_start1:ind_stop1+1] # +1 is to close the gap to 360deg
        r = r[ind_start2:ind_stop2]
        data = data[ind_start2:ind_stop2,ind_start1:ind_stop1]

        # gets vmin, vmax from raw data
        self.vmin = np.min(data)
        self.vmax = np.max(data)

        # gets dynamic of xrange and yrange
        self.xd = self.x_range[1]-self.x_range[0]
        self.yd = self.y_range[1]-self.y_range[0]
        xd = self.xd
        yd = self.yd
        x_range = self.x_range
        y_range = self.y_range

        # get range and hight (x and y) tick vectors
        self.rad = self.get_tick_vector(self.x_range, self.x_res)
        self.hgt = self.get_tick_vector(self.y_range, self.y_res)

        if self.ax is None:
            # create figure, and setup curvilienar grid etc
            if self.fig is None:
                # create a new figure object
                self.fig = pl.figure(figsize=(8,8),dpi=150)
                self.ax, self.ax2 = self.create_curvilinear_axes()
            else:
                # plot on the figure object which was passed to this function
                self.ax, self.ax2 = self.create_curvilinear_axes()

        #get dpi of fig, needed for automatic calculation of fontsize
        self.dpi = self.fig.get_dpi()
        #print("DPI:", self.dpi)


        # set x and y ax-limits
        self.ax.set_xlim(self.x_range[0], self.x_range[1])
        self.ax.set_ylim(self.y_range[0], self.y_range[1])

        # draw grid, tickmarks and ticklabes for left (y) and bottom (x) axis
        # left that out, user should use grid routines to draw ticks
        #self.xticks(self.x_res)
        #self.yticks(self.y_res)

        # plot xy-axis labels and title if already "published"
        if self.ytitle:
            ytitle = self.ytitle
            if self.yunit:
                ytitle = ytitle + ' ('+ self.yunit + ')'
            self.y_title(ytitle)
        if self.xtitle:
            xtitle = self.xtitle
            if self.xunit:
                xtitle = xtitle + ' ('+ self.xunit + ')'
            self.x_title(xtitle)
            # there is no "convenient" position for the "angle" label, maybe we dont need it at all
            # self.ax1.text(x_range[1],y_range[1] + yd/21.,self.atitle, va='top', ha='right')
        if self.ftitle:
            self.title(self.ftitle, ha="left", x = 0)

        # create rectangular meshgrid for polar data
        X,Y = np.meshgrid(theta,r)

        # plot data to parasite axis
        if self.classes==None:
            # automatic color normalization by vmin and vmax (not recommended) shading='flat', edgecolors='None'
            self.circle = self.ax2.pcolormesh(X, Y, data, rasterized=True, cmap=self.colormap, antialiased=False, **kwargs)
        else:
            # colors are assigned according to class boundaries and colormap argument
            mycmap = pl.get_cmap(self.colormap, lut=len(self.classes))
            mycmap = mpl.colors.ListedColormap(mycmap( np.arange(len(self.classes)-1) ))
            norm   = mpl.colors.BoundaryNorm(self.classes, mycmap.N)
            self.circle = self.ax2.pcolormesh(X, Y, data, rasterized=True, cmap=mycmap, norm=norm, **kwargs)

        return self.circle

    def get_fontsize(self, s=None, *args, **kwargs):
        """ gets fontsize according to given percentage and to actual axis size
            takes dpi of figure into account

        Parameters
        ----------
        s : string
            wanted "fontsize" in percentage of axis size

        Returns
        ----------
        fontsize in points

        """
        if s:
            if not isinstance(s, Size._Base):
                fsize = Size.from_any(s,
                                    fraction_ref=Size.AxesX(self.ax))
        else:
            s="5%"
            if not isinstance(s, Size._Base):
                fsize = Size.from_any(s,
                                    fraction_ref=Size.AxesX(self.ax))

        fs = self.ax.transData.transform((fsize.get_size(self.ax)[0],0))- self.ax.transData.transform((0,0))
        return  fs/(self.dpi/self.mdpi)

    def xticks(self, s, *args, **kwargs):
        """ turns xticks on/off

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """

        fsize = kwargs.pop('fsize','1.5%')
        ticklen = kwargs.pop('ticklen','1%')
        labelpad = kwargs.pop('labelpad','2%')

        if s == False:
            if hasattr(self, 'p_xticks'):
                if self.p_xticks:
                    for item in self.p_xticks:
                        item.remove()
                self.p_xticks = None
        else:
            if hasattr(self, 'p_xticks'):
                if self.p_xticks:
                    for item in self.p_xticks:
                        item.remove()
            self.p_xticks = []
            self.rad = self.get_tick_vector(self.x_range, np.array(s))

            fsize = self.get_fontsize(fsize)[0]
            ticklen = self.get_ypadding(ticklen, self.ax)
            labelpad = self.get_ypadding(labelpad, self.ax)

            for xmaj in self.rad:
                if np.equal(np.mod(xmaj, 1), 0):
                    xmaj = np.int(xmaj)
                text = self.ax.text(xmaj,-labelpad+self.y_range[0],str(xmaj), va='top', ha='center', fontsize=fsize)
                self.p_xticks.append(text)
                line = mpl.lines.Line2D([xmaj,xmaj],[-ticklen+self.y_range[0], self.y_range[0]], color='k')
                line.set_clip_on(False)
                self.ax.add_line(line)
                self.p_xticks.append(line)
            self.xgrid('update')

    def yticks(self, s, *args, **kwargs):
        """ turns yticks on/off

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """

        fsize = kwargs.pop('fsize','1.5%')
        ticklen = kwargs.pop('ticklen','1%')
        labelpad = kwargs.pop('labelpad','2%')

        if s == False:
            if hasattr(self, 'p_yticks'):
                if self.p_yticks:
                    for item in self.p_yticks:
                        item.remove()
                self.p_yticks = None
        else:
            if hasattr(self, 'p_yticks'):
                if self.p_yticks:
                    for item in self.p_yticks:
                        item.remove()
            self.p_yticks = []
            self.hgt = self.get_tick_vector(self.y_range, np.array(s))

            fsize = self.get_fontsize(fsize)[0]
            ticklen = self.get_xpadding(ticklen, self.ax)
            labelpad = self.get_xpadding(labelpad, self.ax)

            for ymaj in self.hgt:
                if np.equal(np.mod(ymaj, 1), 0):
                    ymaj = np.int(ymaj)
                text = self.ax.text(-labelpad+self.x_range[0],ymaj,str(ymaj).rjust(4), va='center', ha='right', fontsize=fsize)
                self.p_yticks.append(text)
                line = mpl.lines.Line2D([-ticklen+self.x_range[0],self.x_range[0]],[ymaj, ymaj], color='k')
                line.set_clip_on(False)
                self.ax.add_line(line)
                self.p_yticks.append(line)
            self.ygrid('update')

    def cartticks(self, s, *args, **kwargs):
        """ turns cartesian ticks on/off (xticks, yticks)

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """
        self.yticks(s)
        self.xticks(s)

    def polticks(self, s, *args, **kwargs):
        """ turns polar ticks on/off (lon, lon2)

        Parameters
        ----------
        s : boolean, string
            True or False, 'on' or 'off'

        Returns
        ----------
        None

        """

        fsize = kwargs.pop('fsize',"2.0%")
        fsize = self.get_fontsize(fsize)[0]
        font = fm.FontProperties()
        font.set_size(fsize)

        if s == 'on' or s == True:
            if self.float_axis:
                self.ax.axis["lon"].set_visible(True)
                self.ax.axis["lon"].major_ticklabels.set_visible(True)
                self.ax.axis["lon"].major_ticks.set_ticksize(5)
                self.ax.axis["lon"].invert_ticklabel_direction()
                self.ax.axis["lon"].major_ticklabels.set_fontproperties(font)

            #if self.ind == "PPI":
            self.ax.axis["lon2"].major_ticklabels.set_visible(True)
            self.ax.axis["lon2"].major_ticks.set_ticksize(5)
            self.ax.axis["lon2"].invert_ticklabel_direction()
            self.ax.axis["lon2"].major_ticklabels.set_fontproperties(font)

            if abs(self.x_range[0]) < self.radial_range[1]:
                left = True
                vert1 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.x_range[0],2)))+abs(self.y_range[0]))/self.yd
                vert2 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.x_range[0],2)))+abs(self.y_range[1]))/self.yd
                path = self.ax.axis["left"].line.get_path()
                vert = path.vertices
                vert[0][1] = 0 if vert2 > 1 else 1 - vert2
                vert[1][1] = 1 if vert1 > 1 else vert1
                self.ax.axis["left"].line.set_path(matplotlib.path.Path(vert))

            else:
                left =False

            if abs(self.x_range[1]) < abs(self.radial_range[1]):
                right = True
                vert1 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.x_range[1],2))) + abs(self.y_range[0]))/self.yd
                vert2 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.x_range[1],2))) + abs(self.y_range[1]))/self.yd
                path = self.ax.axis["right"].line.get_path()
                vert = path.vertices
                vert[1][1] = 1 if vert1 > 1 else vert1
                vert[0][1] = 0 if vert2 > 1 else 1 - vert2
                self.ax.axis["right"].line.set_path(matplotlib.path.Path(vert))

            else:
                right = False

            if abs(self.y_range[0]) < abs(self.radial_range[1]):
                bottom = True
                vert1 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.y_range[0],2)))+abs(self.x_range[0]))/self.xd
                vert2 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.y_range[0],2)))+abs(self.x_range[1]))/self.xd
                path = self.ax.axis["bottom"].line.get_path()
                vert = path.vertices
                vert[1][0] = 1 if vert1 > 1 else vert1
                vert[0][0] = 0 if vert2 > 1 else 1 - vert2
                self.ax.axis["bottom"].line.set_path(matplotlib.path.Path(vert))

            else:
                bottom =False

            if abs(self.y_range[1]) < abs(self.radial_range[1]):
                top = True
                vert1 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.y_range[1],2)))+abs(self.x_range[0]))/self.xd
                vert2 = (math.sqrt(abs(pow(self.radial_range[1],2) - pow(self.y_range[1],2)))+abs(self.x_range[1]))/self.xd
                path = self.ax.axis["top"].line.get_path()
                vert = path.vertices
                vert[0][0] = 0 if vert2 > 1 else 1 - vert2
                vert[1][0] = 1 if vert1 > 1 else vert1
                self.ax.axis["top"].line.set_path(matplotlib.path.Path(vert))
            else:
                top = False

            self.ax.axis["top"].major_ticklabels.set_fontproperties(font)
            self.ax.axis["bottom"].major_ticklabels.set_fontproperties(font)
            self.ax.axis["right"].major_ticklabels.set_fontproperties(font)
            self.ax.axis["left"].major_ticklabels.set_fontproperties(font)


            self.ax.axis["left"].set_visible(left)
            self.ax.axis["right"].set_visible(right)
            self.ax.axis["bottom"].set_visible(bottom)
            self.ax.axis["top"].set_visible(top)
            # make ticklabels of left and bottom axis visible.
            self.ax.axis["left"].major_ticklabels.set_visible(left)
            self.ax.axis["bottom"].major_ticklabels.set_visible(bottom)
            # but set tickmarklength to zero for better presentation
            self.ax.axis["left"].major_ticks.set_ticksize(5)
            self.ax.axis["bottom"].major_ticks.set_ticksize(5)
            # let right and top axis shows ticklabels for 1st coordinate (angle)
            self.ax.axis["left"].get_helper().nth_coord_ticks=0
            self.ax.axis["bottom"].get_helper().nth_coord_ticks=0

            # make ticklabels of right and top axis visible.
            self.ax.axis["right"].major_ticklabels.set_visible(right)
            self.ax.axis["top"].major_ticklabels.set_visible(top)
            # but set tickmarklength to zero for better presentation
            self.ax.axis["right"].major_ticks.set_ticksize(5)
            self.ax.axis["top"].major_ticks.set_ticksize(5)
            # let right and top axis shows ticklabels for 1st coordinate (angle)
            self.ax.axis["right"].get_helper().nth_coord_ticks=0
            self.ax.axis["top"].get_helper().nth_coord_ticks=0

        elif s == 'off' or s == False:
            if self.float_axis:
                self.ax.axis["lon"].major_ticklabels.set_visible(False)
                self.ax.axis["lon"].major_ticks.set_ticksize(0)
            #if self.ind == "PPI":
            self.ax.axis["lon2"].major_ticklabels.set_visible(False)
            self.ax.axis["lon2"].major_ticks.set_ticksize(0)

    def cartgrid(self,s, *args, **kwargs):
        """ turns cartesian grid/axis on/off (x, y)

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """
        self.ax.axis["right"].set_visible(s)
        self.ax.axis["bottom"].set_visible(s)
        self.ax.axis["left"].set_visible(s)
        self.ax.axis["top"].set_visible(s)
        self.xgrid(s)
        self.ygrid(s)

    def polgrid(self,s, *args, **kwargs):
        """ turns polar grid on/off

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """
        if s == 'on' or s == True:
            self.ax.grid(True)
        elif s == 'off' or s == False:
            self.ax.grid(False)

    def xgrid(self, s, *args, **kwargs):
        """ turns xgrid on/off

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """
        if s == 'on' or s == True:
            if hasattr(self, 'p_xgrid'):
                if self.p_xgrid:
                    for item in self.p_xgrid:
                        item.remove()
            self.p_xgrid = []
            for xmaj in self.rad:
                line = self.ax.axvline(x=xmaj,color='k', ls=':')
                self.p_xgrid.append(line)
        elif s == 'off' or s == False:
            if hasattr(self, 'p_xgrid'):
                if self.p_xgrid:
                    for item in self.p_xgrid:
                        item.remove()
                self.p_xgrid = None
                #self.remove(p_xgrid)
        elif s == 'update':
            self.xgrid('on')
        else:
            self.xgrid('on')
            self.xgrid('off')


    def ygrid(self, s, *args, **kwargs):
        """ turns xgrid on/off

        Parameters
        ----------
        s : boolean
            True or False

        Returns
        ----------
        None

        """
        if s == 'on' or s == True:
            if hasattr(self, 'p_ygrid'):
                if self.p_ygrid:
                    for item in self.p_ygrid:
                        item.remove()
            self.p_ygrid = []
            for ymaj in self.hgt:
                line = self.ax.axhline(y=ymaj,color='k', ls=':')
                self.p_ygrid.append(line)
        elif s == 'off' or s == False:
            if hasattr(self, 'p_ygrid'):
                if self.p_ygrid:
                    for item in self.p_ygrid:
                        item.remove()
                self.p_ygrid = None
        elif s == 'update':
            self.ygrid('on')
        else:
            self.ygrid('on')
            self.ygrid('off')

    def get_ypadding(self, pad, ax, *args, **kwargs):
        """ calculates labelpadding in direction of y-axis (e.g. x-label)

        Parameters
        ----------
        pad : string
                padding in percent of ax
        ax : relevant axis


        Returns
        ----------
        padding in axis values

        """

        if not isinstance(pad, Size._Base):
                padding = Size.from_any(pad,
                                    fraction_ref=Size.AxesY(ax))


        p = (self.xd/self.yd) / self.aspect
        return padding.get_size(ax)[0]*p

    def get_xpadding(self, pad, ax, *args, **kwargs):
        """ calculates labelpadding in direction of x-axis (e.g. y-label)

        Parameters
        ----------
        pad : string
                padding in percent of ax
        ax : relevant axis

        Returns
        ----------
        padding in axis values

        """
        if not isinstance(pad, Size._Base):
                padding = Size.from_any(pad,
                                    fraction_ref=Size.AxesX(ax))
        p = (self.yd/self.xd) * self.aspect
        return padding.get_size(ax)[0]

    def title(self, s, *args, **kwargs):
        """ plots figure title

        Parameters
        ----------
        s : string
                Title String

        Keyword args
        ----------
        fsize : fontsize in percent of axis size
        pad : string
                padding in percent of axis size

        Returns
        ----------
        None

        """
        fsize = kwargs.pop('fsize',"2%")
        pad = kwargs.pop('pad',"2%")
        labelpad = self.get_ypadding(pad=pad, ax = self.ax)
        fsize = self.get_fontsize(fsize)[0]

        if hasattr(self, 'p_title'):
            if self.p_title:
                self.p_title.remove()

        self.p_title = self.ax.text(self.x_range[0],labelpad + self.y_range[1],s, fontsize = fsize, va='center', ha='left')

    def x_title(self, s, *args, **kwargs):
        """ plots x axis title

        Parameters
        ----------
        s : string
                Title String

        Keyword args
        ----------
        fsize : fontsize in percent of axis size
        pad : string
                padding in percent of axis size

        Returns
        ----------
        None

        """
        fsize = kwargs.pop('fsize',"2%")
        pad = kwargs.pop('pad',"2%")

        if hasattr(self, 'p_xtitle'):
            if self.p_xtitle:
                self.p_xtitle.remove()

        labelpad = self.get_ypadding(pad=pad, ax = self.ax)
        fsize = self.get_fontsize(fsize)[0]

        self.p_xtitle = self.ax.text(self.xd/2.+self.x_range[0],-labelpad+self.y_range[0],s, fontsize = fsize, va='center', ha='center')

    def y_title(self, s, *args, **kwargs):
        """ plots y axis title

        Parameters
        ----------
        s : string
                Title String

        Keyword args
        ----------
        fsize : fontsize in percent of axis size
        pad : string
                padding in percent of axis size

        Returns
        ----------
        None

        """
        fsize = kwargs.pop('fsize',"2%")
        pad = kwargs.pop('pad',"2%")

        if hasattr(self, 'p_ytitle'):
            if self.p_ytitle:
                self.p_ytitle.remove()

        labelpad = self.get_xpadding(pad=pad, ax = self.ax)
        fsize = self.get_fontsize(fsize)[0]

        self.p_ytitle = self.ax.text(-labelpad + self.x_range[0],self.yd/2.+ self.y_range[0],s, fontsize = fsize, va='center', ha='left', rotation='vertical')

    def z_title(self, s, *args, **kwargs):
        """ plots colorbar title if colorbar is defined

        Returns
        ----------
        None

        """

        fsize = kwargs.pop('fsize',"2%")
        pad = kwargs.pop('pad',"2%")

        labelpad = self.get_xpadding(pad=pad, ax = self.ax)
        fsize = self.get_fontsize(fsize)[0]

        if hasattr(self, 'cbar'):
            self.cbar.set_label(s, size = fsize, *args, **kwargs)

    def copy_right(self, *args, **kwargs):
        """ plot copyright in lower left corner
            check position, its in plot coordinates not figure coordinates

        Keyword args
        ----------
        fsize : fontsize in percent of axis size
        text : string
                Copyright String
        padx : string
                padding in percent of axis size
        pady : string
                padding in percent of axis size

        Returns
        ----------
        None
        """

        fsize = kwargs.pop('fsize',"1%")
        text = kwargs.pop('text',r"""$\copyright\/2013\/ created with WRADLIB$""")
        padx = kwargs.pop('padx',"2%")
        pady = kwargs.pop('pady',"2%")

        padx = self.get_xpadding(padx,self.ax)
        pady = self.get_ypadding(pady,self.ax)
        fsize = self.get_fontsize(fsize)[0]

        if hasattr(self, 'p_copy'):
            if self.p_copy:
                self.p_copy.remove()

        self.p_copy = self.ax.text(self.x_range[0]- padx, - pady + self.y_range[0],text,fontsize=fsize, va='center', ha='left')

    def colorbar(self, *args, **kwargs):
        """ plot colorbar, vertical, right side

        Keyword args
        ----------
        vmin : plot minimum
        vmax : plot maximum
        z_res : colorbar tick resolution
        z_unit : string
                unit
        cbp : string
               padding in percent of axis size
        cbw : string
                width in percent of axis size

        Returns
        ----------
        cbar : colorbar object
        """

        key = ['vmin', 'vmax', 'z_res', 'ztitle', 'zunit']
        key1 = ['cbp', 'fsize', 'cbw']

        if kwargs:
            for k in key:
                if k in kwargs:
                    setattr(self, k, np.array(kwargs[k]))
                    kwargs.pop(k)
            for k in key1:
                if k in kwargs:
                    setattr(self, k, kwargs[k])
                    kwargs.pop(k)

        # get axis, create and add colorbar-cax,
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="0%", axes_class=mpl.axes.Axes)
        cbp = Size.from_any(self.cbp, fraction_ref=Size.Fraction(1/self.aspect, Size.AxesX(self.ax)))
        cbw = Size.from_any(self.cbw, fraction_ref=Size.Fraction(1/self.aspect, Size.AxesX(self.ax)))

        h = [# main axes
             Size.Fraction(1/self.aspect, Size.AxesX(self.ax)),
             cbp,
             cbw,
             ]

        v = [Size.AxesY(self.ax)]

        divider.set_horizontal(h)
        divider.set_vertical(v)

        self.ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
        cax.set_axes_locator(divider.new_locator(nx=2, ny=0))

        self.fig.add_axes(cax)

        # set z_range and plot-clims
        self.z_range = [self.vmin,self.vmax]
        args[0].set_clim(vmin=self.vmin, vmax=self.vmax)

        # get ticks
        z_ticks = get_tick_vector(self.z_range,self.z_res)

        # plot colorbar
        if kwargs:
            if 'ticks' in kwargs:
                self.cbar = self.fig.colorbar(*args, cax=cax, **kwargs)
                z_ticks = kwargs['ticks']
            else:
                self.cbar = self.fig.colorbar(*args, cax=cax, ticks=z_ticks, **kwargs)

        # set font and size
        fsize = self.get_fontsize(self.fsize)[0]
        font = fm.FontProperties()
        font.set_family('sans-serif')
        font.set_size(fsize)

        # plot colorbar title and ticks
        if self.ztitle:
            ztitle = str(self.ztitle)
            if self.zunit:
                ztitle = ztitle +' ('+ str(self.zunit) + ')'
            self.cbar.set_label(ztitle, fontsize=fsize)

        # test for integer and account for annotation with decimal place
        if np.equal(np.mod(np.array(z_ticks), 1), 0).all():
            z_ticks1 = [str(np.int(i)) for i in z_ticks]
        else:
            z_ticks1 = [str(i) for i in z_ticks]
        self.cbar.ax.set_yticklabels(z_ticks1, fontsize=fsize)

        return self.cbar



def plot_scan_strategy(ranges, elevs, site, vert_res=500., maxalt=10000., ax=None):
    """Plot the vertical scanning strategy

    Parameters
    ----------
    ranges : array of ranges
    elevs : array of elevation angles
    site : tuple of site coordinates (longitude, latitude, altitude)
    """
    # just a dummy
    az=np.array([90.])

    polc = util.meshgridN(ranges, az, elevs)

    # get mean height over radar
    lon, lat, alt = georef.polar2lonlatalt_n(polc[0].ravel(), polc[1].ravel(), polc[2].ravel(), site)
    alt = alt.reshape(len(ranges), len(elevs))
    r = polc[0].reshape(len(ranges), len(elevs))

    if ax==None:
        returnax = False
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        returnax = True
    # actual plotting
    for y in np.arange(0,10000.,vert_res):
        ax.axhline(y=y, color="grey")
    for x in ranges:
        ax.axvline(x=x, color="grey")
    for i in range(len(elevs)):
        ax.plot(r[:,i].ravel(), alt[:,i].ravel(), lw=2, color="black")
    pl.ylim(ymax=maxalt)
    ax.tick_params(labelsize="large")
    pl.xlabel("Range (m)", size="large")
    pl.ylabel("Height over radar (m)", size="large")
    for i, elev in enumerate(elevs):
        x = r[:,i].ravel()[-1]+1500.
        y = alt[:,i].ravel()[-1]
        if  y > maxalt:
            ix = np.where(alt[:,i].ravel()<maxalt)[0][-1]
            x = r[:,i].ravel()[ix]
            y = maxalt+100.
        pl.text(x, y, str(elev), fontsize="large")

    if returnax:
        return ax
    pl.show()


def plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, unit="", title="", saveto="", **kwargs):
    """Plot 2-D plan view of <dataxy> together with vertical sections <dataxz> and <datazy>

    Parameters
    ----------
    x : array of x-axis coordinates
    y : array of y-axis coordinates
    z : array of z-axis coordinates
    dataxy : 2d array of shape (len(x), len(y))
    datazx : 2d array of shape (len(z), len(x))
    datazy : 2d array of shape (len(z), len(y))
    unit : string (unit of data arrays)
    title: string
    saveto : file path if figure should be saved
    **kwargs : other kwargs which can be passed to pylab.contourf

    """

    fig = pl.figure(figsize=(10, 10))

    # define axes
    left, bottom, width, height = 0.1, 0.1, 0.6, 0.2
    ax_xy = pl.axes((left, bottom, width, width))
    ax_x  = pl.axes((left, bottom+width, width, height))
    ax_y  = pl.axes((left+width, bottom, height, width))
    ax_cb  = pl.axes((left+width+height+0.02, bottom, 0.02, width))

    # set axis label formatters
    ax_x.xaxis.set_major_formatter(NullFormatter())
    ax_y.yaxis.set_major_formatter(NullFormatter())

    # draw CAPPI
    pl.axes(ax_xy)
    xy = pl.contourf(x,y,dataxy, **kwargs)
    pl.grid(color="grey", lw=1.5)

    # draw colorbar
    cb = pl.colorbar(xy, cax=ax_cb)
    cb.set_label("(%s)" % unit)

    # draw upper vertical profil
    ax_x.contourf(x, z, datazx, **kwargs)

    # draw right vertical profil
    ax_y.contourf(z, y, datazy.T, **kwargs)

    # label axes
    ax_xy.set_xlabel('x (km)')
    ax_xy.set_ylabel('y (km)')
    ax_x.set_xlabel('')
    ax_x.set_ylabel('z (km)')
    ax_y.set_ylabel('')
    ax_y.set_xlabel('z (km)')

    def xycoords(x,pos):
        'The two args are the value and tick position'
        return "%d" % (x/1000.)

    xyformatter = FuncFormatter(xycoords)

    def zcoords(x,pos):
        'The two args are the value and tick position'
        return ( "%.1f" % (x/1000.) ).rstrip('0').rstrip('.')

    zformatter = FuncFormatter(zcoords)

    ax_xy.xaxis.set_major_formatter(xyformatter)
    ax_xy.yaxis.set_major_formatter(xyformatter)
    ax_x.yaxis.set_major_formatter(zformatter)
    ax_y.xaxis.set_major_formatter(zformatter)

    if not title=="":
        # add a title - here, we have to create a new axes object which will be invisible
        # then the invisble axes will get a title
        tax = pl.axes((left, bottom+width+height+0.01, width+height, 0.01), frameon=False, axisbg="none")
        tax.get_xaxis().set_visible(False)
        tax.get_yaxis().set_visible(False)
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


def plot_max_plan_and_vert(x, y, z, data, unit="", title="", saveto="", **kwargs):
    """Plot according to <plot_plan_and_vert> with the maximum values along the three axes of <data>
    """
    plot_plan_and_vert(x, y, z, np.max(data,axis=-3), np.max(data, axis=-2), np.max(data, axis=-1), unit, title, saveto, **kwargs)


def plot_tseries(dtimes, data, ax=None, labels=None, datefmt='%b %d, %H:%M', colors=None, ylabel="", title="", fontsize="medium", saveto="", **kwargs):
    """Plot time series data (e.g. gage recordings)

    Parameters
    ----------
    dtimes : array of datetime objects (time steps)
    data : 2D array of shape ( num time steps, num data series )
    labels : list of strings (names of data series)
    title : string
    kwargs : keyword arguments related to pylab.plot

    """
    if ax==None:
        returnax = False
        fig = pl.figure()
        ax  = fig.add_subplot(1,1,1,  title=title)
    else:
        returnax = True
##    if labels==None:
##        labels = ["series%d"%i for i in range(1, data.shape[1]+1)]
##    for i, label in enumerate(labels):
##        ax.plot_date(mpl.dates.date2num(dtimes),data[:,i],label=label, color=colors[i], **kwargs)
    ax.plot_date(mpl.dates.date2num(dtimes), data, **kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))
    pl.setp(ax.get_xticklabels(), visible=True)
    pl.setp(ax.get_xticklabels(), rotation=-30, horizontalalignment='left')
    ax.set_ylabel(ylabel, size=fontsize)
    ax = set_ticklabel_size(ax,fontsize)
    ax.legend(loc='best')

    if returnax:
        return ax

    if saveto=="":
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

def set_ticklabel_size(ax, size):
    """
    """
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(size)
    return ax

if __name__ == '__main__':
    print 'wradlib: Calling module <vis> as main...'



