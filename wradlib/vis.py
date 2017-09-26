#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

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
   create_cg
   plot_cg_ppi
   plot_cg_rhi
   plot_scan_strategy
   plot_plan_and_vert
   plot_max_plan_and_vert
   plot_tseries
   add_lines
   add_patches

"""

# standard libraries
import os.path as path
import warnings

# site packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from matplotlib import patches
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform
from mpl_toolkits.axisartist import (SubplotHost, ParasiteAxesAuxTrans,
                                     GridHelperCurveLinear)
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
import mpl_toolkits.axisartist.angle_helper as ah
from matplotlib.ticker import NullFormatter, FuncFormatter
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection, PolyCollection

# wradlib modules
from . import georef as georef
from . import util as util


class NorthPolarAxes(PolarAxes):
    """
    A variant of PolarAxes where theta starts pointing north and goes
    clockwise.
    Obsolete since matplotlib version 1.1.0, where the same behaviour may
    be achieved with a reconfigured standard PolarAxes object.
    """
    name = 'northpolar'

    class NorthPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy = np.zeros(tr.shape, np.float_)
            t = tr[:, 0:1]
            r = tr[:, 1:2]
            x = xy[:, 0:1]
            y = xy[:, 1:2]
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
            r = np.sqrt(x * x + y * y)
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


def plot_ppi(data, r=None, az=None, autoext=True,
             site=(0, 0), proj=None, elev=0.,
             fig=None, ax=111, func='pcolormesh',
             cg=False, rf=1., refrac=False,
             **kwargs):
    """Plots a Plan Position Indicator (PPI).

    .. versionchanged:: 0.10.0
       added contour, contourf plotting, added `cg`

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

    The implementation of this plot routine is in cartesian axes and does all
    coordinate transforms beforehand. This allows zooming into the data as well
    as making it easier to plot additional data (like gauge locations) without
    having to convert them to the radar's polar coordinate system.

    Using `cg=True` the plotting is done in a curvelinear grid axes.
    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending which axes object is used.

    ``**kwargs` may be used to try to influence the
    :func:`matplotlib.pyplot.pcolormesh`, :func:`matplotlib.pyplot.contour`,
    :func:`matplotlib.pyplot.contourf` and
    :meth:`wradlib.georef.polar2lonlatalt_n` routines under the hood.

    There is one major caveat concerning the values of `r` and `az`.
    Due to the way :func:`matplotlib.pyplot.pcolormesh` works, `r` should give
    the location of the start of each range bin, while `az` should give the
    angle also at the begin (i.e. 'leftmost') of the beam.
    This might be in contrast to other conventions, which might define ranges
    and angles at the center of bin and beam.
    This affects especially the default values set for `r` and `az`, but ìt
    should be possible to accommodate all other conventions by setting `r` and
    `az` properly.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily, unless proj is set. In
        that case the units must be meters. If None, a default is
        calculated from the dimensions of `data`.
    rf: float
        If present, factor for scaling range axes, defaults to 1.
    az : np.array
        The azimuth angles in degrees. If None, a default is
        calculated from the dimensions of `data`.
    autoext : True | False
        This routine uses :func:`matplotlib.pyplot.pcolormesh` to draw the
        bins.
        As this function needs one set of coordinates more than would usually
        be provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    refrac: True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account. If False, simple
        trigonometry will be used to calculate beam propagation.
        Functionality for this will be provided by function
        :meth:`wradlib.georef.arc_distance_n`. Therefore, if `refrac` is True,
        `r` must be given in meters.
    site : tuple
        Tuple of coordinates of the radar site.
        If `proj` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If `proj` is used, values must be given as (longitude, latitude)
        tuple of geographical coordinates.
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection
        If this parameter is not None, `site` must be set. Then the function
        will attempt to georeference the radar bins and display the PPI in the
        coordinate system defined by the projection string.
    elev : float or array of same shape as az
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
    fig : matplotlib Figure object
        If given, the RHI will be plotted into this figure object. Axes are
        created as needed. If None, a new figure object will be created or
        current figure will be used, depending on "ax".
    ax : matplotlib Axes object | matplotlib grid definition
        If matplotlib Axes object is given, the PPI will be plotted into this
        axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    func : str
        Name of plotting function to be used under the hood.
        Defaults to 'pcolormesh'. 'contour' and 'contourf' can be selected too.
    cg : True | False
        If True, the data will be plotted on curvelinear axes.

    See also
    --------
    wradlib.georef.reproject - for information on projection strings
    wradlib.georef.create_osr - generate pre-defined projection strings

    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the PPI was plotted
    pm : mmatplotlib QuadMesh object | matplotlib QuadContourSet
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    Note
    ----
    If `cg` is True, the `cgax` - curvelinear Axes (r-theta-grid)
    is returned. `caax` - Cartesian Axes (x-y-grid) and `paax` -
    parasite axes object for plotting polar data can be derived like this::

        caax = cgax.parasites[0]
        paax = cgax.parasites[1]

    The function `create_cg` uses the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.


    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_plot_ppi_example.ipynb`,
    and :ref:`notebooks/visualisation/wwradlib_plot_curvelinear_grids.ipynb`.

    """
    # kwargs handling
    kw_polar2lonlatalt_n = {}
    if 're' in kwargs:
        kw_polar2lonlatalt_n['re'] = kwargs.pop('re')
    if 'ke' in kwargs:
        kw_polar2lonlatalt_n['ke'] = kwargs.pop('ke')

    if (proj is not None) & cg:
        cg = False
        warnings.warn(
            "WARNING: `cg` cannot be used with `proj`, falling back.")

    # providing 'reasonable defaults', based on the data's shape
    if r is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    else:
        d1 = np.asanyarray(r.copy())

    if az is None:
        d2 = np.arange(data.shape[0], dtype=np.float)
    else:
        d2 = np.asanyarray(az.copy())

    if autoext & ('pcolormesh' in func):
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

    if 'contour' in func:
        # add first azimuth as last for y and data
        y = np.append(d2, d2[0])
        data = np.vstack((data, data[0][np.newaxis, ...]))
        # move to center
        x += (x[1] - x[0]) / 2
        y += (y[1] - y[0]) / 2

    if refrac & (proj is None):
        # with refraction correction, significant at higher elevations
        # calculate new range values
        x = georef.arc_distance_n(x, elev)

    # axes object is given
    if isinstance(ax, mpl.axes.Axes):
        if cg:
            caax = ax.parasites[0]
            paax = ax.parasites[1]
    else:
        if fig is None:
            if ax is 111:
                # create new figure if there is only one subplot
                fig = pl.figure()
            else:
                # assume current figure
                fig = pl.gcf()
        if cg:
            # create curvelinear axes
            ax, caax, paax = create_cg('PPI', fig, ax)
            # this is in fact the outermost thick "ring"
            ax.axis["lon"] = ax.new_floating_axis(1, np.max(x) / rf)
            ax.axis["lon"].major_ticklabels.set_visible(False)
            # and also set tickmarklength to zero for better presentation
            ax.axis["lon"].major_ticks.set_ticksize(0)
        else:
            ax = fig.add_subplot(ax)

    if cg:
        xx, yy = np.meshgrid(y, x)
        # set bounds to min/max
        xa = yy * np.sin(np.radians(xx)) / rf
        ya = yy * np.cos(np.radians(xx)) / rf
        plax = paax
    else:
        # coordinates for all vertices
        xx, yy = np.meshgrid(x, y)
        plax = ax

    if proj:
        # with georeferencing
        if r is None:
            # if we produced a default, this one is still in 'kilometers'
            # therefore we need to get from km to m
            xx *= 1000
        # latitude longitudes from the polar data still stored in xx and yy
        lon, lat, alt = georef.polar2lonlatalt_n(xx, yy, elev, site,
                                                 **kw_polar2lonlatalt_n)
        # projected to the final coordinate system
        xx, yy = georef.reproject(lon, lat, projection_target=proj)
    else:
        if cg:
            yy = yy / rf
            data = data.transpose()
        else:
            # no georeferencing -> simple trigonometry
            xxx = (xx * np.cos(np.radians(90. - yy)) + site[0]) / rf
            yy = (xx * np.sin(np.radians(90. - yy)) + site[1]) / rf
            xx = xxx

    # plot the stuff
    plotfunc = getattr(plax, func)
    pm = plotfunc(xx, yy, data, **kwargs)

    if cg:
        # show curvelinear and cartesian grids
        ax.set_ylim(np.min(ya), np.max(ya))
        ax.set_xlim(np.min(xa), np.max(xa))
        ax.grid(True)
        caax.grid(True)
    else:
        ax.set_aspect('equal')

    return ax, pm


def plot_ppi_crosshair(site, ranges, angles=None,
                       proj=None, elev=0., ax=None, **kwargs):
    """Plots a Crosshair for a Plan Position Indicator (PPI).

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

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
        These lines will be drawn starting from the center and until the
        largest range.
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection
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

    Keyword Arguments
    -----------------
    line :  dict
        dictionary, which will be passed to the crosshair line objects using
        the standard keyword inheritance mechanism. If not given defaults will
        be used.
    circle : dict
        dictionary, which will be passed to the range circle line objects using
        the standard keyword inheritance mechanism. If not given defaults will
        be used.

    See also
    --------
    wradlib.vis.plot_ppi - plotting a PPI in cartesian coordinates

    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the PPI was plotted

    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_plot_ppi_example.ipynb`.

    """
    # if we didn't get an axes object, find the current one
    if ax is None:
        ax = pl.gca()

    if angles is None:
        angles = [0, 90, 180, 270]

    # set default line keywords
    linekw = dict(color='gray', linestyle='dashed')
    # update with user settings
    linekw.update(kwargs.get('line', {}))

    # set default circle keywords
    circkw = dict(edgecolor='gray', linestyle='dashed', facecolor='none')
    # update with user settings
    circkw.update(kwargs.get('circle', {}))

    # determine coordinates for 'straight' lines
    if proj:
        # projected
        # reproject the site coordinates
        psite = georef.reproject(*site, projection_target=proj)
        # these lines might not be straigt so we approximate them with 10
        # segments. Produce polar coordinates
        rr, az = np.meshgrid(np.linspace(0, ranges[-1], 10), angles)
        # and reproject using polar2lonlatalt to convert from polar
        # to geographic
        nsewx, nsewy = georef.reproject(*georef.polar2lonlatalt_n(rr,
                                                                  az,
                                                                  elev,
                                                                  site)[:2],
                                        projection_target=proj)
    else:
        # no projection
        psite = site
        rr, az = np.meshgrid(np.linspace(0, ranges[-1], 2), angles)
        # use simple trigonometry to calculate coordinates
        nsewx, nsewy = (psite[0] + rr * np.cos(np.radians(90 - az)),
                        psite[1] + rr * np.sin(np.radians(90 - az)))

    # mark the site, just in case nothing else would be drawn
    ax.plot(*psite, marker='+', **linekw)

    # draw the lines
    for i in range(len(angles)):
        ax.add_line(mpl.lines.Line2D(nsewx[i, :], nsewy[i, :], **linekw))

    # draw the range circles
    for r in ranges:
        if proj:
            # produce an approximation of the circle
            x, y = georef.reproject(*georef.polar2lonlatalt_n(r,
                                                              np.arange(360),
                                                              elev,
                                                              site)[:2],
                                    projection_target=proj)
            ax.add_patch(patches.Polygon(np.concatenate([x[:, None],
                                                         y[:, None]],
                                                        axis=1),
                                         **circkw))
        else:
            # in the unprojected case, we may use 'true' circles.
            ax.add_patch(patches.Circle(psite, r, **circkw))

    # there should be not much wrong, setting the axes aspect to equal
    # by default
    ax.set_aspect('equal')

    # return the axes object for later use
    return ax


def plot_rhi(data, r=None, th=None, th_res=None, yoffset=0., autoext=True,
             refrac=True, rf=1., fig=None, ax=111, func='pcolormesh', cg=False,
             **kwargs):
    """Plots a Range Height Indicator (RHI).

    .. versionchanged:: 0.10.0
       added contour, contourf plotting, added `cg`

    The implementation of this plot routine is in cartesian axes and does all
    coordinate transforms beforehand. This allows zooming into the data as well
    as making it easier to plot additional data (like gauge locations) without
    having to convert them to the radar's polar coordinate system.

    Using `cg=True` the plotting is done in a curvelinear grid axes.
    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending which axes object is used.

    `**kwargs` may be used to try to influence the
    :func:`matplotlib.pyplot.pcolormesh`, :func:`matplotlib.pyplot.contour`
    and :func:`matplotlib.pyplot.contourf` routines under the hood.

    Parameters
    ----------
    data : np.array
        The data to be plotted. It is assumed that the first dimension is over
        the elevation angles, while the second dimension is over the range bins
    r : np.array
        The ranges. Units may be chosen arbitrarily. If None, a default is
        calculated from the dimensions of `data`.
    rf: float
        If present, factor for scaling range axis, defaults to 1.
    th : np.array
        The elevation angles in degrees. If None, a default is
        calculated from the dimensions of `data`.
    th_res : float or np.array of same shape as `th`
        In RHI's it happens that the elevation angles are spaced wider than
        the beam width. If this beam width (in degrees) is given in `th_res`,
        plot_rhi will plot the beams accordingly. Otherwise the behavior of
        :func:`matplotlib.pyplot.pcolormesh` assumes all beams to be adjacent
        to each other, which might lead to unexpected results.
    yoffset : float
        Altitude offset that would typically represent the altitude of
        the radar antenna. Units must be consistent with units of `r`.
    autoext : True | False
        This routine uses :func:`matplotlib.pyplot.pcolormesh` to draw
        the bins.
        As this function needs one set of coordinates more than would usually
        provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    refrac : True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account. If False, simple
        trigonometry will be used to calculate beam propagation.
        Functionality for this will be provided by functions
        :meth:`wradlib.georef.arc_distance_n` and
        :meth:`wradlib.georef.beam_height_n`, which assume distances to be
        given in meters. Therefore, if `refrac` is True, `r` must be given
        in meters.
    fig : matplotlib Figure object
        If given, the RHI will be plotted into this figure object. Axes are
        created as needed. If None, a new figure object will be created or
        current figure will be used, depending on "ax".
    ax : matplotlib Axes object | matplotlib grid definition
        If matplotlib Axes object is given, the RHI will be plotted into this
        axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    func : str
        Name of plotting function to be used under the hood.
        Defaults to 'pcolormesh'. 'contour' and 'contourf' can be selected too.
    cg : True | False
        If True, the data will be plotted on curvelinear axes.

    See also
    --------
    create_cg : creation of curvelinear grid axes objects

    Returns
    -------
    ax : matplotlib Axes object
        The axes object into which the RHI was plotted.
    pm : matplotlib QuadMesh object | matplotlib QuadContourSet object
        The result of the plot operation. Necessary, if you want to
        add a colorbar to the plot.

    Note
    ----
    If `cg` is True, the `cgax` - curvelinear Axes (r-theta-grid)
    is returned. `caax` - Cartesian Axes (x-y-grid) and `paax` -
    parasite axes object for plotting polar data can be derived like this::

        caax = cgax.parasites[0]
        paax = cgax.parasites[1]

    The function `create_cg` uses the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.


    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.
    """
    # autogenerate axis dimensions
    if r is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    else:
        d1 = np.asanyarray(r.copy())

    if th is None:
        # assume, data is evenly spaced between 0 and 90 degree
        d2 = np.linspace(0., 90., num=data.shape[0], endpoint=True)
        # d2 = np.arange(data.shape[0], dtype=np.float)
    else:
        d2 = np.asanyarray(th.copy())

    if autoext & ('pcolormesh' in func):
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

    # fix reference for contour functions
    if 'contour' in func:
        x += (x[1] - x[0]) / 2
        y += (y[1] - y[0]) / 2

    # axes object given
    if isinstance(ax, mpl.axes.Axes):
        if cg:
            caax = ax.parasites[0]
            paax = ax.parasites[1]
    else:
        if fig is None:
            # create new figure if there is only one subplot
            if ax is 111:
                fig = pl.figure()
            else:
                fig = pl.gcf()
        if cg:
            # create curvelinear axes
            ax, caax, paax = create_cg('RHI', fig, ax)

            # this is in fact the outermost thick "ring" aka max_range
            ax.axis["lon"] = ax.new_floating_axis(1, np.max(x) / rf)
            ax.axis["lon"].major_ticklabels.set_visible(False)
            # and also set tickmarklength to zero for better presentation
            ax.axis["lon"].major_ticks.set_ticksize(0)
        else:
            ax = fig.add_subplot(ax)

    # coordinates for all vertices
    xx, yy = np.meshgrid(x, y)

    plax = ax
    if refrac:
        # observing air refractivity, so ground distances and beam height
        # must be calculated specially
        xxx = georef.arc_distance_n(xx, yy) / rf
        yyy = georef.beam_height_n(xx, yy) / rf
        if cg:
            plax = caax
    else:
        if cg:
            xxx, yyy = np.meshgrid(y, x)
            yyy /= rf
            img = img.transpose()
            plax = paax
        else:
            # otherwise plane trigonometry will do
            xxx = xx * np.cos(np.radians(yy)) / rf
            yyy = xx * np.sin(np.radians(yy)) / rf

    yyy += yoffset / rf

    # plot the stuff
    plotfunc = getattr(plax, func)
    pm = plotfunc(xxx, yyy, img, **kwargs)

    # return references to important and eventually new objects
    if cg:
        # set bounds to maximum
        ax.set_ylim(0, np.max(x) / rf)
        ax.set_xlim(0, np.max(x) / rf)

        # show curvelinear and cartesian grids
        ax.grid(True)
        caax.grid(True)

    return ax, pm


def create_cg(st, fig=None, subplot=111):
    """ Helper function to create curvelinear grid

    The function makes use of the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.

    Parameters
    ----------
    st : string
        scan type, 'PPI' or 'RHI'
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
        tr = Affine2D().scale(np.pi / 180, 1) + PolarAxes.PolarTransform()

        # build up curvelinear grid
        extreme_finder = ah.ExtremeFinderCycle(20, 20,
                                               lon_cycle=100,
                                               lat_cycle=None,
                                               lon_minmax=(0, np.inf),
                                               lat_minmax=(0, np.inf),
                                               )

        # locator and formatter for angular annotation
        grid_locator1 = ah.LocatorDMS(10.)
        tick_formatter1 = ah.FormatterDMS()

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
        tr = (Affine2D().scale(np.pi / 180, 1) +
              NorthPolarAxes.NorthPolarTransform())

        # build up curvelinear grid
        extreme_finder = ah.ExtremeFinderCycle(20, 20,
                                               lon_cycle=360,
                                               lat_cycle=None,
                                               lon_minmax=(360, 0),
                                               lat_minmax=(0, np.inf),
                                               )

        # locator and formatter for angle annotation
        locs = [i for i in np.arange(0., 359., 10.)]
        grid_locator1 = FixedLocator(locs)
        tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i))
                                              for i in locs]))

        # grid_helper for curvelinear grid
        grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=None,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=None,
                                            )
        # try to set nice locations for range gridlines
        grid_helper.grid_finder.grid_locator2._nbins = 15.0
        grid_helper.grid_finder.grid_locator2._steps = [0, 1, 1.5, 2,
                                                        2.5,
                                                        5,
                                                        10]

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

    # make ticklabels of left and bottom axis invisible,
    # because we are drawing them
    cgax.axis["left"].major_ticklabels.set_visible(False)
    cgax.axis["bottom"].major_ticklabels.set_visible(False)

    # and also set tickmarklength to zero for better presentation
    cgax.axis["left"].major_ticks.set_ticksize(0)
    cgax.axis["bottom"].major_ticks.set_ticksize(0)

    # generate and add parasite axes with given transform
    paax = ParasiteAxesAuxTrans(cgax, tr, "equal")
    # note that paax.transData == tr + cgax.transData
    # Anything you draw in paax will match the ticks and grids of cgax.
    cgax.parasites.append(paax)

    return cgax, caax, paax


@util.deprecated(plot_ppi)
def plot_cg_ppi(data, r=None, az=None, rf=1.0, autoext=True,
                refrac=True, elev=0., fig=None, subplot=111,
                **kwargs):
    """Plots a Plan Position Indicator (PPI) on a curvelinear grid.

    .. deprecated:: 0.11
    Use :meth:`~wradlib.vis.plot_ppi`

    The implementation of this plot routine is in curvelinear grid axes and
    does all coordinate transforms beforehand. This allows zooming into the
    data as well as making it easier to plot additional data (like gauge
    locations).

    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending which axes object is used.

    The function uses create_cg which uses the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.

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
        This routine uses :func:`matplotlib.pyplot.pcolormesh` to draw
        the bins.
        As this function needs one set of coordinates more than would usually
        be provided by `r` and `az`, setting ´autoext´ to True automatically
        extends r and az so that all of `data` will be plotted.
    refrac : True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account. If False, simple
        trigonometry will be used to calculate beam propagation.
        Functionality for this will be provided by function
        :meth:`wradlib.georef.arc_distance_n`. Therefore, if `refrac` is True,
        `r` must be given in meters.
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
        all data in polar format must be plotted to this axis

    pm : matplotlib QuadMesh object
        The result of the pcolormesh operation. Necessary, if you want to
        add a colorbar to the plot.

    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.
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


@util.deprecated(plot_rhi)
def plot_cg_rhi(data, r=None, th=None, th_res=None, autoext=True, refrac=True,
                rf=1., fig=None, subplot=111, **kwargs):
    """Plots a Range Height Indicator (RHI) on a curvelinear grid.

    .. deprecated:: 0.11
    Use :meth:`~wradlib.vis.plot_rhi`

    The implementation of this plot routine is in a curvelinear grid axes and
    does all coordinate transforms beforehand.

    This allows zooming into the data as well as making it easier to plot
    additional cartesian data (like certain special points) without having to
    convert them to the radar's polar coordinate system.

    Plotting in the radar's polar coordinate system is possible as well.

    The function uses create_cg wich uses the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.

    `**kwargs` may be used to try to influence :func:`matplotlib.pcolormesh`
    under the hood.

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
        :func:`matplotlib.pyplot.pcolormesh` assumes all beams to be adjacent
        to each other, which might lead to unexpected results.
    autoext : True | False
        This routine uses matplotlib.pyplot.pcolormesh to draw the bins.
        As this function needs one set of coordinates more than would usually
        provided by `r` and `th`, setting ´autoext´ to True automatically
        extends r and th so that all of `data` will be plotted.
    refrac : True | False
        If True, the effect of refractivity of the earth's atmosphere on the
        beam propagation will be taken into account.
        Functionality for this will be provided by functions
        :meth:`wradlib.georef.arc_distance_n` and
        :meth:`wradlib.georef.beam_height_n`, which
        assume distances to be given in meters. Therefore, if `refrac` is True,
        `r` must be given in meters. Cartesian Axis caax ist used for plotting.
        If False,
        :class:`~matplotlib.projections.polar.PolarAxes.PolarTransform`
        will be used to calculate beam propagation.
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

    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.
    """

    # autogenerate axis dimensions
    if r is None:
        d1 = np.arange(data.shape[1], dtype=np.float)
    else:
        d1 = np.asanyarray(r.copy())

    if th is None:
        # assume, data is evenly spaced between 0 and 90 degree
        d2 = np.linspace(0., 90., num=data.shape[0], endpoint=True)
        # d2 = np.arange(data.shape[0], dtype=np.float)
    else:
        d2 = np.asanyarray(th.copy())

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
        yyy /= rf
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


def plot_scan_strategy(ranges, elevs, site, vert_res=500.,
                       maxalt=10000., ax=None):
    """Plot the vertical scanning strategy

    Parameters
    ----------
    ranges : array of ranges
    elevs : array of elevation angles
    site : tuple of site coordinates (longitude, latitude, altitude)
    """
    # just a dummy
    az = np.array([90.])

    polc = util.meshgridN(ranges, az, elevs)

    # get mean height over radar
    lon, lat, alt = georef.polar2lonlatalt_n(polc[0].ravel(), polc[1].ravel(),
                                             polc[2].ravel(), site)
    alt = alt.reshape(len(ranges), len(elevs))
    r = polc[0].reshape(len(ranges), len(elevs))

    if ax is None:
        returnax = False
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        returnax = True
    # actual plotting
    for y in np.arange(0, 10000., vert_res):
        ax.axhline(y=y, color="grey")
    for x in ranges:
        ax.axvline(x=x, color="grey")
    for i in range(len(elevs)):
        ax.plot(r[:, i].ravel(), alt[:, i].ravel(), lw=2, color="black")
    pl.ylim(ymax=maxalt)
    ax.tick_params(labelsize="large")
    pl.xlabel("Range (m)", size="large")
    pl.ylabel("Height over radar (m)", size="large")
    for i, elev in enumerate(elevs):
        x = r[:, i].ravel()[-1] + 1500.
        y = alt[:, i].ravel()[-1]
        if y > maxalt:
            ix = np.where(alt[:, i].ravel() < maxalt)[0][-1]
            x = r[:, i].ravel()[ix]
            y = maxalt + 100.
        pl.text(x, y, str(elev), fontsize="large")

    if returnax:
        return ax
    pl.show()


def plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, unit="",
                       title="", saveto="", **kwargs):
    """Plot 2-D plan view of <dataxy> together with
    vertical sections <dataxz> and <datazy>

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

    Keyword Arguments
    -----------------
    **kwargs : other kwargs which can be passed to \
               :func:`matplotlib.pyplot.contourf`

    """

    pl.figure(figsize=(10, 10))

    # define axes
    left, bottom, width, height = 0.1, 0.1, 0.6, 0.2
    ax_xy = pl.axes((left, bottom, width, width))
    ax_x = pl.axes((left, bottom + width, width, height))
    ax_y = pl.axes((left + width, bottom, height, width))
    ax_cb = pl.axes((left + width + height + 0.02, bottom, 0.02, width))

    # set axis label formatters
    ax_x.xaxis.set_major_formatter(NullFormatter())
    ax_y.yaxis.set_major_formatter(NullFormatter())

    # draw CAPPI
    pl.axes(ax_xy)
    xy = pl.contourf(x, y, dataxy, **kwargs)
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

    def xycoords(x, pos):
        """The two args are the value and tick position"""
        return "%d" % (x / 1000.)

    xyformatter = FuncFormatter(xycoords)

    def zcoords(x, pos):
        """The two args are the value and tick position"""
        return ("%.1f" % (x / 1000.)).rstrip('0').rstrip('.')

    zformatter = FuncFormatter(zcoords)

    ax_xy.xaxis.set_major_formatter(xyformatter)
    ax_xy.yaxis.set_major_formatter(xyformatter)
    ax_x.yaxis.set_major_formatter(zformatter)
    ax_y.xaxis.set_major_formatter(zformatter)

    if not title == "":
        # add a title - here, we have to create a new axes object which will
        # be invisible then the invisible axes will get a title
        tax = pl.axes((left, bottom + width + height + 0.01,
                       width + height, 0.01), frameon=False, axisbg="none")
        tax.get_xaxis().set_visible(False)
        tax.get_yaxis().set_visible(False)
        pl.title(title)
    if saveto == '':
        # show plot
        pl.show()
        if not pl.isinteractive():
            # close figure eplicitely if pylab is not in interactive mode
            pl.close()
    else:
        # save plot to file
        if (path.exists(path.dirname(saveto))) or (path.dirname(saveto) == ''):
            pl.savefig(saveto)
            pl.close()


def plot_max_plan_and_vert(x, y, z, data, unit="", title="",
                           saveto="", **kwargs):
    """Plot according to <plot_plan_and_vert> with the maximum values
    along the three axes of <data>

    Examples
    --------
    See :ref:`notebooks/workflow/recipe2.ipynb`.
    """
    plot_plan_and_vert(x, y, z, np.max(data, axis=-3), np.max(data, axis=-2),
                       np.max(data, axis=-1),
                       unit, title, saveto, **kwargs)


def plot_tseries(dtimes, data, ax=None, labels=None, datefmt='%b %d, %H:%M',
                 colors=None, ylabel="", title="", fontsize="medium",
                 saveto="", **kwargs):
    """Plot time series data (e.g. gage recordings)

    Parameters
    ----------
    dtimes : array of datetime objects (time steps)
    data : 2D array of shape ( num time steps, num data series )
    labels : list of strings (names of data series)
    title : string
    kwargs : keyword arguments related to :func:`matplotlib.pyplot.plot`

    """
    if ax is None:
        returnax = False
        fig = pl.figure()
        ax = fig.add_subplot(1, 1, 1, title=title)
    else:
        returnax = True
    # if labels==None:
    #    labels = ["series%d"%i for i in range(1, data.shape[1]+1)]
    # for i, label in enumerate(labels):
    #    ax.plot_date(mpl.dates.date2num(dtimes),data[:,i],label=label,
    #                 color=colors[i], **kwargs)
    ax.plot_date(mpl.dates.date2num(dtimes), data, **kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))
    pl.setp(ax.get_xticklabels(), visible=True)
    pl.setp(ax.get_xticklabels(), rotation=-30, horizontalalignment='left')
    ax.set_ylabel(ylabel, size=fontsize)
    ax = set_ticklabel_size(ax, fontsize)
    ax.legend(loc='best')

    if returnax:
        return ax

    if saveto == "":
        # show plot
        pl.show()
        if not pl.isinteractive():
            # close figure explicitely if pylab is not in interactive mode
            pl.close()
    else:
        # save plot to file
        if (path.exists(path.dirname(saveto))) or (path.dirname(saveto) == ''):
            pl.savefig(saveto)
            pl.close()


def set_ticklabel_size(ax, size):
    """
    """
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(size)
    return ax


def add_lines(ax, lines, **kwargs):
    """
    Add lines (points in the form Nx2) to axes

    Add lines (points in the form Nx2) to existing axes ax
    using :class:`matplotlib:matplotlib.collections.LineCollection`.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
    lines : nested :class:`numpy:numpy.ndarray` Nx2 array(s)
    kwargs : :class:`matplotlib:matplotlib.collections.LineCollection`

    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_overlay.ipynb`.
    """
    try:
        ax.add_collection(LineCollection([lines], **kwargs))
    except AssertionError:
        ax.add_collection(LineCollection([lines[None, ...]], **kwargs))
    except ValueError:
        for line in lines:
            add_lines(ax, line, **kwargs)


def add_patches(ax, patch_array, **kwargs):
    """
    Add patches (points in the form Nx2) to axes


    Add patches (points in the form Nx2) to existing axes ax
    using :class:`matplotlib:matplotlib.collections.PolyCollection`.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
    patch_array : nested :class:`numpy:numpy.ndarray` Nx2 array(s)
    kwargs : :class:`matplotlib:matplotlib.collections.PolyCollection`

    Examples
    --------
    See :ref:`notebooks/visualisation/wradlib_overlay.ipynb`.
    """

    try:
        ax.add_collection(PolyCollection([patch_array], **kwargs))
    except AssertionError:
        ax.add_collection(PolyCollection([patch_array[None, ...]], **kwargs))
    except ValueError:
        for patch in patch_array:
            add_patches(ax, patch, **kwargs)


if __name__ == '__main__':
    print('wradlib: Calling module <vis> as main...')
