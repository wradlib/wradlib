#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Visualisation
^^^^^^^^^^^^^

Standard plotting and mapping procedures.



.. autosummary::
   :nosignatures:
   :toctree: generated/

   plot_ppi
   plot_ppi_crosshair
   plot_rhi
   WradlibAccessor
   create_cg
   plot_scan_strategy
   plot_plan_and_vert
   plot_max_plan_and_vert
   add_lines
   add_patches


"""

# standard libraries
import os.path as path
import warnings
import re
import collections

# site packages
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches, axes, lines
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import (SubplotHost, ParasiteAxesAuxTrans,
                                     GridHelperCurveLinear)
import mpl_toolkits.axisartist.angle_helper as ah
from matplotlib.ticker import NullFormatter, FuncFormatter
from matplotlib.collections import LineCollection, PolyCollection
import xarray as xr
from osgeo import osr

# wradlib modules
from . import georef as georef
from .io.xarray import create_xarray_dataarray
from . util import import_optional


@xr.register_dataarray_accessor('wradlib')
class WradlibAccessor(object):
    """Dataarray Accessor for plotting radar moments
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._site = None
        self._proj = None
        if self._obj.sweep_mode in ['azimuth_surveillance', 'PPI']:
            self._mode = 'PPI'
        else:
            self. _mode = 'RHI'

        self.fix_cyclic()

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r'<.+>', '<{}>'.format(self.__class__.__name__),
                      str(self._obj))

    def fix_cyclic(self):
        rays = self._obj.azimuth
        if (360 - (rays[-1] - rays[0])) == (rays[1] - rays[0]):
            self._obj = xr.concat([self._obj, self._obj.isel(time=0)],
                                  dim='time')

    @property
    def site(self):
        if self._site is None:
            self._site = (self._obj.longitude.item(),
                          self._obj.latitude.item(),
                          self._obj.altitude.item())
        return self._site

    @property
    def mode(self):
        return self._mode

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, proj):
        self._proj = proj

    def contour(self, **kwargs):
        kwargs.setdefault('func', 'contour')
        return self.plot(**kwargs)

    def contourf(self, **kwargs):
        kwargs.setdefault('func', 'contourf')
        return self.plot(**kwargs)

    def pcolormesh(self, **kwargs):
        kwargs.setdefault('func', 'pcolormesh')
        return self.plot(**kwargs)

    def plot_rhi(self, **kwargs):
        return self.plot(**kwargs)

    def plot_ppi(self, **kwargs):
        return self.plot(**kwargs)

    def plot(self, ax=111, fig=None, proj=None,
             func='pcolormesh', cmap='viridis', center=False,
             add_colorbar=False, add_labels=False,
             **kwargs):
        """Plot Plan Position Indicator (PPI) or Range Height Indicator (RHI).

        The implementation of this plot routine is in cartesian axes and does
        all coordinate transforms using xarray machinery. This allows zooming
        into the data as well as making it easier to plot additional data
        (like gauge locations) without having to convert them to the radar's
        polar coordinate system.

        Using ``proj=cg`` the plotting is done in a curvelinear grid axes.

        Additional data can be plotted in polar coordinates or cartesian
        coordinates depending which axes object is used.

        ``**kwargs`` may be used to try to influence the
        :func:`matplotlib.pyplot.pcolormesh`,
        :func:`matplotlib.pyplot.contour`,
        :func:`matplotlib.pyplot.contourf` and
        :func:`wradlib.georef.polar.spherical_to_proj` routines under the hood.

        Parameters
        ----------
        proj : cartopy CRS | curvelinear grid dict | None
            cartopy CRS Coordinate Reference System describing projection
            If this parameter is not None, ``site`` must be set properly.
            Then the function will attempt to georeference the radar bins and
            display the PPI in the coordinate system defined by the
            projection string.
        fig : :class:`matplotlib:matplotlib.figure.Figure`
            If given, the PPI/RHI will be plotted into this figure object.
            Axes are created as needed. If None, a new figure object will be
            created or current figure will be used, depending on ``ax``.
        ax : :class:`matplotlib:matplotlib.axes.Axes` | matplotlib grid
        definition
            If matplotlib Axes object is given, the PPI will be plotted into
            this axes object.
            If matplotlib grid definition is given (nrows/ncols/plotnumber),
            axis are created in the specified place.
            Defaults to '111', only one subplot/axis.
        func : str
            Name of plotting function to be used under the hood.
            Defaults to 'pcolormesh'. 'contour' and 'contourf' can be
            selected too.
        cmap : str
            matplotlib colormap string

        Returns
        -------
        pm : :class:`matplotlib:matplotlib.collections.QuadMesh` | \
            :class:`matplotlib:matplotlib.contour.QuadContourSet`
            The result of the plotting function. Necessary, if you want to
            add a colorbar to the plot.

        Note
        ----
        If ``proj`` contains a curvelinear grid dict,
        the ``cgax`` - curvelinear Axes (r-theta-grid) is returned.
        ``caax`` - Cartesian Axes (x-y-grid) and ``paax`` -
        parasite axes object for plotting polar data can be derived like this::

            caax = cgax.parasites[0]
            paax = cgax.parasites[1]

        The function :func:`~wradlib.vis.create_cg` uses the
        Matplotlib AXISARTIST namespace `mpl_toolkits.axisartist`_.

        Here are some limitations to normal Matplotlib Axes. See
        `AxesGridToolkitUserGuide`_.

        Examples
        --------
        See :ref:`/notebooks/visualisation/wradlib_plot_ppi_example.ipynb`,
        and
        :ref:`/notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.

        .. _mpl_toolkits.axisartist:
            https://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
        .. _AxesGridToolkitUserGuide:
            https://matplotlib.org/mpl_toolkits/axes_grid/users/index.html
        """
        cg = False
        caax = None
        paax = None

        # fix for correct zorder of data and grid
        kwargs['zorder'] = kwargs.pop('zorder', 0)

        self.proj = proj

        # handle curvelinear grid properties
        if proj == 'cg' or isinstance(proj, collections.Mapping):
            self.proj = None
            if self.mode == 'PPI':
                cg = {'rot': -450, 'scale': -1}
            else:
                cg = {'rot': 0, 'scale': 1}
            if isinstance(proj, collections.Mapping):
                cg.update(proj)

        if isinstance(proj, osr.SpatialReference):
            raise TypeError(
                "WRADLIB: Currently GDAL OSR SRS are not supported")

        if isinstance(ax, axes.Axes):
            if cg:
                try:
                    caax = ax.parasites[0]
                    paax = ax.parasites[1]
                except AttributeError:
                    raise TypeError(
                        "WRADLIB: If `proj='cg'` `ax` need to be of type"
                        " `mpl_toolkits.axisartist.SubplotHost`")
        else:
            # axes object is given
            if fig is None:
                if ax == 111:
                    # create new figure if there is only one subplot
                    fig = pl.figure()
                else:
                    # assume current figure
                    fig = pl.gcf()
            if cg:
                # create curvelinear axes
                ax, caax, paax = create_cg(fig=fig, subplot=ax, **cg)
                # this is in fact the outermost thick "ring"
                rdiff = self._obj.range[1] - self._obj.range[0]
                ax.axis["lon"] = ax.new_floating_axis(1, (np.max(
                    self._obj.bins) + rdiff / 2.))
                ax.axis["lon"].major_ticklabels.set_visible(False)
                # and also set tickmarklength to zero for better presentation
                ax.axis["lon"].major_ticks.set_ticksize(0)
            else:
                ax = fig.add_subplot(ax, projection=self.proj)

        if cg:
            plax = paax
            infer_intervals = kwargs.pop('infer_intervals', False)
            xp, yp = 'rays', 'bins'
        else:
            plax = ax
            infer_intervals = kwargs.pop('infer_intervals', True)
            if self.mode == 'PPI':
                xp, yp = 'x', 'y'
            else:
                xp, yp = 'gr', 'z'

        # use cartopy, if available
        if hasattr(plax, 'projection'):
            cartopy = import_optional('cartopy')
            map_trans = cartopy.crs.AzimuthalEquidistant(
                central_longitude=self.site[0],
                central_latitude=self.site[1])
            kwargs.update({'transform': map_trans})

        # claim xarray plot function and create plot
        plotfunc = getattr(self._obj.plot, func)
        pm = plotfunc(x=xp, y=yp, ax=plax, cmap=cmap, center=center,
                      add_colorbar=add_colorbar, add_labels=add_labels,
                      infer_intervals=infer_intervals, **kwargs)

        # set cg grids and limits
        if cg:
            if self.mode == 'PPI':
                xlims = np.min(self._obj.x), np.max(self._obj.x)
                ylims = np.min(self._obj.y), np.max(self._obj.y)
            else:
                xlims = np.min(self._obj.gr), np.max(self._obj.gr)
                ylims = np.min(self._obj.z), np.max(self._obj.z)
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.grid(True)
            caax.grid(True)

        if self.mode == 'PPI':
            ax.set_aspect('equal', adjustable='box')

        # set ax as current
        pl.sca(ax)

        return pm


def plot_ppi(data, r=None, az=None, elev=0., site=None, proj=None,
             fig=None, ax=111, func='pcolormesh', rf=1.,
             **kwargs):
    """Plots a Plan Position Indicator (PPI).

    This is a small wrapper around xarray dataarray.
    The radar data, coordinates and metadata is transformed into an
    xarray dataarray. Using the wradlib dataarray accessor the dataarray is
    enabled to plot polar data.

    Using ``proj=cg`` the plotting is done in a curvelinear grid axes.
    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending which axes object is used.

    ``**kwargs`` may be used to try to influence the
    :func:`matplotlib.pyplot.pcolormesh`, :func:`matplotlib.pyplot.contour`,
    :func:`matplotlib.pyplot.contourf` and
    :func:`wradlib.georef.polar.spherical_to_proj` routines under the hood.

    Concerning the values of ``r``, ``az``, ``elev``, ``r`` should
    give the location of the center of each range bin, ``az`` and
    ``elev`` should give the angle at the center of the beam.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        The data to be plotted. It is assumed that the first dimension is over
        the azimuth angles, while the second dimension is over the range bins
    r : :class:`numpy:numpy.ndarray`
        The ranges. Units may be chosen arbitrarily, unless proj is set. In
        that case the units must be meters. If None, a default is
        calculated from the dimensions of ``data``.
    rf: float
        If present, factor for scaling range axes, defaults to 1.
    az : :class:`numpy:numpy.ndarray`
        The azimuth angles in degrees. If None, a default is
        calculated from the dimensions of ``data``.
    elev : float or array of same shape as ``az``
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
    site : tuple or None
        Tuple of coordinates of the radar site.
        If ``proj`` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If ``proj`` is used, values must be given as (longitude, latitude,
        altitude) tuple of geographical coordinates.
        Defaults to None.
    proj : GDAL OSR SRS | cartopy CRS | curvelinear grid dict | None
        GDAL OSR Spatial Reference Object describing projection
        If this parameter is not None, ``site`` must be set. Then the function
        will attempt to georeference the radar bins and display the PPI in the
        coordinate system defined by the projection string.
    fig : :class:`matplotlib:matplotlib.figure.Figure`
        If given, the PPI will be plotted into this figure object. Axes are
        created as needed. If None, a new figure object will be created or
        current figure will be used, depending on ``ax``.
    ax : :class:`matplotlib:matplotlib.axes.Axes` | matplotlib grid definition
        If matplotlib Axes object is given, the PPI will be plotted into this
        axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    func : str
        Name of plotting function to be used under the hood.
        Defaults to 'pcolormesh'. 'contour' and 'contourf' can be selected too.
    cg : bool
        If True, the data will be plotted on curvelinear axes.
        Deprecated, use `proj='cg'`.
    autoext : bool
        Deprecated.
    refrac : bool
        Deprecated.

    See also
    --------
    :func:`wradlib.georef.projection.reproject`
    :func:`wradlib.georef.projection.create_osr`

    Returns
    -------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
        The axes object into which the PPI was plotted
    pm : :class:`matplotlib:matplotlib.collections.QuadMesh` | \
        :class:`matplotlib:matplotlib.contour.QuadContourSet`
        The result of the plotting function. Necessary, if you want to
        add a colorbar to the plot.

    Note
    ----
    If ``cg`` is True, the ``cgax`` - curvelinear Axes (r-theta-grid)
    is returned. ``caax`` - Cartesian Axes (x-y-grid) and ``paax`` -
    parasite axes object for plotting polar data can be derived like this::

        caax = cgax.parasites[0]
        paax = cgax.parasites[1]

    The function :func:`~wradlib.vis.create_cg` uses the
    Matplotlib AXISARTIST namespace `mpl_toolkits.axisartist`_.

    Here are some limitations to normal Matplotlib Axes. See
    `AxesGridToolkitUserGuide`_.

    Examples
    --------
    See :ref:`/notebooks/visualisation/wradlib_plot_ppi_example.ipynb`,
    and
    :ref:`/notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.

    .. _mpl_toolkits.axisartist:
        https://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
    .. _AxesGridToolkitUserGuide:
        https://matplotlib.org/mpl_toolkits/axes_grid/users/index.html
    """
    # DeprecationChecks
    autoext = kwargs.pop('autoext', None)
    if autoext is not None:
        warnings.warn("`autoext` keyword is deprecated will be removed in "
                      "future release. The functionality is now handled by "
                      "`xarray` DataArray automatically.",
                      DeprecationWarning)
        if autoext is False and r is not None:
            r = r[:-1]
        if autoext is False and az is not None:
            az = az[:-1]

    refrac = kwargs.pop('refrac', None)
    if refrac is not None:
        warnings.warn("`refrac` keyword is deprecated will be removed in "
                      "future release. Please use `re`/`ke` keywords.",
                      DeprecationWarning)

    # Check can be removed in release 1.4
    cg = kwargs.pop('cg', None)
    if cg is not None:
        warnings.warn("`cg` keyword is deprecated and will be removed in "
                      "future release. Use `proj='cg' instead.",
                      DeprecationWarning)
        if cg:
            if proj:
                warnings.warn("`cg` cannot be used with `proj`, falling back.")
            else:
                proj = 'cg'

    if site and len(site) < 3:
        warnings.warn("`site` need to be a tuple of coordinates "
                      "(longitude, latitude, altitude)."
                      "Use of `site=(longitude, latitude)` will raise an "
                      "error in future releases.", DeprecationWarning)
        site = (*site, 0)

    # site must be given, if proj is OSR
    if isinstance(proj, osr.SpatialReference) and site is None:
        raise TypeError("WRADLIB: If `proj` is Spatial Reference System "
                        "(GDAL OSR SRS) site need to be given "
                        "as tuple of (longitude, latitude, altitude)")

    # site given without proj
    if site and not proj:
        warnings.warn(
            "WRADLIB: site is given without `proj`, it will be used "
            "as simple xy-offset")

    # re/ke kwargs handling
    kw_spherical = {'re': kwargs.pop('re', None),
                    'ke': kwargs.pop('ke', 4. / 3.)}

    if az is None:
        az = np.arange(data.shape[0], dtype=np.float)
        az += (az[1] - az[0]) / 2.

    if r is None:
        if proj and proj != 'cg':
            warnings.warn("Parameter `r` is None, falling back to `proj=None`."
                          "If using projection, r must be given as "
                          "array with units m.")
            proj = None
        r = np.arange(data.shape[1], dtype=np.float)
        r += (r[1] - r[0]) / 2.

    if np.isscalar(elev):
        elev = np.ones_like(az) * elev

    da = create_xarray_dataarray(data, r=r, phi=az, theta=elev, site=site,
                                 proj=proj, sweep_mode='PPI', rf=rf,
                                 **kw_spherical)

    # fallback to proj=None for GDAL OSR
    if isinstance(proj, osr.SpatialReference):
        proj = None

    pm = da.wradlib.plot_ppi(ax=ax, fig=fig, func=func,
                             proj=proj, **kwargs)

    return pl.gca(), pm


def plot_ppi_crosshair(site, ranges, angles=None,
                       proj=None, elev=0., ax=None, **kwargs):
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
        If ``proj`` is None arbitrary units may be used (such that they fit
        with the underlying PPI plot.
        Otherwise the ranges must be given in meters.
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
    ax : :class:`matplotlib:matplotlib.axes.Axes`
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
    :func:`~wradlib.vis.plot_ppi` - plotting a PPI in cartesian coordinates

    Returns
    -------
    ax :  :class:`matplotlib:matplotlib.axes.Axes`
        The axes object into which the PPI was plotted

    Examples
    --------
    See :ref:`/notebooks/visualisation/wradlib_plot_ppi_example.ipynb`.

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
        # convert from spherical to projection
        coords = georef.spherical_to_proj(rr, az, elev, site, proj=proj)
        nsewx = coords[..., 0]
        nsewy = coords[..., 1]
    else:
        # no projection
        psite = site
        rr, az = np.meshgrid(np.linspace(0, ranges[-1], 2), angles)
        # use simple trigonometry to calculate coordinates
        nsewx, nsewy = (psite[0] + rr * np.cos(np.radians(90 - az)),
                        psite[1] + rr * np.sin(np.radians(90 - az)))

    # mark the site, just in case nothing else would be drawn
    ax.plot(*psite[:2], marker='+', **linekw)

    # draw the lines
    for i in range(len(angles)):
        ax.add_line(lines.Line2D(nsewx[i, :], nsewy[i, :], **linekw))

    # draw the range circles
    if proj:
        # produce an approximation of the circle
        x, y = np.meshgrid(ranges, np.arange(360))
        poly = georef.spherical_to_proj(ranges, np.arange(360), elev, site,
                                        proj=proj)[..., :2]
        poly = np.swapaxes(poly, 0, 1)
        for p in poly:
            ax.add_patch(patches.Polygon(p, **circkw))
    else:
        # in the unprojected case, we may use 'true' circles.
        for r in ranges:
            ax.add_patch(patches.Circle(psite, r, **circkw))

    # there should be not much wrong, setting the axes aspect to equal
    # by default
    ax.set_aspect('equal')

    # return the axes object for later use
    return ax


def plot_rhi(data, r=None, th=None, th_res=None, az=0, site=None,
             proj=None, rf=1., fig=None, ax=111, **kwargs):
    """Plots a Range Height Indicator (RHI).

    This is a small wrapper around xarray dataarray.

    The radar data, coordinates and metadata is transformed into an
    xarray dataarray. Using the wradlib dataarray accessor the dataarray is
    enabled to plot polar data.

    Using ``cg=True`` the plotting is done in a curvelinear grid axes.
    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending which axes object is used.

    ``**kwargs`` may be used to try to influence the
    :func:`matplotlib.pyplot.pcolormesh`, :func:`matplotlib.pyplot.contour`
    and :func:`matplotlib.pyplot.contourf` routines under the hood.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        The data to be plotted. It is assumed that the first dimension is over
        the elevation angles, while the second dimension is over the range bins
    r : :class:`numpy:numpy.ndarray`
        The ranges. Units may be chosen arbitrarily. If None, a default is
        calculated from the dimensions of ``data``.
    rf: float
        If present, factor for scaling range axis, defaults to 1.
    th : :class:`numpy:numpy.ndarray`
        The elevation angles in degrees. If None, a default is
        calculated from the dimensions of ``data``.
    th_res : float or :class:`numpy:numpy.ndarray` of same shape as ``th``
        In RHI's it happens that the elevation angles are spaced wider than
        the beam width. If this beam width (in degrees) is given in ``th_res``,
        plot_rhi will plot the beams accordingly. Otherwise the behavior of
        :func:`matplotlib.pyplot.pcolormesh` assumes all beams to be adjacent
        to each other, which might lead to unexpected results.
    az : float or :class:`numpy:numpy.ndarray` of same shape as ``th``
    site : tuple
        Tuple of coordinates of the radar site.
        If ``proj`` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If ``proj`` is used, values must be given as (longitude, latitude,
        altitude)
        tuple of geographical coordinates.
    proj : osr spatial reference object
        GDAL OSR Spatial Reference Object describing projection
        If this parameter is not None, ``site`` must be set. Then the function
        will attempt to georeference the radar bins in the
        coordinate system defined by the projection string.
    fig : :class:`matplotlib:matplotlib.figure.Figure`
        If given, the RHI will be plotted into this figure object. Axes are
        created as needed. If None, a new figure object will be created or
        current figure will be used, depending on ``ax``.
    ax : :class:`matplotlib:matplotlib.axes.Axes` | matplotlib grid definition
        If matplotlib Axes object is given, the RHI will be plotted into this
        axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    func : str
        Name of plotting function to be used under the hood.
        Defaults to 'pcolormesh'. 'contour' and 'contourf' can be selected too.
    cg : bool
        If True, the data will be plotted on curvelinear axes.
        Deprecated, use `proj='cg'`.
    autoext : bool
        Deprecated.
    refrac : bool
        Deprecated.

    See also
    --------
    :func:`wradlib.vis.create_cg` : creation of curvelinear grid axes objects

    Returns
    -------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
        The axes object into which the RHI was plotted.
    pm : :class:`matplotlib:matplotlib.collections.QuadMesh` | \
        :class:`matplotlib:matplotlib.contour.QuadContourSet`
        The result of the plotting function. Necessary, if you want to
        add a colorbar to the plot.

    Note
    ----
    If ``cg`` is True, the ``cgax`` - curvelinear Axes (r-theta-grid)
    is returned. ``caax`` - Cartesian Axes (x-y-grid) and ``paax`` -
    parasite axes object for plotting polar data can be derived like this::

        caax = cgax.parasites[0]
        paax = cgax.parasites[1]

    The function :func:`~wradlib.vis.create_cg` uses the
    Matplotlib AXISARTIST namespace `mpl_toolkits.axisartist`_.

    Here are some limitations to normal Matplotlib Axes. See
    `AxesGridToolkitUserGuide`_.

    Examples
    --------
    See :ref:`/notebooks/visualisation/wradlib_plot_curvelinear_grids.ipynb`.

    .. _mpl_toolkits.axisartist:
        https://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
    .. _AxesGridToolkitUserGuide:
        https://matplotlib.org/mpl_toolkits/axes_grid/users/index.html
    """
    # DeprecationWarnings
    autoext = kwargs.pop('autoext', None)
    if autoext is not None:
        warnings.warn("`autoext` keyword is deprecated will be removed in "
                      "future release. The functionality is now handled by "
                      "`xarray` DataArray automatically.",
                      DeprecationWarning)

    refrac = kwargs.pop('refrac', None)
    if refrac is not None:
        warnings.warn("`refrac` keyword is deprecated will be removed in "
                      "future release. Please use `re`/`ke` keywords.",
                      DeprecationWarning)

    # Check can be removed in release 1.4
    cg = kwargs.pop('cg', None)
    if cg is not None:
        warnings.warn("`cg` keyword is deprecated and will be removed in "
                      "future release. Use `proj='cg' instead.",
                      DeprecationWarning)
        if cg:
            if proj:
                warnings.warn(
                    "`cg` cannot be used with `proj`, falling back.")
            else:
                proj = 'cg'

    # kwargs handling
    kwargs['zorder'] = kwargs.pop('zorder', 0)
    func = kwargs.pop('func', 'pcolormesh')

    # re/ke kwargs handling
    kw_spherical = {'re': kwargs.pop('re', None),
                    'ke': kwargs.pop('ke', 4. / 3.)}

    if th is None:
        th = np.linspace(0., 90., num=data.shape[0], endpoint=True)
        th += (th[1] - th[0]) / 2.

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
        yl = th - th_res * 0.5
        yu = th + th_res * 0.5
        # glue them together to achieve the proper dimensions for the
        # interlaced array
        th = np.concatenate([yl[None, :], yu[None, :]], axis=0).T.ravel()
    else:
        img = data

    if r is None:
        if proj and proj != 'cg':
            warnings.warn("Parameter `r` is None, falling back to `proj=None`."
                          "If using projection, r must be given as "
                          "array with units m.")
            proj = None
        r = np.arange(data.shape[1], dtype=np.float)
        r += (r[1] - r[0]) / 2.

    if np.isscalar(az):
        az = np.ones_like(th) * az

    da = create_xarray_dataarray(img, r=r, phi=az, theta=th, site=site,
                                 proj=proj, sweep_mode='RHI', rf=rf,
                                 **kw_spherical)

    # fallback to proj=None for GDAL OSR
    if isinstance(proj, osr.SpatialReference):
        proj = None

    pm = da.wradlib.plot_rhi(ax=ax, fig=fig, func=func, proj=proj,
                             **kwargs)

    return pl.gca(), pm


def create_cg(st=None, fig=None, subplot=111, rot=-450, scale=-1,
              angular_spacing=10, radial_spacing=10,
              latmin=0, lon_cycle=360):
    """ Helper function to create curvelinear grid

    The function makes use of the Matplotlib AXISARTIST namespace
    `mpl_toolkits.axisartist \
    <https://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid Toolkit \
    <https://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
    most of the limitations can be overcome.
    See `Matplotlib AxesGrid Toolkit User’s Guide \
    <https://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_.

    Parameters
    ----------
    fig : matplotlib Figure object
        If given, the PPI/RHI will be plotted into this figure object.
        Axes are created as needed. If None a new figure object will
        be created or current figure will be used, depending on "subplot".
    subplot : :class:`matplotlib:matplotlib.gridspec.GridSpec`, \
        matplotlib grid definition
        nrows/ncols/plotnumber, see examples section
        defaults to '111', only one subplot
    rot : float
        Rotation of the source data in degrees, defaults to -450 for PPI,
        use 0 for RHI
    scale : float
        Scale of source data, defaults to -1. for PPI, use 1 for RHI
    angular_spacing : float
        Spacing of the angular grid, defaults to 10.
    radial_spacing : float
        Spacing of the radial grid, defaults to 10.
    latmin : float
        Startvalue for radial grid, defaults to 0.
    lon_cycle : float
        Angular cycle, defaults to 360.

    Returns
    -------
    cgax : matplotlib toolkit axisartist Axes object
        curvelinear Axes (r-theta-grid)
    caax : matplotlib Axes object (twin to cgax)
        Cartesian Axes (x-y-grid) for plotting cartesian data
    paax : matplotlib Axes object (parasite to cgax)
        The parasite axes object for plotting polar data
    """

    if st is not None:
        warnings.warn("ScanType string is deprecated and will be removed in "
                      "future release. Please use `rot` and `scale` keyword "
                      "arguments to specify PPI/RHI. ",
                      DeprecationWarning)
        if st == 'RHI':
            rot = 0
            scale = 1

    # create transformation
    # rotate
    tr_rotate = Affine2D().translate(rot, 0)
    # scale
    tr_scale = Affine2D().scale(scale * np.pi / 180, 1)
    # polar
    tr_polar = PolarAxes.PolarTransform()

    tr = tr_rotate + tr_scale + tr_polar

    # build up curvelinear grid
    extreme_finder = ah.ExtremeFinderCycle(360, 360,
                                           lon_cycle=lon_cycle,
                                           lat_cycle=None,
                                           lon_minmax=None,
                                           lat_minmax=(latmin, np.inf),
                                           )
    # locator and formatter for angular annotation
    grid_locator1 = ah.LocatorDMS(lon_cycle // angular_spacing)
    tick_formatter1 = ah.FormatterDMS()

    # grid_helper for curvelinear grid
    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        grid_locator2=None,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    # try to set nice locations for radial gridlines
    grid_locator2 = grid_helper.grid_finder.grid_locator2
    grid_locator2._nbins = (radial_spacing * 2 + 1) // np.sqrt(2)

    # if there is no figure object given
    if fig is None:
        # create new figure if there is only one subplot
        if subplot == 111:
            fig = pl.figure()
        # otherwise get current figure or create new figure
        else:
            fig = pl.gcf()

    # generate Axis
    cgax = SubplotHost(fig, subplot, grid_helper=grid_helper)
    fig.add_axes(cgax)

    # get twin axis for cartesian grid
    caax = cgax.twin()
    # move axis annotation from right to left and top to bottom for
    # cartesian axis
    caax.toggle_axisline()

    # make right and top axis visible and show ticklabels (curvelinear axis)
    cgax.axis["top", "right"].set_visible(True)
    cgax.axis["top", "right"].major_ticklabels.set_visible(True)

    # make ticklabels of left and bottom axis invisible (curvelinear axis)
    cgax.axis["left", "bottom"].major_ticklabels.set_visible(False)

    # and also set tickmarklength to zero for better presentation
    # (curvelinear axis)
    cgax.axis["top", "right", "left", "bottom"].major_ticks.set_ticksize(0)

    # show theta (angles) on top and right axis
    cgax.axis["top"].get_helper().nth_coord_ticks = 0
    cgax.axis["right"].get_helper().nth_coord_ticks = 0

    # generate and add parasite axes with given transform
    paax = ParasiteAxesAuxTrans(cgax, tr, "equal")
    # note that paax.transData == tr + cgax.transData
    # Anything you draw in paax will match the ticks and grids of cgax.
    cgax.parasites.append(paax)

    return cgax, caax, paax


def plot_scan_strategy(ranges, elevs, site, vert_res=500.,
                       maxalt=10000., ax=None):
    """Plot the vertical scanning strategy

    Parameters
    ----------
    ranges : array of ranges
    elevs : array of elevation angles
    site : tuple of site coordinates (longitude, latitude, altitude)
    vert_res : float
        Vertical resolution in [m]
    maxalt : float
        Maximum altitude in [m]
    ax : :class:`matplotlib:matplotlib.axes.Axes`
        The axes object to be plotted to.
    """
    # just a dummy
    az = np.array([90.])
    coords, _ = georef.spherical_to_xyz(ranges, az, elevs, site)
    alt = coords[..., 2]
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
        ax.plot(ranges, alt[i, :], lw=2, color="black")
    pl.ylim(ymax=maxalt)
    ax.tick_params(labelsize="large")
    pl.xlabel("Range (m)", size="large")
    pl.ylabel("Height over radar (m)", size="large")
    for i, elev in enumerate(elevs):
        x = ranges[-1] + 1500.
        y = alt[i, :][-1]
        if y > maxalt:
            ix = np.where(alt[i, :] < maxalt)[0][-1]
            x = ranges[ix]
            y = maxalt + 100.
        pl.text(x, y, str(elev), fontsize="large")

    if returnax:
        return ax
    pl.show()


def plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, unit="",
                       title="", saveto="", **kwargs):
    """Plot 2-D plan view of ``dataxy`` together with vertical sections \
    ``dataxz`` and ``datazy``

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        array of x-axis coordinates
    y : :class:`numpy:numpy.ndarray`
        array of y-axis coordinates
    z : :class:`numpy:numpy.ndarray`
        array of z-axis coordinates
    dataxy : :class:`numpy:numpy.ndarray`
        2d array of shape (len(x), len(y))
    datazx : :class:`numpy:numpy.ndarray`
        2d array of shape (len(z), len(x))
    datazy : :class:`numpy:numpy.ndarray`
        2d array of shape (len(z), len(y))
    unit : string
        unit of data arrays
    title: string
        figure title
    saveto : string
        file path if figure should be saved

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
                       width + height, 0.01), frameon=False, facecolor="none")
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
    along the three axes of ``data``

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.
    """
    plot_plan_and_vert(x, y, z, np.max(data, axis=-3), np.max(data, axis=-2),
                       np.max(data, axis=-1),
                       unit, title, saveto, **kwargs)


def add_lines(ax, lines, **kwargs):
    """Add lines (points in the form Nx2) to axes

    Add lines (points in the form Nx2) to existing axes ax
    using :class:`matplotlib:matplotlib.collections.LineCollection`.

    Parameters
    ----------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
    lines : :class:`numpy:numpy.ndarray`
        nested Nx2 array(s)
    kwargs : :class:`matplotlib:matplotlib.collections.LineCollection`

    Examples
    --------
    See :ref:`/notebooks/visualisation/wradlib_overlay.ipynb`.
    """
    try:
        ax.add_collection(LineCollection([lines], **kwargs))
    except AssertionError:
        ax.add_collection(LineCollection([lines[None, ...]], **kwargs))
    except ValueError:
        for line in lines:
            add_lines(ax, line, **kwargs)


def add_patches(ax, patch_array, **kwargs):
    """Add patches (points in the form Nx2) to axes

    Add patches (points in the form Nx2) to existing axes ax
    using :class:`matplotlib:matplotlib.collections.PolyCollection`.

    Parameters
    ----------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
        the axes object to plot on
    patch_array : :class:`numpy:numpy.ndarray`
        nested Nx2 array(s)
    kwargs : :class:`matplotlib:matplotlib.collections.PolyCollection`

    Examples
    --------
    See :ref:`/notebooks/visualisation/wradlib_overlay.ipynb`.
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
