#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Visualisation
^^^^^^^^^^^^^

Standard plotting and mapping procedures.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "plot",
    "plot_ppi_crosshair",
    "create_cg",
    "plot_scan_strategy",
    "plot_plan_and_vert",
    "plot_max_plan_and_vert",
    "add_lines",
    "add_patches",
    "VisMethods",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import collections

import numpy as np
from pyproj.crs import CRS
from xradar.georeference import get_crs

from wradlib import georef, io, ipol, util

plt = util.import_optional("matplotlib.pyplot")
axes = util.import_optional("matplotlib.axes")
lines = util.import_optional("matplotlib.lines")
patches = util.import_optional("matplotlib.patches")
coll = util.import_optional("matplotlib.collections")
mpl_proj = util.import_optional("matplotlib.projections")
tick = util.import_optional("matplotlib.ticker")
trans = util.import_optional("matplotlib.transforms")
axisartist = util.import_optional("mpl_toolkits.axisartist")
angle_helper = util.import_optional("mpl_toolkits.axisartist.angle_helper")
osr = util.import_optional("osgeo.osr")
cartopy = util.import_optional("cartopy")
cmweather = util.import_optional("cmweather")

if util.has_import(cmweather):
    wrl_cmap = "HomeyerRainbow"
else:
    wrl_cmap = "turbo"


def plot_ppi_crosshair(
    site, ranges, angles=None, crs=None, elev=0.0, ax=None, **kwargs
):
    """Plots a Crosshair for a Plan Position Indicator (PPI).

    Parameters
    ----------
    site : tuple
        Tuple of coordinates of the radar site.
        If `crs` is not used, this simply becomes the offset for the origin
        of the coordinate system.
        If `crs` is used, values must be given as (longitude, latitude,
        altitude) tuple of geographical coordinates.
    ranges : list
        List of ranges, for which range circles should be drawn.
        If ``crs`` is None arbitrary units may be used (such that they fit
        with the underlying PPI plot).
        Otherwise the ranges must be given in meters.
    angles : list, optional
        List of angles (in degrees) for which straight lines should be drawn.
        These lines will be drawn starting from the center and until the
        largest range. Defaults to [0, 90, 180, 270].
    crs : :py:class:`gdal:osgeo.osr.SpatialReference`
        GDAL OSR Spatial Reference Object describing projection
        The function will calculate lines and circles according to
        georeferenced coordinates taking beam propagation, earth's curvature
        and scale effects due to projection into account.
        Depending on the projection, crosshair lines might not be straight and
        range circles might appear elliptical (also check if the aspect of the
        axes might not also be responsible for this).
    elev : float or :class:`numpy:numpy.ndarray`
        float or array of same shape as az
        Elevation angle of the scan or individual azimuths.
        May improve georeferencing coordinates for larger elevation angles.
        Defaults to 0.
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
    See :ref:`/notebooks/visualisation/plot_ppi.ipynb`.

    """
    # check coordinate tuple
    if site and len(site) < 3:
        raise ValueError(
            "`site` need to be a sequence of coordinates "
            "`longitude`, `latitude`, `altitude`."
        )

    # if we didn't get an axes object, find the current one
    if ax is None:
        ax = plt.gca()

    if angles is None:
        angles = [0, 90, 180, 270]

    # set default line keywords
    linekw = dict(color="gray", linestyle="dashed")
    # update with user settings
    linekw.update(kwargs.get("line", {}))

    # set default circle keywords
    circkw = dict(edgecolor="gray", linestyle="dashed", facecolor="none")
    # update with user settings
    circkw.update(kwargs.get("circle", {}))

    # determine coordinates for 'straight' lines
    if crs:
        # projected
        # reproject the site coordinates
        psite = georef.reproject(*site, trg_crs=crs)
        # these lines might not be straight, so we approximate them with 10
        # segments. Produce polar coordinates
        rr, az = np.meshgrid(np.linspace(0, ranges[-1], 10), angles)
        # convert from spherical to projection
        coords = georef.spherical_to_proj(rr, az, elev, site, crs=crs)
        nsewx = coords[..., 0]
        nsewy = coords[..., 1]
    else:
        # no projection
        psite = site
        rr, az = np.meshgrid(np.linspace(0, ranges[-1], 2), angles)
        # use simple trigonometry to calculate coordinates
        nsewx, nsewy = (
            psite[0] + rr * np.cos(np.radians(90 - az)),
            psite[1] + rr * np.sin(np.radians(90 - az)),
        )

    # mark the site, just in case nothing else would be drawn
    ax.plot(*psite[:2], marker="+", **linekw)

    # draw the lines
    for i in range(len(angles)):
        ax.add_line(lines.Line2D(nsewx[i, :], nsewy[i, :], **linekw))

    # draw the range circles
    if crs:
        # produce an approximation of the circle
        x, y = np.meshgrid(ranges, np.arange(360))
        poly = georef.spherical_to_proj(ranges, np.arange(360), elev, site, crs=crs)[
            ..., :2
        ]
        poly = np.swapaxes(poly, 0, 1)
        for p in poly:
            ax.add_patch(patches.Polygon(p, **circkw))
    else:
        # in the unprojected case, we may use 'true' circles.
        for r in ranges:
            ax.add_patch(patches.Circle(psite, r, **circkw))

    # there should be not much wrong, setting the axes aspect to equal
    # by default
    ax.set_aspect("equal")

    # return the axes object for later use
    return ax


def create_cg(
    *,
    fig=None,
    subplot=111,
    rot=-450,
    scale=-1,
    angular_spacing=10,
    radial_spacing=10,
    latmin=0,
    lon_cycle=360,
):
    """ Helper function to create curvelinear grid

    The function makes use of the Matplotlib
    `AXISARTIST <https://matplotlib.org/stable/api/toolkits/axisartist.html>`_ toolkit.

    Here are some limitations to normal Matplotlib Axes. While using the
    Matplotlib `AxesGrid1 Toolkit \
    <https://matplotlib.org/stable/api/toolkits/axes_grid1.html>`_
    most of the limitations can be overcome.
    See `Overview of axes_grid1 toolkit \
    <https://matplotlib.org/stable/tutorials/toolkits/axes_grid.html>`_.

    Parameters
    ----------
    fig : :py:class:`matplotlib:matplotlib.figure.Figure`
        If given, the PPI/RHI will be plotted into this figure object.
        Axes are created as needed. If None a new figure object will
        be created or current figure will be used, depending on "subplot".
    subplot : :class:`matplotlib:matplotlib.gridspec.SubplotSpec`
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
    cgax : :py:class:`matplotlib:mpl_toolkits.axisartist.axis_artist.AxisArtist`
        curvelinear Axes (r-theta-grid)
    caax : :py:class:`matplotlib:matplotlib.axes.Axes`
        matplotlib Axes object (twin to cgax)
        Cartesian Axes (x-y-grid) for plotting cartesian data
    paax : :py:class:`matplotlib:matplotlib.axes.Axes`
        matplotlib Axes object (parasite to cgax)
        The parasite axes object for plotting polar data
    """
    # create transformation
    # rotate
    tr_rotate = trans.Affine2D().translate(rot, 0)
    # scale
    tr_scale = trans.Affine2D().scale(scale * np.pi / 180, 1)
    # polar
    tr_polar = mpl_proj.PolarAxes.PolarTransform()

    tr = tr_rotate + tr_scale + tr_polar

    # build up curvelinear grid
    extreme_finder = angle_helper.ExtremeFinderCycle(
        360,
        360,
        lon_cycle=lon_cycle,
        lat_cycle=None,
        lon_minmax=None,
        lat_minmax=(latmin, np.inf),
    )
    # locator and formatter for angular annotation
    grid_locator1 = angle_helper.LocatorDMS(lon_cycle // angular_spacing)
    tick_formatter1 = angle_helper.FormatterDMS()

    # grid_helper for curvelinear grid
    grid_helper = axisartist.GridHelperCurveLinear(
        tr,
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
            fig = plt.figure()
        # otherwise get current figure or create new figure
        else:
            fig = plt.gcf()

    # generate Axis
    cgax = fig.add_subplot(
        subplot, axes_class=axisartist.HostAxes, grid_helper=grid_helper
    )

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
    paax = cgax.get_aux_axes(tr, "equal")
    # note that paax.transData == tr + cgax.transData
    # Anything you draw in paax will match the ticks and grids of cgax.
    cgax.parasites.append(paax)

    return cgax, caax, paax


def _height_formatter(x, pos, *, cg=False, scale=1.0, er=6371000.0):
    if not cg:
        er = 0
    x = (x - er) / scale
    fmt_str = f"{x:g}"
    return fmt_str


def _range_formatter(x, pos, *, scale=1.0):
    x = x / scale
    fmt_str = f"{x:g}"
    return fmt_str


def _plot_beam(r, alt, beamradius, *, ax=None, label=None):
    """Plot single beam on ax"""
    if label is None:
        label = ""
    if ax is None:
        ax = plt.gca()
    center = ax.plot(r, alt, "-k", linewidth=0.5, alpha=1.0, label="_Center", zorder=1)
    edge = ax.plot(
        r, (alt + beamradius), ":k", linewidth=0.5, alpha=1.0, label="_Edge", zorder=1
    )
    ax.plot(r, (alt - beamradius), ":k", linewidth=0.2, alpha=1.0, zorder=1)
    fill = ax.fill_between(
        r, (alt - beamradius), (alt + beamradius), label=label, alpha=0.45, zorder=1
    )
    return fill, center, edge


def plot_scan_strategy(
    ranges,
    elevs,
    site,
    *,
    beamwidth=1.0,
    vert_res=500.0,
    maxalt=10000.0,
    range_res=None,
    maxrange=None,
    units="m",
    terrain=None,
    az=0.0,
    cg=False,
    ax=111,
    cmap="tab10",
):
    """Plot the vertical scanning strategy.

    Parameters
    ----------
    ranges : sequence of float or :class:`numpy:numpy.ndarray`
        sequence or array of float ranges
    elevs : sequence of float or :class:`numpy:numpy.ndarray`
        elevation angles
    site : sequence of tuple or :class:`numpy:numpy.ndarray`
        radar site coordinates (longitude, latitude, altitude)
    beamwidth : float
        3dB width of the radar beam, defaults to 1.0 deg.
    vert_res : float
        Vertical resolution in [m].
    maxalt : float
        Maximum altitude in [m].
    range_res : float
        Horizontal resolution in [m].
    maxrange : float
        Maximum range in [m].
    units : str
        Units to plot in, can be 'm' or 'km'. Defaults to 'm'.
    terrain : bool or :class:`numpy:numpy.ndarray`
        If True, downloads srtm data and add orography for given `az`.
    az : float
        Used to specify azimuth for terrain plots.
    cg : bool
        If True, plot in curvelinear grid, defaults to False (cartesian grid).
    ax : :class:`matplotlib:matplotlib.axes.Axes` or :class:`matplotlib:matplotlib.gridspec.SubplotSpec`
        If matplotlib Axes object is given, the scan strategy will be plotted into this
        axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    cmap : str
        matplotlib colormap string.

    Returns
    -------
    ax : :class:`matplotlib:matplotlib.axes.Axes`
        matplotlib Axes or curvelinear Axes (matplotlib toolkit axisartist Axes object,
        r-theta-grid) depending on keyword argument `cg`.
    """

    if units == "m":
        scale = 1.0
    elif units == "km":
        scale = 1000.0
    else:
        raise ValueError(f"Unknown value for `units`.kwarg {units!r}")

    az = np.array([az])

    if maxrange is None:
        maxrange = ranges.max()

    xyz, rad = georef.spherical_to_xyz(ranges, az, elevs, site, squeeze=True)

    add_title = ""
    if terrain is True:
        add_title += f" - Azimuth {az[0]}°"
        ll = georef.reproject(xyz, src_crs=rad)
        # (down-)load srtm data
        ds = io.get_srtm(
            [ll[..., 0].min(), ll[..., 0].max(), ll[..., 1].min(), ll[..., 1].max()],
        )
        rastervalues, rastercoords, crs = georef.extract_raster_dataset(
            ds, nodata=-32768.0
        )
        # map rastervalues to polar grid points
        terrain = ipol.cart_to_irregular_spline(
            rastercoords, rastervalues, ll[-1, ..., :2], order=3, prefilter=False
        )
    if ax == 111:
        fig = plt.figure(figsize=(16, 8))
    else:
        fig = plt.gcf()

    legend2 = {}

    if cg is True:
        ax, caax, paax = create_cg(fig=fig, subplot=ax, rot=0, scale=1)
        # for nice plotting we assume earth_radius = 6371000 m
        # this is the default value
        # todo: make this default in whole codebase
        er = 6371000
        # calculate beam_height and arc_distance for ke=1
        # means line of sight
        ade = georef.bin_distance(ranges, 0, site[2], ke=1.0)
        nn0 = np.zeros_like(ranges)
        ecp = nn0 + er
        # theta (arc_distance sector angle)
        thetap = -np.degrees(ade / er) + 90.0

        # zero degree elevation with standard refraction
        (bes,) = paax.plot(thetap, ecp, "-k", linewidth=3, label="_MSL", zorder=3)
        legend2["MSL"] = bes

        if terrain is not None:
            paax.fill_between(
                thetap, ecp.min() - 2500, ecp + terrain, color="0.75", zorder=2
            )

        # axes layout
        ax.set_xlim(0, np.max(ade))
        ax.set_ylim([ecp.min() - maxalt / 5, ecp.max() + maxalt])
        caax.grid(True, axis="x")
        ax.grid(True, axis="y")
        ax.axis["top"].toggle(all=False)
        gh = ax.get_grid_helper()
        yrange = maxalt + maxalt / 5
        nbins = ((yrange // vert_res) * 2 + 1) // np.sqrt(2)
        gh.grid_finder.grid_locator2._nbins = nbins
    else:
        ax = fig.add_subplot(ax)
        paax = ax
        caax = ax
        if terrain is not None:
            paax.fill_between(ranges, 0, terrain, color="0.75", zorder=2)
        ax.set_xlim(0.0, maxrange)
        ax.set_ylim(0.0, maxalt)
        ax.grid()

    # axes ticks and formatting
    if range_res is not None:
        xloc = range_res
        caax.xaxis.set_major_locator(tick.MultipleLocator(xloc))
    else:
        caax.xaxis.set_major_locator(tick.MaxNLocator())
    yloc = vert_res
    caax.yaxis.set_major_locator(tick.MultipleLocator(yloc))

    import functools

    hform = functools.partial(_height_formatter, cg=cg, scale=scale)
    rform = functools.partial(_range_formatter, scale=scale)
    caax.yaxis.set_major_formatter(tick.FuncFormatter(hform))
    caax.xaxis.set_major_formatter(tick.FuncFormatter(rform))

    # color management
    from cycler import cycler

    NUM_COLORS = len(elevs)
    cmap = plt.get_cmap(cmap)
    if cmap.N >= 256:
        colors = [cmap(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    else:
        colors = cmap.colors
    cycle = cycler(color=colors)
    paax.set_prop_cycle(cycle)

    # correctly handle single/multiple elevations
    if xyz.ndim == 2:
        xyz = xyz[np.newaxis, ...]

    # plot beams
    for i, el in enumerate(elevs):
        alt = xyz[i, ..., 2]
        groundrange = np.sqrt(xyz[i, ..., 0] ** 2 + xyz[i, ..., 1] ** 2)

        if cg:
            plrange = thetap
            plalt = ecp + alt
            beamradius = util.half_power_radius(ranges, beamwidth)
        else:
            plrange = np.insert(groundrange, 0, 0)
            plalt = np.insert(alt, 0, site[2])
            beamradius = util.half_power_radius(plrange, beamwidth)
        _, center, edge = _plot_beam(
            plrange, plalt, beamradius, label=f"{el:4.1f}°", ax=paax
        )

    # legend 1
    handles, labels = paax.get_legend_handles_labels()
    leg1 = ax.legend(
        handles,
        labels,
        prop={"family": "monospace"},
        loc="upper left",
        bbox_to_anchor=(1.04, 1),
        borderaxespad=0,
    )

    # legend 2
    legend2["Center"] = center[0]
    legend2["3 dB"] = edge[0]
    ax.legend(
        legend2.values(),
        legend2.keys(),
        prop={"family": "monospace"},
        loc="lower left",
        bbox_to_anchor=(1.04, 0),
        borderaxespad=0,
    )

    # add legend 1
    ax.add_artist(leg1)

    # set axes labels
    ax.set_title(f"Radar Scan Strategy - {site}" + add_title)
    caax.set_xlabel(f"Range ({units})")
    caax.set_ylabel(f"Altitude ({units})")

    return ax


def plot_plan_and_vert(x, y, z, dataxy, datazx, datazy, *, unit="", title="", **kwargs):
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
    unit : str
        unit of data arrays
    title: str
        figure title

    Keyword Arguments
    -----------------
    **kwargs : dict
        other kwargs which can be passed to :func:`matplotlib:matplotlib.pyplot.contourf`

    """

    plt.figure(figsize=(10, 10))

    # define axes
    left, bottom, width, height = 0.1, 0.1, 0.6, 0.2
    ax_xy = plt.axes((left, bottom, width, width))
    ax_x = plt.axes((left, bottom + width, width, height))
    ax_y = plt.axes((left + width, bottom, height, width))
    ax_cb = plt.axes((left + width + height + 0.02, bottom, 0.02, width))

    # set axis label formatters
    ax_x.xaxis.set_major_formatter(tick.NullFormatter())
    ax_y.yaxis.set_major_formatter(tick.NullFormatter())

    # draw CAPPI
    plt.sca(ax_xy)
    xy = plt.contourf(x, y, dataxy, **kwargs)
    plt.grid(color="grey", lw=1.5)

    # draw colorbar
    cb = plt.colorbar(xy, cax=ax_cb)
    cb.set_label(f"({unit})")

    # draw upper vertical profil
    ax_x.contourf(x, z, datazx, **kwargs)

    # draw right vertical profil
    ax_y.contourf(z, y, datazy.T, **kwargs)

    # label axes
    ax_xy.set_xlabel("x (km)")
    ax_xy.set_ylabel("y (km)")
    ax_x.set_xlabel("")
    ax_x.set_ylabel("z (km)")
    ax_y.set_ylabel("")
    ax_y.set_xlabel("z (km)")

    def xycoords(x, pos):
        """The two args are the value and tick position"""
        return f"{x / 1000:.0f}"

    xyformatter = tick.FuncFormatter(xycoords)

    def zcoords(x, pos):
        """The two args are the value and tick position"""
        return f"{x // 1000:.0f}"

    zformatter = tick.FuncFormatter(zcoords)

    ax_xy.xaxis.set_major_formatter(xyformatter)
    ax_xy.yaxis.set_major_formatter(xyformatter)
    ax_x.yaxis.set_major_formatter(zformatter)
    ax_y.xaxis.set_major_formatter(zformatter)

    if not title == "":
        # add a title - here, we have to create a new axes object which will
        # be invisible then the invisible axes will get a title
        tax = plt.axes(
            (left, bottom + width + height + 0.01, width + height, 0.01),
            frameon=False,
            facecolor="none",
        )
        tax.get_xaxis().set_visible(False)
        tax.get_yaxis().set_visible(False)
        plt.title(title)


def plot_max_plan_and_vert(x, y, z, data, *, unit="", title="", **kwargs):
    """Plot according to <plot_plan_and_vert> with the maximum values
    along the three axes of ``data``

    Examples
    --------
    See :ref:`/notebooks/workflow/recipe2.ipynb`.
    """
    plot_plan_and_vert(
        x,
        y,
        z,
        np.max(data, axis=-3),
        np.max(data, axis=-2),
        np.max(data, axis=-1),
        unit=unit,
        title=title,
        **kwargs,
    )


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
    See :ref:`/notebooks/visualisation/gis_overlay.ipynb`.
    """
    try:
        ax.add_collection(coll.LineCollection([lines], **kwargs))
    except AssertionError:
        ax.add_collection(coll.LineCollection([lines[None, ...]], **kwargs))
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
    See :ref:`/notebooks/visualisation/gis_overlay.ipynb`.
    """

    try:
        ax.add_collection(coll.PolyCollection([patch_array], **kwargs))
    except AssertionError:
        ax.add_collection(coll.PolyCollection([patch_array[None, ...]], **kwargs))
    except ValueError:
        for patch in patch_array:
            add_patches(ax, patch, **kwargs)


def _plot_cg(
    da,
    *,
    ax=111,
    fig=None,
    crs=None,
    func="pcolormesh",
    **kwargs,
):
    if da.sweep_mode == "azimuth_surveillance":
        cg = {"rot": -450, "scale": -1}
    else:
        cg = {"rot": 0, "scale": 1}
    if isinstance(crs, collections.abc.Mapping):
        cg.update(crs)

    if isinstance(ax, axes.Axes):
        try:
            caax = ax.parasites[0]
            paax = ax.parasites[1]
        except AttributeError as err:
            raise TypeError(
                "If `crs='cg'` `ax` need to be of type "
                "`mpl_toolkits.axisartist.SubplotHost`."
            ) from err
    else:
        # axes object is given
        if fig is None:
            if ax == 111:
                # create new figure if there is only one subplot
                fig = plt.figure()
            else:
                # assume current figure
                fig = plt.gcf()
        # create curvelinear axes
        ax, caax, paax = create_cg(fig=fig, subplot=ax, **cg)
        # this is in fact the outermost thick "ring"
        rdiff = da.range[1] - da.range[0]
        ax.axis["lon"] = ax.new_floating_axis(
            1, (np.max(da.bins.values) + rdiff.values / 2.0)
        )
        ax.axis["lon"].major_ticklabels.set_visible(False)
        # and also set tickmarklength to zero for better presentation
        ax.axis["lon"].major_ticks.set_ticksize(0)
        # set clip-box, needed if user adapts x/y-limits (e.g. RHI)
        ax.axis["lon"].set_clip_box(
            trans.TransformedBbox(trans.Bbox([[0, 0], [1, 1]]), ax.transAxes)
        )

    infer_intervals = kwargs.pop("infer_intervals", False)
    if func == "pcolormesh":
        kwargs.update(dict(shading="auto"))

    # strip colorbar kwargs from kwargs
    # we do not let create xarray the colorbar
    add_colorbar = kwargs.pop("add_colorbar", None)
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    kwargs["add_colorbar"] = False

    # claim xarray plot function and create plot
    plotfunc = getattr(da.plot, func)
    pm = plotfunc(
        x="rays",
        y="bins",
        ax=paax,
        infer_intervals=infer_intervals,
        **kwargs,
    )

    # set cg grids and limits, colorbar
    if da.sweep_mode == "azimuth_surveillance":
        xlims = np.min(da.x), np.max(da.x)
        ylims = np.min(da.y), np.max(da.y)
    else:
        xlims = np.min(da.gr), np.max(da.gr)
        ylims = np.min(da.z), np.max(da.z)
    # handle extents and activate grids
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.grid(True)
    caax.set_ylim(ylims)
    caax.set_xlim(xlims)
    caax.grid(True)

    # create our own colorbar for curvelinear grids
    if add_colorbar is not False:
        cbar = plt.colorbar(pm, ax=[ax, caax, paax], **cbar_kwargs)

        def _get_label(da):
            attrs = da.attrs
            name = attrs.get("long_name", da.name)
            units = attrs.get("units", False)
            label = f"{name}"
            if units:
                label += f" [{units}]"
            return label

        xl = da["gr"] if da.sweep_mode == "rhi" else da["x"]
        yl = da["z"] if da.sweep_mode == "rhi" else da["y"]

        caax.set_xlabel(_get_label(xl))
        caax.set_ylabel(_get_label(yl))
        cbar.set_label(_get_label(da))

    # apply box aspect for PPI
    if da.sweep_mode == "azimuth_surveillance":
        ax.set_box_aspect(1)
        caax.set_box_aspect(1)
        paax.set_box_aspect(1)

    # set ax as current
    plt.sca(ax)

    return pm


def plot(
    da,
    *,
    ax=111,
    fig=None,
    crs=None,
    func="pcolormesh",
    **kwargs,
):
    """Plot Plan Position Indicator (PPI) or Range Height Indicator (RHI).

    The implementation of this plot routine is in cartesian axes and does
    all coordinate transforms using xarray machinery. This allows zooming
    into the data as well as making it easier to plot additional data
    (like gauge locations) without having to convert them to the radar's
    polar coordinate system.

    Using ``crs='cg'`` the plotting is done in a curvelinear grid axes.

    Additional data can be plotted in polar coordinates or cartesian
    coordinates depending on which axes object is used.

    ``**kwargs`` may be used to try to influence the
    :func:`matplotlib.pyplot.pcolormesh`,
    :func:`matplotlib.pyplot.contour`,
    :func:`matplotlib.pyplot.contourf` and
    :func:`wradlib.georef.polar.spherical_to_proj` routines under the hood.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to plot
    crs : :py:class:`cartopy.crs.CRS`, dict or None
        cartopy CRS Coordinate Reference System describing projection
        If this parameter is not None, ``site`` must be set properly.
        Then the function will attempt to georeference the radar bins and
        display the PPI in the coordinate system defined by the
        projection string.
    fig : :class:`matplotlib.figure.Figure`
        If given, the PPI/RHI will be plotted into this figure object.
        Axes are created as needed. If None, a new figure object will be
        created or current figure will be used, depending on ``ax``.
    ax : :class:`matplotlib.axes.Axes` or :class:`matplotlib.gridspec.SubplotSpec`
        If matplotlib Axes object is given, the PPI will be plotted into
        this axes object.
        If matplotlib grid definition is given (nrows/ncols/plotnumber),
        axis are created in the specified place.
        Defaults to '111', only one subplot/axis.
    func : str
        Name of plotting function to be used under the hood.
        Defaults to 'pcolormesh'. 'contour' and 'contourf' can be
        selected too.

    Keyword Arguments
    -----------------
    cmap : str, optional
        matplotlib colormap string. Defaults to wradlib default colormap, which
        is either `Homeyer_Rainbow` if `cmweather` is installed or `turbo`.
    zorder : int, optional
        Lower zorder values are drawn first. Defaults to 0.
    kwargs : dict, optional
        Further kwargs, which are propagated to xarray plotting functions.

    Returns
    -------
    pm : :class:`matplotlib:matplotlib.collections.QuadMesh` or \
        :class:`matplotlib:matplotlib.contour.QuadContourSet`
        The result of the plotting function. Necessary, if you want to
        add a colorbar to the plot.

    Note
    ----
    If ``crs`` contains a curvelinear grid dict,
    the ``cgax`` - curvelinear Axes (r-theta-grid) is returned.
    ``caax`` - Cartesian Axes (x-y-grid) and ``paax`` -
    parasite axes object for plotting polar data can be derived like this::

        cgax = plt.gca()
        caax = cgax.parasites[0]
        paax = cgax.parasites[1]

    The function :func:`~wradlib.vis.create_cg` uses the Matplotlib
    `AXISARTIST <https://matplotlib.org/stable/api/toolkits/axisartist.html>`_ namespace.

    Here are some limitations to normal Matplotlib Axes (see
    `AXES_GRID1 <https://matplotlib.org/stable/api/toolkits/axes_grid1.html>`_).

    Examples
    --------
    See :ref:`/notebooks/visualisation/plot_ppi.ipynb`,
    and
    :ref:`/notebooks/visualisation/plot_curvelinear_grids.ipynb`.
    """
    # fix for correct zorder of data and grid
    kwargs["zorder"] = kwargs.pop("zorder", 0)
    kwargs["cmap"] = kwargs.get("cmap", wrl_cmap)

    # handle curvelinear grid properties
    if crs == "cg" or isinstance(crs, collections.abc.Mapping):
        return _plot_cg(da, ax=ax, fig=fig, crs=crs, func=func, **kwargs)

    # convert OSR crs to cartopy crs via pyproj
    if util.has_import(osr) and util.has_import(cartopy):
        if isinstance(crs, osr.SpatialReference):
            proj_crs = CRS.from_wkt(crs.ExportToWkt(["FORMAT=WKT2_2018"]))
            crs = cartopy.crs.CRS(proj_crs)

    # no axes object given
    if not isinstance(ax, axes.Axes):
        # no figure object given
        if fig is None:
            if not plt.get_fignums():
                # create new figure if there is None
                fig = plt.figure()
            else:
                # otherwise assume current figure
                fig = plt.gcf()
        # create axes object from given
        ax = fig.add_subplot(ax, projection=crs)

    infer_intervals = kwargs.pop("infer_intervals", True)
    xp, yp = ("x", "y") if da.sweep_mode == "azimuth_surveillance" else ("gr", "z")

    # use cartopy, if available
    if hasattr(ax, "projection") and util.has_import(cartopy):
        data_crs = get_crs(da.to_dataset(name="array-projection"))
        if data_crs.is_projected:
            map_trans = cartopy.crs.Projection(data_crs)
        elif data_crs.is_geographic:
            map_trans = cartopy.crs.PlateCarree()
        else:
            util.warn(
                "No valid CRS object found. Given CRS object is not projected. "
                "Falling back to cartopy.crs.PlateCarree()."
            )
            map_trans = cartopy.crs.PlateCarree()
        kwargs.update({"transform": map_trans})

    # handle aspect for PPI
    if da.sweep_mode == "azimuth_surveillance":
        ax.set_box_aspect(1)

    # handle colorbar sizes
    # special case contour to align with xarray
    if kwargs.get("add_colorbar", None) is not False and func != "contour":
        cbar_kwargs = kwargs.pop("cbar_kwargs", {})
        if not cbar_kwargs:
            cbar_kwargs.setdefault("fraction", 0.045)
        kwargs["cbar_kwargs"] = cbar_kwargs

    # claim xarray plot function and create plot
    plotfunc = getattr(da.plot, func)
    pm = plotfunc(
        x=xp,
        y=yp,
        ax=ax,
        infer_intervals=infer_intervals,
        **kwargs,
    )

    return pm


class VisMethods(util.XarrayMethods):
    """wradlib xarray SubAccessor methods for visualization."""

    @util.docstring(plot)
    def plot(self, *args, **kwargs):
        if not isinstance(self, VisMethods):
            return plot(self, *args, **kwargs)
        else:
            return plot(self._obj, *args, **kwargs)

    @util.docstring(plot)
    def pcolormesh(self, *args, **kwargs):
        kwargs.setdefault("func", "pcolormesh")
        return self.plot(*args, **kwargs)

    @util.docstring(plot)
    def contour(self, *args, **kwargs):
        kwargs.setdefault("func", "contour")
        return self.plot(*args, **kwargs)

    @util.docstring(plot)
    def contourf(self, *args, **kwargs):
        kwargs.setdefault("func", "contourf")
        return self.plot(*args, **kwargs)


if __name__ == "__main__":
    print("wradlib: Calling module <vis> as main...")
