Release Notes
=============

Please note that as long as :math:`\omega radlib` releases are within the ``0.y.z`` series, the API cannot be considered stable. We will try to avoid sudden API breaks via deprecation warnings. All :math:`\omega radlib` releases come without any warranty. Release notes might be incomplete. See `here <https://github.com/wradlib/wradlib/commits/master>`_ for a complete record of changes.

You can install the latest :math:`\omega radlib` release from PyPI via ``$ pip install wradlib`` or specific version via ``$ pip install wradlib==x.y.z``. The recommended installation process is described in :doc:`gettingstarted`.


Bleeding Edge
-------------

**Highlights**

* Added functions to match the precipitation radar of GPM/TRMM platforms with ground radar observations in 3D (:meth:`wradlib.georef.correct_parallax`, :meth:`wradlib.georef.sat2pol`, :meth:`wradlib.georef.dist_from_orbit`, :meth:`wradlib.qual.get_bb_ratio`, :meth:`wradlib.trafo.s2ku`, :meth:`wradlib.trafo.ku2s`, :meth:`wradlib.util.calculate_polynomial`, :meth:`wradlib.zonalstats.get_clip_mask`)
* Added example notebook of GPM/TRMM-GR matching
* revised docs and build process

**New features**

* New notebook examples covering wradlib-tour, classification
* Added reading WX-format to RADOLAN reader
* Enhanced :meth:`wradlib.io.read_RADOLAN_composite` to also accept file-handles
* Added reading groups to :meth:`wradlib.io.read_generic_netcdf`
* Added :meth:`wradlib.qual.cu_beam_block_frac` to compute cumulative beam blockage
* Added earth curvature display to beam blockage

**Bugfixes**

* Fix documentation inconsistencies
* Fix calculation of pulse volume
* Use dedicated OSR IsSame() in :meth:`wradlib.georef.transform_geometry`
* several minor fixes


Version 0.9.0
-------------

**Highlights**

* examples and tutorials are provided as jupyter notebooks
* docs are directly created from notebooks (see full workflow `here <https://github.com/wradlib/wradlib/wiki/dev.-notebook-workflow>`__)
* structured notebooks with parent notebook, where appropriate
* documentation reviewed and adapted
    - community.rst,
    - dev_setup.rst,
    - gettingstarted.rst,
    - index.rst,
    - notebooks.rst,
    - release_notes.rst
* docstrings reviewed, added links to notebooks where appropriate, cosmetic changes
* package notebooks and converted python scripts with pypi-sdist
* added tutorials on how to use notebooks and a breif Python introduction

**New features**

* deploy untagged MICRO-version changes to test-pypi
* reworked citation handling
* added reading capability of new radolan FZ product
* added use of dedicated wradlib-repo, WRADLIB_DATA env variable and data file retrieving functions
* add ability to read gzipped dx data
* enhanced ``wradlib.io.read_Rainbow`` to read product pixmap data from rainbow5 files

**Bugfixes**

* removed bug reading lists in ``wradlib.io.read_Rainbow``
* fixed offset bug in ``wradlib.georef.sweep_centroids``
* fixed incompatibility issue of ipol module with scipy 0.18.0
* several minor fixes


Version 0.8.0
-------------

**Highlights**

* As of now :math:`\omega radlib` is python3 compatible.
* Docstrings, tests and examples as well as the documentation have been reviewed and fixed.
* main :math:`\omega radlib` repository is now hosted `here on github <https://github.com/wradlib/wradlib>`__.
* :math:`\omega radlib` docs are now hosted `on github, but with custom domain <http://wradlib.org/wradlib-docs/>`_.

**New features**

:math:`\omega radlib` is constantly tested on `travis-ci wradlib <https://travis-ci.org/wradlib/wradlib>`_ within a miniconda python environment with the latest python27, python34 and python35 interpreters on linux OS.
We also check code coverage for all pull requests and additions with `coveralls <https://coveralls.io/github/wradlib/wradlib>`_.

**Deprecated features**

*None.*

**Removed functions**

* `georef.create_projstr`, also removed deprecated decorators


Version 0.7.0
-------------

**Highlights**

New *experimental* module ``zonalstats``: it supports computation of zonal statistics (so far mean and variance) for target polygons. 
Typical applications would be the computation of average catchment rainfall from polar or cartesian grids. Check out the 
`module documentation <http://wradlib.org/wradlib-docs/latest/zonalstats.html>`_ and the new examples.


Version 0.6.0
-------------

**Highlights**

:math:`\omega radlib` functions concerned with georeferencing will only use projection information based on OSR objects. This version will help you to adapt your code base to this change before next minor release: Using deprecated PROJ.4 strings in call to the following functions/classes: ``vis.plot_ppi``, ``vis.plot_ppi_crosshair``, ``georef.projected_bincoords_from_radarspec``, ``verify.PolarNeighbours``, ``vpr.volcoords_from_polar``, ``vpr.volcoords_from_polar_irregular``, ``vpr.make_3D_grid`` will generate a DeprecationWarning and try to correct old calling method at runtime.

Added ability to handle georeferenced raster and vector data

Port of PyRadarMet partial beamblockage calculations by DEM


**New features**

- In georef module a new helper function `create_osr` is added. This function helps to create an osr object for specific projections.
- Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid`` merged from v0.5.1
- Several convenience functions for reading, transforming and projecting as well as plotting rasterfiles and shapefiles including example
- Calculation of partial/cumulative beamblockage with example
- The behaviour of wradlib.adjust.AdjustMFB has been changed. Control parameters are no longer passed via the ``__call__`` function, but, as for all other adjustment classes, via the initialisation of an adjustment object. Documentation has been revised to make the behaviour more transparent. The parameter 'biasby' has been replaced by a dictionary mfb_args that carries all parameters that control the behaviour of AdjustMFB.


**Deprecated features**

Function `georef.create_projstr` is deprecated.

**Removed functions**

*None.*


Version 0.5.1
-------------

**Highlights**

Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid`` merged from v0.4.2


Version 0.5.0
-------------

**Highlights**

From version ``0.6.0`` on, :math:`\omega radlib` functions concerned with georeferencing will only use projection information based on OSR objects. This version will help you to adapt your code base to this change before using version ``0.6.0``: Any use of proj4 strings will generate a deprecation warning with the following functions/classes: ``vis.plot_ppi``,``vis.plot_ppi_crosshair``, ``georef.projected_bincoords_from_radarspec``, ``verify.PolarNeighbours``, ``vpr.volcoords_from_polar``, ``vpr.volcoords_from_polar_irregular``, ``vpr.make_3D_grid``.

**New features**

- Two functions exposed documentation library section: :doc:`generated/wradlib.io.read_safnwc` and :doc:`generated/wradlib.vis.plot_max_plan_and_vert`
- New features, changes and deprecations will be addressed in the documentation in the future. This is by highlighting them as *New in Version X.Y.Z*, *Changed in Version X.Y.Z* or *Deprecated since Version X.Y.Z*

**Deprecated features**

*None.*

**Removed functions**

*None.*


Version 0.4.2
-------------

**Highlights**

Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid``


Version 0.4.1
-------------

**Highlights**

From now on, :math:`\omega radlib` will generate warnings if keyword parameters of :math:`\omega radlib` functions will be or have been changed. This is achieved by using ``wradlib.util.apichange_kwargs`` as a decorator (see ``apichange_example.py`` for examples how these warnings might look like). Please take these warnings seriously and adapt your applications before stepping to a new :math:`\omega radlib` release.


Version 0.4.0
-------------

**Highlights**

- Fixed a broken proj4 string for DWD radolan polarstereographic projection in function ``wradlib.georef.create_projstr``
- Added RADOLAN tutorial to the documentation. Together with that came new function to retrieve the RADOLAN composite grid.
- Adding section ``Release notes`` to the documentation.

**New features**

- comprehensive RADOLAN tutorial, examples, and example data: http://wradlib.org/wradlib-docs/latest/tutorial_radolan_format.html
- enhanced :doc:`generated/wradlib.io.read_RADOLAN_composite` to read EX product
- :doc:`generated/wradlib.georef.get_radolan_grid`

**Deprecated features**

*None.*

**Removed functions**

*None.*


Version 0.3.0
-------------

**Highlights**

Visually most strikingly, we moved to a new style in our online documentation. 

However, the most important change introduced with this release was to remove a number of deprecated functions that will not be supported anymore (see list below). Users who want to use these functions need to fall back to ``0.2.0`` (not recommended). Accordingly, examples and documentation has been revised in order to remove all remaining usage of deprecated functions and also fix some documentation issues.

In addition, we removed three outdated tutorial (on clutter detection, convertion and rainfall accumulation) and replaced the two latter by a more concise tutorial "Converting reflectivity to rainfall".

Finally, we use one "central" bibliography for literature cross-referencing now (see http://wradlib.org/wradlib-docs/latest/zreferences.html).

**New features**

- New style of online docs (http://wradlib.org/wradlib-docs), using sphinx_rtd_theme
- Added Tutorial http://wradlib.org/wradlib-docs/latest/tutorial_get_rainfall.html
- New organisation of bibliography: http://wradlib.org/wradlib-docs/latest/zreferences.html

**Deprecated features**

*None*

**Removed functions**

The following functions/classes that were marked as deprecated before have been removed with this release:

- ``wradlib.dp.fill_phidp``
- ``wradlib.dp.process_raw_phidp``
- ``wradlib.georef.polar2latlon``
- ``wradlib.georef.__pol2latlon``
- ``wradlib.georef.polar2latlonalt``
- ``wradlib.georef.polar2latlonalt_n``
- ``wradlib.georef.project``
- ``wradlib.vis.PolarPlot``
- ``wradlib.vis.polar_plot2``
- ``wradlib.vis.polar_plot``
- ``wradlib.vis.CartesianPlot``
- ``wradlib.vis.cartesian_plot``
- ``wradlib.vis.get_tick_vector``
- ``wradlib.vis.create_curvilinear_axes``
- ``wradlib.vis.rhi_plot``
- ``wradlib.vis.cg_plot``
- ``wradlib.vis.rhi_plot``


Version 0.2.0
-------------

``0.2.0`` is the first new release of :math:`\omega radlib` after more than three years of ``0.0.1``. The changes that have accumulated during this time are too many to list them here. Let's just say that from here on, we will keep track of changes in :math:`\omega radlib` releases more thoroughly.


Version 0.1.1
-------------

``0.1.1`` was the first experimental :math:`\omega radlib` release.
