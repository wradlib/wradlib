*******
RADOLAN
*******

RADOLAN is abbreviated from the german **RA**\ dar-\ **O**\ n\ **L**\ ine-\ **AN**\ eichung, which means Radar-Online-Adjustment.

Using it's `network of 17 weather radar <http://www.dwd.de/bvbw/generator/DWDWWW/Content/Oeffentlichkeit/TI/TI2/Downloads/Standorttabelle,templateId=raw,property=publicationFile.pdf/Standorttabelle.pdf>`_ the German Weather Service provides many products for high resolution precipitation analysis and forecast. A comprehensive product list can be found in chapter :ref:`radolan_composite_products`.

These composite products are distributed in the :ref:`radolan_binary_data_format` with an ASCII header. All composites are available in :ref:`polar_stereo_projection` which will be discussed in the chapter :ref:`radolan_grid`.

Quick Start
===========

All RADOLAN composite products can be read by the following function::

   data, metadata = io.read_RADOLAN_composite("mydrive:/path/to/my/file/filename")

Here, ``data`` is a two dimensional integer or float array of shape (number of rows, number of columns). ``metadata`` is a dictionary which provides metadata from the files header section, e.g. using the keys *producttype*, *datetime*, *intervalseconds*, *nodataflag*.

The :ref:`radolan_grid` coordinates can be calculated with :func:`wradlib.georef.get_radolan_grid()`.

In `the following short example <https://bitbucket.org/wradlib/wradlib/src/default/examples/radolan_quickstart_example.py>`_ the RW-product is shown in the RADOLAN :ref:`polar_stereo_projection`::

    # import section
    import wradlib as wrl
    import numpy as np
    import matplotlib.pyplot as pl

    # load radolan files
    rw_filename = '../../examples/data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

    # print the available attributes
    print("RW Attributes:", rwattrs)

    # mask invalid values
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    rwdata = np.ma.masked_equal(rwdata, -9999)

    # get coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900,900)
    x = radolan_grid_xy[:,:,0]
    y = radolan_grid_xy[:,:,1]

    # create quick plot with colorbar and title
    pl.pcolormesh(x, y, rwdata, cmap="spectral")
    cb = pl.colorbar(shrink=0.75)
    cb.set_label("mm/h")
    pl.title('RADOLAN RW Product Polar Stereo \n' + rwattrs['datetime'].isoformat())
    pl.grid(color='r')


.. plot::

    import wradlib as wrl
    import numpy as np
    import matplotlib.pyplot as pl
    # load radolan files
    rw_filename = '../../examples/data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    # print the available attributes
    print("RW Attributes:", rwattrs)
    # do some masking
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    rwdata = np.ma.masked_equal(rwdata, -9999)
    # Get coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900,900)
    x = radolan_grid_xy[:,:,0]
    y = radolan_grid_xy[:,:,1]
    pl.pcolormesh(x, y, rwdata, cmap="spectral")
    # add colorbar and title
    cb = pl.colorbar(shrink=0.75)
    cb.set_label("mm/h")
    pl.title('RADOLAN RW Product Polar Stereo \n' + rwattrs['datetime'].isoformat())
    pl.grid(color='r')

A much more comprehensive section using several RADOLAN composites is shown in chapter :ref:`radolan_examples`.

.. _radolan_binary_data_format:

RADOLAN binary data format
==========================

The radolan composite files consists of an ascii header containing all needed information to decode the following binary data block. *wradlib* provides :func:`wradlib.io.read_RADOLAN_composite` to read the data.

The function :func:`wradlib.io.parse_DWD_quant_composite_header` takes care of correctly decoding the ascii header. All available header information is transferred into the metadata dictionary.

.. _radolan_composite_products:

RADOLAN composite products
==========================

A few products including RW and SF are available free of charge at this `DWD FTP Server <ftp://ftp-cdc.dwd.de/pub/CDC/grids_germany/>`_. A full list of RADOLAN products can be found in the `DWD RADOLAN/RADVOR-OP Kompositformat Version 2.2.2 <http://www.dwd.de/bvbw/generator/DWDWWW/Content/Wasserwirtschaft/Unsere__Leistungen/Radarniederschlagsprodukte/RADOLAN/RADOLAN__RADVOR__OP__Komposit__format__pdf,templateId=raw,property=publicationFile.pdf/RADOLAN_RADVOR_OP_Komposit_format_pdf.pdf>`_.

Currently, most of the RADOLAN composites have a spatial resolution of 1km x 1km, with the :ref:`national_composits` (R- and S-series) being 900 x 900 km grids, and the :ref:`european_composits` 1500 x 1400 km grids. The polar-stereographic projection is described in the chapter :ref:`radolan_grid`.

.. _national_composits:

National Composits
------------------

The common national products (across Germany) with a range of 900 km by 900 km are presented in the following table:

.. tabularcolumns:: |L|L|L|L|L]

+----+-------+-------+------------------------+-------------------------------------+
| ID |  INT  | avail | Filename               | Description                         |
+====+=======+=======+========================+=====================================+
| RX | 5 min | 5 min | | raa01-rx_10000-      | | original radardata in qualitative |
|    |       |       | | YYMMDDhhmm-dwd---bin | | RVP6-units (1 byte coded)         |
+----+-------+-------+------------------------+-------------------------------------+
| RZ | 5 min | 5 min | | raa01-rz_10000-      | | radardata after correction of PBB |
|    |       |       | | YYMMDDhhmm-dwd---bin | | converted to rainrate with        |
|    |       |       |                        | | improved Z-R-relation             |
+----+-------+-------+------------------------+-------------------------------------+
| RY | 5 min | 5 min | | raa01-ry_10000-      | | radardata after correction with   |
|    |       |       | | YYMMDDhhmm-dwd---bin | | Quality-composit (QY)             |
+----+-------+-------+------------------------+-------------------------------------+
| RH |  1 h  | 5 min | | raa01-rh_10000-      | | 1 h summation of RZ-composit      |
|    |       |       | | YYMMDDhhmm-dwd---bin |                                     |
+----+-------+-------+------------------------+-------------------------------------+
| RB |  1 h  | hh:50 | | raa01-rb_10000-      | | 1 h summation with preadjustment  |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| RW |  1 h  | hh:50 | | raa01-rw_10000-      | | 1 h summation with standard       |
|    |       |       | | YYMMDDhhmm-dwd---bin | | adjustment "best of two"          |
+----+-------+-------+------------------------+-------------------------------------+
| RL |  1 h  | hh:50 | | raa01-rl_10000-      | | 1 h summation with adjustment     |
|    |       |       | | YYMMDDhhmm-dwd---bin | | by Merging                        |
+----+-------+-------+------------------------+-------------------------------------+
| RU |  1 h  | hh:50 | | raa01-ru_10000-      | | 1 h summation with standard and   |
|    |       |       | | YYMMDDhhmm-dwd---bin | | merging adjustment "best of three"|
+----+-------+-------+------------------------+-------------------------------------+
| SQ |  6 h  | hh:50 | | raa01-sq_10000-      | | 6 h summation of RW               |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| SH | 12 h  | hh:50 | | raa01-sh_10000-      | | 12 h summation of RW              |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| SF | 24 h  | hh:50 | | raa01-sf_10000-      | | 24 h summation of RW              |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| W1 | 7 d   | 05:50 |                        | | 7 d summation of RW               |
+----+-------+-------+------------------------+-------------------------------------+
| W2 | 14 d  | 05:50 |                        | | 14 d summation of RW              |
+----+-------+-------+------------------------+-------------------------------------+
| W3 | 21 d  | 05:50 |                        | | 21 d summation of RW              |
+----+-------+-------+------------------------+-------------------------------------+
| W4 | 30 d  | 05:50 |                        | | 30 d summation of RW              |
+----+-------+-------+------------------------+-------------------------------------+

.. _european_composits:

Central European Composits
--------------------------

The common central european products with a range of 1500 km by 1400 km are presented in the following table:

+----+-------+-------+------------------------+-------------------------------------+
| ID |  INT  | avail | Filename               | Description                         |
+====+=======+=======+========================+=====================================+
| EX | 5 min | 5 min | | raa01-ex_10000-      | | analogue RX                       |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| EZ | 5 min | 5 min | | raa01-ez_10000-      | | analogue RZ                       |
|    |       |       | | YYMMDDhhmm-dwd---bin | |                                   |
+----+-------+-------+------------------------+-------------------------------------+
| EY | 5 min | 5 min | | raa01-ey_10000-      | | analogue EY after correction with |
|    |       |       | | YYMMDDhhmm-dwd---bin | | Quality-composit                  |
+----+-------+-------+------------------------+-------------------------------------+
| EH |  1 h  | hh:50 | | raa01-eh_10000-      | | analogue RH  (no preadjustment)   |
|    |       |       | | YYMMDDhhmm-dwd---bin | | 1 h summation of EY-composite     |
+----+-------+-------+------------------------+-------------------------------------+
| EB |  1 h  | hh:50 | | raa01-eb_10000-      | | analogue RB  (with preadjustment) |
|    |       |       | | YYMMDDhhmm-dwd---bin | | 1 h summation                     |
+----+-------+-------+------------------------+-------------------------------------+
| EW |  1 h  | hh:50 | | raa01-ew_10000-      | | analogue RW  (full adjustment)    |
|    |       |       | | YYMMDDhhmm-dwd---bin | | 1 h summation                     |
+----+-------+-------+------------------------+-------------------------------------+


.. _radolan_grid:

RADOLAN Grid
============

.. _polar_stereo_projection:

Polar Stereographic Projection
------------------------------

The projected composite raster is equidistant with a grid-spacing of 1.0 km in most cases. There are composites which have 2.0 km grid-spacing (e.g. PC).

There are three different grid sizes, the well-known 900 rows by 900 columns (normal), 1500 rows by 1400 columns (extended, european) and 460 rows by 460 columns (small).

Common to all is that the plane of projection intersects the earth sphere at :math:`\phi_0` = 60.0 :math:`^{\circ}` N. The cartesian co-ordinate system is aligned parallel to the :math:`\lambda_0` = 10.0 :math:`^{\circ}` E meridian.

The reference point ( :math:`\lambda_m` , :math:`\phi_m` ) is 9.0 :math:`^{\circ}` E and 51.0 :math:`^{\circ}` N, which is the center of the two smaller grids. The extended grid has an offset in respect to this reference point of 350km by 150km.

The earth as sphere with an radius of 6370.04 km is used for all calculations.

With formulas (1), (2) and (3) the geographic reference points (lambda, phi) can be converted to projected cartesian coordinates. The calculated (x y) is the distance vector to the origign of the cartesian coordinate system (north pole).

.. math::  x = R * M(\phi) * cos(\phi) * sin(\lambda - \lambda_0)
   :label: f1

.. math::  y = -R * M(\phi) * cos(\phi) * cos(\lambda - \lambda_0)
   :label: f2

.. math::  M(\phi) =  \frac {1 + sin(\phi_0)} {1 + sin(\phi)}
   :label: f3


Assumed the point (10.0 :math:`^{\circ}` E, 90.0 :math:`^{\circ}` N) is defined as coordinate system origin. Then all ccordinates can be calculated with the known grid-spacing d as:

.. math:: x = x_0 + d * (j - j_0)
   :label: f4

.. math:: y = y_0 + d * (i - i_0)
   :label: f5

with i, j as cartesian indices.

wradlib provides the convenience function `util.get_radolan_grid` which returns the radolan grid for further processing. It takes an (nrows, ncols)-tuple and returns the projected cartesian coordinates or the wgs84 coordinates (keyword arg wgs84=True) as numpy ndarray (nrows x ncols x 2).

Inverse Polar Stereographic Projection
--------------------------------------

The geographic coordinates of specific datapoints can be calculated by using the cartesian coordinates (x,y) and the following formulas:

.. math:: \lambda = \arctan\left(\frac {-x} {y}\right) + \lambda_0
   :label: f6

.. math:: \phi = \arcsin\left(\frac {R^2 * \left(1 + \sin\phi_0\right)^2 - \left(x^2 + y^2\right)} {R^2 * \left(1 + \sin\phi_0\right)^2 + \left(x^2 + y^2\right)}\right)
   :label: f7

Within *wradlib* the `georef.reproject` function can be used to convert the radolan grid data from xy-space to lonlat-space and back.

Radolan-projection in various standard formats
----------------------------------------------

WKT-String
^^^^^^^^^^

The German Weather Service provides a `WKT-string <https://kunden.dwd.de/geoserver/web/?wicket:bookmarkablePage=:org.geoserver.web.demo.SRSDescriptionPage&code=EPSG:1000001>`_. This WKT (well known text) is used to create the osr-object representation of the radolan projection.

For the scale_factor the intersection of the projection plane with the earth sphere at 60.0 :math:`^{\circ}` N has to be taken into account:

.. math:: scale\_factor = \frac {1 + \sin\left(60.^{\circ}\right)} {1 + \sin\left(90.^{\circ}\right)} = 0.93301270189
   :label: f8

Also, the PROJECTION["Stereographic_North_Pole"] isn't known within GDAL/OSR. It has to be changed to the known PROJECTION["polar_stereographic"].

Finally we yield the Radolan Projection as WKT-string::

    PROJCS["Radolan projection",
      GEOGCS["Radolan Coordinate System",
        DATUM["Radolan Kugel",
          SPHEROID["Erdkugel", 6370040.0, 0.0]],
        PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],
        UNIT["degree", 0.017453292519943295],
        AXIS["Longitude", EAST],
        AXIS["Latitude", NORTH]],
      PROJECTION["polar_stereographic"],
      PARAMETER["central_meridian", 10.0],
      PARAMETER["Standard_Parallel_1", 60.0],
      PARAMETER["scale_factor", 0.93301270189],
      PARAMETER["false_easting", 0.0],
      PARAMETER["false_northing", 0.0],
      UNIT["m*1000.0", 1000.0],
      AXIS["X", EAST],
      AXIS["Y", NORTH],
      AUTHORITY["EPSG","1000001"]]


PROJ.4
^^^^^^

Using the above WKT-String the PROJ.4 representation can be derived as:

PROJ.4-String::

    +proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189
    +x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs

This PROJ.4-string is used within *wradlib* to create the osr-object by using the two helper-functions :func:`wradlib.georef.create_projstr` and :func:`wradlib.georef.proj4_to_osr`::

    # create radolan projection osr object
    dwd_string = georef.create_projstr("dwd-radolan")
    proj_stereo = georef.proj4_to_osr(dwd_string)


.. _radolan_examples:

Examples
========

In this section examples are provided to get familiar with several RADOLAN products.

Attention is paid to:

* :ref:`ex_radolan_radarloc`
* :ref:`ex_radolan_header`
* :ref:`ex_radolan_projection`
* :ref:`ex_radolan_products`
* :ref:`ex_radolan_underlay`
* :ref:`ex_radolan_overlay`
* :ref:`ex_radolan_gauges`
* :ref:`ex_radolan_google`
* :ref:`ex_radolan_netcdf`

.. _ex_radolan_radarloc:

DWD-Radar Network
-----------------

In `this example script <https://bitbucket.org/wradlib/wradlib/src/default/examples/radolan_radarloc_example.py>`_ the RW-product is shown in WGS84 and the RADOLAN :ref:`polar_stereo_projection`. All for the compositing process used radars are extracted from the metadata and plotted with their respective maximum range rings and location information.

.. plot::

    import wradlib as wrl
    import matplotlib.pyplot as pl
    import numpy as np
    import matplotlib as mpl
    import os
    from osgeo import osr

    def get_radar_locations():

        radars = {}
        radar = {}
        radar['name'] = 'ASR Dresden'
        radar['wmo'] = 10487
        radar['lon'] = 13.76347
        radar['lat'] = 51.12404
        radar['alt'] = 261
        radars['ASD'] = radar

        radar = {}
        radar['name'] = 'Boostedt'
        radar['wmo'] = 10132
        radar['lon'] = 10.04687
        radar['lat'] = 54.00438
        radar['alt'] = 124.56
        radars['BOO'] = radar

        radar = {}
        radar['name'] = 'Dresden'
        radar['wmo'] = 10488
        radar['lon'] = 13.76865
        radar['lat'] = 51.12465
        radar['alt'] = 263.36
        radars['DRS'] = radar

        radar = {}
        radar['name'] = 'Eisberg'
        radar['wmo'] = 10780
        radar['lon'] = 12.40278
        radar['lat'] = 49.54066
        radar['alt'] = 798.79
        radars['EIS'] = radar

        radar = {}
        radar['name'] = 'Emden'
        radar['wmo'] = 10204
        radar['lon'] = 7.02377
        radar['lat'] = 53.33872
        radar['alt'] = 58
        radars['EMD'] = radar

        radar = {}
        radar['name'] = 'Essen'
        radar['wmo'] = 10410
        radar['lon'] = 6.96712
        radar['lat'] = 51.40563
        radar['alt'] = 185.10
        radars['ESS'] = radar

        radar = {}
        radar['name'] = 'Feldberg'
        radar['wmo'] = 10908
        radar['lon'] = 8.00361
        radar['lat'] = 47.87361
        radar['alt'] = 1516.10
        radars['FBG'] = radar

        radar = {}
        radar['name'] = 'Flechtdorf'
        radar['wmo'] = 10440
        radar['lon'] = 8.802
        radar['lat'] = 51.3112
        radar['alt'] = 627.88
        radars['FLD'] = radar

        radar = {}
        radar['name'] = 'Hannover'
        radar['wmo'] = 10339
        radar['lon'] = 9.69452
        radar['lat'] = 52.46008
        radar['alt'] = 97.66
        radars['HNR'] = radar

        radar = {}
        radar['name'] = 'Neuhaus'
        radar['wmo'] = 10557
        radar['lon'] = 11.13504
        radar['lat'] = 50.50012
        radar['alt'] = 878.04
        radars['NEU'] = radar

        radar = {}
        radar['name'] = 'Neuheilenbach'
        radar['wmo'] = 10605
        radar['lon'] = 6.54853
        radar['lat'] = 50.10965
        radar['alt'] = 585.84
        radars['NHB'] = radar

        radar = {}
        radar['name'] = 'Offenthal'
        radar['wmo'] = 10629
        radar['lon'] = 8.71293
        radar['lat'] = 49.9847
        radar['alt'] = 245.80
        radars['OFT'] = radar

        radar = {}
        radar['name'] = 'Proetzel'
        radar['wmo'] = 10392
        radar['lon'] = 13.85821
        radar['lat'] = 52.64867
        radar['alt'] = 193.92
        radars['PRO'] = radar

        radar = {}
        radar['name'] = 'Memmingen'
        radar['wmo'] = 10950
        radar['lon'] = 10.21924
        radar['lat'] = 48.04214
        radar['alt'] = 724.40
        radars['MEM'] = radar

        radar = {}
        radar['name'] = 'Rostock'
        radar['wmo'] = 10169
        radar['lon'] = 12.05808
        radar['lat'] = 54.17566
        radar['alt'] = 37
        radars['ROS'] = radar

        radar = {}
        radar['name'] = 'Isen'
        radar['wmo'] = 10873
        radar['lon'] = 12.10177
        radar['lat'] = 48.1747
        radar['alt'] = 677.77
        radars['ISN'] = radar

        radar = {}
        radar['name'] = 'Tuerkheim'
        radar['wmo'] = 10832
        radar['lon'] = 9.78278
        radar['lat'] = 48.58528
        radar['alt'] = 767.62
        radars['TUR'] = radar

        radar = {}
        radar['name'] = 'Ummendorf'
        radar['wmo'] = 10356
        radar['lon'] = 11.17609
        radar['lat'] = 52.16009
        radar['alt'] = 183
        radars['UMM'] = radar

        return radars

    def ex_radolan_radarloc():

        # load radolan file
        rw_filename = '../../examples/data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
        rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

        # print the available attributes
        print("RW Attributes:", rwattrs)

        # mask data
        sec = rwattrs['secondary']
        rwdata.flat[sec] = -9999
        rwdata = np.ma.masked_equal(rwdata, -9999)

        # create radolan projection object
        dwd_string = wrl.georef.create_projstr("dwd-radolan")
        proj_stereo = wrl.georef.proj4_to_osr(dwd_string)

        # create wgs84 projection object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        # get radolan grid
        radolan_grid_xy = wrl.georef.get_radolan_grid(900,900)
        x1 = radolan_grid_xy[:,:,0]
        y1 = radolan_grid_xy[:,:,1]

        # convert to lonlat
        radolan_grid_ll = wrl.georef.reproject(radolan_grid_xy, projection_source=proj_stereo, projection_target=proj_wgs)
        lon1 = radolan_grid_ll[:,:,0]
        lat1 = radolan_grid_ll[:,:,1]

        # plot two projections side by side
        fig1 = pl.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        pm = ax1.pcolormesh(lon1, lat1, rwdata, cmap='spectral')
        cb = fig1.colorbar(pm, shrink=0.75)
        cb.set_label("mm/h")
        pl.xlabel("Longitude ")
        pl.ylabel("Latitude")
        pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat() + '\n WGS84')
        pl.xlim((lon1[0,0],lon1[-1,-1]))
        pl.ylim((lat1[0,0],lat1[-1,-1]))
        pl.grid(color='r')

        fig2 = pl.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        pm = ax2.pcolormesh(x1, y1, rwdata, cmap='spectral')
        cb = fig2.colorbar(pm, shrink=0.75)
        cb.set_label("mm/h")
        pl.xlabel("x [km]")
        pl.ylabel("y [km]")
        pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat() + '\n Polar Stereographic Projection')
        pl.xlim((x1[0,0],x1[-1,-1]))
        pl.ylim((y1[0,0],y1[-1,-1]))
        pl.grid(color='r')

        # range array 150 km
        print("Max Range: ", rwattrs['maxrange'])
        r = np.arange(1, 151)*1000
        # azimuth array 1 degree spacing
        az = np.linspace(0,360,361)[0:-1]

        # get radar dict
        radars = get_radar_locations()

        # iterate over all radars in rwattrs
        # plot range rings and radar location for the two projections
        for id in rwattrs['radarlocations']:

            # get radar coords etc from dict
            # repair Ummendorf ID
            if id == 'umd':
                id = 'umm'
            radar = radars[id.upper()]

            # build polygons for maxrange rangering
            polygons = wrl.georef.polar2polyvert(r, az, (radar['lon'], radar['lat']))
            polygons.shape = (len(az), len(r), 5, 2)
            polygons_ll = polygons[:,-1,:,:]

            # reproject to radolan polar stereographic projection
            polygons_xy = wrl.georef.reproject(polygons_ll, projection_source=proj_wgs, projection_target=proj_stereo)

            # create PolyCollections and add to respective axes
            polycoll = mpl.collections.PolyCollection(polygons_ll, closed=True, edgecolors='r', facecolors='r')
            ax1.add_collection(polycoll, autolim=True)
            polycoll = mpl.collections.PolyCollection(polygons_xy, closed=True, edgecolors='r', facecolors='r')
            ax2.add_collection(polycoll, autolim=True)

            # plot radar location and information text
            ax1.plot(radar['lon'], radar['lat'], 'r+')
            ax1.text(radar['lon'], radar['lat'], id, color='r')

            # reproject lonlat radar location coordinates to polar stereographic projection
            x_loc, y_loc = wrl.georef.reproject(radar['lon'], radar['lat'], projection_source=proj_wgs, projection_target=proj_stereo)
            # plot radar location and information text
            ax2.plot(x_loc, y_loc, 'r+')
            ax2.text(x_loc, y_loc, id, color='r')

        pl.tight_layout()
        pl.show()

    # =======================================================
    if __name__ == '__main__':
        ex_radolan_radarloc()


.. _ex_radolan_header:

RADOLAN composite header
------------------------

In `this example script <https://bitbucket.org/wradlib/wradlib/src/default/examples/radolan_header_example.py>`_ we extract and show header information from several RADOLAN-products. First we load data and metadata of RX,EX,RW and SF-products::

    import wradlib as wrl
    import os

    # load radolan file
    rx_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rx_10000-1408102050-dwd---bin.gz'
    ex_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-ex_10000-1408102050-dwd---bin.gz'
    rw_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    sf_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-sf_10000-1408102050-dwd---bin.gz'

    rxdata, rxattrs = wrl.io.read_RADOLAN_composite(rx_filename)
    exdata, exattrs = wrl.io.read_RADOLAN_composite(ex_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    sfdata, sfattrs = wrl.io.read_RADOLAN_composite(sf_filename)

Then, we print the RX metadata::

    # print the available attributes
    print("RX Attributes:")
    for key, value in rxattrs.iteritems():
        print(key +':', value)

    RX Attributes:
    ('maxrange:', '150 km')
    ('radarlocations:', ['boo', 'ros', 'emd', 'hnr', 'umd', 'pro', 'ess', 'asd', 'neu', 'nhb', 'oft', 'tur', 'isn', 'fbg', 'mem', 'bdy'])
    ('nrow:', 900)
    ('intervalseconds:', 300)
    ('cluttermask:', array([], dtype=int64))
    ('precision:', 1.0)
    ('datetime:', datetime.datetime(2014, 8, 10, 20, 50))
    ('ncol:', 900)
    ('radolanversion:', '2.13.1')
    ('producttype:', 'RX')
    ('nodataflag:', -9999)
    ('datasize:', 810000)
    ('radarid:', '10000')

Then, we print the EX metadata::

    print("----------------------------------------------------------------")
    # print the available attributes
    print("EX Attributes:")
    for key, value in exattrs.iteritems():
        print(key +':', value)

    EX Attributes:
    ('maxrange:', '128 km')
    ('radarlocations:', ['sin', 'rom', 'vir', 'bor', 'nld', 'zav', 'wid', 'sui', 'abv', 'ave', 'tra', 'arc', 'ncy', 'bgs', 'bla', 'sly', 'sem', 'boo', 'ros', 'emd', 'hnr', 'umd', 'pro', 'ess', 'asd', 'neu', 'nhb', 'oft', 'tur', 'isn', 'fbg', 'mem', 'bdy', 'ska'])
    ('nrow:', 1500)
    ('intervalseconds:', 300)
    ('cluttermask:', array([], dtype=int64))
    ('precision:', 1.0)
    ('datetime:', datetime.datetime(2014, 8, 10, 20, 50))
    ('ncol:', 1400)
    ('radolanversion:', '2.13.1')
    ('producttype:', 'EX')
    ('nodataflag:', -9999)
    ('datasize:', 2100000)
    ('radarid:', '10000')

Then, we print the RW metadata::

    # print the available attributes
    print("RW Attributes:")
    for key, value in rwattrs.iteritems():
        print(key +':', value)

    RW Attributes:
    ('maxrange:', '150 km')
    ('radarlocations:', ['boo', 'ros', 'emd', 'hnr', 'umd', 'pro', 'ess', 'asd', 'neu', 'nhb', 'oft', 'tur', 'isn', 'fbg', 'mem'])
    ('nrow:', 900)
    ('intervalseconds:', 3600)
    ('cluttermask:', array([], dtype=int64))
    ('precision:', 0.1)
    ('datetime:', datetime.datetime(2014, 8, 10, 20, 50))
    ('ncol:', 900)
    ('radolanversion:', '2.13.1')
    ('producttype:', 'RW')
    ('nodataflag:', -9999)
    ('datasize:', 1620000)
    ('radarid:', '10000')
    ('secondary:', array([   799,    800,    801, ..., 806263, 806264, 807163]))

Finally, we print the SF metadata::

    # print the available attributes
    print("SF Attributes:")
    for key, value in sfattrs.iteritems():
        print(key +':', value)

    SF Attributes:
    ('maxrange:', '150 km')
    ('radarlocations:', ['boo', 'ros', 'emd', 'hnr', 'umd', 'pro', 'ess', 'asd', 'neu', 'nhb', 'oft', 'tur', 'isn', 'fbg', 'mem'])
    ('nrow:', 900)
    ('intervalseconds:', 86400)
    ('cluttermask:', array([], dtype=int64))
    ('precision:', 0.1)
    ('datetime:', datetime.datetime(2014, 8, 10, 20, 50))
    ('ncol:', 900)
    ('radolanversion:', '2.13.1')
    ('producttype:', 'SF')
    ('nodataflag:', -9999)
    ('datasize:', 1620000)
    ('radarid:', '10000')
    ('secondary:', array([   188,    189,    190, ..., 809566, 809567, 809568]))

The metadata information reflects the different measurement time intervals, the different radar stations involved and differences in serveral other header information.

.. _ex_radolan_projection:

RADOLAN Projection
------------------

In `this example script <https://bitbucket.org/wradlib/wradlib/src/default/examples/radolan_projection_example.py>`_ we calculate the RADOLAN Grid and print their bounding box coordinates in different projections::

    import wradlib as wrl
    from osgeo import osr

    # create radolan grid coordinates
    # add 1 to each dimension to get upper left corner coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(901,901)

    # create radolan projection osr object
    dwd_string = wrl.georef.create_projstr("dwd-radolan")
    proj_stereo = wrl.georef.proj4_to_osr(dwd_string)

    # create wgs84 projection osr object
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)

    # create Gauss Krueger zone 3 projection osr object
    proj_gk3 = osr.SpatialReference()
    proj_gk3.ImportFromEPSG(31467)

    # transform radolan polar stereographic projection to wgs84 and then to gk3
    radolan_grid_ll = wrl.georef.reproject(radolan_grid_xy, projection_source=proj_stereo, projection_target=proj_wgs)
    radolan_grid_gk = wrl.georef.reproject(radolan_grid_ll, projection_source=proj_wgs, projection_target=proj_gk3)

    # get coordinates for easy access
    lon_wgs0 = radolan_grid_ll[:,:,0]
    lat_wgs0 = radolan_grid_ll[:,:,1]
    x_gk3 = radolan_grid_gk[:,:,0]
    y_gk3 = radolan_grid_gk[:,:,1]
    x_rad = radolan_grid_xy[:,:,0]
    y_rad = radolan_grid_xy[:,:,1]

Then, we print the RADOLAN x,y Grid Coordinates::

    print("source radolan x,y-coordinates")
    print(u"       {0}      {1} ".format('x [km]', 'y [km]'))
    print("ll: {:10.4f} {:10.3f} ".format(x_rad[0,0], y_rad[0,0]))
    print("lr: {:10.4f} {:10.3f} ".format(x_rad[0,-1], y_rad[0,-1]))
    print("ur: {:10.4f} {:10.3f} ".format(x_rad[-1,-1], y_rad[-1,-1]))
    print("ul: {:10.4f} {:10.3f} ".format(x_rad[-1,0], y_rad[-1,0]))

Output::

    source radolan x,y-coordinates
           x [km]      y [km]
    ll:  -523.4622  -4658.645
    lr:   376.5378  -4658.645
    ur:   376.5378  -3758.645
    ul:  -523.4622  -3758.645

Then, we print the RADOLAN lon,lat Grid Coordinates::

    print("transformed radolan lonlat-coordinates")
    print(u"       {0}   {1} ".format(u'lon [\N{DEGREE SIGN}E]', u'lat [\N{DEGREE SIGN}N]'))
    print("ll: {:10.4f} {:10.4f} ".format(lon_wgs0[0,0], lat_wgs0[0,0]))
    print("lr: {:10.4f} {:10.4f} ".format(lon_wgs0[0,-1], lat_wgs0[0,-1]))
    print("ur: {:10.4f} {:10.4f} ".format(lon_wgs0[-1,-1], lat_wgs0[-1,-1]))
    print("ul: {:10.4f} {:10.4f} ".format(lon_wgs0[-1,0], lat_wgs0[-1,0]))

Output::

    transformed radolan lonlat-coordinates
           lon [°E]   lat [°N]
    ll:     3.5889    46.9526
    lr:    14.6209    47.0705
    ur:    15.7208    54.7405
    ul:     2.0715    54.5877

Finally, we print the RADOLAN gk3 Grid Coordinates::

    print("transformed radolan gk3-coordinates")
    print(u"     {0}     {1} ".format('easting [m]', 'northing [m]'))
    print("ll: {:10.0f} {:10.0f} ".format(x_gk3[0,0], y_gk3[0,0]))
    print("lr: {:10.0f} {:10.0f} ".format(x_gk3[0,-1], y_gk3[0,-1]))
    print("ur: {:10.0f} {:10.0f} ".format(x_gk3[-1,-1], y_gk3[-1,-1]))
    print("ul: {:10.0f} {:10.0f} ".format(x_gk3[-1,0], y_gk3[-1,0]))

Output::

    transformed radolan gk3-coordinates
         easting [m]  northing [m]
    ll:    3088210      5215765
    lr:    3926971      5230000
    ur:    3932597      6088666
    ul:    3052511      6072990

.. _ex_radolan_products:

RADOLAN products showcase
-------------------------

In `this example script <https://bitbucket.org/wradlib/wradlib/src/default/examples/examples/radolan_products_example.py>`_ we show several RADOLAN products:

.. plot::

    import wradlib as wrl
    import matplotlib.pyplot as pl
    import numpy as np
    import os

    # load radolan file
    rx_filename = '../../examples/data/radolan/raa01-rx_10000-1408102050-dwd---bin.gz'
    ex_filename = '../../examples/data/radolan/raa01-ex_10000-1408102050-dwd---bin.gz'
    rw_filename = '../../examples/data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    sf_filename = '../../examples/data/radolan/raa01-sf_10000-1408102050-dwd---bin.gz'

    rxdata, rxattrs = wrl.io.read_RADOLAN_composite(rx_filename)
    exdata, exattrs = wrl.io.read_RADOLAN_composite(ex_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    sfdata, sfattrs = wrl.io.read_RADOLAN_composite(sf_filename)

    # mask invalid values
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    sec = sfattrs['secondary']
    sfdata.flat[sec] = -9999

    rxdata = np.ma.masked_equal(rxdata, -9999) / 2 - 32.5
    exdata = np.ma.masked_equal(exdata, -9999) / 2 - 32.5
    rwdata = np.ma.masked_equal(rwdata, -9999)
    sfdata = np.ma.masked_equal(sfdata, -9999)

    # Get coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900,900)
    radolan_egrid_xy = wrl.georef.get_radolan_grid(1500,1400)
    x = radolan_grid_xy[:,:,0]
    y = radolan_grid_xy[:,:,1]

    xe = radolan_egrid_xy[:,:,0]
    ye = radolan_egrid_xy[:,:,1]

    # plot RX product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, rxdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("dBZ")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RX Product single scan\n' + rxattrs['datetime'].isoformat())
    pl.xlim((x[0,0],x[-1,-1]))
    pl.ylim((y[0,0],y[-1,-1]))
    pl.grid(color='r')

    # plot EX product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(xe, ye, exdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("dBZ")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('EX Product single scan - extended grid\n' + exattrs['datetime'].isoformat())
    pl.xlim((xe[0,0],xe[-1,-1]))
    pl.ylim((ye[0,0],ye[-1,-1]))
    pl.grid(color='r')

    # plot RW product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, rwdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("mm/h")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RW Product 1h rain accumulation\n' + rwattrs['datetime'].isoformat())
    pl.xlim((x[0,0],x[-1,-1]))
    pl.ylim((y[0,0],y[-1,-1]))
    pl.grid(color='r')

    # plot SF product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, sfdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("mm / 24h")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('SF Product 24h rain accumulation\n' + sfattrs['datetime'].isoformat())
    pl.xlim((x[0,0],x[-1,-1]))
    pl.ylim((y[0,0],y[-1,-1]))
    pl.grid(color='r')

This example will be extended if more products are available.

.. _ex_radolan_underlay:

Digital Elevation Model Underlay
--------------------------------

Example follows soon...

.. _ex_radolan_overlay:

River Network Overlay
---------------------

Example follows soon...

.. _ex_radolan_gauges:

Rain Gauges Overlay
-------------------

Example follows soon...

.. _ex_radolan_google:

Export to Google Maps
---------------------

Example follows soon...

.. _ex_radolan_netcdf:

Export to NetCDF
----------------

Example follows soon...


Acknowledgements
================

This tutorial was prepared with material from the `DWD RADOLAN/RADVOR-OP Kompositformat Version 2.2.2 <http://www.dwd.de/bvbw/generator/DWDWWW/Content/Wasserwirtschaft/Unsere__Leistungen/Radarniederschlagsprodukte/RADOLAN/RADOLAN__RADVOR__OP__Komposit__format__pdf,templateId=raw,property=publicationFile.pdf/RADOLAN_RADVOR_OP_Komposit_format_pdf.pdf>`_.
We also wish to thank Elmar Weigl, German Weather Service, for providing the extensive set of example data and his valuable information about the RADOLAN products.