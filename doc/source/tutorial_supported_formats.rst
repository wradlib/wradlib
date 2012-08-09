****************************
Supported radar data formats
****************************

The binary encoding of many radar products is a major obstacle for many potential radar users. Often, decoder software is not easily available. In case formats are documented, the implementation of decoders is a major programming effort. This tutorial provides an overview of the data formats currently supported by wradlib. We seek to continuously enhance the range of supported formats, so this document is only a snapshot. If you need a specific file format to be supported by wradlib, please `raise an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_ of type *enhancement*. You can provide support by adding documents which help to decode the format, e.g. format reference documents or software code in other languages for decoding the format.

At the moment, *supported format* means that the radar format can be read and further processed by wradlib. Normally, wradlib will return an array of data values and a dictionary of metadata - if the file contains any. wradlib does not support encoding to any specific file formats, yet! This might change in the future, but it is not a priority. However, you can use Python's netCDF4 or h5py packages to encode the results of your analysis to standard self-describing file formats such as netCDF or hdf5. If you have Python(x,y) installed on your machine, these packages are readily available to you.

In the following, we will provide an overview of file formats which can be currently read by wradlib. Reading weather radar files is done via the :doc:`io` module. There you will find a complete function reference. So normally, you will start by::

   import wradlib.io as io


German Weather Service: DX format
---------------------------------
The German Weather Service uses the DX file format to encode local radar sweeps. DX data are in polar coordinates. The naming convention is as follows: *raa00-dx_<location-id>-<YYMMDDHHMM>-<location-abreviation>---bin or raa00-dx_<location-id>-<YYYYMMDDHHMM>-<location-abreviation>---bin*.
:doc:`tutorial_reading_dx` provides an extensive introduction into working with DX data. For now, we would just like to know how to read the data::

   data, metadata = io.readDX("mydrive:/path/to/my/file/filename")

Here, ``data`` is a two dimensional array of shape (number of azimuth angles, number of range gates). This means that the number of rows of the array corresponds to the number of azimuth angles of the radar sweep while the number of columns corresponds to the number of range gates per ray.


German Weather Service: RADOLAN (quantitative) composit
-------------------------------------------------------
The quantitative composite format of the DWD (German Weather Service) was established in the course of the `RADOLAN project <http://www.dwd.de/radolan>`_. Most quantitative composite products from the DWD are distributed in this format, e.g. the R-series (RX, RY, RH, RW, ...), the S-series (SQ, SH, SF, ...), and the E-series (European quantitative composite, e.g. EZ, EH, EB). Please see the `composite format description <http://www.dwd.de/bvbw/generator/DWDWWW/Content/Wasserwirtschaft/en/Unsere__Leistungen__en/Radarniederschlagsprodukte__en/RADOLAN__en/RADOLAN__RADVOR__OP__Komposit__format__2__2__1__pdf,templateId=raw,property=publicationFile.pdf/RADOLAN_RADVOR_OP_Komposit_format_2_2_1_pdf.pdf>`_ for a full reference and a full table of products (unfortunately only in German language).

Currently, the RADOLAN composites have a spatial resolution of 1km x 1km, with the national composits (R- and S-series) being 900 x 900 grids, and the European composits 1500 x 1400 grids. The projection is polar-stereographic. The products can be read by the following function::

   data, metadata = io.read_RADOLAN_composite("mydrive:/path/to/my/file/filename")

Here, ``data`` is a two dimensional integer array of shape (number of rows, number of columns). Different product types might need different levels of postprocessing, e.g. if the product contains rain rates or accumulations, you will normally have to divide data by factor 10. ``metadata`` is again a dictionary which provides metadata from the files header section, e.g. using the keys *producttype*, *datetime*, *intervalseconds*, *nodataflag*. Masking the NoData (or missing) values can be done by::
	
	import numpy as np
	maskeddata = np.ma.masked_equal(data, attrs["nodataflag"])
	

OPERA BUFR
----------
The Binary Universal Form for the Representation of meteorological data (BUFR) is a binary data format maintained by the World Meteorological Organization (WMO). The BUFR format was adopted by the OPERA program for the representation of weather radar data. This module provides a wrapper around the OPERA BUFR software, currently only for decoding BUFR files. If you intend to work with BUFR data, we recommend reading `OPERA's BUFR software documentation <http://www.knmi.nl/opera/bufr/doc/bufr_sw_desc.pdf>`_. Please note that the way the BUFR software is wrapped has to be considered very preliminary. Due to yet unsolved problems with the BUFR software API, wradlib simply calls the executable for BUFR deoding (decbufr) and read and parses corresponding the file output. This is of course inefficient from a computational perpective. we hope to come up with a new solution in the near future. However, the wradlib BUFR interface is plain simple::

   descnames, descvals, data = io.read_BUFR("mydrive:/path/to/my/file/filename")
   
Basically, a BUFR file consists of a set of *descriptors* which contain all the relevant metadata and a data section. The *descriptors* are identified as a tuple of three integers. The meaning of these tupels is described in the BUFR tables which come with the software. There are generic BUFR tables provided by the WMO, but it is also possible to define so called *local tables* - which was done by the OPERA consortium for the purpose of radar data representation.

:doc:`wradlib.io.read_BUFR` returns a three element tuple. The first element is a dictionary which relates the *descriptor identifiers* to comprehensible *descriptor names*. The second element is a dictionary which relates the *descriptor names* to *descriptor values*. E.g. if the *descriptor identifier* was (0, 30, 21), the *descriptor name* would be 'Number of pixels per row' and the *descriptor value* could be an integer which actually specifies the number of rows of a grid. The third element of the return tuple is the actual data array. It is a multi-dimensional numpy array of which the shape depends on the descriptor specifications (mostly it will be 2-dimensional).

**Gotchas**: At the moment, the BUFR implementation in wradlib has the potential to give you some trouble. It has only been tested on Windows 7 under Python 2.6, yet. The key is that the BUFR software has to be successfully compiled in the course of wradlib installation (via *python setup.py install*). Compilation requires *gcc* and *make*. Both is pre-installed on most Linux machines, and can be installed on Windows using the `MinGW compiler suite <http://www.mingw.org/wiki/Getting_Started>`_. **If you are using Python(x,y)**, gcc and make should already be available on your machine! You can check this by opening a console window and typing ``gcc --version`` and ``mingw32-make --version``. For **Linux**, the makefile is available and we hope that the installation process works. But we never tested it! Please give us your feedback how it works under Linux by sending an e-mail to wradlib-users@googlegroups.com or by `raising an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_.


OPERA HDF5 (ODIM_H5)
--------------------
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ is a data model, library, and file format for storing and managing data. The `OPERA 3 program <http://www.knmi.nl/opera>`_ developed a convention (or information model) on how to store and exchange radar data in hdf5 format. It is based on the work of `COST Action 717 <http://www.smhi.se/hfa_coord/cost717>` and is used e.g. in real-time operations in the Nordic countries. This OPERA Data and Information Model (ODIM) is documented e.g. in this `report <http://www.knmi.nl/opera/opera3/OPERA_2008_03_WP2.1b_ODIM_H5_v2.1.pdf>`_ and in a `UML representation <http://www.knmi.nl/opera/opera3/OPERA_2008_18_WP2.1b_ODIM_UML.pdf>`_.

The implementation of the OPERA HDF5 format in wradlib is an ongoing effort. We hope to present a first reader soon...you can already have a look at the construction site in the source under ``wradlib.io.read_OPERA_hdf5``... 


NetCDF
------
The NetCDF format also claims to be self-describing. However, as for all such formats, the developers of netCDF also admit that "[...] the mere use of netCDF is not sufficient to make data self-describing and meaningful to both humans and machines [...]". The program that reads the data needs to know about the expected content. Different radar operators or data distributors will use different naming conventions and data hierarchies. Even though Python provides a decent netCDF library (netcdf4), wradlib will need to provide different interfaces to netCDF files offered by different distributors.

**NetCDF files exported by the EDGE software**
EDGE is a commercial software for radar control and data analysis provided by the Entreprise Electronics Corporation. It allows for netCDF data export. The resulting files can be read by::

   data, metadata = io.read_EDGE_netcdf("mydrive:/path/to/my/file/filename") 



