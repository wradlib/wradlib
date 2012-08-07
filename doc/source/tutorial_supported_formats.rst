****************************
Supported radar data formats
****************************

This tutorial provides an overview of the data formats currently supported by wradlib. We seek to continuously enhance the range of supported formats, so this document is only a snapshot. If you need a specific file format to be supported by wradlib, please `raise an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_ of type *enhancement*. You can provide support by adding documents which help to decode the format, e.g. format reference documents or software code in other languages for decoding the format.

At the moment, *supported format* means that the radar format can be read and further processed by wradlib. Normally, wradlib will return an array of data values and dictionaries of metadata - if the file contains any. At the moment, *wradlib does not support encoding to any specific file formats!* This might change in the future, however, it is not a priority. However, you can use the Python netCDF4 or h5py packages to encode the resultws of your analysis to standard self-describing file formats such as netCDF or hdf5. If you have Python(x,y) installed on your machine, these packages are readiliy available to you.

In the following, we will provide an overview of file formats which can be currently read by wradlib. Reading weather radar files is done via the :doc:`io` module. There you will find a complete function reference. So normally, you will start by::

   import wradlib.io as io


German Weather Service: DX format
---------------------------------
The German Weather Service uses the DX file format to encode local radar sweeps. DX data are in polar coordinates. The naming convention is as follows: *raa00-dx_<location-id>-<YYMMDDHHMM>-<location-abreviation>---bin or raa00-dx_<location-id>-<YYYYMMDDHHMM>-<location-abreviation>---bin*.
:doc:`tutorial_reading_dx` provides an extensive introduction into working with DX data. For now, we would just like to know how to read the data::

   data, metadata = io.readDX("mydrive:/path/to/my/file/filename")

Here, data is a two dimensional array of shape (number of azimuth angles, number of range gates). This means that the number of rows of the array corresponds to the number of azimuth angles of the radar sweep while the number of columns corresponds to the number of range gates per ray.


BUFR
----
The Binary Universal Form for the Representation of meteorological data (BUFR) is a binary data format maintained by the World Meteorological Organization (WMO). The BUFR format was adopted by the OPERA program for the representation of weather radar data. This module provides a wrapper around the OPERA BUFR software, currently only for decoding BUFR files. If you intend to work with BUFR data, we recommend reading `OPERA's BUFR software documentation <http://www.knmi.nl/opera/bufr/doc/bufr_sw_desc.pdf>`_. Please note that the way the BUFR software is wrapped has to be considered very preliminary. Due to yet unsolved problems with the BUFR software API, wradlib simply calls the executable for BUFR deoding (decbufr) and read and parses corresponding the file output. This is of course inefficient from a computational perpective. we hope to come up with a new solution in the near future. However, the wradlib BUFR interface is plain simple::

   descnames, descvals, data = io.read_BUFR("mydrive:/path/to/my/file/filename")
   
Basically, a BUFR file consists of a set of *descriptors* which contain all the relevant metadata and a data section. The *descriptors* are identified as a tuple of three integers. The meaning of these tupels is described in the BUFR tables which come with the software. There are generic BUFR tables provided by the WMO, but it is also possible to define so called *local tables* - which was done by the OPERA consortium for the purpose of radar data representation.

:doc:`wradlib.io.read_BUFR` returns a three element tuple. The first element is a dictionary which relates the *descriptor identifiers* to comprehensible *descriptor names*. The second element is a dictionary which relates the *descriptor names* to *descriptor values*. E.g. if the *descriptor identifier* was (0, 30, 21), the *descriptor name* would be 'Number of pixels per row' and the *descriptor value* could be an integer which actually specifies the number of rows of a grid. The third element of the return tuple is the actual data array. It is a multi-dimensional numpy array of which the shape depends on the descriptor specifications (mostly it will be 2-dimensional).

**Gotchas**: At the moment, the BUFR implementation in wradlib has the potential to give you some trouble. It has only been tested on Windows 7 under Python 2.6, yet. The key is that the BUFR software has to be successfully compiled in the course of wradlib installation (via *python setup.py install*). Compilation requires *gcc* and *make*. Both is pre-installed on most Linux machines, and can be installed on Windows using the `MinGW compiler suite <http://www.mingw.org/wiki/Getting_Started>`_. **If you are using Python(x,y)**, gcc and make should already be available on your machine! You can check this by opening a console window and typing ``gcc --version`` and ``mingw32-make --version``. For **Linux**, the makefile is available and we hope that the installation process works. But we never tested it! Please give us your feedback how it works under Linux by sending an e-mail to wradlib-users@googlegroups.com or by `raising an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_.


NetCDF
------
The NetCDF format also claims to be self-describing. However, as for all such formats, the developers of netCDF also admit that "[...] the mere use of netCDF is not sufficient to make data self-describing and meaningful to both humans and machines [...]". The program that reads the data needs to know about the expected content. Different radar operators or data distributors will use different naming conventions and data hierarchies. Even though Python provides a decent netCDF library (netcdf4), wradlib will need to provide different interfaces to netCDF files offered by different distributors.

**NetCDF files exported by the EDGE software**
EDGE is a commercial software for radar control and data analysis provided by the Entreprise Electronics Corporation. It allows for netCDF data export. The resulting files can be read by::

   data, metadata = read_EDGE_netcdf("mydrive:/path/to/my/file/filename") 



