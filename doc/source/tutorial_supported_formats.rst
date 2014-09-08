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
The quantitative composite format of the DWD (German Weather Service) was established in the course of the `RADOLAN project <http://www.dwd.de/RADOLAN>`_. Most quantitative composite products from the DWD are distributed in this format, e.g. the R-series (RX, RY, RH, RW, ...), the S-series (SQ, SH, SF, ...), and the E-series (European quantitative composite, e.g. EZ, EH, EB). Please see the `composite format description <http://www.dwd.de/bvbw/generator/DWDWWW/Content/Wasserwirtschaft/en/Unsere__Leistungen__en/Radarniederschlagsprodukte__en/RADOLAN__en/RADOLAN__RADVOR__OP__Komposit__format__2__2__1__pdf,templateId=raw,property=publicationFile.pdf/RADOLAN_RADVOR_OP_Komposit_format_2_2_1_pdf.pdf>`_ for a full reference and a full table of products (unfortunately only in German language).

Currently, the RADOLAN composites have a spatial resolution of 1km x 1km, with the national composits (R- and S-series) being 900 x 900 grids, and the European composits 1500 x 1400 grids. The projection is polar-stereographic. The products can be read by the following function::

   data, metadata = io.read_RADOLAN_composite("mydrive:/path/to/my/file/filename")

Here, ``data`` is a two dimensional integer array of shape (number of rows, number of columns). Different product types might need different levels of postprocessing, e.g. if the product contains rain rates or accumulations, you will normally have to divide data by factor 10. ``metadata`` is again a dictionary which provides metadata from the files header section, e.g. using the keys *producttype*, *datetime*, *intervalseconds*, *nodataflag*. Masking the NoData (or missing) values can be done by::

    import numpy as np
    maskeddata = np.ma.masked_equal(data, metadata["nodataflag"])


OPERA BUFR
----------
The Binary Universal Form for the Representation of meteorological data (BUFR) is a binary data format maintained by the World Meteorological Organization (WMO). The BUFR format was adopted by the OPERA program for the representation of weather radar data. This module provides a wrapper around the OPERA BUFR software, currently only for decoding BUFR files. If you intend to work with BUFR data, we recommend reading `OPERA's BUFR software documentation <http://www.knmi.nl/opera/bufr/doc/bufr_sw_desc.pdf>`_. Please note that the way the BUFR software is wrapped has to be considered very preliminary. Due to yet unsolved problems with the BUFR software API, wradlib simply calls the executable for BUFR deoding (decbufr) and read and parses corresponding the file output. This is of course inefficient from a computational perpective. we hope to come up with a new solution in the near future. However, the wradlib BUFR interface is plain simple::

   data, metadata = io.read_BUFR("mydrive:/path/to/my/file/filename")
   
Basically, a BUFR file consists of a set of *descriptors* which contain all the relevant metadata and a data section. The *descriptors* are identified as a tuple of three integers. The meaning of these tupels is described in the BUFR tables which come with the software. There are generic BUFR tables provided by the WMO, but it is also possible to define so called *local tables* - which was done by the OPERA consortium for the purpose of radar data representation.

:doc:`wradlib.io.read_BUFR` returns a two element tuple. The first element (``data``) of the return tuple is the actual data array. It is a multi-dimensional numpy array of which the shape depends on the descriptor specifications (mostly it will be 2-dimensional). The second element (``metadata``) is a tuple of two dictionaries (descnames, descvals). *descnames* relates the *descriptor identifiers* to comprehensible *descriptor names*. *descvals* relates the *descriptor names* to *descriptor values*. E.g. if the *descriptor identifier* was (0, 30, 21), the *descriptor name* would be 'Number of pixels per row' and the *descriptor value* could be an integer which actually specifies the number of rows of a grid. Just try::

    # Gives the descriptor name for each descriptor ID tuple
    print metadata[0]
    # Gives the descriptor value for each descriptor name
    print metadata[1]
    # Gives the descriptor value for a particular descriptor ID tuple, in this case (0, 30, 21)
    print metadata[1][ metadata[0][(0, 30, 21)] ]

**Gotchas**: At the moment, the BUFR implementation in wradlib has the potential to give you some trouble. It has only been tested on Windows 7 under Python 2.6, yet. The key is that the BUFR software has to be successfully compiled in the course of wradlib installation (via *python setup.py install*). Compilation requires *gcc* and *make*. Both is pre-installed on most Linux machines, and can be installed on Windows using the `MinGW compiler suite <http://www.mingw.org/wiki/Getting_Started>`_. **If you are using Python(x,y)**, gcc and make should already be available on your machine! You can check this by opening a console window and typing ``gcc --version`` and ``mingw32-make --version``. For **Linux**, the makefile is available and we hope that the installation process works. But we never tested it! Please give us your feedback how it works under Linux by sending an e-mail to wradlib-users@googlegroups.com or by `raising an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_.


OPERA HDF5 (ODIM_H5)
--------------------
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ is a data model, library, and file format for storing and managing data. The `OPERA 3 program <http://www.knmi.nl/opera>`_ developed a convention (or information model) on how to store and exchange radar data in hdf5 format. It is based on the work of `COST Action 717 <http://www.smhi.se/hfa_coord/cost717>`_ and is used e.g. in real-time operations in the Nordic European countries. The OPERA Data and Information Model (ODIM) is documented e.g. in this `report <http://www.knmi.nl/opera/opera3/OPERA_2008_03_WP2.1b_ODIM_H5_v2.1.pdf>`_ and in a `UML representation <http://www.knmi.nl/opera/opera3/OPERA_2008_18_WP2.1b_ODIM_UML.pdf>`_. Make use of these documents in order to understand the organization of OPERA hdf5 files!

The hierarchical nature of HDF5 can be described as being similar to directories, files, and links on a hard-drive. Actual metadata are stored as so-called *attributes*, and these attributes are organized together in so-called *groups*. Binary data are stored as so-called *datasets*. As for ODIM_H5, the ``root`` (or top level) group contains three groups of metadata: these are called ``what`` (object, information model version, and date/time information), ``where`` (geographical information), and ``how`` (quality and optional/recommended metadata). For a very simple product, e.g. a CAPPI, the data is organized in a group called ``dataset1`` which contains another group called ``data1`` where the actual binary data are found in ``data``. In analogy with a file system on a hard-disk, the HDF5 file containing this simple product is organized like this::

    /
    /what
    /where
    /how
    /dataset1
    /dataset1/data1
    /dataset1/data1/data

The philosophy behind the wradlib interface to OPERA's data model is very straightforward: wradlib simply translates the complete file structure to *one* dictionary and returns this dictionary to the user. Thus, the potential complexity of the stored data is kept and it is left to the user how to proceed with this data. The keys of the output dictionary are strings that correspond to the "directory trees" shown above. Each key ending with ``/data`` points to a Dataset (i.e. a numpy array of data). Each key ending with ``/what``, ``/where`` or ``/how`` points to another dictionary of metadata. The entire output can be obtained by::

    fcontent = io.read_OPERA_hdf5("mydrive:/path/to/my/file/filename")

The user should inspect the output obtained from his or her hdf5 file in order to see how access those items which should be further processed. In order to get a readable overview of the output dictionary, one can use the pretty printing module::

    # which keyswords can be used to access the content?
    print fcontent.keys()
    # print the entire content including values of data and metadata
    # (numpy arrays will not be entirely printed)
    import pprint as pp
    pp.pprint(fcontent)

Please note that in order to experiment with such datasets, you can download hdf5 sample data from the `Odyssey page <http://www.knmi.nl/opera/odc.html>`_ of the `OPERA 3 homepage <http://www.knmi.nl/opera>`_.

GAMIC HDF5
----------
GAMIC refers to the commercial `GAMIC Enigma V3 MURAN software <http://www.gamic.com/cgi-bin/info.pl?link=softwarebrowser3>`_ which exports data in hdf5 format. The concept is quite similar to the above `OPERA HDF5 (ODIM_H5)`_ format. Such a file (typical ending: *.mvol*) can be read by::

    data, metadata = io.read_GAMIC_hdf5("mydrive:/path/to/my/file/filename")

While metadata represents the usual dictionary of metadata, the data variable is a dictionary which might contain several numpy arrays with the keywords of the dictionary indicating different moments.

NetCDF
------
The NetCDF format also claims to be self-describing. However, as for all such formats, the developers of netCDF also admit that "[...] the mere use of netCDF is not sufficient to make data self-describing and meaningful to both humans and machines [...]". The program that reads the data needs to know about the expected content. Different radar operators or data distributors will use different naming conventions and data hierarchies. Even though Python provides a decent netCDF library (netcdf4), wradlib will need to provide different interfaces to netCDF files offered by different distributors.

**NetCDF files exported by the EDGE software**

EDGE is a commercial software for radar control and data analysis provided by the Enterprise Electronics Corporation. It allows for netCDF data export. The resulting files can be read by::

   data, metadata = io.read_EDGE_netcdf("mydrive:/path/to/my/file/filename") 

Gematronik Rainbow
------------------
Rainbow refers to the commercial `RAINBOW®5 APPLICATION SOFTWARE <http://www.gematronik.com/products/radar-components/rainbowR-5/>`_ which exports data in an XML flavour, which due to binary data blobs violates XML standard. Gematronik provided python code for implementing this reader in wradlib, which is very much appreciated.

The philosophy behind the wradlib interface to Gematroniks data model is very straightforward: wradlib simply translates the complete xml file structure to *one* dictionary and returns this dictionary to the user. Thus, the potential complexity of the stored data is kept and it is left to the user how to proceed with this data. The keys of the output dictionary are strings that correspond to the "xml nodes" and "xml attributes". Each ``data`` key points to a Dataset (i.e. a numpy array of data). Such a file (typical ending: *.vol* or *.azi*) can be read by::

    fcontent = io.read_Rainbow("mydrive:/path/to/my/file/filename")

The user should inspect the output obtained from his or her Rainbow file in order to see how access those items which should be further processed. In order to get a readable overview of the output dictionary, one can use the pretty printing module::

    # which keyswords can be used to access the content?
    print fcontent.keys()
    # print the entire content including values of data and metadata
    # (numpy arrays will not be entirely printed)
    import pprint as pp
    pp.pprint(fcontent)

You can check this :download:`example script <../../examples/load_rainbow.py>` for getting a first impression.
