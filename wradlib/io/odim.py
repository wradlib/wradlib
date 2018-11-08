#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""odim module.

Defines radar data following the OPERA data information model (ODIM).

More information on the `opera website`_

.. _opera website: http://eumetnet.eu/activities/observations-programme/current-activities/opera

"""

import datetime as dt

import h5py
import numpy as np
import osgeo.osr
import wradlib

import rmiradlib.util

class Base(object):
    """Characteristics common to all weather radar data

       This is implemented as follows:
           - dataset values are stored as a list of numpy arrays
             (each data in a given dataset as the same dimension)
           - metadata are stored as nested dictionaries
           - timestamps are stored as datetime object for convenience

    """
    def __init__(self, values=None, dtype=None, nominal=None, source=None,
                       product=None, timestamp=None, quantity=None,
                       gain=None, offset=None, nodata=None, undetect=None):
        """ Initialisation of the data and metadata.
        
        Parameters
        ----------
        filename : string
        values : obj:`list` of :obj:`nparray`
            arrays of values representing each dataset
        dtype : 
            object type according to table 2
        nominal : str
            nominal time associated with the data
        source : `str`
            Variable-length string containing pairs of identifier
            types and their values, separated by a colon.
        product : obj:`list` of :obj:`str`
            product abbreviation according to table 14
        timestamp : obj:`list` of (:obj:`datetime`,:obj:`datetime`)
            starting and ending time of the dataset
        quantity : obj:`list` of :obj:`str`
            quantity abbreviation according to table 16
        gain : obj:`list` of :obj:`float`
            Coefficient ’a’ in y=ax+b used to convert to unit.
            Default value is 1.0.
        offset : obj:`list` of :obj:`float`
            Coefficient ’b’ in y=ax+b used to convert to unit.
            Default value is 0.0.
        nodata : obj:`list` of :obj:`float`
            Raw value used to denote areas void of data
        undetect : obj:`list` of :obj:`float`
            Raw value used to denote areas below the measurement
            detection threshold

        """

        if values is None:
            return

        self.metadata = {"Conventions":"ODIM_H5/V2_2"}

        self.values = values
        self.ndataset = len(product)
        self.ndata = len(quantity)

        self.datasets = ["dataset%s" %(d + 1) for d in range(self.ndataset)]
        self.datas = ["data%s" %(d + 1) for d in range(self.ndata)]

        self.init_metadata()

        self.dtype = dtype
        self.nominal = nominal
        self.source = source
        self.set_root_info()

        self.product = product
        self.timestamp = timestamp
        self.set_dataset_info()

        self.quantity = quantity
        if gain is None:
            gain = [1] * self.ndata
        self.gain = gain
        if offset is None:
            offset = [0] * self.ndata
        self.offset = offset
        if nodata is None:
            nodata = [None] * self.ndata
        if undetect is None:
            undetect = [None] * self.ndata
        self.nodata = nodata
        self.undetect = undetect
        self.set_data_info()

    def __eq__(self, other):
        """Test the equality of data and metadata.

        """
        for key in self.__dict__.keys():
            val1 = self.__dict__[key]
            val2 = other.__dict__[key]
            if isinstance(val1,np.ndarray):
                print("array in key : %s" %(key))
                try:
                    np.testing.assert_almost_equal(val1, val2, decimal=5)
                    test = True
                except AssertionError as e:
                    print("arrays differ")
                    print(e)
                    return False

            elif isinstance(val1, osgeo.osr.SpatialReference):
                test = (val1.ExportToProj4() == val2.ExportToProj4())
            else:
                test = (val1 == val2)
            if not test:
                print("values differ for key : %s" %(key))
                return False
        return True

    def init_metadata(self):
        """Initialisation of the metadata.

        """
        self.metadata["what"] = {}
        self.metadata["where"] = {}
        self.metadata["how"] = {}
        for dataset in self.datasets:
            self.metadata[dataset] = {"what":{},"where":{},"how":{}}
            for data in self.datas:
                self.metadata[dataset][data] = {"what":{},"where":{},"how":{}}

    def read(self, filename, values=True, metadata=True, get=True):
        """ Read data and metadata from an ODIM file.
        
        Parameters
        ----------    
        filename : string
            name of the ODIM file to open
        values : bool
            switch to read the values
        metadata : bool
            switch to read the metadata into a dict
        get : bool
            switch to get the metadata into attributes

        """

        self.filehandler = h5py.File(filename, 'r')

        self.ndataset = self.count_odim_datasets()
        self.ndata = self.count_odim_datas()
        self.datasets = ["dataset%s" %(d + 1) for d in range(self.ndataset)]
        self.datas = ["data%s" %(d + 1) for d in range(self.ndata)]

        if metadata:
            self.metadata = {"Conventions":"ODIM_H5/V2_2"}
            self.read_metadata()
            if get:
                self.get_metadata()
        if values:
            self.read_values()

    def count_odim_datasets(self, keys=None):
        """Count the number of datasets in an ODIM file.

        """
        keys = self.filehandler.keys()
        datasets = [key for key in keys if 'dataset' in key]
        count = len(datasets)
        return(count)

    def count_odim_datas(self, keys=None):
        """Count the number of datas in an ODIM file.

        """

        if keys is None:
            keys = self.filehandler["dataset1"].keys()
        datas = [key for key in keys if 'data' in key]
        count = len(datas)

        return(count)

    def read_values(self, quantity=None, slicing=None,
                             decode=True, precision=None, toarray=True):
        """Read values of a given dataset and data.
 
        Parameters
        ----------
        quantity : obj:`list` of :obj:`str`
            list of quantity to select
        slicing : numpy slice
            data slice to extract
        decode : bool 
            True to decode the data using metadata
        precision : string
            precision for the numpy array
        toarray : bool
            True to try to convert to an array           

        """
        datas = self.datas
        if quantity is not None:
            datas = [self.datas[i] for i in range(self.ndata) if self.quantity[i] in quantity]
            self.quantity = quantity
            self.ndata = len(quantity)

        values = []
        for dataset in self.datasets:
            vald = []
            for data in datas:

                val = self.filehandler[dataset][data]["data"]
                if slicing is None:
                    val = val[:]
                else:
                    val = val[slicing]


                vald.append(val)
            values.append(np.squeeze(np.array(vald, dtype=precision)))
        self.datas = datas

        self.values = values

        if decode:
            self.decode_values()

        if toarray:
            try:
                self.values = np.array(values)
            except:
                pass

    def decode_values(self, nodata=np.nan):
        """Decode data values.
    
        Parameters
        ----------
        nodata : float
            value to assign for no data

           By default NODATA is set to NAN values.

        """
        for idataset in range(self.ndataset):
            vals = self.values[idataset]
            if self.ndata == 1:
                vals = [vals]
            for v in range(self.ndata):
                val = vals[v].astype(float)
                if self.nodata[v] is not None:
                    inodata = (val == self.nodata[v])
                val = self.offset[v] + val * self.gain[v]
                if self.nodata[v] is not None:
                    val[inodata] = nodata
                vals[v] = val
        return()

    def read_metadata(self, group=None, metadata=None):
        """Read metadata to a dictionary
 
        Parameters
        ----------
        group : str
            metadata group
        metadata : dict
            dict to fill in

        """
        if group is None:
            group = self.filehandler
        if metadata is None:
            metadata = self.metadata
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                continue
            metadata[key] = {}
            self.read_metadata(group[key], metadata[key])
        for key, value in group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            metadata[key] = value

    def get_metadata(self):
        """ Get metadata into attributes.
        
        """
        self.get_root_info()
        self.get_dataset_info()
        self.get_data_info()

    def get_root_info(self):
        """Get root information.

        Returns
        -------
        attrs : dict
            dictionary with root information

        """

        what = self.metadata["what"]
        
        dtype = what["object"]

        date = what["date"]
        time = what["time"]
        nominal = rmiradlib.util.parse_date_time(date,time)

        source = what["source"]

        attrs = {"dtype":dtype, "nominal":nominal, "source":source}

        for key in attrs.keys():
            setattr(self, key, attrs[key])

    def get_dataset_info(self):
        """Get dataset info.

        Returns
        -------
        attrs : dict
            dictionary with dataset information


        """
        timestamp = []
        product = []
        for dataset in self.datasets:
            what = self.metadata[dataset]["what"]
            date = what["startdate"]
            time = what["starttime"]
            start = rmiradlib.util.parse_date_time(date,time)

            date = what["enddate"]
            time = what["endtime"]
            end = rmiradlib.util.parse_date_time(date,time)
            
            timestamp.append((start, end))

            product.append(what["product"])

        attrs = {"timestamp":timestamp, "product":product}

        for key in attrs.keys():
            setattr(self, key, attrs[key])

    def get_data_info(self):
        """Get data info.

        Returns
        -------
        attrs : dict
            dictionary with data information


        """
        default = {"quantity":None,
                    "gain":1,
                    "offset":0,
                    "nodata":None,
                    "undetect":None}
        attrs = {}
        for key in default.keys():
            attrs[key] = []
        for data in self.datas:
            what = self.metadata["dataset1"][data]["what"]
            for key in default.keys():
                if key in what:
                    attrs[key].append(what[key])
                else:
                    attrs[key].append(default[key])

        for key in attrs.keys():
            setattr(self, key, attrs[key])

    def write(self, filename):
        """ Write data and metadata to an ODIM file.

        Parameters
        ----------    
        filename : string
            name of the ODIM file to write

        """

        self.filehandler = h5py.File(filename, 'w')

        self.set_root_info()
        self.set_dataset_info()
        self.set_data_info()

        self.write_metadata()

        self.write_values()

    def write_values(self):
        """ Write values to ODIM file.

        """
        values = [self.values[v] for v in range(self.ndataset)]
        if self.ndata == 1:
            values = [[v] for v in self.values]
        for idataset in range(self.ndataset):
            dataset = self.datasets[idataset]
            for idata in range(self.ndata):
                data = self.datas[idata]
                val = values[idataset][idata]
                val = (val - self.offset[idata]) / self.gain[idata]
                if self.nodata[idata] is not None:
                    nans = np.isnan(val)
                    val[nans] = self.nodata[idata]
                group = self.filehandler[dataset][data]
                group.create_dataset("data", data=val, compression="gzip")

    def write_metadata(self, metadata=None, group=None):
        """ Write metadata to an ODIM file.

        Parameters
        ----------
        group : str
            metadata group
        metadata : dict
            dict to fill in

        """
        if metadata is None:
            metadata = self.metadata
        if group is None:
            group = self.filehandler

        for key, value in metadata.items():
            if isinstance(value, dict):
                group2 = group.create_group(key)
                self.write_metadata(value,group2)
            else:
                if isinstance(value, str):
                    value = np.string_(value)
                group.attrs[key] = value

    def set_root_info(self):
        """Set the top level information.

        """
        what = self.metadata["what"]

        what["object"] = self.dtype

        date = dt.datetime.strftime(self.nominal,"%Y%m%d")
        time = dt.datetime.strftime(self.nominal,"%H%M%S")
        what["date"] = date
        what["time"] = time

        what["source"] = self.source

    def set_dataset_info(self):
        """Set the datasets information.

        """

        for idataset in range(self.ndataset):
            dataset = self.datasets[idataset] 
            what = self.metadata[dataset]["what"]

            what["product"] = self.product[idataset]

            ts = self.timestamp[idataset]
            date = dt.datetime.strftime(ts[0],"%Y%m%d")
            time = dt.datetime.strftime(ts[0],"%H%M%S")
            what["startdate"] = date
            what["starttime"] = time    
            date = dt.datetime.strftime(ts[1],"%Y%m%d")
            time = dt.datetime.strftime(ts[1],"%H%M%S")
            what["enddate"] = date
            what["endtime"] = time

    def set_data_info(self):

        """ Set the data related information.
        
        """

        keys = ["quantity", "gain", "offset", "nodata", "undetect"]
        for idata in range(self.ndata):
            data = self.datas[idata]
            for key in keys:
                val = getattr(self, key)[idata]
                if val is None:
                    continue
                for idataset in range(self.ndataset):
                    dataset = self.datasets[idataset]
                    self.metadata[dataset][data]["what"][key] = val

class Pvol(Base):
    """Definition of 3D radar volume data

    """
    def __init__(self, elangle=None, rstart=None, rscale=None, a1gate=None, **baseargs):
        """ Initialisation of the data and metadata.
        
        Parameters
        ----------
        elangle : obj:`list` of :obj:`float`
            Antenna elevation angle (degrees) above the horizon.
        rstart : obj:`list` of :obj:`float`
            The range (km) of the start of the first range bin
        rscale : obj:`list` of :obj:`float`
            The distance in meters between two successive range bins
        a1gate : obj:`list` of :obj:`int`
            Index of the first azimuth gate radiated in the scan
            Default value is 0.

        """

        if elangle is None:
            return

        values = baseargs["values"]
        self.nbins = [values[idataset].shape[-1] for idataset in range(len(values))]
        self.nrays = [values[idataset].shape[-2] for idataset in range(len(values))]
        
        if a1gate is None:
            a1gate = [0]*len(elangle)
        self.a1gate = a1gate

        if rstart is None:
            rstart = [0]*len(elangle)
        self.rstart = rstart

        self.rscale = rscale
        self.elangle = elangle

        super(Pvol, self).__init__(dtype="PVOL", **baseargs)

    def set_dataset_info(self):
        """Set the dataset related information.

        """

        super(Pvol, self).set_dataset_info()

        attrs = ["a1gate", "rscale", "rstart", "elangle", "nbins", "nrays"]
        for idataset in range(self.ndataset):
            dataset = self.datasets[idataset] 
            where = self.metadata[dataset]["where"]
            for attr in attrs:
                where[attr] = getattr(self, attr)[idataset]

    def get_dataset_info(self):
        """Get dataset info.

        """

        super(Pvol, self).get_dataset_info()

        attrs = {}
        keys = ["a1gate", "rscale", "rstart", "elangle", "nbins", "nrays"]
        default = {"a1gate":1, "rstart":0}
        for key in keys:
            attrs[key] = []

        for idataset in range(self.ndataset):
            dataset = self.datasets[idataset] 
            where = self.metadata[dataset]["where"] 
            for key in keys:
                if key in where:
                    attrs[key].append(where[key])
                else:
                    attrs[key].append(default[key])

        for key in attrs.keys():
            setattr(self, key, attrs[key])

class Image(Base):
    """Definition of 2-D cartesian radar image data

       Datasets have the same geolocalisation.
    """
    def __init__(self, geotransform=None, projection=None, **baseargs):
        """ Initialisation of the data and metadata.
        
        Parameters
        ----------
        geotransform : obj:`np array`
            coefficients for transforming between pixel/line (P,L) raster space,
            and projection coordinates (Xp,Yp) space (GDAL).
            Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
            Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
        projection : osr object
            projection definition

        """

        if geotransform is None:
            return

        self.projection = projection
        
        self.geotransform = np.array(geotransform)

        super(Image, self).__init__(dtype="IMAGE", **baseargs)

    def set_dataset_info(self):
        """Set the dataset related information.

        Parameters
        ----------
        geotransform : obj:`np array`
            coefficients for transforming between pixel/line (P,L) raster space,
            and projection coordinates (Xp,Yp) space (GDAL).
            Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
            Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
        projection : osr object
            projection definition

        """

        super(Image, self).set_dataset_info()

        xmin, dx, none, ymax, none, dy = self.geotransform
        where = self.metadata["where"]
        where["xscale"] = dx
        where["yscale"] = -dy # since our origin is at the top left corner
        where["ysize"], where["xsize"] = self.values[0].shape[-2:]
        xmax = xmin + dx * where["ysize"]
        ymin = ymax + dy * where["xsize"]
        corners = ["UL","UR","LR","LL"]
        projected = np.array([[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]])
        geo = wradlib.georef.reproject(projected,projection_source=self.projection)
        for i in range(4):
            where["%s_lon" %(corners[i])] = geo[i][0]
            where["%s_lat" %(corners[i])] = geo[i][1]
        where["projdef"] = self.projection.ExportToProj4()

    def read(self, filename, values=True, metadata=True):
        """ Read data and metadata from an ODIM file.
        
        Parameters
        ----------    
        filename : string
            name of the ODIM file to open
        values : bool
            switch to read the values
        metadata : bool
            switch to read the metadata

        """

        super(Image, self).read(filename, values, metadata)
        if metadata:
            self.projection = self.get_projection()
            self.geotransform = self.get_geotransform()

    def get_projection(self):
        """Get projection from where information"""

        try:
            projection = self.metadata["dataset1"]["where"]["projdef"]
        except KeyError:
            projection = self.metadata["where"]["projdef"]
        projection = wradlib.georef.proj4_to_osr(str(projection))
        return(projection)

    def get_geotransform(self):
        """Get transformation matrix from where information"""
        try:
            where = self.metadata["dataset1"]["where"]
            dx = float(where["xscale"])
        except KeyError:
            where = self.metadata["where"]
            dx = float(where["xscale"])
        dy = -float(where["yscale"])
        try:
            xmin = where["UL_x"]
            ymax = where["UL_y"]
        except:
            geo = np.array([where["UL_lon"],where["UL_lat"]])
            xmin, ymax = wradlib.georef.reproject(geo, projection_target=self.projection)
        geotransform = [xmin, dx, 0, ymax ,0, dy]
        return(geotransform)

# Newline at end of file
