"""Tests for odim module"""

import datetime
import os

import numpy as np
import osgeo.osr
import pytest
import wradlib

import rmiradlib.odim as odim

class TestBase(object):
    @pytest.fixture(autouse=True)

    def sample_base(self):
        source = "NOD:bewid"
        nominal = datetime.datetime.strptime("20130101030000","%Y%m%d%H%M%S")
        return(source, nominal)

    def sample_coding(self, ndata):
        gain = [20] * ndata
        offset = [50] * ndata
        nodata = [-1] * ndata
        return(gain, offset, nodata)
    
class TestPvol(TestBase):
    @pytest.fixture(autouse=True)

    def sample_volume(self):
        attrs = {}
        attrs["source"], attrs["nominal"] = self.sample_base()
        attrs["quantity"] = ["DBZH","VRADH","WRADH"]
        attrs["product"] = ['SCAN'] * 5
        attrs["values"] = np.random.random((5,3,360,500))*5000
        attrs["values"][0,:,0,0] = np.nan
        start = [attrs["nominal"] + datetime.timedelta(secs) for secs in range(0,100,20)]
        attrs["timestamp"] = [(s,s+datetime.timedelta(20)) for s in start]
        attrs["gain"], attrs["offset"], attrs["nodata"] = self.sample_coding(ndata=3)
        attrs["elangle"] = list(range(1,6))
        attrs["rscale"] = list(range(1,6))
        return(attrs)

    def test_read_write(self,tmpdir):
        filename = tmpdir.mkdir("work").join("pvol.hdf")
        attrs = self.sample_volume()
        pvol = odim.Pvol(**attrs)
        pvol.write(filename)
        copy = odim.Pvol()
        copy.read(filename)
        assert(pvol == copy)

class TestImage(TestBase):
    @pytest.fixture(autouse=True)

    def sample_image(self):
        attrs = {}
        attrs["source"], attrs["nominal"] = self.sample_base()
        attrs["quantity"] = ["ACRR"]
        attrs["product"] = ['SURF', 'MAX']
        attrs["values"] = np.random.randint(0,255,(2,500,500))
        attrs["timestamp"] = [(attrs["nominal"], attrs["nominal"])]*2
        attrs["geotransform"] = [-250000, 1000, 0, 250000, 0, -1000]
        proj4str="+proj=aeqd +lat_0=51.1 +lon_0=5.2 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        projection = osgeo.osr.SpatialReference()
        projection.ImportFromProj4(proj4str)
        attrs["projection"] = projection
        return(attrs)

    def test_read_write(self,tmpdir):
        filename = tmpdir.mkdir("work").join("image.hdf")
        attrs = self.sample_image()
        image = odim.Image(**attrs)
        image.write(filename)
        copy = odim.Image()
        copy.read(filename)
        assert(image == copy)

