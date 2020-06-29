# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import datetime
import gzip
import io as sio
import os
import subprocess
import sys
import tempfile
import zlib

import deprecation
import numpy as np
import pytest
import xarray as xr

from wradlib import georef, io, util, zonalstats

from . import requires_data


class TestDX:
    # testing functions related to read_dx
    @requires_data
    def test__get_timestamp_from_filename(self):
        filename = "raa00-dx_10488-200608050000-drs---bin"
        assert io.radolan._get_timestamp_from_filename(filename) == datetime.datetime(
            2006, 8, 5, 0
        )
        filename = "raa00-dx_10488-0608050000-drs---bin"
        assert io.radolan._get_timestamp_from_filename(filename) == datetime.datetime(
            2006, 8, 5, 0
        )

    @requires_data
    def test_get_dx_timestamp(self):
        filename = "raa00-dx_10488-200608050000-drs---bin"
        assert (
            io.radolan.get_dx_timestamp(filename).__str__()
            == "2006-08-05 00:00:00+00:00"
        )
        filename = "raa00-dx_10488-0608050000-drs---bin"
        assert (
            io.radolan.get_dx_timestamp(filename).__str__()
            == "2006-08-05 00:00:00+00:00"
        )

    def test_parse_dx_header(self):
        header = (
            b"DX021655109080608BY54213VS 2CO0CD2CS0EP0.30.30.40.50."
            b"50.40.40.4MS999~ 54( 120,  46) 43-31 44 44 50 50 54 52 "
            b"52 42 39 36  ~ 53(  77,  39) 34-31 32 44 39 48 53 44 45 "
            b"35 28 28  ~ 53(  98,  88)-31-31-31 53 53 52 53 53 53 32-31"
            b" 18  ~ 57(  53,  25)-31-31 41 52 57 54 52 45 42 34 20 20  "
            b"~ 55(  37,  38)-31-31 55 48 43 39 50 51 42 15 15  5  ~ "
            b"56( 124,  19)-31 56 56 56 52 53 50 50 41 44 27 28  ~ "
            b"47(  62,  40)-31-31 46 42 43 40 47 41 34 27 16 10  ~ "
            b"46( 112,  62)-31-31 30 33 44 46 46 46 46 33 38 23  ~ "
            b"44( 100, -54)-31-31 41 41 38 44 43 43 28 35 30  6  ~ "
            b"47( 104,  75)-31-31 45 47 38 41 41 30 30 15 15  8  ^ "
            b"58( 104, -56) 58 58 58 58 53 37 37  9 15-31-31-31  ^ "
            b"58( 123,  16) 56-31 58 58 46 52 49 35 44 14 32  0  ^ "
            b"57(  39,  38)-31 55 53 57 55 27 29 18 11  1  1-31  ^ "
            b"54( 100,  85)-31-31 54 54 46 50-31-31 17-31-31-31  ^ "
            b"53(  71,  39)-31-31 46 53 52 34 34 40 32 32 23  0  ^ "
            b"53( 118,  49)-31-31 51 51 53 52 48 42 39 29 24-31  ` "
            b"28(  90,  43)-31-31 27 27 28 27 27 19 24 19  9  9  ` "
            b"42( 114,  53)-31-31 36 36 40 42 40 40 34 34 37 30  ` "
            b"54(  51,  27)-31-31 49 49 54 51 45 39 40 34.."
        )
        head = ""
        for c in sio.BytesIO(header):
            head += str(c.decode())
        io.radolan.parse_dx_header(head)

    def test_unpack_dx(self):
        pass

    @requires_data
    def test_read_dx(self):
        filename = "dx/raa00-dx_10908-0806021655-fbg---bin.gz"
        dxfile = util.get_wradlib_data_file(filename)
        data, attrs = io.radolan.read_dx(dxfile)


class TestMisc:
    def test_write_polygon_to_text(self):
        poly1 = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        poly2 = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        polygons = [poly1, poly2]
        res = [
            "Polygon\n",
            "0 0\n",
            "0 0.000000 0.000000 0.000000 0.000000\n",
            "1 0.000000 1.000000 0.000000 1.000000\n",
            "2 1.000000 1.000000 0.000000 2.000000\n",
            "3 0.000000 0.000000 0.000000 0.000000\n",
            "1 0\n",
            "0 0.000000 0.000000 0.000000 0.000000\n",
            "1 0.000000 1.000000 0.000000 1.000000\n",
            "2 1.000000 1.000000 0.000000 2.000000\n",
            "3 0.000000 0.000000 0.000000 0.000000\n",
            "END\n",
        ]
        tmp = tempfile.NamedTemporaryFile()
        name = tmp.name
        tmp.close()
        io.misc.write_polygon_to_text(name, polygons)
        assert open(name, "r").readlines() == res

    def test_pickle(self):
        arr = np.zeros((124, 248), dtype=np.int16)
        tmp = tempfile.NamedTemporaryFile()
        name = tmp.name
        tmp.close()
        io.misc.to_pickle(name, arr)
        res = io.misc.from_pickle(name)
        np.testing.assert_allclose(arr, res)

    def test_get_radiosonde(self):
        date = datetime.datetime(2013, 7, 1, 15, 30)
        res1 = np.array(
            [
                (
                    1000.0,
                    153.0,
                    17.4,
                    13.5,
                    13.5,
                    78.0,
                    78.0,
                    9.81,
                    200.0,
                    6.0,
                    290.6,
                    318.5,
                    292.3,
                )
            ],
            dtype=[
                ("PRES", "<f8"),
                ("HGHT", "<f8"),
                ("TEMP", "<f8"),
                ("DWPT", "<f8"),
                ("FRPT", "<f8"),
                ("RELH", "<f8"),
                ("RELI", "<f8"),
                ("MIXR", "<f8"),
                ("DRCT", "<f8"),
                ("SKNT", "<f8"),
                ("THTA", "<f8"),
                ("THTE", "<f8"),
                ("THTV", "<f8"),
            ],
        )

        res2 = {
            "Station identifier": "EDZE",
            "Station number": 10410,
            "Observation time": datetime.datetime(2013, 7, 1, 12, 0),
            "Station latitude": 51.4,
            "Station longitude": 6.96,
            "Station elevation": 153.0,
            "Showalter index": 6.1,
            "Lifted index": 0.58,
            "LIFT computed using virtual temperature": 0.52,
            "SWEAT index": 77.7,
            "K index": 11.7,
            "Cross totals index": 13.7,
            "Vertical totals index": 28.7,
            "Totals totals index": 42.4,
            "Convective Available Potential Energy": 6.9,
            "CAPE using virtual temperature": 17.78,
            "Convective Inhibition": 0.0,
            "CINS using virtual temperature": 0.0,
            "Equilibrum Level": 597.86,
            "Equilibrum Level using virtual temperature": 589.7,
            "Equivalent potential temp [K] of the LCL": 315.05,
            "Level of Free Convection": 931.41,
            "LFCT using virtual temperature": 934.07,
            "Bulk Richardson Number": 0.24,
            "Bulk Richardson Number using CAPV": 0.62,
            "Temp [K] of the Lifted Condensation Level": 284.16,
            "Pres [hPa] of the Lifted Condensation Level": 934.07,
            "Mean mixed layer potential temperature": 289.76,
            "Mean mixed layer mixing ratio": 8.92,
            "1000 hPa to 500 hPa thickness": 5537.0,
            "Precipitable water [mm] for entire sounding": 19.02,
        }

        res3 = {
            "PRES": "hPa",
            "HGHT": "m",
            "TEMP": "C",
            "DWPT": "C",
            "FRPT": "C",
            "RELH": "%",
            "RELI": "%",
            "MIXR": "g/kg",
            "DRCT": "deg",
            "SKNT": "knot",
            "THTA": "K",
            "THTE": "K",
            "THTV": "K",
        }
        import urllib

        try:
            with pytest.raises(ValueError):
                data, meta = io.misc.get_radiosonde(10411, date)
            data, meta = io.misc.get_radiosonde(10410, date)
        except urllib.error.HTTPError:
            print("HTTPError while retrieving radiosonde data, test skipped!")
        else:
            assert data[0] == res1[0]
            quant = meta.pop("quantity")
            assert meta == res2
            assert meta == res2
            assert quant == res3

    @requires_data
    def test_get_membership_functions(self):
        filename = util.get_wradlib_data_file("misc/msf_xband.gz")
        msf = io.misc.get_membership_functions(filename)
        res = np.array(
            [
                [6.000e00, 5.000e00, 1.000e01, 3.500e01, 4.000e01],
                [6.000e00, -7.458e-01, -4.457e-01, 5.523e-01, 8.523e-01],
                [6.000e00, 7.489e-01, 7.689e-01, 9.236e-01, 9.436e-01],
                [6.000e00, -5.037e-01, -1.491e-01, -1.876e-01, 1.673e-01],
                [6.000e00, -5.000e00, 0.000e00, 4.000e01, 2.000e03],
            ]
        )
        assert msf.shape == (11, 5, 55, 5)
        np.testing.assert_array_equal(msf[0, :, 8, :], res)


class TestHDF5:
    def test_to_hdf5(self):
        arr = np.zeros((124, 248), dtype=np.int16)
        metadata = {"test": 12.0}
        tmp = tempfile.NamedTemporaryFile()
        name = tmp.name
        tmp.close()
        io.hdf.to_hdf5(name, arr, metadata=metadata)
        res, resmeta = io.hdf.from_hdf5(name)
        np.testing.assert_allclose(arr, res)
        assert metadata == resmeta

        with pytest.raises(KeyError):
            io.hdf.from_hdf5(name, "NotAvailable")

    @requires_data
    def test_read_safnwc(self):
        filename = "hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5"
        safnwcfile = util.get_wradlib_data_file(filename)
        io.gdal.read_safnwc(safnwcfile)

        command = "rm -rf test1.h5"
        subprocess.check_call(command, shell=True)
        command = "h5copy -i {} -o test1.h5 -s CT -d CT".format(safnwcfile)
        subprocess.check_call(command, shell=True)
        with pytest.raises(KeyError):
            io.gdal.read_safnwc("test1.h5")

    @requires_data
    def test_read_gpm(self):
        filename1 = (
            "gpm/2A-CS-151E24S154E30S.GPM.Ku.V7-20170308.20141206-"
            "S095002-E095137.004383.V05A.HDF5"
        )
        gpm_file = util.get_wradlib_data_file(filename1)
        filename2 = "hdf5/IDR66_20141206_094829.vol.h5"
        gr2gpm_file = util.get_wradlib_data_file(filename2)
        gr_data = io.netcdf.read_generic_netcdf(gr2gpm_file)
        dset = gr_data["dataset{0}".format(2)]
        nray_gr = dset["where"]["nrays"]
        ngate_gr = dset["where"]["nbins"].astype("i4")
        elev_gr = dset["where"]["elangle"]
        dr_gr = dset["where"]["rscale"]
        lon0_gr = gr_data["where"]["lon"]
        lat0_gr = gr_data["where"]["lat"]
        alt0_gr = gr_data["where"]["height"]
        coord = georef.sweep_centroids(nray_gr, dr_gr, ngate_gr, elev_gr)
        coords = georef.spherical_to_proj(
            coord[..., 0], coord[..., 1], coord[..., 2], (lon0_gr, lat0_gr, alt0_gr)
        )
        lon = coords[..., 0]
        lat = coords[..., 1]
        bbox = zonalstats.get_bbox(lon, lat)
        io.hdf.read_gpm(gpm_file, bbox)

    @requires_data
    def test_read_trmm(self):
        # define TRMM data sets
        trmm_2a23_file = util.get_wradlib_data_file(
            "trmm/2A-CS-151E24S154E30S.TRMM.PR.2A23.20100206-"
            "S111425-E111526.069662.7.HDF"
        )
        trmm_2a25_file = util.get_wradlib_data_file(
            "trmm/2A-CS-151E24S154E30S.TRMM.PR.2A25.20100206-"
            "S111425-E111526.069662.7.HDF"
        )

        filename2 = "hdf5/IDR66_20141206_094829.vol.h5"
        gr2gpm_file = util.get_wradlib_data_file(filename2)
        gr_data = io.netcdf.read_generic_netcdf(gr2gpm_file)
        dset = gr_data["dataset{0}".format(2)]
        nray_gr = dset["where"]["nrays"]
        ngate_gr = dset["where"]["nbins"].astype("i4")
        elev_gr = dset["where"]["elangle"]
        dr_gr = dset["where"]["rscale"]
        lon0_gr = gr_data["where"]["lon"]
        lat0_gr = gr_data["where"]["lat"]
        alt0_gr = gr_data["where"]["height"]
        coord = georef.sweep_centroids(nray_gr, dr_gr, ngate_gr, elev_gr)
        coords = georef.spherical_to_proj(
            coord[..., 0], coord[..., 1], coord[..., 2], (lon0_gr, lat0_gr, alt0_gr)
        )
        lon = coords[..., 0]
        lat = coords[..., 1]
        bbox = zonalstats.get_bbox(lon, lat)

        io.hdf.read_trmm(trmm_2a23_file, trmm_2a25_file, bbox)

    @requires_data
    def test_read_generic_hdf5(self):
        filename = "hdf5/IDR66_20141206_094829.vol.h5"
        h5_file = util.get_wradlib_data_file(filename)
        io.hdf.read_generic_hdf5(h5_file)

    @requires_data
    def test_read_opera_hdf5(self):
        filename = "hdf5/IDR66_20141206_094829.vol.h5"
        h5_file = util.get_wradlib_data_file(filename)
        io.hdf.read_opera_hdf5(h5_file)

    @requires_data
    def test_read_gamic_hdf5(self):
        ppi = "hdf5/2014-08-10--182000.ppi.mvol"
        rhi = "hdf5/2014-06-09--185000.rhi.mvol"
        filename = (
            "gpm/2A-CS-151E24S154E30S.GPM.Ku.V7-20170308.20141206-"
            "S095002-E095137.004383.V05A.HDF5"
        )

        h5_file = util.get_wradlib_data_file(ppi)
        io.hdf.read_gamic_hdf5(h5_file)
        h5_file = util.get_wradlib_data_file(rhi)
        io.hdf.read_gamic_hdf5(h5_file)
        h5_file = util.get_wradlib_data_file(filename)
        with pytest.raises(KeyError):
            io.hdf.read_gamic_hdf5(h5_file)


class TestRadolan:
    def test_get_radolan_header_token(self):
        keylist = [
            "BY",
            "VS",
            "SW",
            "PR",
            "INT",
            "GP",
            "MS",
            "LV",
            "CS",
            "MX",
            "BG",
            "ST",
            "VV",
            "MF",
            "QN",
            "VR",
            "U",
        ]
        head = io.radolan.get_radolan_header_token()
        for key in keylist:
            assert head[key] is None

    def test_get_radolan_header_token_pos(self):
        header = (
            "RW030950100000814BY1620130VS 3SW   2.13.1PR E-01"
            "INT  60GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,"
            "asd,neu,nhb,oft,tur,isn,fbg,mem>"
        )

        test_head = io.radolan.get_radolan_header_token()
        test_head["PR"] = (43, 48)
        test_head["GP"] = (57, 66)
        test_head["INT"] = (51, 55)
        test_head["SW"] = (32, 41)
        test_head["VS"] = (28, 30)
        test_head["MS"] = (68, 128)
        test_head["BY"] = (19, 26)

        head = io.radolan.get_radolan_header_token_pos(header)
        assert head == test_head

        header = (
            "RQ210945100000517BY1620162VS 2SW 1.7.2PR E-01"
            "INT 60GP 900x 900VV 0MF 00000002QN 001"
            "MS 67<bln,drs,eis,emd,ess,fbg,fld,fra,ham,han,muc,"
            "neu,nhb,ros,tur,umd>"
        )
        test_head = {
            "BY": (19, 26),
            "VS": (28, 30),
            "SW": (32, 38),
            "PR": (40, 45),
            "INT": (48, 51),
            "GP": (53, 62),
            "MS": (85, 153),
            "LV": None,
            "CS": None,
            "MX": None,
            "BG": None,
            "ST": None,
            "VV": (64, 66),
            "MF": (68, 77),
            "QN": (79, 83),
            "VR": None,
            "U": None,
        }
        head = io.radolan.get_radolan_header_token_pos(header)
        assert head == test_head

    def test_decode_radolan_runlength_line(self):
        # fmt: off
        testarr = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
                   9., 9., 9.,
                   9., 9., 9., 9., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # fmt: on
        testline = (
            b"\x10\x98\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9"
            b"\xf9\xf9\xf9\xf9\xf9\xf9\xd9\n"
        )
        testline1 = b"\x10\n"
        testattrs = {"ncol": 460, "nodataflag": 0}
        arr = np.frombuffer(testline, np.uint8).astype(np.uint8)
        line = io.radolan.decode_radolan_runlength_line(arr, testattrs)
        np.testing.assert_allclose(line, testarr)
        arr = np.frombuffer(testline1, np.uint8).astype(np.uint8)
        line = io.radolan.decode_radolan_runlength_line(arr, testattrs)
        np.testing.assert_allclose(line, [0] * 460)

    def test_read_radolan_runlength_line(self):
        testline = (
            b"\x10\x98\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9"
            b"\xf9\xf9\xf9\xf9\xf9\xf9\xd9\n"
        )
        testarr = np.frombuffer(testline, np.uint8).astype(np.uint8)
        fid, temp_path = tempfile.mkstemp()
        tmp_id = open(temp_path, "wb")
        tmp_id.write(testline)
        tmp_id.close()
        tmp_id = open(temp_path, "rb")
        line = io.radolan.read_radolan_runlength_line(tmp_id)
        tmp_id.close()
        os.close(fid)
        os.remove(temp_path)
        np.testing.assert_allclose(line, testarr)

    @requires_data
    def test_decode_radolan_runlength_array(self):
        filename = "radolan/misc/raa00-pc_10015-1408030905-dwd---bin.gz"
        pg_file = util.get_wradlib_data_file(filename)
        pg_fid = io.radolan.get_radolan_filehandle(pg_file)
        header = io.radolan.read_radolan_header(pg_fid)
        attrs = io.radolan.parse_dwd_composite_header(header)
        data = io.radolan.read_radolan_binary_array(pg_fid, attrs["datasize"])
        attrs["nodataflag"] = 255
        arr = io.radolan.decode_radolan_runlength_array(data, attrs)
        assert arr.shape == (460, 460)

    @requires_data
    def test_read_radolan_binary_array(self):
        filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
        rw_file = util.get_wradlib_data_file(filename)
        rw_fid = io.radolan.get_radolan_filehandle(rw_file)
        header = io.radolan.read_radolan_header(rw_fid)
        attrs = io.radolan.parse_dwd_composite_header(header)
        data = io.radolan.read_radolan_binary_array(rw_fid, attrs["datasize"])
        assert len(data) == attrs["datasize"]

        rw_fid = io.radolan.get_radolan_filehandle(rw_file)
        header = io.radolan.read_radolan_header(rw_fid)
        attrs = io.radolan.parse_dwd_composite_header(header)
        with pytest.raises(IOError):
            io.radolan.read_radolan_binary_array(rw_fid, attrs["datasize"] + 10)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="no gunzip on windows")
    @requires_data
    def test_get_radolan_filehandle(self):
        filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
        rw_file = util.get_wradlib_data_file(filename)
        rw_fid = io.radolan.get_radolan_filehandle(rw_file)
        assert rw_file == rw_fid.name

        command = "gunzip -k -f {}".format(rw_file)
        subprocess.check_call(command, shell=True)

        rw_fid = io.radolan.get_radolan_filehandle(rw_file[:-3])
        assert rw_file[:-3] == rw_fid.name

    def test_read_radolan_header(self):
        rx_header = (
            b"RW030950100000814BY1620130VS 3SW   2.13.1PR E-01"
            b"INT  60GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,"
            b"asd,neu,nhb,oft,tur,isn,fbg,mem>"
        )

        buf = sio.BytesIO(rx_header)
        with pytest.raises(EOFError):
            io.radolan.read_radolan_header(buf)

        buf = sio.BytesIO(rx_header + b"\x03")
        header = io.radolan.read_radolan_header(buf)
        assert header == rx_header.decode()

    def test_parse_dwd_composite_header(self):
        rx_header = (
            "RW030950100000814BY1620130VS 3SW   2.13.1PR E-01INT  60"
            "GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,asd,neu,nhb,"
            "oft,tur,isn,fbg,mem>"
        )
        test_rx = {
            "maxrange": "150 km",
            "radarlocations": [
                "boo",
                "ros",
                "emd",
                "hnr",
                "pro",
                "ess",
                "asd",
                "neu",
                "nhb",
                "oft",
                "tur",
                "isn",
                "fbg",
                "mem",
            ],
            "nrow": 900,
            "intervalseconds": 3600,
            "precision": 0.1,
            "datetime": datetime.datetime(2014, 8, 3, 9, 50),
            "ncol": 900,
            "radolanversion": "2.13.1",
            "producttype": "RW",
            "radarid": "10000",
            "datasize": 1620001,
        }

        pg_header = (
            "PG030905100000814BY20042LV 6  1.0 19.0 28.0 37.0 46.0 "
            "55.0CS0MX 0MS 82<boo,ros,emd,hnr,pro,ess,asd,neu,nhb,"
            "oft,tur,isn,fbg,mem,czbrd> are used, BG460460"
        )
        test_pg = {
            "radarlocations": [
                "boo",
                "ros",
                "emd",
                "hnr",
                "pro",
                "ess",
                "asd",
                "neu",
                "nhb",
                "oft",
                "tur",
                "isn",
                "fbg",
                "mem",
                "czbrd",
            ],
            "nrow": 460,
            "level": [1.0, 19.0, 28.0, 37.0, 46.0, 55.0],
            "datetime": datetime.datetime(2014, 8, 3, 9, 5),
            "ncol": 460,
            "producttype": "PG",
            "radarid": "10000",
            "nlevel": 6,
            "indicator": "near ground level",
            "imagecount": 0,
            "datasize": 19889,
        }

        rq_header = (
            "RQ210945100000517BY1620162VS 2SW 1.7.2PR E-01"
            "INT 60GP 900x 900VV 0MF 00000002QN 001"
            "MS 67<bln,drs,eis,emd,ess,fbg,fld,fra,ham,han,muc,"
            "neu,nhb,ros,tur,umd>"
        )

        test_rq = {
            "producttype": "RQ",
            "datetime": datetime.datetime(2017, 5, 21, 9, 45),
            "radarid": "10000",
            "datasize": 1620008,
            "maxrange": "128 km",
            "radolanversion": "1.7.2",
            "precision": 0.1,
            "intervalseconds": 3600,
            "nrow": 900,
            "ncol": 900,
            "radarlocations": [
                "bln",
                "drs",
                "eis",
                "emd",
                "ess",
                "fbg",
                "fld",
                "fra",
                "ham",
                "han",
                "muc",
                "neu",
                "nhb",
                "ros",
                "tur",
                "umd",
            ],
            "predictiontime": 0,
            "moduleflag": 2,
            "quantification": 1,
        }

        sq_header = (
            "SQ102050100000814BY1620231VS 3SW   2.13.1PR E-01"
            "INT 360GP 900x 900MS 62<boo,ros,emd,hnr,umd,pro,ess,"
            "asd,neu,nhb,oft,tur,isn,fbg,mem> ST 92<asd 6,boo 6,"
            "emd 6,ess 6,fbg 6,hnr 6,isn 6,mem 6,neu 6,nhb 6,oft 6,"
            "pro 6,ros 6,tur 6,umd 6>"
        )

        test_sq = {
            "producttype": "SQ",
            "datetime": datetime.datetime(2014, 8, 10, 20, 50),
            "radarid": "10000",
            "datasize": 1620001,
            "maxrange": "150 km",
            "radolanversion": "2.13.1",
            "precision": 0.1,
            "intervalseconds": 21600,
            "nrow": 900,
            "ncol": 900,
            "radarlocations": [
                "boo",
                "ros",
                "emd",
                "hnr",
                "umd",
                "pro",
                "ess",
                "asd",
                "neu",
                "nhb",
                "oft",
                "tur",
                "isn",
                "fbg",
                "mem",
            ],
            "radardays": [
                "asd 6",
                "boo 6",
                "emd 6",
                "ess 6",
                "fbg 6",
                "hnr 6",
                "isn 6",
                "mem 6",
                "neu 6",
                "nhb 6",
                "oft 6",
                "pro 6",
                "ros 6",
                "tur 6",
                "umd 6",
            ],
        }

        yw_header = (
            "YW070235100001014BY1980156VS 3SW   2.18.3PR E-02"
            "INT   5U0GP1100x 900MF 00000000VR2017.002"
            "MS 61<boo,ros,emd,hnr,umd,pro,ess,asd,neu,"
            "nhb,oft,tur,isn,fbg,mem>"
        )

        test_yw = {
            "producttype": "YW",
            "datetime": datetime.datetime(2014, 10, 7, 2, 35),
            "radarid": "10000",
            "datasize": 1980000,
            "maxrange": "150 km",
            "radolanversion": "2.18.3",
            "precision": 0.01,
            "intervalseconds": 300,
            "intervalunit": 0,
            "nrow": 1100,
            "ncol": 900,
            "moduleflag": 0,
            "reanalysisversion": "2017.002",
            "radarlocations": [
                "boo",
                "ros",
                "emd",
                "hnr",
                "umd",
                "pro",
                "ess",
                "asd",
                "neu",
                "nhb",
                "oft",
                "tur",
                "isn",
                "fbg",
                "mem",
            ],
        }

        rx = io.radolan.parse_dwd_composite_header(rx_header)
        pg = io.radolan.parse_dwd_composite_header(pg_header)
        rq = io.radolan.parse_dwd_composite_header(rq_header)
        sq = io.radolan.parse_dwd_composite_header(sq_header)
        yw = io.radolan.parse_dwd_composite_header(yw_header)

        for key, value in rx.items():
            assert value == test_rx[key]
        for key, value in pg.items():
            if type(value) == np.ndarray:
                np.testing.assert_allclose(value, test_pg[key])
            else:
                assert value == test_pg[key]
        for key, value in rq.items():
            if type(value) == np.ndarray:
                np.testing.assert_allclose(value, test_rq[key])
            else:
                assert value == test_rq[key]
        for key, value in sq.items():
            if type(value) == np.ndarray:
                np.testing.assert_allclose(value, test_sq[key])
            else:
                assert value == test_sq[key]
        for key, value in yw.items():
            if type(value) == np.ndarray:
                np.testing.assert_allclose(value, test_yw[key])
            else:
                assert value == test_yw[key]

    @requires_data
    def test_read_radolan_composite(self):
        filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
        rw_file = util.get_wradlib_data_file(filename)
        test_attrs = {
            "maxrange": "150 km",
            "radarlocations": [
                "boo",
                "ros",
                "emd",
                "hnr",
                "pro",
                "ess",
                "asd",
                "neu",
                "nhb",
                "oft",
                "tur",
                "isn",
                "fbg",
                "mem",
            ],
            "nrow": 900,
            "intervalseconds": 3600,
            "precision": 0.1,
            "datetime": datetime.datetime(2014, 8, 3, 9, 50),
            "ncol": 900,
            "radolanversion": "2.13.1",
            "producttype": "RW",
            "nodataflag": -9999,
            "datasize": 1620000,
            "radarid": "10000",
        }

        # test for complete file
        data, attrs = io.radolan.read_radolan_composite(rw_file)
        assert data.shape == (900, 900)

        for key, value in attrs.items():
            if type(value) == np.ndarray:
                assert value.dtype in [np.int32, np.int64]
            else:
                assert value == test_attrs[key]

        # Do the same for the case where a file handle is passed
        # instead of a file name
        with gzip.open(rw_file) as fh:
            data, attrs = io.radolan.read_radolan_composite(fh)
            assert data.shape == (900, 900)

        for key, value in attrs.items():
            if type(value) == np.ndarray:
                assert value.dtype in [np.int32, np.int64]
            else:
                assert value == test_attrs[key]

        # test for loaddata=False
        data, attrs = io.radolan.read_radolan_composite(rw_file, loaddata=False)
        assert data is None
        for key, value in attrs.items():
            if type(value) == np.ndarray:
                assert value.dtype == np.int64
            else:
                assert value == test_attrs[key]
        with pytest.raises(KeyError):
            attrs["nodataflag"]

        filename = "radolan/misc/raa01-rx_10000-1408102050-dwd---bin.gz"
        rx_file = util.get_wradlib_data_file(filename)
        data, attrs = io.radolan.read_radolan_composite(rx_file)

        filename = "radolan/misc/raa00-pc_10015-1408030905-dwd---bin.gz"
        pc_file = util.get_wradlib_data_file(filename)
        data, attrs = io.radolan.read_radolan_composite(pc_file)

        # xarray test
        filename = "radolan/misc/raa01-rx_10000-1408102050-dwd---bin.gz"
        rx_file = util.get_wradlib_data_file(filename)
        data, attrs = io.radolan.read_radolan_composite(rx_file, loaddata="xarray")
        assert data.RX.shape == (900, 900)
        assert data.dims == {"x": 900, "y": 900}
        assert data.RX.dims == ("y", "x")
        assert data.time.values == np.datetime64("2014-08-10T20:50:00.000000000")


class TestRainbow:
    @requires_data
    def test_read_rainbow(self):
        filename = "rainbow/2013070308340000dBuZ.azi"
        rb_file = util.get_wradlib_data_file(filename)
        with pytest.raises(IOError):
            io.rainbow.read_rainbow("test")
        # Test reading from file name
        rb_dict = io.rainbow.read_rainbow(rb_file)
        assert rb_dict["volume"]["@datetime"] == "2013-07-03T08:33:55"
        # Test reading from file handle
        with open(rb_file, "rb") as rb_fh:
            rb_dict = io.rainbow.read_rainbow(rb_fh)
            assert rb_dict["volume"]["@datetime"] == "2013-07-03T08:33:55"

    def test_find_key(self):
        indict = {
            "A": {
                "AA": {"AAA": 0, "X": 1},
                "AB": {"ABA": 2, "X": 3},
                "AC": {"ACA": 4, "X": 5},
                "AD": [{"ADA": 4, "X": 2}],
            }
        }
        outdict = [
            {"X": 1, "AAA": 0},
            {"ABA": 2, "X": 3},
            {"X": 5, "ACA": 4},
            {"ADA": 4, "X": 2},
        ]

        assert list(io.rainbow.find_key("X", indict)) == outdict
        assert list(io.rainbow.find_key("Y", indict)) == []

    def test_decompress(self):
        dstring = b"very special compressed string"
        cstring = zlib.compress(dstring)
        assert io.rainbow.decompress(cstring) == dstring

    def test_get_rb_data_layout(self):
        assert io.rainbow.get_rb_data_layout(8) == (1, ">u1")
        assert io.rainbow.get_rb_data_layout(16) == (2, ">u2")
        assert io.rainbow.get_rb_data_layout(32) == (4, ">u4")
        with pytest.raises(ValueError):
            io.rainbow.get_rb_data_layout(128)

    def test_get_rb_data_layout_big(self):
        from unittest.mock import patch

        with patch("sys.byteorder", "big"):
            assert io.rainbow.get_rb_data_layout(8) == (1, "<u1")
            assert io.rainbow.get_rb_data_layout(16) == (2, "<u2")
            assert io.rainbow.get_rb_data_layout(32) == (4, "<u4")

    def test_get_rb_data_attribute(self):
        xmltodict = util.import_optional("xmltodict")
        data = xmltodict.parse(
            (
                '<slicedata time="13:30:05" date="2013-04-26">'
                '#<rayinfo refid="startangle" blobid="0" '
                'rays="361" depth="16"/> '
                '#<rawdata blobid="1" rays="361" type="dBuZ" '
                'bins="400" min="-31.5" max="95.5" '
                'depth="8"/> #</slicedata>'
            )
        )
        data = list(io.rainbow.find_key("@blobid", data))
        assert io.rainbow.get_rb_data_attribute(data[0], "blobid") == 0
        assert io.rainbow.get_rb_data_attribute(data[1], "blobid") == 1
        assert io.rainbow.get_rb_data_attribute(data[0], "rays") == 361
        assert io.rainbow.get_rb_data_attribute(data[1], "rays") == 361
        assert io.rainbow.get_rb_data_attribute(data[1], "bins") == 400
        with pytest.raises(KeyError):
            io.rainbow.get_rb_data_attribute(data[0], "Nonsense")
        assert io.rainbow.get_rb_data_attribute(data[0], "depth") == 16

    def test_get_rb_blob_attribute(self):
        xmltodict = util.import_optional("xmltodict")
        xmldict = xmltodict.parse(
            '<BLOB blobid="0" size="737" compression="qt"></BLOB>'
        )
        assert io.rainbow.get_rb_blob_attribute(xmldict, "compression") == "qt"
        assert io.rainbow.get_rb_blob_attribute(xmldict, "size") == "737"
        assert io.rainbow.get_rb_blob_attribute(xmldict, "blobid") == "0"
        with pytest.raises(KeyError):
            io.rainbow.get_rb_blob_attribute(xmldict, "Nonsense")

    def test_get_rb_data_shape(self):
        xmltodict = util.import_optional("xmltodict")
        data = xmltodict.parse(
            (
                '<slicedata time="13:30:05" date="2013-04-26">'
                '#<rayinfo refid="startangle" blobid="0" '
                'rays="361" depth="16"/> #<rawdata blobid="1" '
                'rays="361" type="dBuZ" bins="400" '
                'min="-31.5" max="95.5" depth="8"/> #<flagmap '
                'blobid="2" rows="800" type="dBuZ" '
                'columns="400" min="-31.5" max="95.5" '
                'depth="6"/> #<defect blobid="3" type="dBuZ" '
                'columns="400" min="-31.5" max="95.5" '
                'depth="6"/> #<rawdata2 '
                'blobid="4" rows="800" type="dBuZ" '
                'columns="400" min="-31.5" max="95.5" '
                'depth="8"/> #</slicedata>'
            )
        )
        data = list(io.rainbow.find_key("@blobid", data))
        assert io.rainbow.get_rb_data_shape(data[0]) == 361
        assert io.rainbow.get_rb_data_shape(data[1]) == (361, 400)
        assert io.rainbow.get_rb_data_shape(data[2]) == (800, 400, 6)
        assert io.rainbow.get_rb_data_shape(data[4]) == (800, 400)
        with pytest.raises(KeyError):
            io.rainbow.get_rb_data_shape(data[3])

    def test_map_rb_data(self):
        indata = b"0123456789"
        outdata8 = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57], dtype=np.uint8)
        outdata16 = np.array([12337, 12851, 13365, 13879, 14393], dtype=np.uint16)
        outdata32 = np.array([808530483, 875902519], dtype=np.uint32)
        np.testing.assert_allclose(io.rainbow.map_rb_data(indata, 8), outdata8)
        np.testing.assert_allclose(io.rainbow.map_rb_data(indata, 16), outdata16)
        np.testing.assert_allclose(io.rainbow.map_rb_data(indata, 32), outdata32)
        flagdata = b"1"
        np.testing.assert_allclose(
            io.rainbow.map_rb_data(flagdata, 1), [0, 0, 1, 1, 0, 0, 0, 1]
        )

    def test_get_rb_blob_data(self):
        datastring = b'<BLOB blobid="0" size="737" compression="qt"></BLOB>'
        with pytest.raises(EOFError):
            io.rainbow.get_rb_blob_data(datastring, 1)

    @requires_data
    def test_get_rb_blob_from_file(self):
        filename = "rainbow/2013070308340000dBuZ.azi"
        rb_file = util.get_wradlib_data_file(filename)
        rbdict = io.rainbow.read_rainbow(rb_file, loaddata=False)
        rbblob = rbdict["volume"]["scan"]["slice"]["slicedata"]["rawdata"]
        # Check reading from file handle
        with open(rb_file, "rb") as rb_fh:
            data = io.rainbow.get_rb_blob_from_file(rb_fh, rbblob)
            assert data.shape[0] == int(rbblob["@rays"])
            assert data.shape[1] == int(rbblob["@bins"])
            with pytest.raises(IOError):
                io.rainbow.get_rb_blob_from_file("rb_fh", rbblob)
        # Check reading from file path
        data = io.rainbow.get_rb_blob_from_file(rb_file, rbblob)
        assert data.shape[0] == int(rbblob["@rays"])
        assert data.shape[1] == int(rbblob["@bins"])
        with pytest.raises(IOError):
            io.rainbow.get_rb_blob_from_file("rb_fh", rbblob)

    @requires_data
    def test_get_rb_file_as_string(self):
        filename = "rainbow/2013070308340000dBuZ.azi"
        rb_file = util.get_wradlib_data_file(filename)
        with open(rb_file, "rb") as rb_fh:
            rb_string = io.rainbow.get_rb_file_as_string(rb_fh)
            assert rb_string
            with pytest.raises(IOError):
                io.rainbow.get_rb_file_as_string("rb_fh")

    def test_get_rb_header(self):
        rb_header = (
            b'<volume version="5.34.16" '
            b'datetime="2013-07-03T08:33:55"'
            b' type="azi" owner="RainAnalyzer"> '
            b'<scan name="analyzer.azi" time="08:34:00" '
            b'date="2013-07-03">'
        )

        buf = sio.BytesIO(rb_header)
        with pytest.raises(IOError):
            io.rainbow.get_rb_header(buf)

    @requires_data
    def test_get_rb_header_from_file(self):
        filename = "rainbow/2013070308340000dBuZ.azi"
        rb_file = util.get_wradlib_data_file(filename)
        with open(rb_file, "rb") as rb_fh:
            rb_header = io.rainbow.get_rb_header(rb_fh)
            assert rb_header["volume"]["@version"] == "5.34.16"


class TestRaster:
    def test_gdal_create_dataset(self):
        testfunc = io.gdal.gdal_create_dataset
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        with pytest.raises(TypeError):
            testfunc("AIG", tmp)
        from osgeo import gdal

        with pytest.raises(TypeError):
            testfunc(
                "AAIGrid", tmp, cols=10, rows=10, bands=1, gdal_type=gdal.GDT_Float32
            )
        testfunc("GTiff", tmp, cols=10, rows=10, bands=1, gdal_type=gdal.GDT_Float32)
        testfunc(
            "GTiff",
            tmp,
            cols=10,
            rows=10,
            bands=1,
            gdal_type=gdal.GDT_Float32,
            remove=True,
        )

    @requires_data
    def test_write_raster_dataset(self):
        filename = "geo/bonn_new.tif"
        geofile = util.get_wradlib_data_file(filename)
        ds = io.gdal.open_raster(geofile)
        io.gdal.write_raster_dataset(geofile + "asc", ds, "AAIGrid")
        io.gdal.write_raster_dataset(geofile + "asc", ds, "AAIGrid", remove=True)
        with pytest.raises(TypeError):
            io.gdal.write_raster_dataset(geofile + "asc1", ds, "AIG")

    @requires_data
    def test_open_raster(self):
        filename = "geo/bonn_new.tif"
        geofile = util.get_wradlib_data_file(filename)
        io.gdal.open_raster(geofile, "GTiff")


class TestVector:
    @requires_data
    def test_open_vector(self):
        filename = "shapefiles/agger/agger_merge.shp"
        geofile = util.get_wradlib_data_file(filename)
        io.gdal.open_vector(geofile)
        io.gdal.open_vector(geofile, "ESRI Shapefile")


class TestIris:
    @requires_data
    def test_open_iris(self):
        filename = "sigmet/cor-main131125105503.RAW2049"
        sigmetfile = util.get_wradlib_data_file(filename)
        data = io.iris.IrisRawFile(sigmetfile, loaddata=False)
        assert isinstance(data.rh, io.iris.IrisRecord)
        assert isinstance(data.fh, np.memmap)
        data = io.iris.IrisRawFile(sigmetfile, loaddata=True)
        assert data._record_number == 511
        assert data.filepos == 3139584

    @requires_data
    def test_read_iris(self):
        filename = "sigmet/cor-main131125105503.RAW2049"
        sigmetfile = util.get_wradlib_data_file(filename)
        data = io.iris.read_iris(sigmetfile, loaddata=True, rawdata=True)
        data_keys = [
            "product_hdr",
            "product_type",
            "ingest_header",
            "nsweeps",
            "nrays",
            "nbins",
            "data_types",
            "data",
            "raw_product_bhdrs",
        ]
        product_hdr_keys = ["structure_header", "product_configuration", "product_end"]
        ingest_hdr_keys = [
            "structure_header",
            "ingest_configuration",
            "task_configuration",
            "spare_0",
            "gparm",
            "reserved",
        ]
        data_types = [
            "DB_DBZ",
            "DB_VEL",
            "DB_ZDR",
            "DB_KDP",
            "DB_PHIDP",
            "DB_RHOHV",
            "DB_HCLASS",
        ]
        assert list(data.keys()) == data_keys
        assert list(data["product_hdr"].keys()) == product_hdr_keys
        assert list(data["ingest_header"].keys()) == ingest_hdr_keys
        assert data["data_types"] == data_types

        data_types = ["DB_DBZ", "DB_VEL"]
        selected_data = [1, 3, 8]
        loaddata = {"moment": data_types, "sweep": selected_data}
        data = io.iris.read_iris(sigmetfile, loaddata=loaddata, rawdata=True)
        assert list(data["data"][1]["sweep_data"].keys()) == data_types
        assert list(data["data"].keys()) == selected_data

    @requires_data
    def test_IrisRecord(self):
        filename = "sigmet/cor-main131125105503.RAW2049"
        sigmetfile = util.get_wradlib_data_file(filename)
        data = io.iris.IrisRecordFile(sigmetfile, loaddata=False)
        # reset record after init
        data.init_record(1)
        assert isinstance(data.rh, io.iris.IrisRecord)
        assert data.rh.pos == 0
        assert data.rh.recpos == 0
        assert data.rh.recnum == 1
        rlist = [23, 0, 4, 0, 20, 19, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(data.rh.read(10, 2), rlist)
        assert data.rh.pos == 20
        assert data.rh.recpos == 10
        data.rh.pos -= 20
        np.testing.assert_array_equal(data.rh.read(20, 1), rlist)
        data.rh.recpos -= 10
        np.testing.assert_array_equal(data.rh.read(5, 4), rlist)

    def test_decode_bin_angle(self):
        assert io.iris.decode_bin_angle(20000, 2) == 109.86328125
        assert io.iris.decode_bin_angle(2000000000, 4) == 167.63806343078613

    def decode_array(self):
        data = np.arange(0, 11)
        np.testing.assert_array_equal(
            io.iris.decode_array(data),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        np.testing.assert_array_equal(
            io.iris.decode_array(data, offset=1.0),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        )
        np.testing.assert_array_equal(
            io.iris.decode_array(data, scale=0.5),
            [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        )
        np.testing.assert_array_equal(
            io.iris.decode_array(data, offset=1.0, scale=0.5),
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
        )
        np.testing.assert_array_equal(
            io.iris.decode_array(data, offset=1.0, scale=0.5, offset2=-2.0),
            [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        )
        data = np.array(
            [0, 1, 255, 1000, 9096, 22634, 34922, 50000, 65534], dtype=np.uint16
        )
        np.testing.assert_array_equal(
            io.iris.decode_array(data, scale=1000, tofloat=True),
            [0.0, 0.001, 0.255, 1.0, 10.0, 100.0, 800.0, 10125.312, 134184.96],
        )

    def test_decode_kdp(self):
        np.testing.assert_array_almost_equal(
            io.iris.decode_kdp(np.arange(-5, 5, dtype="int8"), wavelength=10.0),
            [
                12.243229,
                12.880858,
                13.551695,
                14.257469,
                15.0,
                -0.0,
                -15.0,
                -14.257469,
                -13.551695,
                -12.880858,
            ],
        )

    def test_decode_phidp(self):
        np.testing.assert_array_almost_equal(
            io.iris.decode_phidp(
                np.arange(0, 10, dtype="uint8"), scale=254.0, offset=-1
            ),
            [
                -0.70866142,
                0.0,
                0.70866142,
                1.41732283,
                2.12598425,
                2.83464567,
                3.54330709,
                4.2519685,
                4.96062992,
                5.66929134,
            ],
        )

    def test_decode_phidp2(self):
        np.testing.assert_array_almost_equal(
            io.iris.decode_phidp2(
                np.arange(0, 10, dtype="uint16"), scale=65534.0, offset=-1
            ),
            [
                -0.00549333,
                0.0,
                0.00549333,
                0.01098666,
                0.01648,
                0.02197333,
                0.02746666,
                0.03295999,
                0.03845332,
                0.04394665,
            ],
        )

    def test_decode_sqi(self):
        np.testing.assert_array_almost_equal(
            io.iris.decode_sqi(np.arange(0, 10, dtype="uint8"), scale=253.0, offset=-1),
            [
                np.nan,
                0.0,
                0.06286946,
                0.08891084,
                0.1088931,
                0.12573892,
                0.14058039,
                0.1539981,
                0.16633696,
                0.17782169,
            ],
        )

    def test_decode_rainrate2(self):
        vals = np.array(
            [0, 1, 2, 255, 1000, 9096, 22634, 34922, 50000, 65534, 65535],
            dtype="uint16",
        )
        prod = io.iris.SIGMET_DATA_TYPES[13]
        np.testing.assert_array_almost_equal(
            io.iris.decode_array(vals.copy(), **prod["fkw"]),
            [
                -1.00000000e-04,
                0.00000000e00,
                1.00000000e-04,
                2.54000000e-02,
                9.99000000e-02,
                9.99900000e-01,
                9.99990000e00,
                7.99999000e01,
                1.01253110e03,
                1.34184959e04,
                1.34201343e04,
            ],
        )

    def test_decode_time(self):
        timestring = b"\xd1\x9a\x00\x000\t\xdd\x07\x0b\x00\x19\x00"
        assert (
            io.iris.decode_time(timestring).isoformat() == "2013-11-25T11:00:35.352000"
        )

    def test_decode_string(self):
        assert io.iris.decode_string(b"EEST\x00\x00\x00\x00") == "EEST"

    def test__get_fmt_string(self):
        fmt = "12sHHi12s12s12s6s12s12sHiiiiiiiiii2sH12sHB1shhiihh80s16s12s48s"
        assert io.iris._get_fmt_string(io.iris.PRODUCT_CONFIGURATION) == fmt


class TestNetcdf:
    @requires_data
    def test_read_edge_netcdf(self):
        filename = "netcdf/edge_netcdf.nc"
        edgefile = util.get_wradlib_data_file(filename)
        data, attrs = io.netcdf.read_edge_netcdf(edgefile)
        data, attrs = io.netcdf.read_edge_netcdf(edgefile, enforce_equidist=True)

        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        with pytest.raises(Exception):
            io.netcdf.read_edge_netcdf(ncfile)
        with pytest.raises(Exception):
            io.netcdf.read_edge_netcdf("test")

    @requires_data
    def test_read_generic_netcdf(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        io.netcdf.read_generic_netcdf(ncfile)
        with pytest.raises(IOError):
            io.netcdf.read_generic_netcdf("test")
        filename = "sigmet/cor-main131125105503.RAW2049"
        ncfile = util.get_wradlib_data_file(filename)
        with pytest.raises(IOError):
            io.netcdf.read_generic_netcdf(ncfile)

        filename = "hdf5/IDR66_20100206_111233.vol.h5"
        ncfile = util.get_wradlib_data_file(filename)
        io.netcdf.read_generic_netcdf(ncfile)

        filename = "netcdf/example_cfradial_ppi.nc"
        ncfile = util.get_wradlib_data_file(filename)
        io.netcdf.read_generic_netcdf(ncfile)


class TestXarray:
    @deprecation.fail_if_not_removed
    def test_create_xarray_dataarray(self):
        img = np.zeros((360, 10), dtype=np.float32)
        r = np.arange(0, 100000, 10000)
        az = np.arange(0, 360)
        th = np.zeros_like(az)
        proj = georef.projection.epsg_to_osr(4326)
        with pytest.raises(TypeError):
            io.xarray.create_xarray_dataarray(img)
        with pytest.warns(DeprecationWarning):
            io.xarray.create_xarray_dataarray(img, r, az, th, proj=proj)

    @requires_data
    def test_iter(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        assert len(cf) == cf.sweep
        assert len(cf) == 9

    @requires_data
    def test_del(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        for k in list(cf):
            del cf[k]
        assert cf == {}

    @requires_data
    def test_read_cfradial(self):
        sweep_names = [
            "sweep_1",
            "sweep_2",
            "sweep_3",
            "sweep_4",
            "sweep_5",
            "sweep_6",
            "sweep_7",
            "sweep_8",
            "sweep_9",
        ]
        fixed_angles = np.array(
            [0.4999, 1.0986, 1.8018, 2.5983, 3.598, 4.7021, 6.4984, 9.1022, 12.7991]
        )
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        np.testing.assert_array_almost_equal(
            cf.root.sweep_fixed_angle.values, fixed_angles
        )
        cfnames, cfangles = zip(*cf.sweeps)
        assert sweep_names == list(cfnames)
        np.testing.assert_array_almost_equal(fixed_angles, np.array(cfangles))
        assert cf.sweep == 9
        assert cf.location == (120.43350219726562, 22.52669906616211, 45.00000178813934)
        assert cf.version == "1.2"
        assert cf.Conventions == (
            "CF/Radial instrument_parameters "
            "radar_parameters radar_calibration "
            "geometry_correction"
        )

        assert repr(cf) == repr(cf._sweeps)

    @requires_data
    def test_read_odim(self):
        fixed_angles = np.array([0.3, 0.9, 1.8, 3.3, 6.0])
        filename = "hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf"
        h5file = util.get_wradlib_data_file(filename)
        cf = io.xarray.OdimH5(h5file)
        np.testing.assert_array_almost_equal(
            cf.root.sweep_fixed_angle.values, fixed_angles
        )
        filename = "hdf5/knmi_polar_volume.h5"
        h5file = util.get_wradlib_data_file(filename)
        cf = io.xarray.OdimH5(h5file)
        with pytest.raises(AttributeError):
            cf = io.xarray.OdimH5(h5file, flavour="None")

    @requires_data
    def test_read_gamic(self):
        time_cov = ("2014-08-10T18:23:35Z", "2014-08-10T18:24:05Z")
        filename = "hdf5/2014-08-10--182000.ppi.mvol"
        h5file = util.get_wradlib_data_file(filename)
        with pytest.raises(AttributeError):
            io.xarray.OdimH5(h5file)
        cf = io.xarray.OdimH5(h5file, flavour="GAMIC")
        assert str(cf.root.time_coverage_start.values) == time_cov[0]
        assert str(cf.root.time_coverage_end.values) == time_cov[1]

        filename = "hdf5/2014-06-09--185000.rhi.mvol"
        h5file = util.get_wradlib_data_file(filename)
        cf = io.xarray.OdimH5(h5file, flavour="GAMIC")
        cf = io.xarray.OdimH5(h5file, flavour="GAMIC", strict=False)

    @requires_data
    def test_odim_roundtrip(self):
        filename = "hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf"
        odimfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.OdimH5(odimfile)
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        cf.to_odim(tmp)
        cf2 = io.xarray.OdimH5(tmp)
        xr.testing.assert_equal(cf.root, cf2.root)
        for i in range(1, 6):
            key = "sweep_{}".format(i)
            xr.testing.assert_equal(cf[key], cf2[key])
        # test write after del, file lockage
        del cf2
        cf.to_odim(tmp)

    @requires_data
    def test_cfradial_roundtrip(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        cf.to_cfradial2(tmp)
        cf2 = io.xarray.CfRadial(tmp)
        xr.testing.assert_equal(cf.root, cf2.root)
        for i in range(1, 10):
            key = "sweep_{}".format(i)
            xr.testing.assert_equal(cf[key], cf2[key])
        # test write after del, file lockage
        del cf2
        cf.to_cfradial2(tmp)

    @requires_data
    def test_cfradial_odim_roundtrip(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        cf.to_odim(tmp)
        cf2 = io.xarray.OdimH5(tmp)
        xr.testing.assert_allclose(
            cf.root.sweep_fixed_angle, cf2.root.sweep_fixed_angle
        )
        xr.testing.assert_allclose(
            cf.root.time_coverage_start, cf2.root.time_coverage_start
        )
        drop = ["longitude", "latitude", "altitude", "sweep_mode"]
        xr.testing.assert_allclose(
            cf["sweep_1"].drop_vars(drop).sweep_number,
            cf2["sweep_1"].drop_vars(drop).sweep_number,
        )

        tmp1 = tempfile.NamedTemporaryFile(mode="w+b").name
        cf2.to_cfradial2(tmp1)
        cf3 = io.xarray.CfRadial(tmp1)
        xr.testing.assert_allclose(
            cf.root.time_coverage_start, cf3.root.time_coverage_start
        )

    @requires_data
    def test_georeference(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile, georef=True)
        tmp = tempfile.NamedTemporaryFile(mode="w+b").name
        cf.to_cfradial2(tmp)
        cf2 = io.xarray.CfRadial(tmp, georef=True)
        swp1 = cf["sweep_1"].copy()
        cf["sweep_1"] = cf["sweep_1"].drop_vars(["x", "y", "z", "gr", "rays", "bins"])
        cf.georeference()
        xr.testing.assert_equal(swp1, cf["sweep_1"])
        xr.testing.assert_equal(swp1, cf2["sweep_1"])

    @requires_data
    def test_root_key_warnings(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        with pytest.warns(DeprecationWarning):
            cf["root"]

    @requires_data
    def test_to_odim_warning(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        cf.root = None
        with pytest.warns(UserWarning):
            cf.to_odim("test.h5")

    @requires_data
    def test_to_cfradial2_warning(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        cf.root = None
        with pytest.warns(UserWarning):
            cf.to_cfradial2("test.nc")

    @requires_data
    def test_setitem_warning(self):
        filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
        ncfile = util.get_wradlib_data_file(filename)
        cf = io.xarray.CfRadial(ncfile)
        with pytest.warns(UserWarning):
            cf["test"] = None

    @requires_data
    def test_odim_errors(self):
        filename = "netcdf/edge_netcdf.nc"
        ncfile = util.get_wradlib_data_file(filename)
        with pytest.raises(TypeError):
            io.xarray.OdimH5(ncfile)

        filename = "netcdf/example_cfradial_ppi.nc"
        ncfile = util.get_wradlib_data_file(filename)
        with pytest.raises(AttributeError):
            io.xarray.OdimH5(ncfile)

    @requires_data
    def test_netcdf4_errors(self):
        filename = "hdf5/2014-08-10--182000.ppi.mvol"
        h5file = util.get_wradlib_data_file(filename)
        with pytest.raises(AttributeError):
            io.xarray.CfRadial(h5file, flavour="Cf/Radial3")
        with pytest.raises(AttributeError):
            io.xarray.CfRadial(h5file)


class TestDem:
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="known break on windows")
    @requires_data
    def test_get_srtm(self):
        targets = ["N51W001", "N51E000", "N51E001", "N52W001", "N52E000", "N52E001"]
        targets = ["%s.hgt.zip" % (f) for f in targets]

        opts = {"region": "Eurasia"}
        extent = [-0.3, 1.5, 51.4, 52.5]
        datasets = io.dem.get_srtm(extent, merge=False, download=opts)
        filelist = [os.path.basename(d.GetFileList()[0]) for d in datasets]
        assert targets == filelist

        targets = ["S02E015", "S02E016", "S01E015", "S01E016", "N00E015", "N00E016"]
        targets = ["%s.hgt.zip" % (f) for f in targets]

        opts = {"region": "Africa"}
        extent = [15.3, 16.6, -1.4, 0.4]
        datasets = io.dem.get_srtm(extent, merge=False, download=opts)
        filelist = [os.path.basename(d.GetFileList()[0]) for d in datasets]
        assert targets == filelist

        merged = io.dem.get_srtm(extent)

        xsize = (datasets[0].RasterXSize - 1) * 2 + 1
        ysize = (datasets[0].RasterXSize - 1) * 3 + 1
        assert merged.RasterXSize == xsize
        assert merged.RasterYSize == ysize

        geo = merged.GetGeoTransform()
        resolution = 3 / 3600
        ulcx = 15 - resolution / 2
        ulcy = 1 + resolution / 2
        geo_ref = [ulcx, resolution, 0, ulcy, 0, -resolution]
        np.testing.assert_array_almost_equal(geo, geo_ref)
