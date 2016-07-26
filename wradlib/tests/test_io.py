# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import unittest
import wradlib as wrl
import numpy as np
import zlib
import tempfile
import os
import datetime
import io


class IOTest(unittest.TestCase):
    # testing functions related to readDX
    def test__getTimestampFromFilename(self):
        filename = 'raa00-dx_10488-200608050000-drs---bin'
        self.assertEqual(wrl.io._getTimestampFromFilename(filename),
                         datetime.datetime(2006, 8, 5, 0))
        filename = 'raa00-dx_10488-0608050000-drs---bin'
        self.assertEqual(wrl.io._getTimestampFromFilename(filename),
                         datetime.datetime(2006, 8, 5, 0))

    def test_getDXTimestamp(self):
        filename = 'raa00-dx_10488-200608050000-drs---bin'
        self.assertEqual(wrl.io.getDXTimestamp(filename).__str__(),
                         '2006-08-05 00:00:00+00:00')
        filename = 'raa00-dx_10488-0608050000-drs---bin'
        self.assertEqual(wrl.io.getDXTimestamp(filename).__str__(),
                         '2006-08-05 00:00:00+00:00')

    def test_unpackDX(self):
        pass

    def test_readDX(self):
        pass

    def test_writePolygon2Text(self):
        poly1 = [[0., 0., 0., 0.], [0., 1., 0., 1.], [1., 1., 0., 2.],
                 [0., 0., 0., 0.]]
        poly2 = [[0., 0., 0., 0.], [0., 1., 0., 1.], [1., 1., 0., 2.],
                 [0., 0., 0., 0.]]
        polygons = [poly1, poly2]
        res = ['Polygon\n', '0 0\n', '0 0.000000 0.000000 0.000000 0.000000\n',
               '1 0.000000 1.000000 0.000000 1.000000\n',
               '2 1.000000 1.000000 0.000000 2.000000\n',
               '3 0.000000 0.000000 0.000000 0.000000\n', '1 0\n',
               '0 0.000000 0.000000 0.000000 0.000000\n',
               '1 0.000000 1.000000 0.000000 1.000000\n',
               '2 1.000000 1.000000 0.000000 2.000000\n',
               '3 0.000000 0.000000 0.000000 0.000000\n', 'END\n']
        tmp = tempfile.NamedTemporaryFile()
        wrl.io.writePolygon2Text(tmp.name, polygons)
        self.assertEqual(open(tmp.name, 'r').readlines(), res)


class PickleTest(unittest.TestCase):
    def test_pickle(self):
        arr = np.zeros((124, 248), dtype=np.int16)
        tmp = tempfile.NamedTemporaryFile()
        wrl.io.to_pickle(tmp.name, arr)
        res = wrl.io.from_pickle(tmp.name)
        self.assertTrue(np.allclose(arr, res))


class HDF5Test(unittest.TestCase):
    def test_to_hdf5(self):
        arr = np.zeros((124, 248), dtype=np.int16)
        metadata = {'test': 12.}
        tmp = tempfile.NamedTemporaryFile()
        wrl.io.to_hdf5(tmp.name, arr, metadata=metadata)
        res, resmeta = wrl.io.from_hdf5(tmp.name)
        self.assertTrue(np.allclose(arr, res))
        self.assertDictEqual(metadata, resmeta)


class RadolanTest(unittest.TestCase):
    def test_get_radolan_header_token(self):
        keylist = ['BY', 'VS', 'SW', 'PR', 'INT', 'GP',
                   'MS', 'LV', 'CS', 'MX', 'BG']
        head = wrl.io.get_radolan_header_token()
        for key in keylist:
            self.assertIsNone(head[key])

    def test_get_radolan_header_token_pos(self):
        header = ('RW030950100000814BY1620130VS 3SW   2.13.1PR E-01'
                  'INT  60GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,'
                  'asd,neu,nhb,oft,tur,isn,fbg,mem>')

        test_head = wrl.io.get_radolan_header_token()
        test_head['PR'] = (43, 48)
        test_head['GP'] = (57, 66)
        test_head['INT'] = (51, 55)
        test_head['SW'] = (32, 41)
        test_head['VS'] = (28, 30)
        test_head['MS'] = (68, 128)
        test_head['BY'] = (19, 26)

        head = wrl.io.get_radolan_header_token_pos(header)
        self.assertDictEqual(head, test_head)

    def test_decode_radolan_runlength_line(self):
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

        testline = (b'\x10\x98\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9'
                    b'\xf9\xf9\xf9\xf9\xf9\xf9\xd9\n')
        testattrs = {'ncol': 460, 'nodataflag': 0}
        arr = np.fromstring(testline, np.uint8).astype(np.uint8)
        line = wrl.io.decode_radolan_runlength_line(arr, testattrs)
        self.assertTrue(np.allclose(line, testarr))

    def test_read_radolan_runlength_line(self):
        testline = (b'\x10\x98\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9'
                    b'\xf9\xf9\xf9\xf9\xf9\xf9\xd9\n')
        testarr = np.fromstring(testline, np.uint8).astype(np.uint8)
        fid, temp_path = tempfile.mkstemp()
        tmp_id = open(temp_path, 'wb')
        tmp_id.write(testline)
        tmp_id.close()
        tmp_id = open(temp_path, 'rb')
        line = wrl.io.read_radolan_runlength_line(tmp_id)
        tmp_id.close()
        os.close(fid)
        os.remove(temp_path)
        self.assertTrue(np.allclose(line, testarr))

    def test_decode_radolan_runlength_array(self):
        filename = 'radolan/misc/raa00-pc_10015-1408030905-dwd---bin.gz'
        pg_file = wrl.util.get_wradlib_data_file(filename)
        pg_fid = wrl.io.get_radolan_filehandle(pg_file)
        header = wrl.io.read_radolan_header(pg_fid)
        attrs = wrl.io.parse_DWD_quant_composite_header(header)
        data = wrl.io.read_radolan_binary_array(pg_fid, attrs['datasize'])
        attrs['nodataflag'] = 255
        arr = wrl.io.decode_radolan_runlength_array(data, attrs)
        self.assertEqual(arr.shape, (460, 460))

    def test_read_radolan_binary_array(self):
        filename = 'radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz'
        rw_file = wrl.util.get_wradlib_data_file(filename)
        rw_fid = wrl.io.get_radolan_filehandle(rw_file)
        header = wrl.io.read_radolan_header(rw_fid)
        attrs = wrl.io.parse_DWD_quant_composite_header(header)
        data = wrl.io.read_radolan_binary_array(rw_fid, attrs['datasize'])
        self.assertEqual(len(data), attrs['datasize'])

        rw_fid = wrl.io.get_radolan_filehandle(rw_file)
        header = wrl.io.read_radolan_header(rw_fid)
        attrs = wrl.io.parse_DWD_quant_composite_header(header)
        self.assertRaises(
            IOError,
            lambda: wrl.io.read_radolan_binary_array(rw_fid,
                                                     attrs['datasize'] + 10))

    def test_get_radolan_filehandle(self):
        filename = 'radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz'
        rw_file = wrl.util.get_wradlib_data_file(filename)
        rw_fid = wrl.io.get_radolan_filehandle(rw_file)
        self.assertEqual(rw_file, rw_fid.name)

    def test_read_radolan_header(self):
        rx_header = (b'RW030950100000814BY1620130VS 3SW   2.13.1PR E-01'
                     b'INT  60GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,'
                     b'asd,neu,nhb,oft,tur,isn,fbg,mem>')

        buf = io.BytesIO(rx_header + b"\x03")
        header = wrl.io.read_radolan_header(buf)
        self.assertEqual(header, rx_header.decode())

    def test_parse_DWD_quant_composite_header(self):
        rx_header = ('RW030950100000814BY1620130VS 3SW   2.13.1PR E-01INT  60'
                     'GP 900x 900MS 58<boo,ros,emd,hnr,pro,ess,asd,neu,nhb,'
                     'oft,tur,isn,fbg,mem>')
        test_rx = {'maxrange': '150 km',
                   'radarlocations': ['boo', 'ros', 'emd', 'hnr', 'pro',
                                      'ess', 'asd', 'neu', 'nhb', 'oft',
                                      'tur', 'isn', 'fbg', 'mem'],
                   'nrow': 900, 'intervalseconds': 3600, 'precision': 0.1,
                   'datetime': datetime.datetime(2014, 8, 3, 9, 50),
                   'ncol': 900,
                   'radolanversion': '2.13.1', 'producttype': 'RW',
                   'radarid': '10000',
                   'datasize': 1620001, }

        pg_header = ('PG030905100000814BY20042LV 6  1.0 19.0 28.0 37.0 46.0 '
                     '55.0CS0MX 0MS 82<boo,ros,emd,hnr,pro,ess,asd,neu,nhb,'
                     'oft,tur,isn,fbg,mem,czbrd> are used, BG460460')
        test_pg = {
            'radarlocations': ['boo', 'ros', 'emd', 'hnr', 'pro', 'ess', 'asd',
                               'neu',
                               'nhb', 'oft', 'tur', 'isn', 'fbg', 'mem',
                               'czbrd'],
            'nrow': 460, 'level': [1., 19., 28., 37., 46., 55.],
            'datetime': datetime.datetime(2014, 8, 3, 9, 5), 'ncol': 460,
            'producttype': 'PG', 'radarid': '10000', 'nlevel': 6,
            'indicator': 'near ground level', 'imagecount': 0,
            'datasize': 19889}

        rx = wrl.io.parse_DWD_quant_composite_header(rx_header)
        pg = wrl.io.parse_DWD_quant_composite_header(pg_header)

        for key, value in rx.items():
            self.assertEqual(value, test_rx[key])
        for key, value in pg.items():
            if type(value) == np.ndarray:
                self.assertTrue(np.allclose(value, test_pg[key]))
            else:
                self.assertEqual(value, test_pg[key])

    def test_read_RADOLAN_composite(self):
        filename = 'radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz'
        rw_file = wrl.util.get_wradlib_data_file(filename)
        test_attrs = {'maxrange': '150 km',
                      'radarlocations': ['boo', 'ros', 'emd', 'hnr', 'pro',
                                         'ess', 'asd', 'neu', 'nhb', 'oft',
                                         'tur', 'isn', 'fbg', 'mem'],
                      'nrow': 900, 'intervalseconds': 3600,
                      'precision': 0.1,
                      'datetime': datetime.datetime(2014, 8, 3, 9, 50),
                      'ncol': 900, 'radolanversion': '2.13.1',
                      'producttype': 'RW', 'nodataflag': -9999,
                      'datasize': 1620000, 'radarid': '10000'}

        # test for complete file
        data, attrs = wrl.io.read_RADOLAN_composite(rw_file)
        self.assertEqual(data.shape, (900, 900))

        for key, value in attrs.items():
            if type(value) == np.ndarray:
                self.assertIn(value.dtype, [np.int32, np.int64])
            else:
                self.assertEqual(value, test_attrs[key])

        # test for loaddata=False
        data, attrs = wrl.io.read_RADOLAN_composite(rw_file, loaddata=False)
        self.assertEqual(data, None)
        for key, value in attrs.items():
            if type(value) == np.ndarray:
                self.assertEqual(value.dtype, np.int64)
            else:
                self.assertEqual(value, test_attrs[key])
        self.assertRaises(KeyError, lambda: attrs['nodataflag'])


class RainbowTest(unittest.TestCase):
    def test_read_rainbow(self):
        pass

    def test_find_key(self):
        indict = {'A': {'AA': {'AAA': 0, 'X': 1},
                        'AB': {'ABA': 2, 'X': 3},
                        'AC': {'ACA': 4, 'X': 5}}}
        outdict = [{'X': 1, 'AAA': 0}, {'X': 5, 'ACA': 4}, {'ABA': 2, 'X': 3}]
        try:
            self.assertCountEqual(list(wrl.io.find_key('X', indict)), outdict)
            self.assertCountEqual(list(wrl.io.find_key('Y', indict)), [])
        except AttributeError:
            self.assertItemsEqual(list(wrl.io.find_key('X', indict)), outdict)
            self.assertItemsEqual(list(wrl.io.find_key('Y', indict)), [])

    def test_decompress(self):
        dstring = b'very special compressed string'
        cstring = zlib.compress(dstring)
        self.assertEqual(wrl.io.decompress(cstring), dstring)

    def test_get_RB_data_layout(self):
        self.assertEqual(wrl.io.get_RB_data_layout(8), (1, '>u1'))
        self.assertEqual(wrl.io.get_RB_data_layout(16), (2, '>u2'))
        self.assertEqual(wrl.io.get_RB_data_layout(32), (4, '>u4'))
        self.assertRaises(ValueError, lambda: wrl.io.get_RB_data_layout(128))

    def test_get_RB_data_attribute(self):
        xmltodict = wrl.util.import_optional('xmltodict')
        data = xmltodict.parse(('<slicedata time="13:30:05" date="2013-04-26">'
                                '#<rayinfo refid="startangle" blobid="0" '
                                'rays="361" depth="16"/> '
                                '#<rawdata blobid="1" rays="361" type="dBuZ" '
                                'bins="400" min="-31.5" max="95.5" '
                                'depth="8"/> #</slicedata>'))
        data = list(wrl.io.find_key('@blobid', data))
        self.assertEqual(wrl.io.get_RB_data_attribute(data[0], 'blobid'), 0)
        self.assertEqual(wrl.io.get_RB_data_attribute(data[1], 'blobid'), 1)
        self.assertEqual(wrl.io.get_RB_data_attribute(data[0], 'rays'), 361)
        self.assertEqual(wrl.io.get_RB_data_attribute(data[1], 'rays'), 361)
        self.assertEqual(wrl.io.get_RB_data_attribute(data[1], 'bins'), 400)
        self.assertRaises(KeyError,
                          lambda: wrl.io.get_RB_data_attribute(data[0],
                                                               'Nonsense'))
        self.assertEqual(wrl.io.get_RB_data_attribute(data[0], 'depth'), 16)

    def test_get_RB_blob_attribute(self):
        xmltodict = wrl.util.import_optional('xmltodict')
        xmldict = xmltodict.parse(
            '<BLOB blobid="0" size="737" compression="qt"></BLOB>')
        self.assertEqual(wrl.io.get_RB_blob_attribute(xmldict, 'compression'),
                         'qt')
        self.assertEqual(wrl.io.get_RB_blob_attribute(xmldict, 'size'), '737')
        self.assertEqual(wrl.io.get_RB_blob_attribute(xmldict, 'blobid'), '0')
        self.assertRaises(KeyError,
                          lambda: wrl.io.get_RB_blob_attribute(xmldict,
                                                               'Nonsense'))

    def test_get_RB_data_shape(self):
        xmltodict = wrl.util.import_optional('xmltodict')
        data = xmltodict.parse(('<slicedata time="13:30:05" date="2013-04-26">'
                                '#<rayinfo refid="startangle" blobid="0" '
                                'rays="361" depth="16"/> #<rawdata blobid="1" '
                                'rays="361" type="dBuZ" bins="400" '
                                'min="-31.5" max="95.5" depth="8"/> #<flagmap '
                                'blobid="2" rows="800" type="dBuZ" '
                                'columns="400" min="-31.5" max="95.5" '
                                'depth="6"/> #<defect blobid="3" type="dBuZ" '
                                'columns="400" min="-31.5" max="95.5" '
                                'depth="6"/> #</slicedata>'))
        data = list(wrl.io.find_key('@blobid', data))
        self.assertEqual(wrl.io.get_RB_data_shape(data[0]), 361)
        self.assertEqual(wrl.io.get_RB_data_shape(data[1]), (361, 400))
        self.assertEqual(wrl.io.get_RB_data_shape(data[2]), (800, 400, 6))
        self.assertRaises(KeyError, lambda: wrl.io.get_RB_data_shape(data[3]))

    def test_map_RB_data(self):
        indata = b'0123456789'
        outdata8 = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
                            dtype=np.uint8)
        outdata16 = np.array([12337, 12851, 13365, 13879, 14393],
                             dtype=np.uint16)
        outdata32 = np.array([808530483, 875902519], dtype=np.uint32)
        self.assertTrue(np.allclose(wrl.io.map_RB_data(indata, 8), outdata8))
        self.assertTrue(np.allclose(wrl.io.map_RB_data(indata, 16), outdata16))
        self.assertTrue(np.allclose(wrl.io.map_RB_data(indata, 32), outdata32))

    def test_get_RB_blob_data(self):
        datastring = b'<BLOB blobid="0" size="737" compression="qt"></BLOB>'
        self.assertRaises(EOFError,
                          lambda: wrl.io.get_RB_blob_data(datastring, 1))

    def test_get_RB_blob_from_file(self):
        filename = 'rainbow/2013070308340000dBuZ.azi'
        rb_file = wrl.util.get_wradlib_data_file(filename)
        rbdict = wrl.io.read_Rainbow(rb_file, loaddata=False)
        rbblob = rbdict['volume']['scan']['slice']['slicedata']['rawdata']
        data = wrl.io.get_RB_blob_from_file(rb_file, rbblob)
        self.assertEqual(data.shape[0], int(rbblob['@rays']))
        self.assertEqual(data.shape[1], int(rbblob['@bins']))
        self.assertRaises(IOError,
                          lambda: wrl.io.get_RB_blob_from_file('rb_file',
                                                               rbblob))

    def test_get_RB_file_as_string(self):
        filename = 'rainbow/2013070308340000dBuZ.azi'
        rb_file = wrl.util.get_wradlib_data_file(filename)
        rb_string = wrl.io.get_RB_file_as_string(rb_file)
        self.assertTrue(rb_string)
        self.assertRaises(IOError,
                          lambda: wrl.io.get_RB_file_as_string('rb_file'))

    def test_get_RB_header(self):
        filename = 'rainbow/2013070308340000dBuZ.azi'
        rb_file = wrl.util.get_wradlib_data_file(filename)
        rb_header = wrl.io.get_RB_header(rb_file)
        self.assertEqual(rb_header['volume']['@version'], '5.34.16')
        self.assertRaises(IOError,
                          lambda: wrl.io.get_RB_header('rb_file'))


if __name__ == '__main__':
    unittest.main()
