

import unittest
import wradlib.io as io
import xmltodict
import numpy as np
import zlib


class IOTest(unittest.TestCase):
    # testing functions related to readDX
    def test_getTimestampFromFilename(self):
        pass

    def test_getDXTimestamp(self):
        pass

    def test_unpackDX(self):
        pass

    def test_readDX(self):
        pass

class RainbowTest(unittest.TestCase):

    def test_read_rainbow(self):
        pass
    def test_find_key(self):
        inDict = {'A': {'AA': {'AAA': 0, 'X': 1},
                       'AB': {'ABA': 2, 'X': 3},
                        'AC': {'ACA': 4, 'X': 5}}}
        outDict = [{'X': 1, 'AAA': 0}, {'X': 5, 'ACA': 4}, {'ABA': 2, 'X': 3}]
        self.assertEqual(list(io.find_key('X', inDict)), outDict)
        self.assertEqual(list(io.find_key('Y', inDict)), [])
    def test_decompress(self):
        dString = 'very special compressed string'
        cString = zlib.compress(dString)
        self.assertEqual(io.decompress(cString), dString)

    def test_get_RB_data_layout(self):
        self.assertEqual(io.get_RB_data_layout(8), (1, '>u1'))
        self.assertEqual(io.get_RB_data_layout(16), (2, '>u2'))
        self.assertEqual(io.get_RB_data_layout(32),  (4 , '>u4'))
        self.assertRaises(ValueError, lambda: io.get_RB_data_layout(128))
    def test_get_RB_data_attribute(self):
        data = xmltodict.parse('<slicedata time="13:30:05" date="2013-04-26"> \
        <rayinfo refid="startangle" blobid="0" rays="361" depth="16"/> \
        <rawdata blobid="1" rays="361" type="dBuZ" bins="400" min="-31.5" max="95.5" depth="8"/> \
        </slicedata>')
        data = list(io.find_key('@blobid', data))
        self.assertEqual(io.get_RB_data_attribute(data[0], 'blobid'), 0)
        self.assertEqual(io.get_RB_data_attribute(data[1], 'blobid'), 1)
        self.assertEqual(io.get_RB_data_attribute(data[0], 'rays'), 361)
        self.assertIsNone(io.get_RB_data_attribute(data[0], 'bins'))
        self.assertEqual(io.get_RB_data_attribute(data[1], 'rays'), 361)
        self.assertEqual(io.get_RB_data_attribute(data[1], 'bins'), 400)
        self.assertRaises(KeyError, lambda: io.get_RB_data_attribute(data[0], 'Nonsense'))
        self.assertEqual(io.get_RB_data_attribute(data[0], 'depth'), 16)
    def test_get_RB_blob_attribute(self):
        xmldict = xmltodict.parse('<BLOB blobid="0" size="737" compression="qt"></BLOB>')
        self.assertEqual(io.get_RB_blob_attribute(xmldict, 'compression'), 'qt')
        self.assertEqual(io.get_RB_blob_attribute(xmldict, 'size'), '737')
        self.assertEqual(io.get_RB_blob_attribute(xmldict, 'blobid'), '0')
        self.assertRaises(KeyError, lambda: io.get_RB_blob_attribute(xmldict, 'Nonsense'))
    def test_map_RB_data(self):
        inData='0123456789'
        outData8 = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57], dtype=np.uint8)
        outData16 = np.array([12337, 12851, 13365, 13879, 14393], dtype=np.uint16)
        outData32 = np.array([808530483, 875902519], dtype=np.uint32)
        self.assertTrue(np.allclose(io.map_RB_data(inData, 8), outData8))
        self.assertTrue(np.allclose(io.map_RB_data(inData, 16), outData16))
        self.assertTrue(np.allclose(io.map_RB_data(inData, 32), outData32))
    def test_get_RB_blob_data(self):
        datastring = '<BLOB blobid="0" size="737" compression="qt"></BLOB>'
        self.assertRaises(EOFError, lambda: io.get_RB_blob_data(datastring, 1))


if __name__ == '__main__':
    unittest.main()