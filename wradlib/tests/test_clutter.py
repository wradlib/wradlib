import numpy as np
import wradlib.clutter as cl
import unittest


#-------------------------------------------------------------------------------
# testing the filter helper function
#-------------------------------------------------------------------------------
class TestClutter(unittest.TestCase):

    # def test_filter_gabella_a_trdefault(self):
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data)
    #     self.assertTrue(result == 4)
    #
    # def test_filter_gabella_a_tr1(self):
    #     pass
    #     data = np.arange(9)
    #     data[4] = 10
    #     result = cl.filter_gabella_a(data, tr1=5)
    #     self.assertTrue(result == 3)
            #-------------------------------------------------------------------------------
    # testing the first part of the filter
    #-------------------------------------------------------------------------------
    def filter_setup(self):
        img = np.zeros((10,10), dtype=np.float32)
        img[2,2] = 10    # isolated pixel
        img[3,8:10] = 10 # line
        img[5,:] = 5     # spike
        img[7:9,7:9] = 5 # precip field
        pass

    def test_filter_gabella_a(self):
        pass