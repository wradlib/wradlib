import numpy as np
import wradlib.util as util
import unittest


#-------------------------------------------------------------------------------
# testing the filter helper function
#-------------------------------------------------------------------------------
class TestUtil(unittest.TestCase):

    def img_setup(self):
        img = np.zeros((36,10), dtype=np.float32)
        img[2,2] = 1    # isolated pixel
        img[5,6:8] = 1 # line
        img[20,:] = 1     # spike
        img[9:12,4:7] = 1 # precip field
        #img[15:17,5:7] = np.nan # nodata as nans
        self.img = img
        pass
    def test_filter_window_polar(self):
        self.img_setup()
        np.set_printoptions(precision=3)
        rscale = 250
        nrays, nbins = self.img.shape
        ascale = 2*np.pi/self.img.shape[0]
        mean = util.filter_window_polar(self.img,300,"maximum",rscale)
        correct = np.array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.],
                            [ 0.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.],
                            [ 1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                            [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        self.assertTrue((mean == correct).all())

    def test_half_power_radius(self):
        hpr = util.half_power_radius(np.arange(0,100000,10000), 1.0)
        res = np.array([0., 87.266, 174.533, 261.799, 349.066, 436.332,
                        523.599, 610.865, 698.132, 785.398])
        self.assertTrue(np.allclose(hpr,res))

class FindBboxIndicesTest(unittest.TestCase):

    def setUp(self):
        xarr = np.linspace(500,1000, num=6)
        yarr = np.linspace(550,950, num=9)

        gridx, gridy = np.meshgrid(xarr, yarr)

        self.grid = np.dstack((gridx,gridy))
        self.outside = [400,400,1100,1100]
        self.inside1 = [599, 599, 901, 901]
        self.inside2 = [601, 601, 899, 899]


    def test_find_bbox_indices(self):

        bbind = util.find_bbox_indices(self.grid, self.outside)
        self.assertTrue(np.array_equal(bbind, [0, 0, self.grid.shape[1], self.grid.shape[0]]))

        bbind = util.find_bbox_indices(self.grid, self.inside1)
        self.assertTrue(np.array_equal(bbind, [0,0, self.grid.shape[1], self.grid.shape[0]]))

        bbind = util.find_bbox_indices(self.grid, self.inside2)
        self.assertTrue(np.array_equal(bbind, [1, 1, 5, 8]))



