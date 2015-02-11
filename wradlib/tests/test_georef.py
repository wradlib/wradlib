#-------------------------------------------------------------------------------
# Name:        test_georef
# Purpose:     testing file for the wradlib.georef module
#
# Authors:     Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import unittest
import wradlib.georef as georef
import numpy as np
from osgeo import osr


class GetGridsTest(unittest.TestCase):

    def setUp(self):
        # calculate xy and lonlat grids with georef function
        self.radolan_grid_xy = georef.get_radolan_grid(901,901)
        self.radolan_grid_ll = georef.get_radolan_grid(901,901, wgs84=True)

    def test_get_radolan_grid(self):

        # create radolan projection osr object
        dwd_string = georef.create_projstr("dwd-radolan")
        proj_stereo = georef.proj4_to_osr(dwd_string)

        # create wgs84 projection osr object
        proj_wgs = osr.SpatialReference()
        proj_wgs.ImportFromEPSG(4326)

        # transform radolan polar stereographic projection to wgs84 and wgs84 to polar stereographic
        # using osr transformation routines
        radolan_grid_ll = georef.reproject(self.radolan_grid_xy, projection_source=proj_stereo,
                                               projection_target=proj_wgs)
        radolan_grid_xy = georef.reproject(self.radolan_grid_ll, projection_source=proj_wgs,
                                               projection_target=proj_stereo)

        # check source and target arrays for equality
        self.assertTrue(np.allclose(radolan_grid_ll, self.radolan_grid_ll))
        self.assertTrue(np.allclose(radolan_grid_xy, self.radolan_grid_xy))

if __name__ == '__main__':
    unittest.main()