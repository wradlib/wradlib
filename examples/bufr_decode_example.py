#-------------------------------------------------------------------------------
# Name:        vis_example
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.vis as vis
import wradlib.bufr as bufr


if __name__ == '__main__':

    import numpy as np
    # This is our test BUFR file
    buffile = "data/test.buf"
    # decode BUFR file
    descnames, descvals, vals = decodebufr(buffile)
    # print the BUFR descriptor dictionaries
    print descnames
    print descvals
    # plot the image
    classes = [-32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 70]
    vis.cartesian_plot(vals, title='Reflectivity', unit='dBZ', colormap='spectral', classes=classes, extend='max')
