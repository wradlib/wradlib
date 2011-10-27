#-------------------------------------------------------------------------------
# Name:        clutter_gabella_example
# Purpose:
#
# Author:      Thomas Pfaff
#
# Created:     26.10.2011
# Copyright:   (c) Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import wradlib.vis as vis
import wradlib.clutter as clutter
import os
print os.getcwd()


if __name__ == '__main__':
    # load the example data
    import numpy as np
    testdata = np.loadtxt('data/polar_dBZ.dat')

    # calculate the clutter map
    clmap = clutter.filter_gabella(testdata,
                                   wsize=5,
                                   thrsnorain=0.,
                                   tr1=6.,
                                   n_p=8,
                                   tr2=1.3)

    # visualize the result
    vis.polar_plot(clmap,
                   title='cluttermap',
                   saveto='./clutter_gabella_example.png')
