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

import pylab as pl
# just making sure that the plots immediately pop up
pl.interactive(True)
import wradlib.vis as vis
import wradlib.clutter as clutter
import os


def ex_clutter_gabella():
    # load the example data
    import numpy as np
    # Todo: link right data set
    testdata = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/polar_dBZ_fbg.gz')

    # calculate the clutter map
    clmap = clutter.filter_gabella(testdata,
                                   wsize=5,
                                   thrsnorain=0.,
                                   tr1=6.,
                                   n_p=8,
                                   tr2=1.3)

    # visualize the result
    ax, pm = vis.plot_ppi(clmap)
    ax.set_title('cluttermap')
    pl.show()

if __name__ == '__main__':
    ex_clutter_gabella()