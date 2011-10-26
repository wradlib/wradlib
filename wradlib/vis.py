#-------------------------------------------------------------------------------
# Name:        vis
# Purpose:
#
# Authors:     Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:     26.10.2011
# Copyright:   (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Visualisation
^^^^^^^^^^^^^

Standard plotting and mapping procedures

.. autosummary::
   :nosignatures:
   :toctree: generated/

   polar_plot


"""

import os.path as path
import numpy as np
import pylab as pl

def polar_plot(data, title='', saveto=None, saveformat='png'):
    """Plots data from a polar grid

    Parameters:
    -----------
    data : 2-d array - polar grid data to be plotted
            1st dimension must be azimuth angles, 2nd must be ranges!
    title : string - a title of the plot
    """
    n_theta, n_r = data.shape
    R = 1.
    theta = np.linspace(0, 2*np.pi, n_theta+1)
    r = np.linspace(0., R, n_r + 1)

    # plot as pcolormesh
    fig = pl.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="polar", aspect=1.)
    pl.jet()
    circle = ax.pcolormesh(theta+np.pi/2, r, np.fliplr(np.transpose(data)),rasterized=True)
    pl.colorbar(circle, shrink=0.75)
    pl.title(title)
    if saveto==None:
        # show plot
        pl.show()
        pl.close()
    else:
        # save plot to file
        if path.exists(path.dirname(saveto)):
            pl.savefig(saveto+'.png')


if __name__ == '__main__':
    print 'wradlib: Calling module <vis> as main...'
