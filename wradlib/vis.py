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
"""

def polar_plot(data):
    n_r, n_theta = data.shape
    R = 1.
    theta = np.linspace(0, 2*np.pi, n_theta+1)
    r = np.linspace(0., R, n_r + 1)

    # plot as pcolormesh
    pl.figure(figsize=(8,8))
    pl.subplot(1,1,1, projection="polar", aspect=1.)
    pl.jet()
    pl.pcolormesh(theta+np.pi/2, r, np.fliplr((data)),rasterized=True)
    pl.colorbar(shrink=0.75)
    pl.show()
    pl.close()


if __name__ == '__main__':
    print 'wradlib: Calling module <vis> as main...'
