#-------------------------------------------------------------------------------
# Name:        plot_ppi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_ppi
#
# Author:      Thomas Pfaff
#
# Created:     09.01.2014
# Copyright:   (c) Thomas Pfaff 2014
# Licence:     The MIT License
#-------------------------------------------------------------------------------
import numpy as np
# importing most matplotlib routines at once
import pylab as pl
# just making sure that the plots immediately pop up
pl.interactive(True)
# well, it's a wradlib example
import wradlib
import os

def ex_plot_ppi():
    # a polar scan
    img = np.loadtxt(os.path.dirname(__file__)+'/' + 'data/polar_dBZ_tur.gz')

    # the simplest call, everything else some default
    pl.figure()
    wradlib.vis.plot_ppi(img)
    pl.title('Simple PPI')

    # now let's just plot a sector of data
    pl.figure()
    # for this, we need to give the ranges and azimuths explicitly
    # and one more than we pass on in the data, because we also may not use
    # the autoext-feature, and otherwise the last row and column of our data
    # would not be plotted
    r = np.arange(40, 81)
    az = np.arange(200,251)
    wradlib.vis.plot_ppi(img[200:250, 40:80], r, az, autoext=False)
    pl.title('Sector PPI')


    # Let's get a bit more complex
    pl.figure()
    # we introduce a site offset
    wradlib.vis.plot_ppi(img, site=(10, 20))
    # and plot a crosshair over our data
    # we overwrite the default angles, adding the line at 45 degrees
    # also we overwrite some line properties (don't ask for good taste here)
    wradlib.vis.plot_ppi_crosshair(site=(10, 20),
                                   ranges=[40,80,120],
                                   angles=[0,45,90,180,270],
                                   kwds={'line':dict(color='white',
                                                     linestyle='solid')})
    pl.title('Offset and Custom Crosshair')

    # adding georeferencing
    pl.figure()
    # using the proj keyword we tell the function to:
    # - interpret the site coordinates as longitude/latitude
    # - reproject the coordinates to the dwd-radolan composite coordinate system
    wradlib.vis.plot_ppi(img, site=(10., 45.),
                         proj=wradlib.georef.create_projstr('dwd-radolan'))
    # now the crosshair must also observe the projection
    # in addition the ranges must now be given in meters
    # we now also change the circles to a different color and linestyle
    # observe the different methods to define the dictionaries.
    # they are completely equivalent. Your choice, which you like better
    wradlib.vis.plot_ppi_crosshair(site=(10., 45.),
                                   ranges=[40000, 80000, 128000],
                                   kwds={'line':dict(color='black'),
                                         'circle':{'edgecolor':'blue',
                                                   'linestyle':'dotted'},
                                        },
                                   proj=wradlib.georef.create_projstr('dwd-radolan')
                                   )
    pl.title('Georeferenced/Projected PPI')

    # some side effects of georeferencing
    pl.figure()
    # Transplanting the radar virtually moves it away from the central meridian
    # of the projection (which is 10 degrees east)
    # Due north now does not point straight upwards on the map
    wradlib.vis.plot_ppi(img, site=(45., 7.),
                         proj=wradlib.georef.create_projstr('dwd-radolan'))
    # The crosshair shows this.
    # for the case that the lines should actually become curved, they are
    # implemented as a piecewise linear curve with 10 vertices
    # The same is true for the range circles, but with more vertices, of course
    wradlib.vis.plot_ppi_crosshair(site=(45., 7.),
                                   ranges=[64000, 128000],
                                   kwds={'line':dict(color='black'),
                                         'circle':{'edgecolor':'darkgray',
                                                   'linestyle':'dotted'},
                                        },
                                   proj=wradlib.georef.create_projstr('dwd-radolan')
                                   )
    pl.title('Projection Side Effects')

    # now you might wonder, how to annotate these plots.
    # The functions don't provide anything for this, as it can be much more flexibly
    # done outside, using standard matplotlib tools
    # returning to the simple call
    pl.figure()
    ppi, pm = wradlib.vis.plot_ppi(img)
    ppi.set_xlabel('easting [km]')
    ppi.set_ylabel('northing [km]')
    ppi.set_title('PPI manipulations/colorbar')
    # you can now also zoom - either programmatically or interactively
    ppi.set_xlim(-80, -20)
    ppi.set_ylim(-80, 0)
    # as the function returns the axes- and 'mappable'-objects colorbar needs,
    # adding a colorbar is easy
    pl.colorbar(pm, ax=ppi)


if __name__ == '__main__':
    ex_plot_ppi()

