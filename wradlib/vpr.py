#-------------------------------------------------------------------------------
# Name:        vpr
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
Vertical Profile of Reflectivity (VPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Precipitation is 3-dimensional in space. The vertical distribution of precipitation
(and thus reflectivity) is typically non-uniform. As the height of the radar beam
increases with the distance from the radar location (beam elevation, earth curvature),
one sweep samples from different heights. The effects of the non-uniform VPR and
the different sampling heights need to be accounted for if we are interested in
the precipiation near the ground or in defined heights. This module is intended
to provide a set of tools to account for these effects.

The first step will normally be to reference the polar volume data in a 3-dimensional
Cartesian coordinate system. The three dimensional Cartesian coordinates of the
original polar volume data can be computed using :doc:`volcoords_from_polar <generated/wradlib.vpr.volcoords_from_polar>`.

Then, we can create regular 3-D grids in order to analyse the vertical profile
of reflectivity or rainfall intensity. For some applications you might want
to create so-called `Constant Altitude Plan Position Indicators (CAPPI)
<http://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator>`_ in order
to make radar observations at different distances from the radar more comparable.
Basically, a CAPPI is simply one slice out of a 3-D volume grid. AnalogousBy the way, we will
refer to the elements in a three dimensional Cartesian grid as *voxels*. In wradlib,
you can create :doc:`CAPPIs <generated/wradlib.vpr.CAPPI>` (and :doc:`Pseudo CAPPIs
<generated/wradlib.vpr.PseudoCAPPI>`) for different altitudes at once.

Here's an example how a set of CAPPIs can be created from synthetic polar volume data::

    import wradlib
    import numpy as np

    # define elevation and azimuth angles, ranges, radar site coordinates, projection
    elevs  = np.array([0.5,1.5,2.4,3.4,4.3,5.3,6.2,7.5,8.7,10,12,14,16.7,19.5])
    azims  = np.arange(0., 360., 1.)
    ranges = np.arange(0., 120000., 1000.)
    sitecoords = (14.924218,120.255547,500.)
    projstr = wradlib.georef.create_projstr("utm", zone=51, hemisphere="north")

    # create Cartesian coordinates corresponding the location of the polar volume bins
    polxyz  = wradlib.vpr.volcoords_from_polar(sitecoords, elevs, azims, ranges, projstr)
    poldata = wradlib.vpr.synthetic_polar_volume(polxyz)
    # this is the shape of our polar volume
    polshape = (len(elevs),len(azims),len(ranges))

    # now we define the coordinates for the 3-D grid (the CAPPI layers)
    x = np.linspace(polxyz[:,0].min(), polxyz[:,0].max(), 120)
    y = np.linspace(polxyz[:,1].min(), polxyz[:,1].max(), 120)
    z = np.arange(500.,10500.,500.)
    xyz = wradlib.util.gridaspoints(x, y, z)
    gridshape = (len(x), len(y), len(z))

    # create an instance of the CAPPI class and use it to create a series of CAPPIs
    gridder = wradlib.vpr.CAPPI(polxyz, xyz, maxrange=ranges.max(), polshape=polshape, Ipclass=wradlib.ipol.Idw)
    gridded = np.ma.masked_invalid( gridder(poldata) ).reshape(gridshape)

    # plot results
    levels = np.linspace(0,100,25)
    wradlib.vis.plot_max_plan_and_vert(x, y, z, gridded, levels=levels, cmap=pl.cm.spectral)


.. autosummary::
   :nosignatures:
   :toctree: generated/

   volcoords_from_polar
   CAPPI
   PseudoCAPPI

"""

import numpy as np
import wradlib.georef as georef
import wradlib.ipol as ipol
import wradlib.util as util
import wradlib.io as io
from scipy.spatial import cKDTree
import os



##class CartesianVolume():
##    """Create 3-D regular volume grid in Cartesian coordinates from polar data with multiple elevation angles
##
##    Parameters
##    ----------
##    polcoords : array of shape (number of bins, 3)
##    cartcoords : array of shape (number of voxels, 3)
##    polshape : shape of the original volume (num elevation angles, num azimuth angles, num range bins)
##        size must correspond to length of polcoords
##    maskfile : path to an hdf5 file (default: empty string)
##        File should contain a boolean array which masks the "blind" areas of the volume scan
##    Ipclass : an interpolation class from wradlib.ipol
##    ipargs : keyword arguments corresponding to Ipclass
##
##    Returns
##    -------
##    output : float ndarray of shape (number of levels, number of x coordinates, number of y coordinates)
##
##    """
##    def __init__(self, polcoords, cartcoords, polshape, maxrange, pseudocappi=False, maskfile="", Ipclass=ipol.Idw, **ipargs):
##        self.Ipclass        = Ipclass
##        self.ipargs         = ipargs
##        # create a default instance of interpolator
##        print "Creating 3D interpolator...this is still very slow."
##        self.ip             = Ipclass(src=polcoords, trg=cartcoords, **ipargs)
##        self.ispcappi       = pseudocappi
##        try:
##            # read mask from pickled file
##            self.mask = io.from_hdf5(maskfile)[0]
##            # check whether mask is consistent with the data
##            if not len(self.mask)==len(cartcoords):
##                raise Exception()
##            print "Load mask from file <%s>: successful" % maskfile
##        except:
##            self.mask = self.create_mask(polcoords, cartcoords, polshape, maxrange)
##            if not maskfile=="":
##                try:
##                    io.to_hdf5(maskfile, self.mask, dtype="bool")
##                    print "Save mask to file <%s>: successful" % maskfile
##                except:
##                    pass
##
##    def __call__(self, data):
##        """Interpolates the polar data to 3-dimensional Cartesian coordinates
##
##        Parameters
##        ----------
##        data : 1-d array of length (num voxels,)
##
##        """
##        ipdata = self.ip(data)
##        ipdata[self.mask] = np.nan
##        return ipdata
##
##    def create_mask(self, polcoords, cartcoords, polshape, maxrange):
##        """Identifies all the "blind" voxels of a Cartesian 3D-volume grid
##        """
##        print "Creating volume mask from scratch...this is still very slow."
##        # Identify voxels beyond the maximum range
##        center = np.array([np.mean(polcoords[:,0]), np.mean(polcoords[:,1]), np.min(polcoords[:,2])]).reshape((-1,3))
##        in_range = ((cartcoords-center)**2).sum(axis=-1) <= maxrange**2
##        if not self.ispcappi:
##            # Identify those grid altitudes above the maximum scanning angle
##            maxelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[-1].ravel(),polcoords[:,1].reshape(polshape)[-1].ravel(),polcoords[:,2].reshape(polshape)[-1].ravel())).transpose()
##            alt_interpolator = ipol.Nearest(maxelevcoords, cartcoords)
##            maxalt = alt_interpolator(maxelevcoords[:,2])
##            # Identify those grid altitudes below the minimum scanning angle
##            minelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[0].ravel(),polcoords[:,1].reshape(polshape)[0].ravel(),polcoords[:,2].reshape(polshape)[0].ravel())).transpose()
##            alt_interpolator = ipol.Nearest(minelevcoords, cartcoords)
##            minalt = alt_interpolator(minelevcoords[:,2])
##            # mask those values above the maximum and below the minimum scanning angle
##            return np.logical_not( np.logical_and(np.logical_and(cartcoords[:,2]<=maxalt, cartcoords[:,2]>=minalt), in_range) )
##        else:
##            return np.logical_not( in_range )


class CartesianVolume():
    """Create 3-D regular volume grid in Cartesian coordinates from polar data with multiple elevation angles

    Parameters
    ----------
    polcoords : array of shape (num bins, 3)
    cartcoords : array of shape (num voxels, 3)
    polshape : shape of the original volume (num elevation angles, num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    maskfile : path to an hdf5 file (default: empty string) which has been created by the self._set_mask method.
        File should contain a boolean array which masks the "blind" areas of the volume scan
    Ipclass : an interpolation class from wradlib.ipol
    ipargs : keyword arguments corresponding to Ipclass

    Returns
    -------
    output : float ndarray of shape (num levels, num x coordinates, num y coordinates)

    """
    def __init__(self, polcoords, cartcoords, polshape=None, maxrange=None, maskfile="", Ipclass=ipol.Idw, **ipargs):
        # create an instance of the Interpolation class
        self.ip             = Ipclass(src=polcoords, trg=cartcoords, **ipargs)
        # Set the mask which masks the blind voxels of the 3-D volume grid
        self._set_mask(cartcoords, polcoords, polshape, maxrange, maskfile)

    def __call__(self, data):
        """Interpolates the polar data to 3-dimensional Cartesian coordinates

        Parameters
        ----------
        data : 1-d array of length (num radar bins in volume,)
            The length of this array must be the same as len(polcoords)

        Returns
        -------
        output : 1-d array of length (num voxels,)

        """
        # Interpolate data in 3-D
        ipdata = self.ip(data)
        # Mask the "blind" voxels with NaN
        ipdata[self.mask] = np.nan

        return ipdata

    def _set_mask(self, cartcoords, polcoords=None, polshape=None, maxrange=None, maskfile=""):
        """Sets the mask which masks blind voxels in the volume grid

        Computing the mask can still be time consuming, particularly from CAPPIs.
        Thus, if possible, the mask will be loaded from a file given by *maskfile*.
        Otherwise, the mask will be computed by the *_get_mask* method. The *_get_mask*
        method is typically not inherited but has to be defined for a specific product.
        Examples are given in the CAPPI._get_mask and PseudoCAPPI._get_mask.

        Parameters
        ----------
        cartcoords :
        polcoords :
        polshape :
        maxrange :
        maskfile :

        """
        if maskfile=="":
            self.mask = self._get_mask(cartcoords, polcoords, polshape, maxrange, maskfile)
            print ""
        else:
            try:
                # read mask from pickled file
                self.mask = io.from_hdf5(maskfile)[0]
                # check whether mask is consistent with the data
                if not len(self.mask)==len(cartcoords):
                    raise Exception()
                print "Load mask from file <%s>: successful" % maskfile
            except:
                print "Creating mask from sratch...this might still be slow."
                self.mask = self._get_mask(cartcoords, polcoords, polshape, maxrange, maskfile)
                if not maskfile=="":
                    try:
                        io.to_hdf5(maskfile, self.mask, dtype="bool")
                        print "Save mask to file <%s>: successful" % maskfile
                    except:
                        print "Cannot save mask to file..."
                        raise Exception()
    def _get_mask(self, cartcoords, polcoords=None, polshape=None, maxrange=None, maskfile=None):
        """Returns a mask (the base class only contains a dummy function which masks nothing)

        This method needs to be replaced for inherited classes such as CAPPI or PseudoCAPPI

        Parameters
        ----------
        cartcoords :
        polcoords :
        polshape :
        maxrange :

        Returns
        -------
        output : Boolean array of length (num voxels,)

        """
        return np.repeat(False, len(cartcoords))



class CAPPI(CartesianVolume):
    """Create a Constant Altitude Plan Position Indicator (CAPPI)

    A CAPPI gives the value of a target variable (typically reflectivity in dBZ,
    but here also other variables such as e.g. rainfall intensity) in a defined
    altitude.

    In order to create a CAPPI , you first have to create an instance of this class.
    Calling this instance with the actual polar volume data will return the CAPPI grid.

    Parameters
    ----------
    polcoords : coordinate array of shape (num bins, 3)
        Represents the 3-D coordinates of the orginal radar bins
    cartcoords : coordinate array of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    polshape : shape of the original polar volume (num elevation angles, num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    maskfile : path to an hdf5 file (default: empty string) which has been created by the self._set_mask method.
        File should contain a boolean array which masks the "blind" areas of the volume scan
    Ipclass : an interpolation class from wradlib.ipol
    ipargs : keyword arguments corresponding to Ipclass

    """
    def _get_mask(self, cartcoords, polcoords, polshape, maxrange, maskfile):
        """Masks the "blind" voxels of the Cartesian 3D-volume grid
        """
        return np.logical_not(
              np.logical_not( out_of_range(polcoords, cartcoords, maxrange) )
            & np.logical_not( below_radar(polcoords, cartcoords, polshape)  )
            & np.logical_not( above_radar(polcoords, cartcoords, polshape)  )
        )


class PseudoCAPPI(CartesianVolume):
    """Create a Pseudo-CAPPI Constant Altitude Plan Position Indicator (CAPPI)

    The difference to a :doc:`CAPPI <wradlib.vpr.CAPPI>` is that the blind area *below* and *above* the radar
    are not masked, but filled by interpolation. Only the areas beyond the *range*
    of the radar are masked out. As a result, "blind" areas below the radar are
    particularly filled from the lowest available elevation angle.

    In order to create a Pseudo CAPPI , you first have to create an instance of this class.
    Calling this instance with the actual polar volume data will return the Pseudo CAPPI grid.

    Parameters
    ----------
    polcoords : coordinate array of shape (num bins, 3)
        Represents the 3-D coordinates of the orginal radar bins
    cartcoords : coordinate array of shape (num voxels, 3)
        Represents the 3-D coordinates of the Cartesian grid
    polshape : shape of the original polar volume (num elevation angles, num azimuth angles, num range bins)
        size must correspond to length of polcoords
    maxrange : float
        The maximum radar range (must be the same for each elevation angle)
    maskfile : path to an hdf5 file (default: empty string) which has been created by the self._set_mask method.
        File should contain a boolean array which masks the "blind" areas of the volume scan
    Ipclass : an interpolation class from wradlib.ipol
    ipargs : keyword arguments corresponding to Ipclass

    """
    def _get_mask(self, cartcoords, polcoords, polshape, maxrange, maskfile):
        """Masks the "blind" voxels of the Cartesian 3D-volume grid
        """
        return np.logical_not(
            np.logical_not( out_of_range(polcoords, cartcoords, maxrange) ) )


def out_of_range(polcoords, cartcoords, maxrange):
    """Flags the region outside the radar range

    Paramters
    ---------
    polcoords : array of 3-D coordinates with shape (num bins, 3)
    cartcoords : array of 3-D coordinates with shape (num voxels, 3)
    maxrange : maximum range (meters)

    Returns
    -------
    output : 1-D Boolean array of length len(cartcoords)

    """
    center = np.array([np.mean(polcoords[:,0]), np.mean(polcoords[:,1]), np.min(polcoords[:,2])]).reshape((-1,3))
    return ((cartcoords-center)**2).sum(axis=-1) > maxrange**2


def below_radar(polcoords, cartcoords, polshape):
    """Flags the region below the lowest elevation angle ("below the radar")

    ATTENTION: We need to tune performance here!

    Paramters
    ---------
    polcoords : array of 3-D coordinates with shape (num bins, 3)
    cartcoords : array of 3-D coordinates with shape (num voxels, 3)
    polshape : original shape of the polar volume (num elevations, num azimuths, num ranges)

    Returns
    -------
    output : 1-D Boolean array of length len(cartcoords)

    """
    minelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[0].ravel(),polcoords[:,1].reshape(polshape)[0].ravel(),polcoords[:,2].reshape(polshape)[0].ravel())).transpose()
    interpolator = ipol.Nearest(minelevcoords, cartcoords)
    minalt = interpolator(minelevcoords[:,2])
    return cartcoords[:,2] < minalt


def above_radar(polcoords, cartcoords, polshape):
    """Flags the region above the upper elevation angle (Cone of Silence)

    See also http://www.radartutorial.eu/18.explanations/ex47.en.html for an explanation
    of the Cone of Silence

    ATTENTION: We need to tune performance here!

    Paramters
    ---------
    polcoords : array of 3-D coordinates with shape (num bins, 3)
    cartcoords : array of 3-D coordinates with shape (num voxels, 3)
    polshape : original shape of the polar volume (num elevations, num azimuths, num ranges)

    Returns
    -------
    output : 1-D Boolean array of length len(cartcoords)

    """
    maxelevcoords = np.vstack((polcoords[:,0].reshape(polshape)[-1].ravel(),polcoords[:,1].reshape(polshape)[-1].ravel(),polcoords[:,2].reshape(polshape)[-1].ravel())).transpose()
    interpolator = ipol.Nearest(maxelevcoords, cartcoords)
    maxalt = interpolator(maxelevcoords[:,2])
    return cartcoords[:,2] > maxalt



def volcoords_from_polar(sitecoords, elevs, azimuths, ranges, projstr=None):
    """Create Cartesian coordinates for the polar volume bins

    Parameters
    ----------
    sitecoords : sequence of three floats indicating the radar position
       (latitude in decimal degrees, longitude in decimal degrees, height a.s.l. in meters)
    elevs : sequence of elevation angles
    azimuths : sequence of azimuth angles
    ranges : sequence of ranges
    projstr : proj.4 projection string

    Returns
    -------
    output : array of shape (num volume bins, 3)

    """
    # make sure that elevs is an array
    elevs = np.array([elevs]).ravel()
    # create polar grid
    el, az, r = util.meshgridN(elevs, azimuths, ranges)
    # get geographical coordinates
    lats, lons, z = georef.polar2latlonalt(r, az, el, sitecoords, re=6370040.)
    # get projected horizontal coordinates
    x, y = georef.project(lats, lons, projstr)
    # create standard shape
    coords = np.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return coords


def volcoords_from_polar_irregular(sitecoords, elevs, azimuths, ranges, projstr=None):
    """Create Cartesian coordinates for the polar volume bins

    Parameters
    ----------
    sitecoords : sequence of three floats indicating the radar position
       (latitude in decimal degrees, longitude in decimal degrees, height a.s.l. in meters)
    elevs : sequence of elevation angles
    azimuths : sequence of azimuth angles
    ranges : sequence of ranges
    projstr : proj.4 projection string

    Returns
    -------
    output : array of shape (num volume bins, 3)

    """
    # check structure: Are azimuth angles and range bins the same for each elevation angle?
    oneaz4all = True
    onerange4all = True
    #   check elevs array, first: must be one-dimensional
    try:
        elevs = np.array(elevs)
    except:
        print "Could not create an array from argument <elevs>."
        print "The following exception was raised:"
        raise
    assert (elevs.ndim==1) and (elevs.dtype!=np.dtype("object")), "Argument <elevs> in wradlib.wolcoords_from_polar must be a 1-D array."
    #   now: is there one azimuths array for all elevation angles or one for each?
    try:
        azimuths = np.array(azimuths)
    except:
        print "Could not create an array from argument <azimuths>."
        print "The following exception was raised:"
        raise
    if len(azimuths)==len(elevs):
        # are the items of <azimuths> arrays themselves?
        isseq = [util.issequence( elem ) for elem in azimuths]
        assert not ( (False in isseq) and (True in isseq) ), "Argument <azimuths> contains both iterable and non-iterable items."
        if True in isseq:
            # we expect one azimuth array for each elevation angle
            oneaz4all = False
    #   now: is there one ranges array for all elevation angles or one for each?
    try:
        ranges = np.array(ranges)
    except:
        print "Could not create an array from argument <ranges>."
        print "The following exception was raised:"
        raise
    if len(ranges)==len(elevs):
        # are the items of <azimuths> arrays themselves?
        isseq = [util.issequence( elem ) for elem in ranges]
        assert not ( (False in isseq) and (True in isseq) ), "Argument <azimuths> contains both iterable and non-iterable items."
        if True in isseq:
            # we expect one azimuth array for each elevation angle
            onerange4all = False
    if oneaz4all and onerange4all:
        # this is the simple way
        return volcoords_from_polar(sitecoords, elevs, azimuths, ranges, projstr)
    # No simply way, so we need to construct the coordinates arrays for each elevation angle
    #   but first adapt input arrays to this task
    if onerange4all:
        ranges = np.array([ranges for i in range(len(elevs))])
    if oneaz4all:
        azimuths = np.array([azimuths for i in range(len(elevs))])
    #   and second create the corresponding polar volume grid
    el=np.array([])
    az=np.array([])
    r =np.array([])
    for i, elev in enumerate(elevs):
        az_tmp, r_tmp = np.meshgrid(azimuths[i], ranges[i])
        el = np.append(el, np.repeat(elev, len(azimuths[i])*len(ranges[i])) )
        az = np.append(az, az_tmp.ravel())
        r  = np.append(r,  r_tmp.ravel())
    # get geographical coordinates
    lats, lons, z = georef.polar2latlonalt(r, az, el, sitecoords, re=6370040.)
    # get projected horizontal coordinates
    x, y = georef.project(lats, lons, projstr)
    # create standard shape
    coords = np.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return coords



def synthetic_polar_volume(coords):
    """Returns a synthetic polar volume
    """
    x = coords[:,0] * 10 / np.max(coords[:,0])
    y = coords[:,1] * 10 / np.max(coords[:,1])
    z = coords[:,2] * 10 / np.max(coords[:,2])
    out = np.abs(np.sin(x*y*z)/(x*y*z))
    out = out * 100./ out.max()
    return out


if __name__ == '__main__':
    print 'wradlib: Calling module <vpr> as main...'


