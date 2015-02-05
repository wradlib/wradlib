#-------------------------------------------------------------------------------
# Name:        qual
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
Data Quality
^^^^^^^^^^^^

This module will serve two purposes:

#. provide routines to create simple radar data quality related fields.
#. provide routines to decide which radar pixel to choose based on the
   competing information in different quality fields.

Data is supposed to be stored in 'aligned' arrays. Aligned here means that
all fields are structured such that in each field the data for a certain index
is representative for the same physical target.

Therefore no assumptions are made on the dimensions or shape of the input
fields except that they exhibit the numpy ndarray interface.

.. autosummary::
   :nosignatures:
   :toctree: generated/

    beam_height_ft
    beam_height_ft_doviak
    pulse_volume

"""

import numpy as np


def beam_height_ft(ranges, elevations, degrees=True, re=6371000):
    """Calculates the height of a radar beam above the antenna according to
    the 4/3 (four-thirds -> ft) effective Earth radius model.
    The formula was taken from Collier :cite:`Collier1996`.

    Parameters
    ----------
    ranges : array
        the distances of each bin from the radar [m]

    elevations : array
        the elevation angles of each bin from the radar [degrees or radians]

    degrees : bool
        if True (the default) elevation angles are given in degrees and will
        be converted to radians before calculation. If False no transformation
        will be done and elevations has to be given in radians.

    re : float
        earth radius [m]

    Returns
    -------
    output : height of the beam [m]

    Notes
    -----
    The shape of `elevations` and `ranges` may differ in which case numpy's
    broadcasting rules will apply and the shape of `output` will be that of
    the broadcast arrays. See the numpy documentation on how broadcasting works.

    """
    if degrees:
        elev = np.deg2rad(elevations)
    else:
        elev = elevations

    return ((ranges**2*np.cos(elev)**2)/(2*(4./3.)*re))+ranges*np.sin(elev)


def beam_height_ft_doviak(ranges, elevations, degrees=True, re=6371000):
    """Calculates the height of a radar beam above the antenna according to \
    the 4/3 (four-thirds -> ft) effective Earth radius model.
    The formula was taken from Doviak :cite:`Doviak1993`.

    Parameters
    ----------
    ranges : array
        the distances of each bin from the radar [m]

    elevations : array
        the elevation angles of each bin from the radar [degrees or radians]

    degrees : bool
        if True (the default) elevation angles are assumed to be given in
        degrees and will
        be converted to radians before calculation. If False no transformation
        will be done and `elevations` has to be given in radians.

    re : float
        earth radius [m]

    Returns
    -------
    output : height of the beam [m]

    Notes
    -----
    The shape of `elevations` and `ranges` may differ in which case numpy's
    broadcasting rules will apply and the shape of `output` will be that of
    the broadcast arrays. See the numpy documentation on how broadcasting works.

    """
    if degrees:
        elev = np.deg2rad(elevations)
    else:
        elev = elevations

    reft = (4./3.)*re

    return np.sqrt(ranges**2 + reft**2 + 2*ranges*reft*np.sin(elev)) - reft


def pulse_volume(ranges, h, theta):
    """Calculates the sampling volume of the radar beam per bin depending on \
    range and aperture.

    We assume a cone frustum which has the volume V=(pi/3)*h*(R**2 + R*r + r**2).
    R and r are the radii of the two frustum surface circles. Assuming that the
    pulse width is small compared to the range, we get R=r=tan(theta*pi/180)*range.
    Thus, the pulse volume simpy becomes a the volume of a cylinder with
    V=pi * h * range**2 * tan(theta*pi/180)**2

    Parameters
    ----------
    ranges : array
        the distances of each bin from the radar [m]
    h : float
        pulse width (which corresponds to the range resolution [m])
    theta : float
        the aperture angle (beam width) of the radar beam [degree]

    Returns
    -------
    output : volume of radar bins at each range in `ranges` [m**3]

    """
    return np.pi * h * (ranges**2) * (np.tan( np.radians(theta) ))**2


if __name__ == '__main__':
    print 'wradlib: Calling module <qual> as main...'
