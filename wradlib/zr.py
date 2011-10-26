#-------------------------------------------------------------------------------
# Name:        zr
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""Module zr takes care of transforming reflectivity
into rainfall rates and vice versa
"""

def z2r(z, a=200., b=1.6, opts=None):
    """Calculates rain rates from radar reflectivities using
    a power law Z/R relationship Z = a*R**b

    Parameters
    ----------
    z : a float or an ndarray of floats
        values reflectivity
    opts : list of strings - recognized values:
            'dwd':  a/b combination derived
                    by the German Weather Service
                    is used a=256, b=1.42 instead
                    of the Marshall-Palmer defaults.
    """
    if opts == None:
        return (z/a)**(1./b)
    if 'dwd' in opts:
        a1 = 256.
        b1 = 1.42
        return (z/a1)**(1./b1)


if __name__ == '__main__':
    print 'wradlib: Calling module <zr> as main...'
