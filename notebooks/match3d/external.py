#!/usr/bin/env python

import numpy as np

# Space-born precipitation radar parameters
pr_pars = {"trmm": {
   "zt": 402500.,   # orbital height of TRMM (post boost)   APPROXIMATION!
   "dr": 250. ,    # gate spacing of TRMM
    }, "gpm": {
   "zt": 407000.,   # orbital height of GPM                 APPROXIMATION!
   "dr": 125.      # gate spacing of GPM
}}



def correct_parallax(pr_xy, nray, nbin, drt, alpha):

    # get x,y-grids
    pr_x = pr_xy[...,0]
    pr_y = pr_xy[...,1]
    
    # create range array from ground to sat
    prng = np.arange(nbin) * drt
    print("PRANGE:", prng.shape, prng.max())

    # calculate height of bin
    zp = prng * np.cos(np.deg2rad(alpha))[:,np.newaxis]
    # calculate bin ground xy-displacement length
    ds = prng * np.sin(np.deg2rad(alpha))[:, np.newaxis]
    print("HEIGHT:", zp.shape, zp.max())

    # calculate x,y-differences between ground coordinate 
    # and center ground coordinate [25th element]
    xdiff = pr_x[:,24][:, np.newaxis] - pr_x 
    ydiff = pr_y[:,24][:, np.newaxis] - pr_y 
    print("XDIFF:", xdiff.shape)

    # assuming ydiff and xdiff being a triangles adjacent and 
    # opposite this calculates the xy-angle of the PR scan
    ang = np.arctan2(ydiff, xdiff)
    print("Ang:", ang.shape)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.cos(ang)[...,np.newaxis]
    dy = ds * np.sin(ang)[...,np.newaxis]
    print("DX:", dx.shape)

    # add displacement to PR ground coordinates
    pr_xp = dx + pr_x[..., np.newaxis]
    pr_yp = dy + pr_y[..., np.newaxis]
    print("XP:", pr_xp.shape)
    
    return np.stack((pr_xp, pr_yp, np.repeat(zp[np.newaxis,...],pr_xp.shape[0], axis=0)), axis=3), prng, zp


def sat2pol(prcoords, grcoords, re):
    
    # calculate arc length
    s = np.sqrt(np.sum(prcoords[...,0:2]**2, axis=-1))
    
    # calculate arc angle
    gamma = s / re
    
    # calculate theta (elevation-angle)
    numer = np.cos(gamma) - (re+grcoords[2])/(re+prcoords[...,2])
    denom = np.sin(gamma)
    theta = np.rad2deg(np.arctan(numer/denom))
    
    # calculate SlantRange r
    r = (re+prcoords[...,2])*denom/np.cos(np.deg2rad(theta))
    
    # calculate Azimuth phi
    phi = 90 - np.rad2deg(np.arctan2(prcoords[..., 1], prcoords[..., 0]))
    phi[phi <= 0] += 360
    
    return r, theta, phi


def dist_from_orbit(zt, alpha, r_pr_inv):
	"""Returns range distances of PR bins (in meters) as seen from the orbit.
	"""
	return(zt/np.cos(np.radians(alpha))[:, np.newaxis] - r_pr_inv)	

def get_bb_ratio(pr_data, zp):

    zbb = pr_data['zbb']
    bbwidth = pr_data['bbwidth']
    quality = pr_data['quality']
    print("ZBB", zbb.shape, np.nanmin(zbb), np.nanmax(zbb))
    print("BBWidth", bbwidth.shape, np.nanmin(bbwidth), np.nanmax(bbwidth))

    # parameters for bb detection
    ibb = (zbb > 0) & (bbwidth > 0) & (quality == 1)
    
    # set non-bb-pixels to np.nan
    zbb[~ibb] = np.nan 
    bbwidth[~ibb] = np.nan 
    
    # get median of bb-pixels
    zbb_m = np.nanmedian(zbb)
    bbwidth_m = np.nanmedian(bbwidth)
    print("MEDIAN:", zbb_m, bbwidth_m)
    
    # approximation of melting layer top and bottom
    zmlt=zbb_m+bbwidth/2.
    zmlb=zbb_m-bbwidth/2.
    print("ZMLT:", zmlt.shape)

    # get ratio connected to brightband height
    # ratio <= 0: below ml
    # 0 < ratio < 1 : between ml
    # 1 <= ratio: above ml
    ratio = (zp - zmlb[:, :, np.newaxis]) / (zmlt - zmlb)[:, :, np.newaxis]
    print("RATIO:", ratio.shape)
    
    return ratio, zbb


def calculate_polynomial(data, w):
    #print("Data:", data.shape, w.shape)
    res = np.zeros_like(data)
    for i, c in enumerate(w):
        #print(i, res.shape, c.shape, (data**i).shape)
        res += c * data**i
    #print(w.shape)
    #res1 = (w[0] +
    #        w[1] * data +            
    #        w[2] * data**2 +
    #        w[3] * data**3 +
    #        w[4] * data**4)
    #print(res1 - res)
    return res


# Coefficients to transform from Ku band to S-band for snow and rain
ku2s = { 
    "snow": np.array([0.185074, 1.01378, -0.00189212]),
    "rain": np.array([-1.50393, 1.07274, 0.000165393])
}
    

# TODO: Reformat from function to dictionary
def s_ku_coefficients():
    
    # Set coefficients for conversion from Ku-band to S-band
    
    # Snow
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Snow
    as0=[4.78e-2, 4.12e-2, 8.12e-2, 1.59e-1, 2.87e-1, 4.93e-1, 8.16e-1, 1.31e+0, 2.01e+0, 2.82e+0, 1.74e-1]
    as1=[ 1.23e-2, 3.66e-3, 2.00e-3, 9.42e-4, 5.29e-4, 5.96e-4, 1.22e-3, 2.11e-3, 3.34e-3, 5.33e-3, 1.35e-2]
    as2=[-3.50e-4, 1.17e-3, 1.04e-3, 8.16e-4, 6.59e-4, 5.85e-4, 6.13e-4, 7.01e-4, 8.24e-4, 1.01e-3,-1.38e-3]
    as3=[-3.30e-5,-8.08e-5,-6.44e-5,-4.97e-5,-4.15e-5,-3.89e-5,-4.15e-5,-4.58e-5,-5.06e-5,-5.78e-5, 4.74e-5]
    as4=[ 4.27e-7, 9.25e-7,7.41e-7, 6.13e-7, 5.80e-7, 6.16e-7, 7.12e-7, 8.22e-7, 9.39e-7, 1.10e-6, 0.00e+0]

    a_s = np.array([as0, as1, as2, as3, as4])
    
    # Hail
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Hail
    ah0=[ 4.78e-2, 1.80e-1, 1.95e-1, 1.88e-1, 2.36e-1, 2.70e-1, 2.98e-1, 2.85e-1, 1.75e-1, 4.30e-2, 8.80e-2]
    ah1=[ 1.23e-2,-3.73e-2,-3.83e-2,-3.29e-2,-3.46e-2,-2.94e-2,-2.10e-2,-9.96e-3,-8.05e-3,-8.27e-3, 5.39e-2]
    ah2=[-3.50e-4, 4.08e-3, 4.14e-3, 3.75e-3, 3.71e-3, 3.22e-3, 2.44e-3, 1.45e-3, 1.21e-3, 1.66e-3,-2.99e-4]
    ah3=[-3.30e-5,-1.59e-4,-1.54e-4,-1.39e-4,-1.30e-4,-1.12e-4,-8.56e-5,-5.33e-5,-4.66e-5,-7.19e-5, 1.90e-5]
    ah4=[ 4.27e-7, 1.59e-6, 1.51e-6, 1.37e-6, 1.29e-6, 1.15e-6, 9.40e-7, 6.71e-7, 6.33e-7, 9.52e-7, 0.00e+0]

    a_h = np.array([ah0, ah1, ah2, ah3, ah4])#.T

    return a_s, a_h

def fix_for_cband(refp, refp_ss, refp_sh):
    delta_s = (refp_ss - refp) * 5.3 / 10.0
    refp_ss = refp + deltas
    delta_h = (refp_sh - refp) * 5.3 / 10.0
    refp_sh = refp + deltah
    return refp_ss, refp_sh


	