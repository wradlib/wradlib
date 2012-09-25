#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      heistermann
#
# Created:     28.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.adjust as adjust

if __name__ == '__main__':

    import numpy as np
    import pylab as pl

    # 1-d example including all available adjustment methods
    # --------------------------------------------------------------------------
    # gage and radar coordinates
    obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    radar_coords = np.arange(0,101)
    # true rainfall
    truth = np.abs(np.sin(0.1*radar_coords))
    # radar error
    erroradd = np.random.uniform(0,0.5,len(radar_coords))
    errormult= 1.1
    # radar observation
    radar = errormult*truth + erroradd
    # gage observations are assumed to be perfect
    obs = truth[obs_coords]
    # add a missing value to observations (just for testing)
    obs[1] = np.nan
    # adjust the radar observation by additive model
    add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=1)
    add_adjusted = add_adjuster(obs, radar)
    # adjust the radar observation by multiplicative model
    mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=1)
    mult_adjusted = mult_adjuster(obs, radar,0.)
    # adjust the radar observation by MFB
    mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=1)
    mfb_adjusted = mfb_adjuster(obs, radar,0.)
    # adjust the radar observation by AdjustMixed
    mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=1)
    mixed_adjusted = mixed_adjuster(obs, radar)
    line1 = pl.plot(radar_coords, radar, 'k-', label="raw radar")
    line2 = pl.plot(obs_coords, obs, 'ro', label="gage obs")
    line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="adjusted by AdjustAdd")
    line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="adjusted by AdjustMultiply")
    line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="adjusted by AdjustMFB")
    line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="adjusted by AdjustMixed")
    pl.legend()
    pl.show()


##    # 2d example
##    #---------------------------------------------------------------------------
##    # Creating synthetic data
##    # --------------------------------------------------------------------------
##    # number of points of raw per axis (total number of points of raw will be num_raw**2)
##    num_raw = 100
##    num_obs = 50
##    # making raw coordinates
##    raw_coords = np.meshgrid(np.linspace(0,100,num_raw), np.linspace(0,100,num_raw))
##    raw_coords = np.vstack((raw_coords[0].ravel(), raw_coords[1].ravel())).transpose()
##    # making raw data
##    raw = np.abs(np.sin(0.1*raw_coords).sum(axis=1))
##    # indices for creating obs from raw
##    obs_ix = np.random.uniform(low=0, high=num_raw**2, size=num_obs).astype('i4')
##    # creating obs_coordinates
##    obs_coords = raw_coords[obs_ix]
##    # creating obs data by perturbing raw
##    obs = raw[obs_ix]+np.random.uniform(low=-1., high=1, size=len(obs_ix))
##    obs = np.abs(obs)
##
##    #---------------------------------------------------------------------------
##    # Gage adjustment
##    #---------------------------------------------------------------------------
##    adjuster = adjust.AdjustAdd(obs_coords, raw_coords, stat='median', p_idw=2.)
##    result = adjuster(obs, raw)
##
##    #---------------------------------------------------------------------------
##    # Plotting
##    #---------------------------------------------------------------------------
##    # maximum value for normalisation
##    maxval = np.max(np.concatenate((raw, obs, result)).ravel())
##    # open figure
##    fig = pl.figure()
##    # adding subplot for unadjusted
##    ax = fig.add_subplot(221, aspect='equal')
##    raw_plot = ax.scatter(raw_coords[:,0], raw_coords[:,1], c=raw, vmin=0, vmax=maxval, edgecolor='none')
##    ax.scatter(obs_coords[:,0], obs_coords[:,1], c=obs.ravel(), marker='s', s=50, vmin=0, vmax=maxval)
##    pl.colorbar(raw_plot)
##    pl.title('Raw field and observations')
##    # adding subplot for adjusted
##    ax = fig.add_subplot(222, aspect='equal')
##    raw_plot = ax.scatter(raw_coords[:,0], raw_coords[:,1], c=result, vmin=0, vmax=maxval, edgecolor='none')
##    ax.scatter(obs_coords[:,0], obs_coords[:,1], c=obs.ravel(), marker='s', s=50, vmin=0, vmax=maxval)
##    pl.colorbar(raw_plot)
##    pl.title('Adjusted field and observations')
##    # scatter plot raw vs. obs
##    ax = fig.add_subplot(223, aspect='equal')
##    ax.scatter(obs, raw[obs_ix])
##    ax.plot([0,maxval],[0,maxval],'-', color='grey')
##    pl.title('Scatter plot raw vs. obs')
##    ax.set_xlim(left=0)
##    ax.set_ylim(bottom=0)
##    # scatter adjusted vs. raw (for control purposes)
##    ax = fig.add_subplot(224, aspect='equal')
##    ax.scatter(obs, result[obs_ix])
##    ax.plot([0,maxval],[0,maxval],'-', color='grey')
##    ax.set_xlim(left=0)
##    ax.set_ylim(bottom=0)
##    pl.title('Scatter plot adjusted vs. obs')
##
##    pl.show()



