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
import wradlib.verify as verify
import wradlib.ipol as ipol

def ex_adjust():

    import numpy as np
    import pylab as pl

    # 1-d example including all available adjustment methods
    # --------------------------------------------------------------------------
    # gage and radar coordinates
    obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    radar_coords = np.arange(0,101)
    # true rainfall
    truth = np.abs(1.5+np.sin(0.075*radar_coords)) + np.random.uniform(-0.1,0.1,len(radar_coords))
    # radar error
    erroradd = 0.7*np.sin(0.2*radar_coords+10.)
    errormult = 0.75 + 0.015*radar_coords
    noise = np.random.uniform(-0.05,0.05,len(radar_coords))
    # radar observation
    radar = errormult*truth + erroradd + noise
    # gage observations are assumed to be perfect
    obs = truth[obs_coords]
    # add a missing value to observations (just for testing)
    obs[1] = np.nan
    # number of neighbours to be used
    nnear_raws=3
    # adjust the radar observation by additive model
    add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=nnear_raws)
    add_adjusted = add_adjuster(obs, radar)
    # adjust the radar observation by multiplicative model
    mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=nnear_raws)
    mult_adjusted = mult_adjuster(obs, radar)
    # adjust the radar observation by MFB
    mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=nnear_raws)
    mfb_adjusted = mfb_adjuster(obs, radar)
    # adjust the radar observation by AdjustMixed
    mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=nnear_raws)
    mixed_adjusted = mixed_adjuster(obs, radar)
    # plotting
    line0 = pl.plot(radar_coords, radar, 'k-', label="Unadjusted radar", linewidth=2., linestyle="dashed")
    pl.xlabel("Distance (km)")
    pl.ylabel("Rainfall intensity (mm/h)")
    line1 = pl.plot(radar_coords, truth, 'k-', label="True rainfall", linewidth=2.)
    line2 = pl.plot(obs_coords, obs, 'o', label="Gage observation", markersize=10.0, markerfacecolor="grey")
    line3 = pl.plot(radar_coords, add_adjusted, '-', color="red", label="Additive adjustment")
    line4 = pl.plot(radar_coords, mult_adjusted, '-', color="green", label="Multiplicative adjustment")
    line5 = pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="Mean Field Bias adjustment")
    line6 = pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="Mixed (mult./add.) adjustment")
    pl.legend(prop={'size':12})
    pl.show()

    # Verification for this example
    rawerror  = verify.ErrorMetrics(truth, radar)
    mfberror  = verify.ErrorMetrics(truth, mfb_adjusted)
    adderror  = verify.ErrorMetrics(truth, add_adjusted)
    multerror = verify.ErrorMetrics(truth, mult_adjusted)
    mixerror  = verify.ErrorMetrics(truth, mixed_adjusted)

    # Verification reports
    maxval = 4.
    fig = pl.figure(figsize=(14,8))
    ax = fig.add_subplot(231, aspect=1.)
    rawerror.report(ax=ax, unit="mm", maxval=maxval)
    ax.text(0.2, 0.9*maxval, "Unadjusted radar")
    ax = fig.add_subplot(232, aspect=1.)
    adderror.report(ax=ax, unit="mm", maxval=maxval)
    ax.text(0.2, 0.9*maxval, "Additive adjustment")
    ax = fig.add_subplot(233, aspect=1.)
    multerror.report(ax=ax, unit="mm", maxval=maxval)
    ax.text(0.2, 0.9*maxval, "Multiplicative adjustment")
    ax = fig.add_subplot(234, aspect=1.)
    mixerror.report(ax=ax, unit="mm", maxval=maxval)
    ax.text(0.2, 0.9*maxval, "Mixed (mult./add.) adjustment")
    mixerror.report(ax=ax, unit="mm", maxval=maxval)
    ax = fig.add_subplot(235, aspect=1.)
    mfberror.report(ax=ax, unit="mm", maxval=maxval)
    ax.text(0.2, 0.9*maxval, "Mean Field Bias adjustment")
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
##    adjuster = adjust.AdjustAdd(obs_coords, raw_coords, stat='median', Ipclass=ipol.Nearest)
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

if __name__ == '__main__':
    ex_adjust()

