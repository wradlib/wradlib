#-------------------------------------------------------------------------------
# Name:        adjust_example.py
# Purpose:
#
# Author:      heistermann
#
# Created:     28.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.adjust as adjust
import wradlib.verify as verify
import wradlib.util as util

def ex_adjust():

    import numpy as np
    import pylab as pl

    ###########################################################################
    # 1d Example ##############################################################
    ###########################################################################

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
    pl.plot(radar_coords, radar, 'k-', label="Unadjusted radar", linewidth=2., linestyle="dashed")
    pl.xlabel("Distance (km)")
    pl.ylabel("Rainfall intensity (mm/h)")
    pl.plot(radar_coords, truth, 'k-', label="True rainfall", linewidth=2.)
    pl.plot(obs_coords, obs, 'o', label="Gage observation", markersize=10.0, markerfacecolor="grey")
    pl.plot(radar_coords, add_adjusted, '-', color="red", label="Additive adjustment")
    pl.plot(radar_coords, mult_adjusted, '-', color="green", label="Multiplicative adjustment")
    pl.plot(radar_coords, mfb_adjusted, '-', color="orange", label="Mean Field Bias adjustment")
    pl.plot(radar_coords, mixed_adjusted, '-', color="blue", label="Mixed (mult./add.) adjustment")
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


    ###########################################################################
    # 2d Example ##############################################################
    ###########################################################################

    # a) CREATE SYNTHETIC DATA ------------------------------------------------
    
    # grid axes
    xgrid = np.arange(0,10)
    ygrid = np.arange(20,30)
    
    # number of observations
    num_obs = 10

    # create grid
    gridshape = len(xgrid), len(ygrid)
    grid_coords = util.gridaspoints(ygrid, xgrid)     

    # Synthetic true rainfall
    truth = np.abs(10.*np.sin(0.1*grid_coords).sum(axis=1))

    # Creating radar data by perturbing truth with multiplicative and additive error
    # YOU CAN EXPERIMENT WITH THE ERROR STRUCTURE
    radar = 0.6 * truth + 1.*np.random.uniform(low=-1., high=1, size=len(truth))
    radar[radar<0.] = 0.

    # indices for creating obs from raw (random placement of gauges)
    obs_ix = np.random.uniform(low=0, high=len(grid_coords), size=num_obs).astype('i4')

    # creating obs_coordinates
    obs_coords = grid_coords[obs_ix]

    # creating gauge observations from truth
    obs = truth[obs_ix]

    
    # b) GAUGE ADJUSTMENT -----------------------------------------------------
    
    # Mean Field Bias Adjustment
    mfbadjuster = adjust.AdjustMFB(obs_coords, grid_coords)
    mfbadjusted = mfbadjuster(obs, radar)
    
    # Additive Error Model
    addadjuster = adjust.AdjustAdd(obs_coords, grid_coords)
    addadjusted = addadjuster(obs, radar)
    
    # Multiplicative Error Model
    multadjuster = adjust.AdjustMultiply(obs_coords, grid_coords)
    multadjusted = multadjuster(obs, radar)

    # c) PLOTTING

    # Maximum value (used for normalisation of colorscales)    
    maxval = np.max(np.concatenate((truth, radar, obs, addadjusted)).ravel())
    
    # Helper functions for repeated plotting tasks
    def scatterplot(x, y, title):
        """Quick and dirty helper function to produce scatter plots
        """
        pl.scatter(x, y)
        pl.plot([0,1.2*maxval],[0,1.2*maxval],'-', color='grey')
        pl.xlabel("True rainfall (mm)")
        pl.ylabel("Estimated rainfall (mm)")
        pl.xlim(0,maxval+0.1*maxval)
        pl.ylim(0, maxval+0.1*maxval)
        pl.title(title)
        
    def gridplot(data, title):
        """Quick and dirty helper function to produce a grid plot
        """
        xplot = np.append(xgrid,xgrid[-1]+1.)-0.5
        yplot = np.append(ygrid,ygrid[-1]+1.)-0.5
        grd = ax.pcolormesh(xplot, yplot, data.reshape(gridshape),vmin=0, vmax=maxval)
        ax.scatter(obs_coords[:,0], obs_coords[:,1], c=obs.ravel(), marker='s', s=50, vmin=0, vmax=maxval)
        pl.colorbar(grd, shrink=0.7)
        pl.title(title)

    # open figure
    fig = pl.figure(figsize=(10,10))
    
    # True rainfall
    ax = fig.add_subplot(331, aspect='equal')
    gridplot(truth, 'True rainfall')

    # Unadjusted radar rainfall
    ax = fig.add_subplot(332, aspect='equal')
    gridplot(radar, 'Radar rainfall')

    # Scatter plot radar vs. observations
    ax = fig.add_subplot(333, aspect='equal')
    scatterplot(truth, radar, 'Radar vs. Truth (red: Gauges)')
    pl.plot(obs, radar[obs_ix], linestyle="None", marker="o", color="red")

    # Adjusted radar rainfall (MFB)
    ax = fig.add_subplot(334, aspect='equal')
    gridplot(mfbadjusted, 'Adjusted (MFB)')

    # Adjusted radar rainfall (additive)
    ax = fig.add_subplot(335, aspect='equal')
    gridplot(addadjusted, 'Adjusted (Add.)')

    # Adjusted radar rainfall (multiplicative)
    ax = fig.add_subplot(336, aspect='equal')
    gridplot(multadjusted, 'Adjusted (Mult.)')

    # Adjusted (MFB) vs. radar (for control purposes)
    ax = fig.add_subplot(337, aspect='equal')
    #scatterplot(obs, mfbadjusted[obs_ix], 'Adjusted (MFB) vs. Gauges\n(no x-validation!)')
    scatterplot(truth, mfbadjusted, 'Adjusted (MFB) vs. Truth')
    
    # Adjusted (Add) vs. radar (for control purposes)
    ax = fig.add_subplot(338, aspect='equal')
    #scatterplot(obs, addadjusted[obs_ix], 'Adjusted (Add.) vs. Gauges\n(no x-validation!)')
    scatterplot(truth, addadjusted, 'Adjusted (Add.) vs. Truth')
    
    # Adjusted (Mult.) vs. radar (for control purposes)
    ax = fig.add_subplot(339, aspect='equal')
    #scatterplot(obs, multadjusted[obs_ix], 'Adjusted (Mult.) vs. Gauges\n(no x-validation!)')
    scatterplot(truth, multadjusted, 'Adjusted (Mult.) vs. Truth')
    
    pl.tight_layout()
    pl.show()

if __name__ == '__main__':
    ex_adjust()

