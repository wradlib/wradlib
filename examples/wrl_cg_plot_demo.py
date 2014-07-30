# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Reading polar volume data
# Purpose:
#
# Author:
#
# Created:     10.06.2013
# Copyright:   (c) kai_muehlbauer 2013
# Licence:     MIT
#-------------------------------------------------------------------------------
#!/usr/bin/env python

from __future__ import print_function
import matplotlib as mpl
import wradlib as wrl
import pylab as pl
# just making sure that the plots immediately pop up
pl.interactive(True)
import numpy as np
import os

def demo_plot_ppi_01():

    # read the data (sample file in wradlib/examples/data)
    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    # define arrays of polar coordinate arrays (azimuth and range)
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    # get the scan data for this elevation
    #   here, you can do all the processing on the 2-D polar level
    #   e.g. clutter elimination, attenuation correction, ...
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]
    print("Data:", data.shape, np.max(data), np.mean(data))

    #cut off noise, use 0dBZ as threshold
    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    #set dpi
    pdpi = 150

    #create figure
    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    #create cg_plot instance as ppi
    ppi = wrl.vis.cg_plot(ind='PPI',fig=fig0)

    #plot data, set ranges accordingly
    circle = ppi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,360], \
                      # range to plot in x-direction and tick resolution
                      x_range=[-320,320.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[-320,320], y_res = 40., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick resolution
                      a_res = 10, \
                      # aspect, should be 1 for ppi
                      aspect=1.0, \
                      # floating axis
                      faxis=None)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = ppi.colorbar(circle, cbp="5%", fsize="1.5%", cbw="3%", z_res = 10., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    ppi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot cartesian grids and ticks
    ppi.xticks(ppi.x_res, fsize="1.5%")
    ppi.yticks(ppi.y_res, fsize="1.5%")
    ppi.cartgrid(True)

    # plot Title
    ppi.title("PPI Curvilinear Grid Demonstration", pad="7%", fsize="3%" , ha="left")
    # plot Y-Title
    ppi.y_title("Y-Range", pad="7%", fsize="2.0%")
    # plot X-Title
    ppi.x_title("X-Range", pad="5%", fsize='2.0%')

    ppi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_ppi_01.png", dpi=pdpi, bbox_inches=0)

def demo_plot_ppi_02():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 150

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    ppi = wrl.vis.cg_plot(ind='PPI',fig=fig0)

    #plot data, set ranges accordingly
    circle = ppi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,360], \
                      # range to plot in x-direction and tick resolution
                      x_range=[-320,320.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[-320,320], y_res = 40., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick resolution
                      a_res = 10, \
                      # aspect, should be 1 for ppi
                      aspect=1.0, \
                      # floating axis
                      faxis=37)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = ppi.colorbar(circle, cbp="5%", fsize="1.5%", cbw="3%", z_res = 10., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    ppi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot polar ticks and grid
    ppi.polgrid(True)
    ppi.polticks(True)

    # plot Title
    ppi.title("PPI Curvilinear Grid Demonstration 02", pad="7%", fsize="3%" , ha="left")

    ppi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_ppi_02.png", dpi=pdpi, bbox_inches=0)

def demo_plot_ppi_03():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 300

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    ppi = wrl.vis.cg_plot(ind='PPI',fig=fig0)

    #plot data, set ranges accordingly
    circle = ppi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,360], \
                      # range to plot in x-direction and tick resolution
                      x_range=[-320,320.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[-320,320], y_res = 40., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick vector
                      a_res = [30,60,120,150,210,240,300,330], \
                      # aspect, should be 1 for ppi
                      aspect=1.0, \
                      # floating axis
                      faxis=None)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = ppi.colorbar(circle, cbp="5%", fsize="1.5%", cbw="3%", z_res = 10., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    ppi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot polar grid and ticks
    ppi.polgrid(True)
    ppi.polticks(True)

    # plot cartesian grids and ticks
    ppi.xticks(ppi.x_res, fsize="1.5%")
    ppi.yticks(ppi.y_res, fsize="1.5%")
    ppi.cartgrid(True)

    # plot Title
    ppi.title("PPI Curvilinear Grid Demonstration 03", pad="7%", fsize="3%" , ha="left")

    ppi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_ppi_03.png", dpi=pdpi, bbox_inches=0)

def demo_plot_ppi_04():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 300

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    ppi = wrl.vis.cg_plot(ind='PPI',fig=fig0)

    #plot data, set ranges accordingly
    circle = ppi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,360], \
                      # range to plot in x-direction and tick resolution
                      x_range=[-320,280.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[-320,80], y_res = 40., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick vector
                      a_res = [0,30,60,120,150,180,210,240,270,300,330], \
                      # aspect, should be 1 for ppi
                      aspect=1.0, \
                      # floating axis
                      faxis=90)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = ppi.colorbar(circle, cbp="7%", fsize="1.5%", cbw="5%", z_res = 10., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    ppi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot polar grid and ticks
    ppi.polticks(True)
    ppi.polgrid(True)

    # plot Title
    ppi.title("PPI Curvilinear Grid Demonstration 04", pad="7%", fsize="3%" , ha="left")

    ppi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_ppi_04.png", dpi=pdpi, bbox_inches=0)

def demo_plot_ppi_05():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 300

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    ppi = wrl.vis.cg_plot(ind='PPI',fig=fig0)

    #plot data, set ranges accordingly
    circle = ppi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[120,210], \
                      # range to plot in x-direction and tick resolution
                      x_range=[-320,320.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[-320,0], y_res = 40., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick vector
                      a_res = [0,30,60,120,150,180,210,240,270,300,330], \
                      # aspect, should be 1 for ppi
                      aspect=1.0, \
                      # floating axis
                      faxis=None)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = ppi.colorbar(circle, cbp="5%", fsize="1.5%", cbw="3%", z_res = 5., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    ppi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot cartesian grids and ticks
    ppi.xticks(ppi.x_res, fsize="1.5%")
    ppi.yticks(ppi.y_res, fsize="1.5%")
    ppi.cartgrid(True)

    # plot Title
    ppi.title("PPI Curvilinear Grid Demonstration 05", pad="7%", fsize="3%" , ha="left")
    # plot Y-Title
    ppi.y_title("Y-Range", pad="9%", fsize="2.0%")
    # plot X-Title
    ppi.x_title("X-Range", pad="6%", fsize='2.0%')

    ppi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_ppi_05.png", dpi=pdpi, bbox_inches=0)

def demo_plot_rhi_01():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 300

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    rhi = wrl.vis.cg_plot(ind='RHI',fig=fig0)

    #plot data, set ranges accordingly
    circle = rhi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,90], \
                      # range to plot in x-direction and tick resolution
                      x_range=[0,200.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[40,120], y_res = 20., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick vector
                      a_res = [0,5, 10, 15, 30, 40, 50, 60, 80, 90], \
                      # aspect, should be 1 for ppi
                      aspect=.5, \
                      # floating axis
                      faxis=None)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = rhi.colorbar(circle, cbp="7%", fsize="1.5%", cbw="3%", z_res = 5., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    rhi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot cartesian grids and ticks
    rhi.xticks(rhi.x_res, fsize="1.5%")
    rhi.yticks(rhi.y_res, fsize="1.5%")
    rhi.cartgrid(True)

    # plot Title
    rhi.title("RHI Curvilinear Grid Demonstration 01", pad="7%", fsize="3%" , ha="left")
    # plot Y-Title
    rhi.y_title("Y-Range", pad="9%", fsize="2.0%")
    # plot X-Title
    rhi.x_title("X-Range", pad="6%", fsize='2.0%')

    rhi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_rhi_01.png", dpi=pdpi, bbox_inches=0)

def demo_plot_rhi_02():

    raw = wrl.io.read_OPERA_hdf5(os.path.dirname(__file__) + '/' + "data/knmi_polar_volume.h5")
    where = raw["dataset%d/where"%(1)]
    what  = raw["dataset%d/data1/what"%(1)]
    az = np.roll(np.arange(0.,360.,360./where["nrays"]), -where["a1gate"])
    r  = np.arange(where["rstart"], where["rstart"]+where["nbins"]*where["rscale"], where["rscale"])
    data = what["offset"] + what["gain"] * raw["dataset%d/data1/data"%(1)]

    noise_indices = np.where(data <= 0.0)
    data[noise_indices] = np.nan
    cmap = mpl.cm.jet
    masked_array = np.ma.array(data, mask=np.isnan(data))
    cmap.set_bad('w',-64.0)

    pdpi = 300

    fig0 = pl.figure(figsize=(8,8), dpi=pdpi)
    fig0.patch.set_facecolor(None)
    fig0.patch.set_alpha(0)

    rhi = wrl.vis.cg_plot(ind='RHI',fig=fig0)

    #plot data, set ranges accordingly
    circle = rhi.plot(masked_array, \
                      # range of the raw data in radial direction
                      data_range=[0.,320.], \
                      # range of the raw data in azimutal direction
                      theta_range=[0,90], \
                      # range to plot in x-direction and tick resolution
                      x_range=[0,200.], x_res = 40., \
                      # range to plot in y-direction and tick resolution
                      y_range=[0,80], y_res = 20., \
                      # range of the raw data to plot
                      radial_range = [0.,320.], \
                      # azimutal tick vector
                      a_res = [0,5, 10, 15,20, 25, 30, 40, 50, 60, 80, 90], \
                      # aspect, should be 1 for ppi
                      aspect=2.0, \
                      # floating axis
                      faxis=None)

    # create colorbar
    # cbp - padding, percent of x-axis
    # cbw - width, percent of x-axis
    cbar = rhi.colorbar(circle, cbp="7%", fsize="1.5%", cbw="3%", z_res = 5., ztitle='Reflectivity', zunit='dBZ', extend='both', vmin=-0.0, vmax=40.0)

    # overwrite colorbar label
    # pad - padding, percent of x-axis
    # fsize - fontsize, approx. percent of x-axis
    rhi.z_title("Reflectivity (dBZ)", fsize="2%", pad="1%")

    # plot cartesian grids and ticks
    rhi.xticks(rhi.x_res, fsize="1.5%")
    rhi.yticks(rhi.y_res, fsize="1.5%")
    rhi.cartgrid(True)

    # plot Title
    rhi.title("RHI Curvilinear Grid Demonstration 02", pad="7%", fsize="3%" , ha="left")
    # plot Y-Title
    rhi.y_title("Y-Range", pad="9%", fsize="2.0%")
    # plot X-Title
    rhi.x_title("X-Range", pad="6%", fsize='2.0%')

    rhi.copy_right(text=r"$\copyright$ 2013 created with WRADLIB", padx="7%", pady="7%", fsize="1.5%")

    #fig0.savefig("demo_rhi_02.png", dpi=pdpi, bbox_inches=0)

def ex_cg_plot():
    demo_plot_ppi_01()
    demo_plot_ppi_02()
    demo_plot_ppi_03()
    demo_plot_ppi_04()
    demo_plot_ppi_05()
    demo_plot_rhi_01()
    demo_plot_rhi_02()

if __name__ == '__main__':
    ex_cg_plot()

