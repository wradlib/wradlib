---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.17.3
---

```{raw-cell}
:tags: [hide-cell]
This notebook is part of the wradlib documentation: https://docs.wradlib.org.

Copyright (c) wradlib developers.
Distributed under the MIT License. See LICENSE.txt for more info.
```

# A Typical Workflow For Radar-Based Rainfall Estimation


Raw, unprocessed reflectivity products can already provide useful visual information about the spatial distribution of rainfall fields. However, in order to use weather radar observations for quantitative studies (e.g. in hydrological modelling or for assimilation into Numerical Weather Prediction models), the data has to be carefully processed in order to account for typical errors sources such as ground echoes (clutter), attenuation of the radar signal, or uncertainties in the Z/R relationship.

Moreover, it might be necessary to transfer the data from polar coordinates to cartesian grids, or to combine observations from different radar locations in overlapping areas on a common grid (composition). And in the end, you would typically like to visualise the spatial rainfall distribution on a map. Many users also need to quantify the potential error (uncertainty) of their data-based rainfall estimation.

These are just some steps that might be necessary in order to make radar data useful in a specific quantitative application environment. All steps together are typically referred to as a *"radar data processing chain"*. $\omega radlib$ was designed to support you in establishing your own processing chain, suited to your specific requirements. In the following, we will provide an outline of a typical processing chain, step-by-step. You might not need all steps for your own workflow, or you might need steps which are not yet included here.

```{code-cell} python
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

## Introduction

Consider this just as an example. We will not go into detail for each step in this section, but refer to more detailed tutorials (if available) or the corresponding entry in the library reference. Most of the steps have a corresponding $\omega radlib$ module. In order to access the functions of $\omega radlib$, you have to import $\omega radlib$ in your Python environment:

```{code-cell} python
import wradlib as wrl
```

If you have trouble with that import, please head back to the [Installation](https://docs.wradlib.org/en/latest/installation.html) section.


```{note}
The data used in this tutorial can be found in the [wradlib-data repository](https://github.com/wradlib/wradlib-data). Follow [these instructions](https://github.com/wradlib/wradlib-data/blob/main/README.md) to install and use this data files.
```

```{warning}
Be aware that applying an algorithm for error correction does not guarantee that the error is totally removed. Error correction procedures are suceptible to errors, too. Not only might they fail to *remove* the error. They might also introduce *new* errors. The trade-off between costs (introduction of new errors) and benefits (error reduction) can turn out differently for different locations, different points in time, or different rainfall situations.
```

## Reading the data


The binary encoding of many radar products is a major obstacle for many potential radar users. Often, decoder software is not easily available. $\omega radlib$ supports a couple of formats such as the ODIM_H5 implementation, NetCDF, and some formats used by the German Weather Service. With the porting of [xarray](https://xarray.dev/)-based code from $\omega radlib$ to [xradar](https://docs.openradarscience.org/projects/xradar/en/stable/) the range of supported formats will surely increase in the future.

Ever since the basic data type used in $\omega radlib$ is a multi-dimensional array, the numpy.ndarray. Such an array might e.g. represent a polar or cartesian grid, or a series of rain gage observations. Metadata are normally managed as Python dictionaries. In order to read the content of a data file into a numpy array, you would normally use the {mod}`wradlib.io` module. With the evolution in the scientific Python stack (eg. xarray) $\omega radlib$ adopted DataTree, Dataset and DataArray to conveniently hold radar data together with coordinates and metadata (now available in [xradar](https://docs.openradarscience.org/projects/xradar/en/stable/)). In the following example, a local PPI from the German Weather Service, a DX file, is read and converted to an xarray.DataArray and plotted:


The ``metadata`` object can be inspected via keywords. The ``data`` object contains the actual data, in this case a polar grid with 360 azimuth angles and 128 range bins.

```{code-cell} python
filename = wradlib_data.DATASETS.fetch("dx/raa00-dx_10908-0806021655-fbg---bin.gz")
data, metadata = wrl.io.read_dx(filename)
print(data.shape)
print(metadata.keys())
```

Using xarray we can conveniently hold the reflectivity data as well as the needed coordinates in one convenient structure.

```{code-cell} python
radar_location = (8.003611, 47.873611, 1516.0)
da = wrl.georef.create_xarray_dataarray(
    data,
    r=np.arange(500, data.shape[1] * 1000 + 500, 1000),
    phi=metadata["azim"],
    theta=metadata["elev"],
    site=radar_location,
    sweep_mode="azimuth_surveillance",
)
```

We can use xarray directly to create a nice plot.

```{code-cell} python
da.plot()
```

## Georeferencing and Projection


In order to define the horizontal and vertical position of the radar bins, we need to retrieve the corresponding 3-dimensional coordinates in terms of longitude, latitude and altitude. This information is required e.g. if the positions should be plotted on a map. It is also required for constructing [CAPPIs](https://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator). The position of a radar bin in 3-dimensional space depends on the position of the radar device, the elevation and azimuth angle of the radar beam, the range of the bin, and the assumed influence of atmospheric refraction on the beam propagation. For the sample data used above, the position of the radar device is the Feldberg in Germany (8.005, 47.8744, 1517):


We have the ability to easily georeference all radar bins (here with an azimuthal equidistant projection). This adds the projection as `crs_wkt` coordinate to the Dataset/DataArray.

```{code-cell} python
da = da.wrl.georef.georeference()
display(da)
```

$\omega radlib$ supports the projection between geographical coordinates (lon/lat) and other reference systems. It uses GDAL/OSR Spatial References Objects as function parameters. Basically, you have to create the OSR-object by using GDAL-capabilities or one of the provided helper functions. We recommend the creation using [EPSG numbers](https://epsg.io/):

```{code-cell} python
# UTM Zone 32, EPSG-Number 32632
utm = wrl.georef.epsg_to_osr(32632)
print(utm)
da_utm = da.wrl.georef.reproject(trg_crs=utm)
da_utm.plot(x="x", y="y")
```

Second, you can provide a string which represents the projection - based on the [PROJ library](https://proj.org). You can [look up projection strings](https://epsg.io/), but for some projections, $\omega radlib$ helps you to define a projection string. In the following example, the target projection is 'dwd-radolan':

```{code-cell} python
radolan = wrl.georef.create_osr("dwd-radolan")
da_radolan = da.wrl.georef.reproject(trg_crs=radolan)
da_radolan.plot(x="x", y="y")
```

```{seealso}
Get more info in the library reference section {mod}`wradlib.georef`.
```

For plotting we also can use the $\omega radlib$ plot-function, which detects the data as PPI and plots it.

```{code-cell} python
pm = da.wrl.vis.plot(add_colorbar=True)
```

```{note}
Get more info in the section [Supported radar data formats](../fileio/fileio) and in the library reference section {mod}`wradlib.io`.
```

## Clutter removal


Clutter are non-meteorological echos. They are caused by the radar beam hitting objects on the earth's surface (e.g. mountain or hill tops, houses, wind turbines) or in the air (e.g. airplanes, birds). These objects can potentially cause high reflectivities due large scattering cross sections. Static clutter, if not efficiently removed by Doppler filters, can cause permanent echos which could introduce severe bias in quantitative applications. Thus, an efficient identification and removal of clutter is mandatory e.g. for hydrological studies. Clutter removal can be based on static maps or dynamic filters. Normally, static clutter becomes visible more clearly in rainfall accumulation maps over periods of weeks or months. We recommend such accumulations to create static clutter maps which can in turn be used to remove the static clutter from an image and fill the resulting gaps by interpolation.


In the following example, the clutter filter published by [Gabella et al., 2002](https://docs.wradlib.org/en/latest/bibliography.html#gabella2002)) is applied to the single radar sweep of the above example:

```{code-cell} python
clutter = da.wrl.classify.filter_gabella(tr1=12, n_p=6, tr2=1.1)
pm = clutter.wrl.vis.plot(cmap=plt.cm.gray)
plt.title("Clutter Map")
```

The resulting Boolean array ``clutter`` indicates the position of clutter. It can be used to interpolate the values at those positons from non-clutter values, as shown in the following lines:

```{code-cell} python
data_no_clutter = da.wrl.ipol.interpolate_polar(clutter)
pm = data_no_clutter.wrl.vis.plot(add_colorbar=True)  # simple diagnostic plot
```

It is generally recommended to remove the clutter before e.g. gridding the data. Otherwise the clutter signal might be "smeared" over multiple grid cells, resulting into a decrease in detectability.


```{seealso}
Get more info in the library reference section {mod}`wradlib.classify`.
```

## Attenuation correction


Attenuation by wet radome and by heavy rainfall can cause serious underestimation of rainfall for [C-Band and X-Band](https://www.everythingweather.com/weather-radar/bands.shtml) devices. For such radar devices, situations with heavy rainfall require a correction of attenuation effects. <br>
The general approach with single-polarized radars is to use a recursive gate-by-gate approach. See [Kraemer et al., 2008](https://docs.wradlib.org/en/latest/bibliography.html#kraemer2008) for an introduction to this concept. Basically, the specific attenuation ``k`` of the first range gate is computed via a so-called ``k-Z`` relationship. Based on ``k``, the reflectivity of the second range gate is corrected and then used to compute the specific attenuation for the second range gate (and so on). <br>
The concept was first introduced by [Hitschfeld et al., 1954)](https://docs.wradlib.org/en/latest/bibliography.html#hitschfeld1954). Its main drawback is its suceptibility to instable behaviour. $\omega radlib$ provides different implementations which address this problem.


One example is the algorithm published by [Jacobi and Heistermann, 2016](https://docs.wradlib.org/en/latest/bibliography.html#jacobi2016):

```{code-cell} python
pia = data_no_clutter.wrl.atten.correct_attenuation_constrained(
    a_max=1.67e-4,
    a_min=2.33e-5,
    n_a=100,
    b_max=0.7,
    b_min=0.65,
    n_b=6,
    gate_length=1.0,
    constraints=[wrl.atten.constraint_dbz, wrl.atten.constraint_pia],
    constraint_args=[[59.0], [20.0]],
)
data_attcorr = data_no_clutter + pia
```

The first line computes the path integrated attenuation ``pia`` for each radar bin. The second line uses ``pia`` to correct the reflectivity values. Let's inspect the effect of attenuation correction for an azimuth angle of 65 deg:

```{code-cell} python
plt.figure(figsize=(10, 8))
data_attcorr[65].plot(label="attcorr")
data_no_clutter[65].plot(label="no attcorr")
plt.xlabel("km")
plt.ylabel("dBZ")
plt.legend()
```

```{note}
Get more info in the library reference section {mod}`wradlib.atten`. There you will learn to know the algorithms available for attenuation correction and how to manipulate their behaviour by using additional keyword arguments.
```

## Vertical Profile of Reflectivity


Precipitation is 3-dimensional in space. The vertical distribution of precipitation (and thus reflectivity) is typically non-uniform. As the height of the radar beam increases with the distance from the radar location (beam elevation, earth curvature), one sweep samples from different heights. The effects of the non-uniform VPR and the different sampling heights need to be accounted for if we are interested in the precipitation near the ground or in defined altitudes.

```{seealso}
Get more info in the library reference section {mod}`wradlib.vpr`. There you will learn how to reference polar volume data, to create CAPPIs and Pseudo CAPPIs, to inspect vertical profiles of reflectivity (UNDER DEVELOPMENT), and to use these for correction (UNDER DEVELOPMENT).
```

## Conversion of Reflectivity into Rainfall


Reflectivity (Z) and precipitation rate (R) can be related in form of a power law $R=a*Z^b$. The parameters ``a`` and ``b`` depend on the type of precipitation in terms of drop size distribution and water temperature. Before applying the Z-R relationship, we need to convert from dBZ to Z:

```{code-cell} python
z = data_attcorr.wrl.trafo.idecibel()
R = z.wrl.zr.z_to_r()  # wrl.zr.z_to_r(wrl.trafo.idecibel(data_attcorr))
```

The above line uses the default parameters ``a=200`` and ``b=1.6`` for the Z-R relationship. In order to compute a rainfall depth from rainfall intensity, we have to specify an integration interval in seconds. In this example, we choose five minutes (300 s), corresponding to the scan interval.

```{code-cell} python
depths = R.wrl.trafo.r_to_depth(300)
depths.wrl.vis.plot()
```

```{seealso}
Get more info in the section [Converting reflectivity to rainfall](wradlib_get_rainfall.ipynb) and in the library reference sections [Z-R Conversions](https://docs.wradlib.org/en/latest/zr.html) and [Data Transformation](https://docs.wradlib.org/en/latest/trafo.html). Here you will learn about the effects of the Z-R parameters ``a`` and ``b``.
```

## Rainfall accumulation


For many applications, accumulated rainfall depths over specific time intervals are required, e.g. hourly or daily accumulations. In the following example, we will use a synthetic time series of 5 minute intervals. Just imagine we have repeated the above procedure for one day of five-minute sweeps and combined the arrays of rainfall depth in a 3-dimensional array of shape ``(number of time steps, number of azimuth angles, number of range gates)``.

Now we want to compute the daily rainfall sum:

```{code-cell} python
sweep_times = wrl.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 300)
np.random.seed(1319622840)
depths_5min = np.random.uniform(size=(len(sweep_times) - 1, 360, 128))
depth_daily = depths_5min.sum(axis=0)
```

Check the shape the resulting array for plausibility:

```{code-cell} python
print(depth_daily.shape)
```

```{seealso}
For more advanced operations on time series, we recommend the [Pandas](https://pandas.pydata.org/) package.
```

## Gridding


Assume you would like to transfer the rainfall intensity from the [above example](#conversion-of-reflectivity-into-rainfall) from polar coordinates to a Cartesian grid, or to an arbitrary set of irregular points in space (e.g. centroids of sub-catchments). You already retrieved the Cartesian coordinates of the radar bins in the previous section [Georeferencing and Projection](#georeferencing-and-projection). Now you only need to define the target coordinates (e.g. a grid) and apply the ``togrid`` function of the {mod}`wradlib.comp` module. In this example, we want our grid only to represent the South-West sector of our radar circle on a 100 x 100 grid. First, we define the target grid coordinates (these must be an array of 100x100 rows with one coordinate pair each):

```{code-cell} python
xgrid = np.linspace(R.x.min().values, R.x.mean().values, 100)
ygrid = np.linspace(R.y.min().values, R.y.mean().values, 100)
cart = xr.Dataset(coords={"x": (["x"], xgrid), "y": (["y"], ygrid)})
cart
```

Now we transfer the polar data to the grid and mask out invalid values for plotting (values outside the radar circle receive NaN):

```{code-cell} python
gridded = data_attcorr.wrl.comp.togrid(
    cart, radius=128000.0, center=(0, 0), interpol=wrl.ipol.Nearest
)
display(gridded)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, aspect="equal")
pm = gridded.plot()
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
```

```{seealso}
Get more info about the function {func}`wradlib.comp.togrid`.
```

## Adjustment by rain gage observations


Adjustment normally refers to using rain gage observations on the ground to correct for errors in the radar-based rainfall estimating. [Goudenhoofdt et al., 2009](https://docs.wradlib.org/en/latest/bibliography.html#goudenhoofdt2009) provide an excellent overview of adjustment procedures. A typical approach is to quantify the error of the radar-based rainfall estimate *at* the rain gage locations, assuming the rain gage observation to be accurate. The error can be assumed to be additive, multiplicative, or a mixture of both. Most approaches assume the error to be heterogeneous in space. Hence, the error at the rain gage locations will be interpolated to the radar bin (or grid) locations and then used to adjust (correct) the raw radar rainfall estimates.

In the following example, we will use an illustrative one-dimensional example with synthetic data (just imagine radar rainfall estimates and rain gage observations along one radar beam).

First, we create the synthetic "true" rainfall (``truth``):

```{code-cell} python
import numpy as np

radar_coords = np.arange(0, 101)
np.random.seed(1319622840)
truth = np.abs(1.5 + np.sin(0.075 * radar_coords)) + np.random.uniform(
    -0.1, 0.1, len(radar_coords)
)
```

The radar rainfall estimate ``radar`` is then computed by imprinting a multiplicative ``error`` on ``truth`` and adding some noise:

```{code-cell} python
error = 0.75 + 0.015 * radar_coords
np.random.seed(1319622840)
radar = error * truth + np.random.uniform(-0.1, 0.1, len(radar_coords))
```

Synthetic gage observations ``obs`` are then created by selecting arbitrary "true" values:

```{code-cell} python
obs_coords = np.array([5, 10, 15, 20, 30, 45, 65, 70, 77, 90])
obs = truth[obs_coords]
```

Now we adjust the ``radar`` rainfall estimate by using the gage observations. First, you create an "adjustment object" from the approach you want to use for adjustment. After that, you can call the object with the actual data that is to be adjusted. Here, we use a multiplicative error model with spatially heterogeneous error (see {mod}`wradlib.adjust.AdjustMultiply`:

```{code-cell} python
adjuster = wrl.adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=3)
adjusted = adjuster(obs, radar)
```

Let's compare the ``truth``, the ``radar`` rainfall estimate and the ``adjusted`` product:

```{code-cell} python
plt.plot(radar_coords, truth, "k-", label="True rainfall", linewidth=2.0)
plt.xlabel("Distance (km)")
plt.ylabel("Rainfall intensity (mm/h)")
plt.plot(
    radar_coords,
    radar,
    "k-",
    label="Raw radar rainfall",
    linewidth=2.0,
    linestyle="dashed",
)
plt.plot(
    obs_coords,
    obs,
    "o",
    label="Gage observation",
    markersize=10.0,
    markerfacecolor="grey",
)
plt.plot(
    radar_coords,
    adjusted,
    "-",
    color="green",
    label="Multiplicative adjustment",
    linewidth=2.0,
)
plt.legend(prop={"size": 12})
```

```{seealso}
Get more info in the library reference section {mod}`wradlib.adjust`. There, you will also learn how to use the built-in *cross-validation* in order to evaluate the performance of the adjustment approach.
```

## Verification and quality control


Typically, radar-based precipitation estimation and the effectiveness of the underlying correction and adjustment methods are verified by comparing the results against rain gage observations on the ground. {mod}`wradlib.verify` module provides procedures not only to extract the radar values at specific gauge locations, but also a set of error metrics which are computed from gage observations and the corresponding radar-based precipitation estimates (including standard metrics such as RMSE, mean error, Nash-Sutcliffe Efficiency). In the following, we will illustrate the usage of error metrics by comparing the "true" rainfall against the raw and adjusted radar rainfall estimates from the above example:

```{code-cell} python
raw_error = wrl.verify.ErrorMetrics(truth, radar)
adj_error = wrl.verify.ErrorMetrics(truth, adjusted)
```

Error metrics can be reported e.g. as follows:

```{code-cell} python
print("Error metrics for unadjusted radar rainfall estimates:")
raw_error.pprint()
print("\nError metrics for adjusted radar rainfall estimates:")
adj_error.pprint()
```

```{seealso}
Get more info in the library reference section {mod}`wradlib.verify`.
```

## Visualisation and mapping


In the above sections [Reading the data](#reading-the-data), [Clutter removal](#clutter-removal), and [Gridding](#gridding) you already saw examples of the $\omega radlib's$ plotting capabilities.


```{seealso}
Get more info in the library reference section {mod}`wradlib.vis`.
```

## Data export to other applications


Once you created a dataset which meets your requirements, you might want to export it to other applications or archives. $\omega radlib$ does not favour or support a specific output format. Basically, you have all the freedom of choice offered by Python and its packages in order to export your data. Arrays can be stored as text or binary files by using numpy functions. You can use the package [NetCDF4](https://unidata.github.io/netcdf4-python/) to write NetCDF files, and the packages [h5py](https://www.h5py.org/) or [PyTables](https://www.pytables.org) to write hdf5 files.

Using [xradar](https://docs.openradarscience.org/projects/xradar/en/stable/) a standardized data export is available. Of course, you can also export data as images. See {mod}`wradlib.vis` for some options.

Export your data array as a text file:

```{code-cell} python
np.savetxt("mydata.txt", data_attcorr)
```

Or as a gzip-compressed text file:

```{code-cell} python
np.savetxt("mydata.gz", data_attcorr)
```

Or as a NetCDF file:

```{code-cell} python
data_attcorr.name = "reflectivity corrected"
data_attcorr.to_netcdf("test.nc", group="sweep_0")
```

```{code-cell} python
with xr.open_dataset("test.nc", group="sweep_0") as ds:
    display(ds)
```

```{note}
An example for hdf5 export will follow.
```
