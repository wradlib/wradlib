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

# RADOLAN Grid


## Polar Stereographic Projection


The projected composite raster is equidistant with a grid-spacing of 1.0 km in most cases. There are composites which have 2.0 km grid-spacing (e.g. PC).

There are five different grid sizes, the well-known 900 rows by 900 columns (normal, 1km grid spacinf), 1500 rows by 1400 columns (extended, 1km grid spacing), 460 rows by 460 columns (small, 2km grid spacing) and the legacy 450 rows by 450 rows (2km grid spacing). Since the RADSYS-E project is finalized an extended national composite with 1100 rows by 900 columns (normal_wx, 1km grid spacing) is available, too.

Common to all is that the plane of projection intersects the earth sphere at $\phi_0 = 60.0^{\circ}N$. The cartesian co-ordinate system is aligned parallel to the $\lambda_0 = 10.0^{\circ}E$ meridian.

The reference point ($\lambda_m$, $\phi_m$) is $9.0^{\circ}E$ and $51.0^{\circ}N$, which is the center of the two smaller grids. The extended european grid has an offset in respect to this reference point of 350km by 150km, the extended national grid 100km by -80km.

The earth as sphere with an radius of 6370.04 km is used for all calculations.

With formulas (1), (2) and (3) the geographic reference points ($\lambda$, $\phi$) can be converted to projected cartesian coordinates. The calculated (x y) is the distance vector to the origign of the cartesian coordinate system (north pole).

$\begin{equation}
x = R * M(\phi) * cos(\phi) * sin(\lambda - \lambda_0)
\tag{1}
\end{equation}$

$\begin{equation}
y = -R * M(\phi) * cos(\phi) * cos(\lambda - \lambda_0)
\tag{2}
\end{equation}$

$\begin{equation}
M(\phi) =  \frac {1 + sin(\phi_0)} {1 + sin(\phi)}
\tag{3}
\end{equation}$

Assumed the point ($10.0^{\circ}E$, $90.0^{\circ}N$) is defined as coordinate system origin. Then all coordinates can be calculated with the known grid-spacing d as:

$\begin{equation}
x = x_0 + d * (j - j_0)
\tag{4}
\end{equation}$

$\begin{equation}
y = y_0 + d * (i - i_0)
\tag{5}
\end{equation}$

with i, j as cartesian indices.

$\omega radlib$ provides the convenience function {func}`wradlib.georef.get_radolan_grid` which returns the radolan grid for further processing. It takes `nrows` and `ncols` as parameters and returns the projected cartesian coordinates or the wgs84 coordinates (keyword arg wgs84=True) as numpy ndarray (nrows x ncols x 2).

```{code-cell} python
import wradlib as wrl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

### 900x900, 1km grid box

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
radolan_grid_ll = wrl.georef.get_radolan_grid(900, 900, wgs84=True)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

### 1100x900, 1km grid box

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(1100, 900)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
radolan_grid_ll = wrl.georef.get_radolan_grid(1100, 900, wgs84=True)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

### 1500x1400, 1km grid box

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(1500, 1400)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
radolan_grid_ll = wrl.georef.get_radolan_grid(1500, 1400, wgs84=True)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

### 460x460, 2km grid box

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(460, 460)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
radolan_grid_ll = wrl.georef.get_radolan_grid(460, 460, wgs84=True)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

### 450x450, 2km grid box

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(450, 450)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
radolan_grid_ll = wrl.georef.get_radolan_grid(450, 450, wgs84=True)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

## Inverse Polar Stereographic Projection


The geographic coordinates of specific datapoints can be calculated by using the cartesian coordinates (x,y) and the following formulas:

$\begin{equation}
\lambda = \arctan\left(\frac {-x} {y}\right) + \lambda_0
\tag{6}
\end{equation}$

$\begin{equation}
\phi = \arcsin\left(\frac {R^2 * \left(1 + \sin\phi_0\right)^2 - \left(x^2 + y^2\right)} {R^2 * \left(1 + \sin\phi_0\right)^2 + \left(x^2 + y^2\right)}\right)
\tag{7}
\end{equation}$


## Standard Formats


### WKT-String


The German Weather Service provides a [WKT-string](https://maps.dwd.de/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.SRSDescriptionPage?0&code=EPSG:1000001). This WKT (well known text) is used to create the osr-object representation of the radolan projection.

For the scale_factor the intersection of the projection plane with the earth sphere at $60.0^{\circ}N$ has to be taken into account:

$\begin{equation}
scale\_factor = \frac {1 + \sin\left(60.^{\circ}\right)} {1 + \sin\left(90.^{\circ}\right)} = 0.93301270189
\tag{8}
\end{equation}$

Also, the `PROJECTION["Stereographic_North_Pole"]` isn't known within GDAL/OSR. It has to be changed to the known `PROJECTION["polar_stereographic"]`.

With these adaptions we finally yield the Radolan Projection as WKT-string. This WKT-string is used within $\omega radlib$ to create the osr-object by using the helper-function {func}`wradlib.georef.create_osr`.

```{code-cell} python
proj_stereo = wrl.georef.create_osr("dwd-radolan")
print(proj_stereo)
```

### PROJ

Using the above WKT-String the PROJ representation can be derived as:

```
+proj = stere + lat_0 = 90 + lat_ts = 90 + lon_0 = 10 + k = 0.93301270189
+x_0 = 0 + y_0 = 0 + a = 6370040 + b = 6370040 + to_meter = 1000 + no_defs
```

This PROJ-string can also be used to create the osr-object by using the helper-function {func}`wradlib.georef.projstr_to_osr`:

```{code-cell} python
# create radolan projection osr object
dwd_string = (
    "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 "
    "+x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs"
)
proj_stereo = wrl.georef.projstr_to_osr(dwd_string)
print(proj_stereo)
```

## Grid Reprojection


Within $\omega radlib$ the {func}`wradlib.georef.reproject` function can be used to convert the RADOLAN grid data from xy-space to lonlat-space and back. First, we need to create the necessary Spatial Reference Objects for the RADOLAN-projection and wgs84.

```{code-cell} python
from osgeo import osr

proj_stereo = wrl.georef.create_osr("dwd-radolan")
print(proj_stereo)
proj_wgs = osr.SpatialReference()
proj_wgs.ImportFromEPSG(4326)
print(proj_wgs)
```

Then, we call `reproject` with the osr-objects as `projection_source` and `projection_target` parameters.

```{code-cell} python
radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
radolan_grid_ll = wrl.georef.reproject(
    radolan_grid_xy, src_crs=proj_stereo, trg_crs=proj_wgs
)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_ll.shape, *radolan_grid_ll[0, 0, :])
)
```

And the other way round.

```{code-cell} python
radolan_grid_xy = wrl.georef.reproject(
    radolan_grid_ll, src_crs=proj_wgs, trg_crs=proj_stereo
)
print(
    "{0}, ({1:.4f}, {2:.4f})".format(radolan_grid_xy.shape, *radolan_grid_xy[0, 0, :])
)
```

In the following example the RADOLAN grid is projected to wgs84 and GaussKrüger Zone3.

```{code-cell} python
# create UTM zone 32 projection osr object
proj_utm32 = osr.SpatialReference()
proj_utm32.ImportFromEPSG(32632)

# transform radolan polar stereographic projection to wgs84 and then to utm zone 32
radolan_grid_ll = wrl.georef.reproject(
    radolan_grid_xy, src_crs=proj_stereo, trg_crs=proj_wgs
)
radolan_grid_utm32 = wrl.georef.reproject(
    radolan_grid_ll, src_crs=proj_wgs, trg_crs=proj_utm32
)

lon_wgs0 = radolan_grid_ll[:, :, 0]
lat_wgs0 = radolan_grid_ll[:, :, 1]

x_utm32 = radolan_grid_utm32[:, :, 0]
y_utm32 = radolan_grid_utm32[:, :, 1]

x_rad = radolan_grid_xy[:, :, 0]
y_rad = radolan_grid_xy[:, :, 1]

print("\n------------------------------")
print("source radolan x,y-coordinates")
print("       {0}      {1} ".format("x [km]", "y [km]"))
print("ll: {:10.4f} {:10.3f} ".format(x_rad[0, 0], y_rad[0, 0]))
print("lr: {:10.4f} {:10.3f} ".format(x_rad[0, -1], y_rad[0, -1]))
print("ur: {:10.4f} {:10.3f} ".format(x_rad[-1, -1], y_rad[-1, -1]))
print("ul: {:10.4f} {:10.3f} ".format(x_rad[-1, 0], y_rad[-1, 0]))
print("\n--------------------------------------")
print("transformed radolan lonlat-coordinates")
print("      {0}  {1} ".format("lon [degE]", "lat [degN]"))
print("ll: {:10.4f}  {:10.4f} ".format(lon_wgs0[0, 0], lat_wgs0[0, 0]))
print("lr: {:10.4f}  {:10.4f} ".format(lon_wgs0[0, -1], lat_wgs0[0, -1]))
print("ur: {:10.4f}  {:10.4f} ".format(lon_wgs0[-1, -1], lat_wgs0[-1, -1]))
print("ul: {:10.4f}  {:10.4f} ".format(lon_wgs0[-1, 0], lat_wgs0[-1, 0]))
print("\n-----------------------------------")
print("transformed radolan utm32-coordinates")
print("     {0}   {1} ".format("easting [m]", "northing [m]"))
print("ll: {:10.0f}   {:10.0f} ".format(x_utm32[0, 0], y_utm32[0, 0]))
print("lr: {:10.0f}   {:10.0f} ".format(x_utm32[0, -1], y_utm32[0, -1]))
print("ur: {:10.0f}   {:10.0f} ".format(x_utm32[-1, -1], y_utm32[-1, -1]))
print("ul: {:10.0f}   {:10.0f} ".format(x_utm32[-1, 0], y_utm32[-1, 0]))
```
