---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
  main_language: python
kernelspec:
  display_name: Python 3
  name: python3
---

# Clutter detection by using space-born cloud images

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import wradlib as wrl
import wradlib_data
import xarray as xr
import xradar as xd
from IPython.display import display
from osgeo import osr
```

## Read the radar data into DataTree

```{code-cell} python
# read the radar volume scan
filename = "hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf"
filename = wradlib_data.DATASETS.fetch(filename)
pvol = xd.io.open_odim_datatree(filename)
display(pvol)
```

## Georeference sweeps

```{code-cell} python
pvol1 = pvol.match("sweep*")
display(pvol1)
vol = []
for sweep in pvol1.values():
    vol.append(sweep.to_dataset().pipe(wrl.georef.georeference))
vol = xr.concat(vol, dim="tilt")
vol = vol.assign_coords(sweep_mode=vol.sweep_mode)
display(vol)
```

## Construct collocated satellite data

```{code-cell} python
proj_radar = osr.SpatialReference()
proj_radar.ImportFromWkt(vol.crs_wkt.attrs["crs_wkt"])
```

```{code-cell} python
filename = "hdf5/SAFNWC_MSG3_CT___201304290415_BEL_________.h5"
filename = wradlib_data.DATASETS.fetch(filename)
```

```{code-cell} python
sat_gdal = wrl.io.read_safnwc(filename)
val_sat = wrl.georef.read_gdal_values(sat_gdal)
coord_sat = wrl.georef.read_gdal_coordinates(sat_gdal)
proj_sat = wrl.georef.read_gdal_projection(sat_gdal)
coord_sat = wrl.georef.reproject(coord_sat, src_crs=proj_sat, trg_crs=proj_radar)
```

```{code-cell} python
coord_radar = np.stack((vol.x, vol.y), axis=-1)
coord_sat[..., 0:2].reshape(-1, 2).shape, coord_radar[..., 0:2].reshape(-1, 2).shape
```

```{code-cell} python
interp = wrl.ipol.Nearest(
    coord_sat[..., 0:2].reshape(-1, 2), coord_radar[..., 0:2].reshape(-1, 2)
)
```

```{code-cell} python
val_sat = interp(val_sat.ravel()).reshape(coord_radar.shape[:-1])
```

## Estimate localisation errors

```{code-cell} python
timelag = 9 * 60
wind = 10
error = np.absolute(timelag) * wind
```

## Identify clutter based on collocated cloudtype

```{code-cell} python
rscale = vol.range.diff("range").median().values
clutter = wrl.classify.filter_cloudtype(
    vol.DBZH, val_sat, scale=rscale, smoothing=error
)
```

## Assign to vol

```{code-cell} python
vol = vol.assign(sat=(["tilt", "azimuth", "range"], val_sat))
vol = vol.assign(clutter=(["tilt", "azimuth", "range"], clutter.values))
display(vol)
```

## Plot the results

```{code-cell} python
fig = plt.figure(figsize=(16, 8))

tilt = 0

ax = fig.add_subplot(131)
pm = vol.DBZH[tilt].wrl.vis.plot(ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Radar reflectivity")

ax = fig.add_subplot(132)
pm = vol.sat[tilt].wrl.vis.plot(ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Satellite cloud classification")

ax = fig.add_subplot(133)
pm = vol.clutter[tilt].wrl.vis.plot(ax=ax)
# plt.colorbar(pm, shrink=0.5)
plt.title("Detected clutter")

fig.tight_layout()
```
