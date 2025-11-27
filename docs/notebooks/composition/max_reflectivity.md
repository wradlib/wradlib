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

# Production of a maximum reflectivity composite from 3 neighboring radars

```{code-cell} python
import tempfile

import xarray
import xradar
import matplotlib.pyplot

import wradlib
import wradlib_data
```

## Get measurements from three belgian radars

```{code-cell} python
filenames = ["bejab.pvol.hdf", "bewid.pvol.hdf", "behel.pvol.hdf"]
paths = [wradlib_data.DATASETS.fetch(f"hdf5/{f}") for f in filenames]
volumes = [xradar.io.backends.odim.open_odim_datatree(p) for p in paths]
```

## Define a raster dataset with projected coordinates

```{code-cell} python
europe_crs = 3035
bounds = [3614000, 2783000, 4200000, 3338000]
resolution = 1000
raster_projected = wradlib.georef.create_raster_xarray(
    crs=europe_crs,
    bounds=bounds,
    resolution=resolution,
)
```

## Define a raster dataset with geographic coordinates

```{code-cell} python
bounds = [0, 48, 9, 53]
resolution = 1000
raster_geographic = wradlib.georef.create_raster_geographic(
    bounds=bounds, resolution=resolution, resolution_in_meters=True
)
```

## Transform lowest sweep into a raster for each radar

```{code-cell} python
raster = {}
raster["projected"] = raster_projected
raster["geographic"] = raster_geographic
rasters = {}
for key in ["projected", "geographic"]:
    metadata = xradar.model.required_sweep_metadata_vars
    rasters_radar = []
    for volume in volumes:
        sweep = volume["sweep_0"].to_dataset()
        sweep = sweep[["DBZH"] + list(metadata)]
        sweep = sweep.sel(range=slice(0, 200e3))
        sweep = xradar.georeference.get_x_y_z(sweep)
        raster_radar = wradlib.comp.sweep_to_raster(
            sweep=sweep,
            raster=raster[key],
        )
        rasters_radar.append(raster_radar)

    for raster_radar in rasters_radar:
        raster_radar["DBZH"].plot(vmin=0, vmax=50)
        matplotlib.pyplot.axis("equal")
        matplotlib.pyplot.show()
    rasters[key] = rasters_radar
```

## Take the maximum reflectivity value from the 3 rasters
```{code-cell} python
for key in ["projected", "geographic"]:
    rasters_concat = xarray.concat(rasters[key], dim="sweep")
    comp = rasters_concat.max(dim="sweep", keep_attrs=True)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        comp.to_netcdf(tmp.name)
    comp = comp.drop_vars("spatial_ref")
    fig, ax = matplotlib.pyplot.subplots()
    comp["DBZH"].plot(ax=ax, vmin=0, vmax=50)
    ax.set_title(key)
```
