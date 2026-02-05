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

# Projections

In this notebook {{wradlib}}'s projection capabilities are highlighted, from several helper functions to full featured coordinate reprojection.

```{code-cell} python3
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import wradlib as wrl
import xradar as xd

warnings.filterwarnings("ignore")
```

## Projection Helpers

- {func}`wradlib.georef.projection.get_default_projection`
- {func}`wradlib.georef.projection.get_earth_projection`
- {func}`wradlib.georef.projection.get_earth_radius`
- {func}`wradlib.georef.projection.get_radar_projection`
- {func}`wradlib.georef.projection.create_crs`
- {func}`wradlib.georef.projection.ensure_crs`

+++

### get_default_projection

```{code-cell} python3
def_proj = wrl.georef.get_default_projection()
print(type(def_proj))
display(def_proj)
```

### get_earth_projection

Defaults to WGS84 ellipsoid.

```{code-cell} python3
earth = wrl.georef.get_earth_projection()
display(earth)
```

```{code-cell} python3
earth2 = wrl.georef.get_earth_projection(model="geoid")
display(earth2)
```

### get_earth_radius

Return the radius of the earth for a given spheroid model at a given latitude.

$R^2 = \frac{a^4 \cos(f)^2 + b^4 \sin(f)^2} {a^2 \cos(f)^2 + b^2 \sin(f)^2}$

```{code-cell} python3
lat = np.arange(-90, 90)
er = wrl.georef.get_earth_radius(lat)
```

```{code-cell} python3
plt.plot(lat, er)
```

### get_radar_projection
Get the native radar projection which is an azimuthal equidistant projection centered at the site using WGS84.

```{code-cell} python3
site = (7, 53)
radar = wrl.georef.get_radar_projection(site)
display(radar)
```

### create_crs

Conveniently create RADOLAN projection objects.

```{code-cell} python3
radolan = wrl.georef.create_crs("dwd-radolan")
display(radolan.to_cf())
```

```{code-cell} python3
radolan = wrl.georef.create_crs("dwd-radolan-wgs84")
display(radolan.to_cf())
```

```{code-cell} python3
radolan = wrl.georef.create_crs("dwd-radolan-wgs84-de1200")
display(radolan.to_cf())
```

```{code-cell} python3
radolan = wrl.georef.create_crs("dwd-radolan-wgs84-rx")
display(radolan.to_cf())
```

### ensure_crs

Helper function to always return correct projection objects. Defaults to {class}`pyproj:pyproj.crs.CRS`. For easy conversion from/to pyproj/cartopy/osgeo.osr.

```{code-cell} python3
radar_pyproj = wrl.georef.ensure_crs(radar, trg="pyproj")
print(type(radar_pyproj))
display(radar_pyproj)
```

```{code-cell} python3
radar_cartopy = wrl.georef.ensure_crs(radar, trg="cartopy")
print(type(radar_cartopy))
display(radar_cartopy)
```

```{code-cell} python3
radar_osr = wrl.georef.ensure_crs(radar, trg="osr")
print(type(radar_osr))
print(radar_osr)
```

## Reproject

{{wradlib}}'s power work horse for coordinate transformation. Relying on {class}`pyproj:pyproj.transformer.Transformer` and {class}`pyproj:pyproj.crs.CRS`.

Can be attributed with Coordinate Reference Systems (CRS) of any provenience.

```{code-cell} python3
swp = (
    xd.model.create_sweep_dataset(rng=150)
    .swap_dims(time="azimuth")
)
swp = swp.assign_coords(sweep_mode="azimuthal_surveillance")
display(swp)
```

```{code-cell} python3
swp = swp.wrl.georef.georeference()
display(swp)
```

```{code-cell} python3
crs = pyproj.CRS.from_epsg(2056)
swp2 = swp.wrl.georef.reproject(trg_crs=crs)
display(swp2)
```

```{code-cell} python3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

swp.z.plot(x="x", y="y", ax=ax1)
ax1.set_title(swp.xradar.get_crs().name)
swp2.z.plot(x="x", y="y", ax=ax2)
ax2.set_title(swp2.xradar.get_crs().name)
fig.tight_layout()
```
