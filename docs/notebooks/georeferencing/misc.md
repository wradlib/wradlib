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

# Radar Bins

```{code-cell} python3
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import wradlib as wrl
import xradar as xd

warnings.filterwarnings("ignore")
```

## Artificial Dataset

```{code-cell} python3
swp = (
    xd.model.create_sweep_dataset(rng=150)
    .swap_dims(time="azimuth")
)
swp = swp.assign_coords(sweep_mode="azimuthal_surveillance")
display(swp)
```

## Radar Bin Altitude

With {func}`wradlib.georef.GeorefMethods.bin_altitude` we can calculate the height of a radar bin taking the refractivity of the atmosphere into account.

Based on {cite}`Doviak1993` the bin altitude is calculated as $h = \sqrt{r^2 + (k_e r_e)^2 + 2 r k_e r_e \sin\theta} - k_e r_e$.

```{code-cell} python3
ke = [0.5, 1., 4/3., 2., 5.]
alt = [swp.wrl.georef.bin_altitude(ke=k) for k in ke]
```

```{code-cell} python3
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot()

for k, ba in zip(ke, alt):
    ba.isel(azimuth=0).plot(ax=ax, label=f"ke={k:.2f}")

ax.legend(loc="best")
ax.set_title("Bin Altitude")
ax.grid()
ax.set_ylim(0, 6500)
ax.set_xlim(0, 150e3)
fig.tight_layout()
```

## Radar Bin Distance

With {func}`wradlib.georef.GeorefMethods.bin_distance` we can calculate the great circle distance from radar site to radar bin over spherical earth, taking the refractivity of the atmosphere into account.

   $s = k_e r_e \arctan\left(\frac{r \cos\theta}{r \cos\theta + k_e r_e + h}\right)$

where $h$ would be the radar site altitude amsl.

```{code-cell} python3
ke = [0.5, 1., 4/3., 2., 5., 50.]
dist = [swp.wrl.georef.bin_distance(ke=k) for k in ke]

for di in dist:
    print(di[0][0].values, di[0][-1].values)
```

## Radar Site Distance

With {func}`wradlib.georef.GeorefMethods.site_distance` we can calculate the great circle distance from bin at certain altitude to the radar site over spherical earth, taking the refractivity of the atmosphere into account. Based on {cite}`Doviak1993` the site distance may be calculated as:

   $s = k_e r_e \arcsin\left(\frac{r \cos\theta}{k_e r_e + h_n(r, \theta, r_e, k_e)}\right)$

where $h_n$ is provided under the hood by by {func}`~wradlib.georef.misc.bin_altitude`.

```{code-cell} python3
ke = [0.5, 1., 4/3., 2., 5., 50.]
dist = [swp.wrl.georef.site_distance(ke=k) for k in ke]

for di in dist:
    print(di[0][0].values, di[0][-1].values)
```
