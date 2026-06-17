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

# System Differential Phase {math}`\Phi_{DP}^{sys}`$

```{code-cell} ipython3
import warnings

from IPython.display import display

warnings.filterwarnings("ignore")
```

Correct retrieval of {math}`\Phi_{DP}^{sys}` (Offset) is crucial for correct processing of further derivations. Normally {math}`\Phi_{DP}^{sys}` is more or less constant. It can have azimuthal/elevational deviations, eg. depending on the antenna near field.

Retrieval of {math}`\Phi_{DP}^{sys}` is sometimes tedious, due to the contamination of the signal with clutter and other artifacts.

# Import Section

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import wradlib as wrl
import open_radar_data
```

# Open Dataset

We use a dataset from Surgavere Radar, Estonia.

```{code-cell} ipython3
filename = open_radar_data.DATASETS.fetch("SUR.202506091000.VOL.h5")
swp = xr.open_dataset(filename, engine="odim", group="sweep_0").set_coords("sweep_mode").wrl.georef.georeference()
display(swp)
```

# Inspect Dataset

```{code-cell} ipython3
display(swp)
```

```{code-cell} ipython3
swp.PHIDP.wrl.vis.plot(vmin=0, vmax=360)
```

# System Differential Phase {math}`\Phi_{DP}^{sys}` via first precipitating bins

This is a most common algorithm. But there are several approaches. All use common $\rho_{HV}$ filtering or other means of reducing unwanted artifacts in {math}`\Phi_{DP}^{sys}`.

1. N consecutive radar bins with {math}`\rho_{HV}` > threshold
2. maximum number of valid bins in a N-size window
3. first N valid bins (not necessarily consecutive)

+++

## Mask source data

Essential pre-step masking unwanted signal.

```{code-cell} ipython3
phimask = swp.PHIDP.where(swp.RHOHV >= 0.9)
```

## N consecutive valid radar bins

1. It finds the first N consecutive precipitating bins per each ray
2. and uses the median of these values to determine the offset per ray.
3. If there are only a few of those radials (<30) per sweep we'll use a default value from a previous sweep
4. Sort the median values from 2. and calculate the median from the smallest 30 (default). That value is considered {math}`\Phi_{DP}^{sys}`.

```{seealso}
{func}`wradlib.dp.system_phidp_block`
```

```{code-cell} ipython3
phisys_block = phimask.wrl.dp.system_phidp_block(rng=2000.)
display(phisys_block)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
phisys_block.start_range.plot(ax=ax1, lw=0.8, c="k")
phisys_block.stop_range.plot(ax=ax1, lw=0.8, c="k")
phimask.plot(ax=ax1, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax1.set_title(rf"$\phi_{{DP}}^{{sys}}$ - {phisys_block.valid_bins[0].values} consecutive valid radar bins")

phisys_block.start_range.plot(ax=ax2, lw=0.8, c="k")
phisys_block.stop_range.plot(ax=ax2, lw=0.8, c="k")
phimask.plot(ax=ax2, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax2.set_title(rf"$\phi_{{DP}}^{{sys}}$ - zoom")
ax2.set_ylim(0, 15e3)
display(phisys_block.sysphi)
fig.tight_layout()
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
# set the lable go clockwise and start from the top
ax.set_theta_zero_location("N")
# clockwise
ax.set_theta_direction(-1)

theta = np.linspace(0, 2 * np.pi, num=phisys_block.dims["azimuth"], endpoint=False)
ax.plot(theta, phisys_block.sysphi_ray, color="b", linewidth=1)
ax.plot(theta, np.ones_like(theta)*phisys_block.sysphi.values, color="r", linewidth=1)
_ = ax.set_title(rf"$\phi_{{DP}}^{{sys}}$")
```

## maximum number of valid radar bins in N-sized window

1. Calculate the number of valid bins for a window of size N along the ray
2. find the position where this valid bin number has it's maximum along the ray
3. and uses the median of these values to determine the offset per ray.
4. a) Sort the median values from 3. and calculate the median from the smallest 30. That value is considered {math}`\Phi_{DP}^{sys}`.
   b) Calculate the median from 3. That value is considered system PhiDP.

```{seealso}
{func}`wradlib.dp.system_phidp_window`
```

```{code-cell} ipython3
phisys_window = phimask.wrl.dp.system_phidp_window(2000.)
display(phisys_window)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
phisys_window.start_range.plot(ax=ax1, lw=0.8, c="k")
phisys_window.stop_range.plot(ax=ax1, lw=0.8, c="k")
phimask.plot(ax=ax1, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax1.set_title(rf"$\phi_{{DP}}^{{sys}}$ - {phisys_window.valid_bins[0].values} valid radar bins")

phisys_window.start_range.plot(ax=ax2, lw=0.8, c="k")
phisys_window.stop_range.plot(ax=ax2, lw=0.8, c="k")
phimask.plot(ax=ax2, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax2.set_title(rf"$\phi_{{DP}}^{{sys}}$ - zoom")
ax2.set_ylim(0, 15e3)
display(phisys_window.sysphi)
fig.tight_layout()
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
# set the lable go clockwise and start from the top
ax.set_theta_zero_location("N")
# clockwise
ax.set_theta_direction(-1)

theta = np.linspace(0, 2 * np.pi, num=phisys_window.dims["azimuth"], endpoint=False)
ax.plot(theta, phisys_window.sysphi_ray, color="b", linewidth=1)
ax.plot(theta, np.ones_like(theta)*phisys_window.sysphi.values, color="r", linewidth=1)
_ = ax.set_title(rf"$\phi_{{DP}}^{{sys}}$")
```

## first N valid bins

1. get the first N valid bins per each ray
2. calculate median from these values
3. a) Sort the median values from 2. and calculate the median from the smallest 30. That value is considered {math}`\Phi_{DP}^{sys}`.
   b) Calculate the median from 2. That value is considered system PhiDP.

```{seealso}
{func}`wradlib.dp.system_phidp_first`
```

```{code-cell} ipython3
phisys_first = phimask.wrl.dp.system_phidp_first(11)
display(phisys_first)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
phisys_first.start_range.plot(ax=ax1, lw=0.8, c="k")
phisys_first.stop_range.plot(ax=ax1, lw=0.8, c="k")
phimask.plot(ax=ax1, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax1.set_title(rf"$\phi_{{DP}}^{{sys}}$ - {phisys_first.valid_bins[0].values} valid radar bins")

phisys_first.start_range.plot(ax=ax2, lw=0.8, c="k")
phisys_first.stop_range.plot(ax=ax2, lw=0.8, c="k")
phimask.plot(ax=ax2, x="azimuth", y="range", cmap="turbo", vmin=0, vmax=140)
ax2.set_title(rf"$\phi_{{DP}}^{{sys}}$ - zoom")
ax2.set_ylim(0, 15e3)
display(phisys_first.sysphi)
fig.tight_layout()
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
# set the lable go clockwise and start from the top
ax.set_theta_zero_location("N")
# clockwise
ax.set_theta_direction(-1)

theta = np.linspace(0, 2 * np.pi, num=phisys_first.dims["azimuth"], endpoint=False)
ax.plot(theta, phisys_first.sysphi_ray, color="b", linewidth=1)
ax.plot(theta, np.ones_like(theta)*phisys_first.sysphi.values, color="r", linewidth=1)
_ = ax.set_title(rf"$\phi_{{DP}}^{{sys}}$")
```

# System Differential Phase {math}`\Phi_{DP}^{sys}` via phase histogram

The idea behind is:

- {math}`\Phi_{DP}^{sys}` is constantly increasing
- {math}`\Phi_{DP}^{sys}` is inherently noisy

That means, the majority of phase measurements (precipitating bins) should lie around {math}`\Phi_{DP}^{sys}`.
It's relatively robust since it does not rely on finding precipitating bins in each ray or the like. Nevertheless, taking a pre-filtered phase as input should return similar results.

```{seealso}
{func}`wradlib.dp.system_phidp_hist`
```

```{code-cell} ipython3
from xhistogram.xarray import histogram as xhist

hlist = []
phase_res = 0.1
bins = (0, 360, phase_res)
```

```{code-cell} ipython3
phisys_hist = swp.PHIDP.wrl.dp.system_phidp_hist(bins=bins)
display(phisys_hist)
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
# set the lable go clockwise and start from the top
ax.set_theta_zero_location("N")
# clockwise
ax.set_theta_direction(-1)

theta = np.linspace(0, 2 * np.pi, num=phisys_first.dims["azimuth"], endpoint=False)
ax.plot(theta, phisys_hist.sysphi_peak_ray, color="b", linewidth=1)
ax.plot(theta, phisys_hist.sysphi_first_ray, color="r", linewidth=1)
ax.plot(theta, np.ones_like(theta)*phisys_hist.sysphi_peak.values, color="b", linewidth=1.0)
ax.plot(theta, np.ones_like(theta)*phisys_hist.sysphi_first.values, color="r", linewidth=1.0)
_ = ax.set_title(rf"$\phi_{{DP}}^{{sys}}$")
ax.set_ylim(60, 100)
```

# Overview and Diagnostic plots

add theta in radians to dataset

```{code-cell} ipython3
theta = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
phisys_block = phisys_block.assign_coords(
    theta=(["azimuth"], theta, {"standard_name": "azimuth angle"})
)
phisys_window = phisys_window.assign_coords(
    theta=(["azimuth"], theta, {"standard_name": "azimuth angle"})
)
phisys_first = phisys_first.assign_coords(
    theta=(["azimuth"], theta, {"standard_name": "azimuth angle"})
)
phisys_hist = phisys_hist.assign_coords(
    theta=(["azimuth"], theta, {"standard_name": "azimuth angle"})
)
```

```{code-cell} ipython3
vmin = 0
vmax = 120
startaz = swp.sortby("time").azimuth[0]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.axvline(np.radians(startaz.values), c="black", lw=1.0)
phisys_block.sysphi_ray.plot(x="theta", c="k", lw=0.5, ax=ax)
phisys_window.sysphi_ray.plot(x="theta", c="m", lw=0.5, ax=ax)
phisys_first.sysphi_ray.plot(x="theta", c="r", lw=0.5, ax=ax)
phisys_hist.sysphi_peak_ray.plot(x="theta", c="b", lw=0.5, ax=ax)
phisys_hist.sysphi_first_ray.plot(x="theta", c="b", lw=0.5, ax=ax)
ax.axhline(phisys_block.sysphi, c="k").get_path()._interpolation_steps = 180
ax.axhline(phisys_window.sysphi, c="m").get_path()._interpolation_steps = 180
ax.axhline(phisys_first.sysphi, c="r").get_path()._interpolation_steps = 180
ax.axhline(phisys_hist.sysphi_peak, c="b", ls="--").get_path()._interpolation_steps = 180
ax.axhline(phisys_hist.sysphi_first, c="b", ls=":").get_path()._interpolation_steps = 180
ax.set_ylim(vmin, vmax)
ax.set_title(rf"$\phi_{{DP}}^{{sys}}$")
plt.tight_layout()
```
