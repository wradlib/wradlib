# wradlib 2.0 migration

## Introduction

For {{wradlib}} 2.0 there have been quite some deprecations and additions. Most of the changes have been announced over the latest wradlib 1 versions. Nevertheless, to make a clean cut it was neccessary to change and remove code and functionality which was not yet being officially deprecated.

The function signatures have been revisited with regard to [PEP570](https://peps.python.org/pep-0570/). Please look out for changes in the parameter-list, there might be changes on parameters being positional only, positional and keyword or keyword-only.

The georeferencing submodule has undergone an alignment process for function parameters. For Coordinate Reference Systems (CRS) `crs`-kwarg is used for geographic CRS as well as projected CRS. If the function needs source and/or target, then `src_crs` and `trg_crs` is used respectively. At the same time the `site`/`sitecoords` have been aligned to `site`.

The early experiments with {{xarray}} reading capabilities (deprecated) have been removed, the more mature {{xarray}} backend code has been ported to {{xradar}}-package from where {{wradlib}} is importing it.

There have also been quite some changes to the visualization code. For several years now, the `plot_ppi`/`plot_rhi`-functions have converted the provided arguments to an
{py:class}`xarray:xarray.DataArray` under the hood. To lessen that code burden these functions are also removed. The user now has to convert the data to {py:class}`xarray:xarray.DataArray`, if not already reading via {{xradar}}. Finally, the unified {py:func}`wradlib.vis.plot`-function manages all plotting.

To bring {{wradlib}}'s integration with {{xarray}} to a new level, many functions have already been bound to either {py:class}`xarray:xarray.DataArray` or {py:class}`xarray:xarray.Dataset` by Accessors as laid out in <inv:xarray:std:doc#internals/extending-xarray>. To remain backwards compatible with users numpy-based workflows the functions have been overloaded using {py:func}`functools.singledispatch`-decorator. Within the overloaded functions the numpy-functions are called via {py:func}`xarray:xarray.apply_ufunc` or the functionality was implemented directly based on xarray. Usage of {{dask}}, and with that lazy-processing, is currently only available on those functions which are already capable. The adaption or rewrite of code can now be done using a unified calling convention for numpy- as well as {{xarray}}-based functions.

In the next sections the additions and deprecations (including breaking changes) are announced on a per module basis.

## adjust - Submodule

No deprecations or additions.

## atten - Submodule

### Additions

Implemented Xarray Accessor `wrl.atten` ({py:class}`wradlib.atten.AttenMethods) with:

- {py:func}`~wradlib.atten.correct_attenuation_constrained`


## classify - Submodule

Merged with code from `clutter`-module.

### Additions

Implemented Xarray Accessor `wrl.classify` ({py:class}`wradlib.classify.ClassifyMethods`) with:

- {py:func}`~wradlib.classify.filter_gabella`
- {py:func}`~wradlib.classify.filter_gabella_a`
- {py:func}`~wradlib.classify.filter_gabella_b`
- {py:func}`~wradlib.classify.histo_cut`
- {py:func}`~wradlib.classify.classify_echo_fuzzy`
- {py:func}`~wradlib.classify.filter_window_distance`

{py:func}`~wradlib.classify.classify_echo_fuzzy` returns probability of meteorological echo instead of boolean clutter mask, use `np.where(prob < thresh, True, False)` to retrieve boolean clutter mask.

## clutter - Submodule

### Deprecations

Functions have been merged with {py:mod}`wradlib.classify`-submodule (see above). Module has been removed.

## comp - Submodule

### Additions

Implemented Xarray Accessor `wrl.comp` ({py:class}`wradlib.comp.CompMethods`) with:

- {py:func}`~wradlib.comp.togrid`
- {py:func}`~wradlib.comp.compose_weighted`

## dp - Submodule

### Additions

Implemented Xarray Accessor `wrl.dp` ({py:class}`wradlib.dp.DpMethods`) with:

- {py:func}`~wradlib.dp.depolarization`
- {py:func}`~wradlib.dp.kdp_from_phidp`
- {py:func}`~wradlib.dp.phidp_kdp_vulpiani`
- {py:func}`~wradlib.dp.texture`
- {py:func}`~wradlib.dp.unfold_phi`
- {py:func}`~wradlib.dp.unfold_phi_vulpiani`

### Deprecations

- `wrl.dp.linear_despeckle` -> {py:func}`wradlib.util.despeckle`
- `wrl.dp.process_raw_phidp_vulpiani` -> {py:func}`wradlib.dp.phidp_kdp_vulpiani`
- `wrl.dp.unfold_phi_naive` -> {py:func}`~wradlib.dp.unfold_phi`


## georef-submodule
### Additions

Implemented Xarray Accessor `wrl.georef` ({py:class}`wradlib.georef.GeorefMethods`) with:

- GeorefMiscMethods
  - {py:func}`~wradlib.georef.misc.bin_altitude`
  - {py:func}`~wradlib.georef.misc.bin_distance`
  - {py:func}`~wradlib.georef.misc.site_distance`

- GeorefPolarMethods
  - {py:func}`~wradlib.georef.polar.georeference`
  - {py:func}`~wradlib.georef.polar.spherical_to_xyz`
  - {py:func}`~wradlib.georef.polar.spherical_to_proj`
  - {py:func}`~wradlib.georef.polar.spherical_to_polyvert`
  - {py:func}`~wradlib.georef.polar.spherical_to_centroids`

- GeorefProjectionMethods
  - {py:func}`~wradlib.georef.projection.get_earth_radius`
  - {py:func}`~wradlib.georef.projection.reproject`

- GeorefRectMethods
  - {py:func}`~wradlib.georef.rect.xyz_to_spherical`

- GeorefSatelliteMethods
  - {py:func}`~wradlib.georef.satellite.correct_parallax`
  - {py:func}`~wradlib.georef.satellite.dist_from_orbit`


### Deprecations

- `wradlib.georef.xarray.georeference_dataset` -> {py:func}`wradlib.georef.polar.georeference`

## io - Submodule

### Deprecations

The xarray based radar readers for polar data have been moved to {{xradar}}-package

- `wradlib.io.ODIMH5`
- `wradlibio.CfRadial`
- `wradlibio.XRadVol`
- `wradlibio.open_odim`
- `wradlibio.XRadSweep`
- `wradlibio.XRadMoment`
- `wradlibio.XRadTimeSeries`
- `wradlibio.XRadVolume`
- `wradlibio.RadarVolume`
- `wradlibio.open_radar_dataset`
- `wradlibio.open_radar_mfdataset`
- `wradlibio.to_netcdf`
- `wradlibio.open_rainbow_dataset`
- `wradlibio.open_rainbow_mfdataset`
- `wradlibio.open_cfradial1_dataset`
- `wradlibio.open_cdradial1_mfdataset`
- `wradlibio.open_cfradial2_dataset`
- `wradlibio.open_cdradial2_mfdataset`
- `wradlibio.open_iris_dataset`
- `wradlibio.open_iris_mfdataset`
- `wradlibio.open_odim_dataset`
- `wradlibio.open_odim_mfdataset`
- `wradlibio.open_gamic_dataset`
- `wradlibio.open_gamic_mfdataset`
- `wradlibio.open_furuno_dataset`
- `wradlibio.open_furuno_mfdataset`
- `wradlibio.CfRadial1BackendEntrypoint`
- `wradlibio.CfRadial2BackendEntrypoint`
- `wradlibio.FurunoBackendEntrypoint`
- `wradlibio.GamicBackendEntrypoint`
- `wradlibio.OdimBackendEntrypoint`
- `wradlibio.RainbowBackendEntrypoint`
- `wradlibio.IrisBackendEntrypoint`
- `wradlib.io.radolan_to_xarray` - {py:func}`wradlib.io.radolan.open_radolan_dataset` or {py:func}`xarray:xarray.open_dataset` with `engine="radolan"`
- `wradlib.io.create_xarray_dataarray` -> {py:func}`wradlib.georef.xarray.create_xarray_dataarray`

### How can I read my data now?

#### Single sweep

```python
swp = xarray.open_dataset(filename, engine=engine, group=group)
```
`engine` would be one `BackendName` defined in {{xradar}}, where currently available are:

- [cfradial1](inv:xradar:std:doc#notebooks/CfRadial1)
- [odim](inv:xradar:std:doc#notebooks/ODIM_H5)
- [gamic](inv:xradar:std:doc#notebooks/GAMIC)
- [rainbow](inv:xradar:std:doc#notebooks/Rainbow)
- [iris](inv:xradar:std:doc#notebooks/Iris)
- [furuno](inv:xradar:std:doc#notebooks/Furuno)

`group` would be a string like `sweep_0` for first sweep, `sweep_1` for second sweep and so forth.

The above command will return an `xarray.Dataset` which is aligned with the CfRadial2/FM301 standard. Please refer to the <inv:xradar:std:doc#model>.

Please also refer to {py:func}`xarray:xarray.open_dataset` documentation.

#### Timeseries of sweeps

```python
ts = xarray.open_mfdataset(
    filelist, concat_dim=time2, engine=engine, group=group, preprocess=preprocess
)
```

`preprocess` is here a function which is applied to each of the retrieved datasets to align them for stacking along the new dimension (`time2`). One use-case would be <inv:xradar:std:doc#notebooks/angle_reindexing>.

Please also refer to {py:func}`xarray:xarray.open_mfdataset` documentation.

#### Single metadata group

The same way different metadata groups can be retrieved. Just require the wanted group with `group`-kwarg.

#### Single Volume

{{wradlib}}'s `RadarVolume` is replaced by {py:class}`datatree:datatree.DataTree`.

```python
vol = xradar.io.open_cfradial1_datatree(filename)
```

Here, as well as above, each backend has its own loading function:

- {py:func}`xradar:xradar.io.backends.cfradial1.open_cfradial1_datatree`
- {py:func}`xradar:xradar.io.backends.odim.open_odim_datatree`
- {py:func}`xradar:xradar.io.backends.gamic.open_gamic_datatree`
- {py:func}`xradar:xradar.io.backends.rainbow.open_rainbow_datatree`
- {py:func}`xradar:xradar.io.backends.iris.open_iris_datatree`
- {py:func}`xradar:xradar.io.backends.furuno.open_furuno_datatree`
- {py:func}`datatree:datatree.open_datatree`

#### Multiple Volumes

This is not yet available out of the box as dedicated functions (like {py:func}`xarray:xarray.open_mfdataset`) but this is worked on at <inv:xradar:std:doc#notebooks/Multi-Volume-Concatenation>.

## ipol - Submodule

### Additions

Implemented Xarray Accessor `wrl.ipol` ({py:class}`wradlib.ipol.IpolMethods`) with:

- {py:func}`~wradlib.ipol.interpolate_polar`

## qual - Submodule
### Additions

Implemented Xarray Accessor `wrl.qual` ({py:class}`wradlib.qual.QualMethods`) with:

- {py:func}`~wradlib.qual.pulse_volume`
- {py:func}`~wradlib.qual.get_bb_ratio`

## trafo - Submodule
### Additions

Implemented Xarray Accessor `wrl.trafo` ({py:class}`wradlib.trafo.TrafoMethods`) with:

- {py:func}`~wradlib.trafo.decibel`
- {py:func}`~wradlib.trafo.idecibel`
- {py:func}`~wradlib.trafo.r_to_depth`

## util - Submodule
### Additions

Implemented Xarray Accessor `wrl.util` ({py:class}`wradlib.util.UtilMethods`) with:

- {py:func}`~wradlib.util.despeckle`
- {py:func}`~wradlib.util.derivate`
- {py:func}`~wradlib.util.dim0`

## verify - Submodule

No deprecations or additions.

## vis - Submodule
### Additions

Implemented Xarray Accessor `wrl.vis` ({py:class}`wradlib.vis.VisMethods`) with:

- {py:func}`~wradlib.vis.plot`
- {py:func}`~wradlib.vis.pcolormesh`
- {py:func}`~wradlib.vis.contour`
- {py:func}`~wradlib.vis.contourf`

### Deprecations

- `plot_ppi`/`plot_rhi` ->  {py:func}`wradlib.georef.xarray.create_xarray_dataarray` and  {py:func}`wradlib.vis.plot` or xarray accessor `da.wrl.vis.plot()`
- removed `saveto`-kwarg from signature of {py:func}`wradlib.vis.plot_plan_and_vert` and {py:func}`wradlib.vis.plot_max_plan_and_vert`

## vpr - Submodule

No deprecations or additions.

## xarray - Submodule
### Additions

New module containing {py:class}`~wradlib.xarray.WradlibXarrayAccessor` implementation.

## zonalstats - Submodule

### Deprecations

- `wradlib.zonalstats.DataSource` -> (py:class}`wradlib.io.vector.VectorSource`

## zr - Submodule

### Additions

Implemented Xarray Accessor `wrl.zr` ({py:class}`wradlib.zr.ZRMethods`) with:

- {py:func}`~wradlib.zr.z_to_r`
- {py:func}`~wradlib.zr.r_to_z`
- {py:func}`~wradlib.zr.z_to_r_enhanced`
