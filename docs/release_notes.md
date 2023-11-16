```{currentmodule} wradlib
```

# Release Notes

Please note that {{wradlib}} releases follow [semantic versioning](https://semver.org/). API breaks will be announced via deprecation warnings. All {{wradlib}} releases come without any warranty. Release notes might be incomplete. See [commit history](https://github.com/wradlib/wradlib/commits/main) for a complete record of changes.

You can install the latest {{wradlib}} release from PyPI via ``$ python -m pip install wradlib`` or specific version via ``$ pip install wradlib==x.y.z``. The recommended installation process is described in {doc}`installation`.

## Version 2.0.3

**Bugfixes**

* FIX: align earth radius in plot_scan_strategy for CG plots ({pull}`655`) by {at}`kmuehlbauer`

## Version 2.0.2

**Maintenance - CI**

* MNT: enable EARTHDATA bearer token for notebook CI runs. By {at}`kmuehlbauer

**Bugfixes**

* FIX: update util.cross_section_ppi to work with wradlib 2.0 (RadarVolume no longer available) ({pull}`650`) by {at}`JulianGiles`
* FIX: disentangle cg/normal plotting for better maintainability and apply explicit colorbar handling for cg plotting ({pull}`652`) by {at}`kmuehlbauer`
* FIX: properly implement bearer token authentication adn function calling convention for plot_scan_strategy with terrain=True ({issue}`651`) by {at}`JulianGiles`, ({pull}`652`) by {at}`kmuehlbauer`



## Version 2.0.1

**Maintenance - CI**

* MNT: add codecov.yml to configure coverage status checks ({pull}`647`) by {at}`kmuehlbauer

**Bugfixes**

* FIX: make pyproj import lazy in georef.polar ({pull}`646`) by {at}`kmuehlbauer`
* FIX: enable radolan backend to work without GDAL/pyproj, by falling back to trigonometric calculations ({issue}`648`, {pull}`649`) by {at}`kmuehlbauer`


## Version 2.0.0

For {{wradlib}} 2.0.0 there have been quite some deprecations and additions. Most of the changes have been announced over the latest  {{wradlib}} 1 versions. Nevertheless, to make a clean cut it was neccessary to change and remove code and functionality which was not yet being officially deprecated.

Please follow up at [wradlib 2.0 migration](./migration.md).

The major new feature is the smooth integration into `xradar`-based reading into `xarray`-based data structures and the convenient use of `xarray`-accessors. The tutorials and examples got a major overhaul to show the capabilities of {{wradlib}} in the light of these new developments.

This version also brings doc-folder back into wradlib repo. The wradlib-docs repo will be archived. wradlib-notebooks repo has been restructured as well, with a single point of access.

**New features**

* ENH: introduce wradlib xarray accessor for easy access of wradlib functions from xarray.DataArray (with dp and zr modules prefilled), adapt tests ({pull}`621`) by {at}`kmuehlbauer`
* DOC: reintroduce doc into wradlib repository ({pull}`619`) by {at}`kmuehlbauer`
* FIX: wrong prediction_time variable in radolan xarray.Dataset ({pull}`639`) by {at}`Hoffmann77`
* wradlib 2.0 preparations ({pull}`635`) by {at}`kmuehlbauer`
* final wradlib 2.0.0-rc.1 preparations ({pull}`636`) by {at}`kmuehlbauer`
* wradlib docstring updates ({pull}`638`) by {at}`kmuehlbauer`
* wradlib 2.0 RC ({pull}`640`) by {at}`kmuehlbauer`

**Maintenance - CI**

* Fixing wradlib version for ReadTheDocs ({pull}`620`) by {at}`kmuehlbauer`
* use setup-micromamba action ({pull}`627`) by {at}`kmuehlbauer`

**Maintenance - Code**

* Remove deprecated code ({pull}`621`) by {at}`kmuehlbauer`

**Bugfixes**

* fix RADOLAN xarray 'prediction_time' for RADVOR datasets ({pull}`639`) by {at}`Hoffmann77`


## Version 1.19.0

This version is most likely the last version before wradlib 2.0. We've switched to use [xradar](https://docs.openradarscience.org/projects/xradar) for reading radar data in polar coordinates. The relevant code has been ported from wradlib to xradar and only a shallow compatibility layer is kept in wradlib.

**New features**

* Enhance Furuno Reader to read WR110 data ({pull}`606`) by {at}`kmuehlbauer`
* New function for cross sections (RHI) from radar volume ({issue}`439`) by {at}`pandasambit15`  and ({pull}`610`) by {at}`JulianGiles`

**Maintenance code**

* Adapt codebase to use xradar, move/remove duplicate code ({pull}`617`) by {at}`kmuehlbauer`

**Maintenance - CI**

* several updates and fixes to the CI ({pull}`604`), ({pull}`613`), ({pull}`616`), ({pull}`618`) by {at}`kmuehlbauer`

**Bugfixes**

* delete tmp-files for VectorSource after closing ({issue}`608`) and ({pull}`609`) by {at}`plvoit`

## Version 1.18.0

**New features**

* Histo cut enhancement ({issue}`602`) by {at}`overeem11`, ({pull}`603`) and ({pull}`605`) by {at}`kmuehlbauer`

**Maintenance code**

* xradar compatibility preparations ({pull}`599`) and ({pull}`604`) by {at}`kmuehlbauer`

**Maintenance - CI**

* MNT: update CI actions, python versions ({pull}`604`) by {at}`kmuehlbauer`

**Bugfixes**

* Changes in # read metadata under get_radiosonde() ({issue}`596`) and ({pull}`597`) by {at}`JulianGiles`
* FIX: cfradial2 coordinates ({pull}`600`) by {at}`kmuehlbauer`

## Version 1.17.0


**New features**

* MNT: use Bearer Token instead of credentials ({pull}`584`) by {at}`kmuehlbauer`
* FIX: enable ODIM reader to read `qualityN` fields (similar to `dataN`) ({pull}`588`) by {at}`kmuehlbauer`
* ENH: read RADOLAN ascii format ({issue}`593`) by {at}`SandeepAllampalli` and ({pull}`594`) by {at}`kmuehlbauer`
* ENH: add RADVOR products RV, RE and RQ ({issue}`591`) by {at}`heistermann` and ({pull}`594`) by {at}`heistermann` and {at}`kmuehlbauer`

**Maintenance code**

* MNT: add pre-commit ({pull}`577`) by {at}`kmuehlbauer`
* Pre Erad2022 ({pull}`580`) by {at}`kmuehlbauer`
* MNT: fix SRTM testing if resource is not available, implement timeout ({issue}`586`) and ({pull}`587`) by {at}`kmuehlbauer`

**Bugfixes**

* Use numpy.linspace in ipol.interpolate_polar ({pull}`576`) by {at}`syedhamidali`
* FIX: explicitely cast ray indices to int in cfradial1 reader ({pull}`579`) by {at}`kmuehlbauer`
* FIX: add missing finally ({pull}`581`) by {at}`kmuehlbauer`
* FIX: cfradial1 reader alignments ({pull}`585`) by {at}`kmuehlbauer`
* FIX: use 'None' instead of ambiguous 'False' (`0`) for comparison ({pull}`595`) by {at}`kmuehlbauer`

## Version 1.16.0


**New features**

* add "ancillary"-keyword to io.radolan._radolan_file ({pull}`574`) by {at}`kmuehlbauer`
* update DWD grids ({pull}`568`) by {at}`kmuehlbauer`
* add open_gpm_dataset ({pull}`571`) by {at}`kmuehlbauer`

**Maintenance - Code**

* update docstring for classify_echo_fuzzy ({pull}`570`) by {at}`swatelet`
* use np.expand_dims instead of np.newaxis to make functions work with xarray DataArray ({pull}`568`) by {at}`kmuehlbauer`

**Maintenance - CI**

* add nc-time-axis to notebook-environment ({pull}`568`) by {at}`kmuehlbauer`

**Bugfixes**

* fix RADOLAN xarray coordinates (which have been off by 0.5km) ({pull}`568`) by {at}`kmuehlbauer`

**Deprecations**

* removes GDAL]( 3 compatibility code ({pull}`568`) by {at}`kmuehlbauer`

## Version 1.15.0


**New features**

* add Furuno backend (``scn`` and ``SCNX`` files) for ``Xarray`` ({pull}`567`) by {at}`kmuehlbauer`

## Version 1.14.0


**New features**

* zonalstats enhancements, new VectorSource class, geopandas connector and more ({pull}`558`) by {at}`kmuehlbauer`

**Maintenance - Code**

* refactor deprecated xarray functionality  ({pull}`533`) by {at}`kmuehlbauer`
* use f-strings where appropriate ({pull}`537`) by {at}`kmuehlbauer`
* remove unnecessary object-inheritance ({pull}`538`) by {at}`kmuehlbauer`
* replace distutils.version.LooseVersion with packaging.version.Version ({pull}`539`) by {at}`kmuehlbauer`
* use dict-literals ({pull}`542`) by {at}`kmuehlbauer`

**Maintenance - Build/CI**

* cancel previous CV builds ({pull}`534`) by {at}`kmuehlbauer`
* use provision-with-micromamba action ({pull}`543`) by {at}`kmuehlbauer`

**Bugfixes**

* remove zero padding of bits in rainbow format (truncate excess bits from flagmap) ({issue}`549`) ({pull}`550`) by {at}`binomaiheu`
* raise ValueError if projection cannot be determined from source dataset ({issue}`551`) ({pull}`553`) {at}`kmuehlbauer`
* output full timeslice when calling to_netcdf with no timestep ({issue}`552`) ({pull}`554`) {at}`kmuehlbauer`
* handle variable number of gates in CfRadial1 backend ({issue}`545`) ({pull}`555`) {at}`kmuehlbauer`
* use radar site altitude in bin_altitude calculation ({issue}`546`) ({pull}`556`) {at}`kmuehlbauer`
* take precision into account for RADOLAN WN product ({issue}`548`) ({pull}`557`) {at}`kmuehlbauer`
* correct elevation for negative angles in iris/sigmet RAW data ({issue}`560`) (reported by Ozan Pirbudak from Turkish Met Service)  ({pull}`561`) {at}`kmuehlbauer`
* fix AttributeError: 'str' object has no attribute 'item' ({issue}`562`) ({pull}`561`) {at}`kmuehlbauer`
* use start date/time if end date/time is missing for ODIM reader ({issue}`563`) (reported by Anna Gabbert from University of Graz) ({pull}`564`) {at}`kmuehlbauer`

## Version 1.13.0


**New features**

* add IRIS/Sigmet backend for ``Xarray`` ({issue}`361`) ({pull}`520`) by {at}`kmuehlbauer`
* add Rainbow backend for ``Xarray`` ({issue}`394`, {issue}`459`) suggested by {at}`wcwoo` and {at}`maxok` ({pull}`522`) by {at}`kmuehlbauer`

**Maintenance**

* optionalize dependencies (dask, gdal, h5netcdf, h5py, netCDF4, requests, xmltodict) ({pull}`531`) by {at}`kmuehlbauer`
* utilize pytest-doctestplus ({pull}`530`) by {at}`kmuehlbauer`
* update deprecated matplotlib functionality ({pull}`530`) by {at}`kmuehlbauer`
* docstring updates in several functions ({pull}`530`, ) by {at}`kmuehlbauer`
* docstring updates in several functions

**Bugfixes**

* use reasonable default values in `io.xarray.to_odim` (gain, offset, nodata, undetect, fillvalue)
* add cf attributes when reading GAMIC files ({pull}`523`) by {at}`kmuehlbauer`
* fix regression in legacy GAMIC reader ({pull}`523`) by {at}`kmuehlbauer`
* catch `dt.accessor` TypeError ({pull}`529`)  by {at}`kmuehlbauer`
* fix thread-lock issue, if dask is not installed ({pull}`531`) by {at}`kmuehlbauer`
* use int instead np.int in radolan header parser ({pull}`531`) by {at}`kmuehlbauer`
* fix several tests ({pull}`531`) by {at}`kmuehlbauer`
* other minor fixes distributed over several PR's

## Version 1.12.0


* withdrawn, please use 1.13.0.

## Version 1.11.0


**New features**

* add support for RADOLAN HG product ({pull}`495`) by {at}`v4lli`
* add %M, %J and %Y RADOLAN products ({issue}`504`) ({pull}`517`) by {at}`kmuehlbauer`

**Maintenance**

* rename master -> main
* fix docstrings (links, types, minor issues) ({pull}`518`) by {at}`kmuehlbauer`
* add .git-blame-ignore-revs ({pull}`519`) by {at}`kmuehlbauer`

**Bugfixes**

* minor fixes in GAMIC and CfRadial readers ({pull}`492`) by {at}`kmuehlbauer`
* use default values for ODIM/OPERA what-group ({pull}`496`) by {at}`kmuehlbauer`
* do not restrict variables, but read all variables for Cf/Radial1 data ({pull}`497`) by {at}`kmuehlbauer`
* correct calculation of angle resolution in ODIM/GAMIC xarray readers reported by {at}`TiemoMathijssen` ({pull}`501`) by {at}`kmuehlbauer`
* add mode-kwarg to radolan coordinates/grid functions ({issue}`499`) reported by {at}`gogglesguy` ({pull}`502`) by {at}`kmuehlbauer`
* add kwarg origin and FutureWarning to IRIS CartesianImage reader ({issue}`493`) ({pull}`503`) by {at}`kmuehlbauer`
* remove unnecessary gridshape kwarg from docstring in CartesianVolume ({issue}`444`) by {at}`fuxb` ({pull}`505`) by {at}`kmuehlbauer`
* correctly handle single/multiple elevations in wradlib.vis.plot_scan_strategy ({pull}`507`) by {at}`kmuehlbauer`
* fix ODIM xarray reader issues ({issue}`513`), ({issue}`514`) ({pull}`515`) by {at}`kmuehlbauer`
* mention dask in all open_*_mfdataset functions ({issue}`510`) by {at}`Guruprasadhegde` ({pull}`516`) by {at}`kmuehlbauer`

## Version 1.10.0

**New features**

* add ODIM/GAMIC/CfRadial backends for ``Xarray`` ({pull}`487`) by {at}`kmuehlbauer`
* add RADOLAN backend for ``Xarray`` ({pull}`480`) by {at}`kmuehlbauer`
* decode IRIS ``DB_XHDR`` as numpy structured array ({issue}`362`) ({pull}`488`) by {at}`kmuehlbauer`

**Maintenance**

* move CI to GitHub Actions ({pull}`477`) by {at}`kmuehlbauer`
* create/use earthdata credentials for srtm data ({pull}`481`) by {at}`kmuehlbauer`
* address numpy DeprecationWarnings ({pull}`479`) by {at}`kmuehlbauer`

**Bugfixes**

* fix _FillValue and GAMIC dynamic range ({pull}`486`) by {at}`kmuehlbauer`
* fix doctest example in vpr-module ({pull}`478`) by {at}`kmuehlbauer`
* fix handle kwarg change scipy.cKDTree ({pull}`474`) by {at}`kmuehlbauer`

## Version 1.9.0

**New features**

* make wradlib.io capable of consuming file-like objects ({issue}`448`) ({pull}`469`) by {at}`kmuehlbauer`
* read truncated RADOLAN composites (reported by {at}`franzmueller`) ({pull}`471`) by {at}`franzmueller` and {at}`kmuehlbauer`

**Maintenance**

* use micromamba on CI to save CI time ({issue}`457`) ({pull}`452`, {pull}`464`, {pull}`465`, {pull}`470`) by {at}`kmuehlbauer`
* add Python 3.9 builds to all CI ({pull}`463`) by {at}`kmuehlbauer`
* adapt to new tsring handling in h5py >= 3.0 ({pull}`468`) by {at}`kmuehlbauer`

**Bugfixes**

* add capability to decode old DX header ({issue}`455`) reported by {at}`GolGosh` ({pull}`467`) by {at}`kmuehlbauer`
* simplify dimension angle handling ODIM/GAMIC ({pull}`462`) by {at}`kmuehlbauer`

## Version 1.8.0

**New features**

* add WN product size (1200,1000) to radolan grid, add test for correct reference point (lower left) ({issue}`448`) reported by {at}`julste` ({pull}`449`) by {at}`kmuehlbauer`
* add `WN` and `YW` products to radolan to xarray converter ({pull}`450`) by {at}`kmuehlbauer`

**Maintenance**

* remove deprecated and unused code and handle upstream deprecations ({pull}`450`) by {at}`kmuehlbauer`

**Bugfixes**

* fix srtm downloads windows path issues and region selection ({pull}`445`) by {at}`kmuehlbauer`
* make `georeference_dataset` work with ND datasets ({pull}`446`) by {at}`kmuehlbauer`

## Version 1.7.4

**Bugfixes**

* update `vis.plot_scan_strategy()` ({issue}`441`) originally reported at [wradlib-users group](https://groups.google.com/g/wradlib-users/c/Vud23QpQtmo/m/ni-e_biVBAAJ) by {at}`pandasambit15` ({pull}`442`) by {at}`jorahu` and {at}`kmuehlbauer`
* add switch to keep elevation data unaltered (DWD terrain following scan) ({issue}`437`, {pull}`443`) by {at}`kmuehlbauer`

## Version 1.7.3

**Bugfixes**

* always translate ODIM attributes to CF attributes ({issue}`373`, {pull}`438`) by {at}`kmuehlbauer`
* specify keys (sweep_groups) which should be saved using to_netcdf ({pull}`440`) by {at}`kmuehlbauer`

**Maintenance**

* pin isort  ({pull}`438`) by {at}`kmuehlbauer`

## Version 1.7.2

**Bugfixes**

* rework ODIM RHI elevation angle retrieval ({pull}`435`) by {at}`kmuehlbauer`

**Maintenance**

* use pytest for testing, implement "@require_data" to be able to run tests in case of missing wradlib-data ({pull}`434`) by {at}`kmuehlbauer`
* enhance azure ci workflow by adding flake8 linter and uploading coverage ({pull}`436`) by {at}`kmuehlbauer`
* minor changes to README.md

## Version 1.7.1

**Maintenance**

* add azure CI tests
* code formatting according to black/isort/flake8, add setup.cfg
* add show_versions
* use new semver parse
* add github templates
* all above done in ({pull}`432`) by {at}`kmuehlbauer`

## Version 1.7.0

**Highlights**

* implement generalized :py:func:`util.derivate` function with improved NaN-handling ({pull}`419`, {pull}`423`) by {at}`kmuehlbauer`
* complete rework of phidp/kdp estimation code (Vulpiani) including new keyword-parameters, handling of ndimensional arrays,
  using ``scipy.integrate.cumtrapz`` instead of ``np.cumsum`` ({pull}`412`, {pull}`422`) by {at}`kmuehlbauer`
* new interpolators on regular grids ({pull}`390`, {pull}`429`, {pull}`430`) by {at}`egouden` and {at}`kmuehlbauer`

**New features**

* reimplement `dp.linear_despeckle` as :py:func:`util.despeckle` ({pull}`420`) by {at}`kmuehlbauer`
* read RHI in ODIM reader ({pull}`424`) by {at}`kmuehlbauer`
* new :py:func:`.get_earth_projection` and :py:func:`.get_radar_projection` functions ({pull}`379`) by {at}`egouden`
* new convenience functions :py:func:`.set_raster_indexing` and :py:func:`.set_coordinate_indexing` ({pull}`429`) by {at}`kmuehlbauer`
* implement rainrate decoder to iris reader ({pull}`428`) by {at}`tsmsalper`

**Bugfixes**

* correct padding and nan-filling for multidimensional arrays in ``dp.texture`` ({pull}`418`) by {at}`kmuehlbauer`
* introduce ``call_kwargs`` in :py:func:`comp.togrid` ({issue}`373`) reported by {at}`jorahu`  ({pull}`425`) by {at}`kmuehlbauer`

## Version 1.6.2

* re-add removed IRIS features ({issue}`415`, {pull}`416`) by {at}`kmuehlbauer`

## Version 1.6.1

* use LooseVersion to check for dependency matching ({issue}`413`, {pull}`414`) by {at}`kmuehlbauer`

## Version 1.6.0

**Highlights**

* improvements of georef.raster module ({pull}`376`) by {at}`egouden`
* implement multi-file ODIMH5-reader/writer ({pull}`397`, {pull}`409` and {pull}`410`) by {at}`kmuehlbauer` and {at}`egouden`
* simplify `zr`-module, add handling of multidimensional arrays ({pull}`408`) by {at}`kmuehlbauer`
* use __all__ in submodules (georef, io) to specify exported/documented functions ({issue}`386`, {pull}`388`) by {at}`kmuehlbauer`

**New features**

* add STATUS product to Iris/Sigmet reader ({pull}`378`) by {at}`kmuehlbauer`
* improvements of georef.raster module ({pull}`376`) by {at}`egouden`
* add PRF and NSAMPLES to ODIM reader ({pull}`393`) by {at}`kmuehlbauer`
* refactor code into `assign_root`-function ({pull}`393`) by {at}`egouden`
* add ODIM WRAD moment ({pull}`406`) by {at}`kmuehlbauer`

**Bugfixes**

* apply correct decoding of VEL, WIDTH and KDP IrisCartesianProductFile ({pull}`378`) by {at}`kmuehlbauer`
* add missing `requests` dependency to CI ({pull}`374`) by {at}`s-m-e`
* correct error in documentation of sweep_centroids ({pull}`383`) by {at}`ElmerJeanpierreLopez`
* adapt `georef.polar.sweep_centroids` to only use angles in degrees ({pull}`385`) by {at}`kmuehlbauer`
* work around issue, where ODIM `startime` == `endtime` ({pull}`391`) by {at}`kmuehlbauer`
* improve handling of equal sized dimensions ({pull}`393`) by {at}`kmuehlbauer`
* use xarray `Dataset.drop_vars` instead of deprecated `Dataset.drop` ({pull}`398`) by {at}`kmuehlbauer`
* use xarray.Dataset.rename instead of rename_dims ({pull}`402`) by {at}`kmuehlbauer`
* add missing `+`-sign in projection string ({pull}`405`) by {at}`kmuehlbauer`
* fix filter_cloudtype (low cloud switch removes everything) ({pull}`401`) by {at}`egouden`
* use Dataset.swap_dims instead of rename ({pull}`406`) by {at}`kmuehlbauer`

## Version 1.5.0

**Highlights**

* consolidation of xarray based functionality, bugfixing, feature adding
* speedup zonal statistics by using `/vsimem`, by creation of spatial and attribute index files as well as by faster reading of attributes and properties

**New features**

* make OdimH5 reader accept list of files and merge into one structure
* make `chunks` available for transparently use dask with OdimH5 and CfRadial readers
* make gdal3 compatible (added by {at}`egouden`)
* use `loaddata='xarray'` kwargs to output data as Xarray Dataset in `read_radolan_composite`
* CI: add Appveyor to run test-suite under Windows OS

**Bugfixes**

* use `importlib` in `import_optional`, correct multidimensional calling in `gradient_along_axis`
* several fixes for OdimH5 and Cf/Radial readers/writers
* set destination projection to destination dataset in `reproject_raster_dataset` (spotted by [wradlib-forum](https://groups.google.com/forum/#!msg/wradlib-users/-dvRhDCjgV0/X0JR4yL3BgAJ))

## Version 1.4.0

**Highlights**

* read sigmet/iris ingest files, redesign of sigmet reader (suggested by {at}`aschueth`)
* enhance/rewrite fuzzy echo classifier (implemented with {at}`overeem11`)

**New features**

* parametrize xarray based OdimH5-reader (suggested by {at}`egouden`)
* add depolarization ratio calculation (implemented by {at}`overeem11`)
* add script for test profiling (added by {at}`egouden`)

**Bugfixes**

* remove unnecessary seek in radolan-reader (suggested by {at}`PPazderski`)
* correct handling of edge cases in `dp.texture` processing (spotted by {at}`overeem11`)
* correct decoding of DB_FLIQUID2 (sigmet-reader) (implemented by {at}`ckaradavut`)
* correct handling of non-precip in 2D hmc (spotted by and fixed with {at}`bzohidov`)
* fix semver handling and install process (suggested by {at}`s-m-e`)
* fix import for MutableMapping (added by {at}`zssherman`)

## Version 1.3.0

**Highlights**

* wradlib is considered Python3 only working with Python >= 3.6
* xarray-powered reader/writer for Cf/Radial 1.X and 2.0 as well as ODIM_H5
* xarray-powered plotting using DataArray via xarray-DataArray Accessor

**New features**

* creation of xarray DataArray from spherical coordinates and radar data
* update test machinery to use pytest (mainly CI use)
* correctly apply `semver`

**Bugfixes**

* beamblockage calculation, precisely detect clear or blocked beam
* catch HTTPError in `test_radiosonde`, graceful skip test
* `spherical_to_xyz` better aligns with input dimensions

## Version 1.2.0

**Highlights**

* significantly speed up functions using interpolation classes
* add `classify` module including 2d membershipfunctions hydrometeor classification
* fix conformance, correctness and consistency issues in wradlib-docs (thanks {at}`CAM-Gerlach`)

**New features**

* add new header token `VR` and `U` to radolan header parser
* add `load_vector`-method to `zonaldata.DataSource`
* enable `zonaldata.ZonaldataBase` to take `DataSource` objects as parameters
* add `get_radiosonde` to `io.misc` to retrieve radiosonde data from internet
* add `minalt` keyword argument to `vpr.make_3d_grid`

**Bugfixes**

* update links, fix typos, improve CI workflow
* fix bug in all adjustment classes when checking for None
* show angle axis curvelinear grid again
* align docstring with actual code and use `sweep` in iris-reader

## Version 1.1.0

**Highlights**

* use with-statement in rainbow-reader
* fix in gpm-reader and rainbow_reader
* fix issues with cg-plot in vis-module
* fix in gdal/ogr exception handling
* update in versioning/release procedure
* automatic build of devel-docs

## Version 1.0.0

**Highlights**

* export ``notebooks`` into dedicated [wradlib-notebooks](https://github.com/wradlib/wradlib-notebooks)
* export ``doc`` into dedicated [wradlib-docs](https://github.com/wradlib/wradlib-docs)
* complete rewrite of CI-integration
* complete rework of modules

## Pre 1.0.0 Versions

Versions before 1.0.0 are available from the [wradlib-old](https://github.com/wradlib/wradlib-old) repository.
