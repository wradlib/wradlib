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

# Reading NetCDF

This reader utilizes [netCDF4-python](https://unidata.github.io/netcdf4-python/).

In this example, we read NetCDF files from different sources using a generic reader from $\omega radlib's$ io module.

```{code-cell} python
import wradlib as wrl
import wradlib_data
from wradlib.io import read_generic_netcdf
from wradlib.util import get_wradlib_data_file
import os
```

## NetCDF Format

The NetCDF format also claims to be self-describing. However, as for all such formats, the developers of netCDF also admit that "[...] the mere use of netCDF is not sufficient to make data self-describing and meaningful to both humans and machines [...]" (see [here](https://www.unidata.ucar.edu/software/netcdf/conventions.html)). Different radar operators or data distributors will use different naming conventions and data hierarchies (i.e. "data models") that the reading program might need to know about.

$\omega radlib$ provides two solutions to address this challenge. The first one ignores the concept of data models and just pulls all data and metadata from a NetCDF file ({func}`wradlib.io.read_generic_netcdf`). The second is designed for a specific data model used by the EDGE software ({func}`wradlib.io.read_edge_netcdf`).

```{warning}
For radar data in CfRadial1 or CfRadial2 format the [openradar community](https://openradarscience.org/) published [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/) where xarray-based readers/writers are implemented. That particular code was ported from $\omega radlib$ to xradar. Please refer to xradar for enhancements for polar radar.

From $\omega radlib$ version 1.19 `CfRadial` reading code is imported from [xradar](https://github.com/openradar/xradar)-package whenever and wherever necessary.

Please read the more indepth notebooks [cfradial1_backend](../backends/cfradial1_backend) and [cfradial2_backend](../backends/cfradial2_backend).
```

## Generic NetCDF reader (includes CfRadial)

$\omega radlib$ provides a function that will virtually read any NetCDF file irrespective of the data model: {func}`wradlib.io.read_generic_netcdf`. It is built upon Python's [netcdf4](https://unidata.github.io/netcdf4-python/) library. {func}`wradlib.io.read_generic_netcdf` will return only one object, a dictionary, that contains all the contents of the NetCDF file corresponding to the original file structure. This includes all the metadata, as well as the so called "dimensions" (describing the dimensions of the actual data arrays) and the "variables" which will contains the actual data. Users can use this dictionary at will in order to query data and metadata; however, they should make sure to consider the documentation of the corresponding data model. {func}`wradlib.io.read_generic_netcdf` has been shown to work with a lot of different data models, most notably **CfRadial** (see [here](https://ncar.github.io/CfRadial/) for details). A typical call to {func}`wradlib.io.read_generic_netcdf` would look like:

```{code-cell} python
fpath = "netcdf/example_cfradial_ppi.nc"
f = wradlib_data.DATASETS.fetch(fpath)
outdict = wrl.io.read_generic_netcdf(f)
for key in outdict.keys():
    print(key)
```

# EDGE NetCDF

EDGE is a commercial software for radar control and data analysis provided by the Enterprise Electronics Corporation. It allows for netCDF data export. The resulting files can be read by {func}`wradlib.io.read_generic_netcdf`, but $\omega radlib$ also provides a specific function, {func}`wradlib.io.read_edge_netcdf` to return metadata and data as separate objects:

```{code-cell} python
fpath = "netcdf/edge_netcdf.nc"
f = wradlib_data.DATASETS.fetch(fpath)
data, metadata = wrl.io.read_edge_netcdf(f)
print(data.shape)
print(metadata.keys())
```

```{code-cell} python
# A little helper function for repeated tasks
def read_and_overview(filename):
    """Read NetCDF using read_generic_netcdf and print upper level dictionary keys"""
    test = read_generic_netcdf(filename)
    print("\nPrint keys for file %s" % os.path.basename(filename))
    for key in test.keys():
        print("\t%s" % key)
```

## CfRadial example from S-Pol research radar TIMREX campaign

See also: https://www.eol.ucar.edu/field_projects/timrex

```{code-cell} python
filename = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
filename = get_wradlib_data_file(filename)
read_and_overview(filename)
```

## Example PPI from Py-ART repository

See also: https://github.com/ARM-DOE/pyart/

```{code-cell} python
filename = "netcdf/example_cfradial_ppi.nc"
filename = get_wradlib_data_file(filename)
read_and_overview(filename)
```

## Example RHI from Py-ART repository
See also: https://github.com/ARM-DOE/pyart/

```{code-cell} python
filename = "netcdf/example_cfradial_rhi.nc"
filename = get_wradlib_data_file(filename)
read_and_overview(filename)
```

## Example EDGE NetCDF export format

```{code-cell} python
filename = "netcdf/edge_netcdf.nc"
filename = get_wradlib_data_file(filename)
read_and_overview(filename)
```
