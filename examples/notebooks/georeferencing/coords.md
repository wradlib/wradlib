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

# Computing cartesian and geographical coordinates for polar data

```{code-cell} python
import numpy as np
import wradlib as wrl
import wradlib_data
import xradar as xd
import warnings

warnings.filterwarnings("ignore")
```

## Read the data

Here, we use an OPERA hdf5 dataset.

```{code-cell} python
filename = "hdf5/20130429043000.rad.bewid.pvol.dbzh.scan1.hdf"
filename = wradlib_data.DATASETS.fetch(filename)
pvol = xd.io.open_odim_datatree(filename)
display(pvol)
```

## Retrieve azimuthal equidistant coordinates and projection

```{code-cell} python
for key in list(pvol.children):
    if "sweep" in key:
        pvol[key].ds = pvol[key].ds.wrl.georef.georeference()
```

```{code-cell} python
pvol["sweep_0"].ds.DBZH.plot(x="x", y="y")
```

## Retrieve geographic coordinates (longitude and latitude)


### Using crs-keyword argument.

```{code-cell} python
for key in list(pvol.children):
    if "sweep" in key:
        pvol[key].ds = pvol[key].ds.wrl.georef.georeference(
            crs=wrl.georef.get_default_projection()
        )
```

```{code-cell} python
ds1 = pvol["sweep_0"].ds.wrl.georef.georeference(
    crs=wrl.georef.get_default_projection()
)
ds1.DBZH.plot(x="x", y="y")
```

### Using reproject

```{code-cell} python
ds2 = pvol["sweep_0"].ds.wrl.georef.reproject(
    trg_crs=wrl.georef.epsg_to_osr(32632),
)
ds2.DBZH.plot(x="x", y="y")
```
