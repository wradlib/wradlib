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

# GAMIC HDF5

```{code-cell} python
import wradlib as wrl
import wradlib_data
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
```

GAMIC refers to the commercial [GAMIC Enigma MURAN software](https://www.gamic.com) which exports data in hdf5 format. The concept is quite similar to the [OPERA HDF5 (ODIM_H5)](read_odim#OPERA-HDF5-(ODIM_H5)) format.

```{warning}
For radar data in GAMIC HDF5 format the [openradar community](https://openradarscience.org/) published [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/) where xarray-based readers/writers are implemented. That particular code was ported from $\omega radlib$ to xradar. Please refer to xradar for enhancements for polar radar.

From $\omega radlib$ version 1.19 `GAMIC HDF5` reading code is imported from [xradar](https://github.com/openradar/xradar)-package whenever and wherever necessary.

Please read the more indepth notebook [gamic_backend](../backends/gamic_backend).
```

Such a file (typical ending: *.mvol*) can be read by:

```{code-cell} python
fpath = "hdf5/2014-08-10--182000.ppi.mvol"
f = wradlib_data.DATASETS.fetch(fpath)
data, metadata = wrl.io.read_gamic_hdf5(f)
```

While metadata represents the usual dictionary of metadata, the data variable is a dictionary which might contain several numpy arrays with the keywords of the dictionary indicating different moments.

```{code-cell} python
print(metadata.keys())
print(metadata["VOL"])
print(metadata["SCAN0"].keys())
```

```{code-cell} python
print(data["SCAN0"].keys())
print(data["SCAN0"]["PHIDP"].keys())
print(data["SCAN0"]["PHIDP"]["data"].shape)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 10))
da = wrl.georef.create_xarray_dataarray(
    data["SCAN0"]["ZH"]["data"]
).wrl.georef.georeference()
im = da.wrl.vis.plot(fig=fig, crs="cg")
```