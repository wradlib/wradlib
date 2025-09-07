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

# Reading HDF5 files with a generic reader

This reader utilizes [h5py](https://docs.h5py.org).

In this example, we read HDF5 files from different sources using a generic reader from $\omega radlib's$ io module.

```{code-cell} python
import wradlib as wrl
import wradlib_data
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

This is a generic hdf5 reader, which will read any hdf5 structure.

```{code-cell} python
fpath = "hdf5/2014-08-10--182000.ppi.mvol"
f = wradlib_data.DATASETS.fetch(fpath)
fcontent = wrl.io.read_generic_hdf5(f)
```

```{code-cell} python
print(fcontent.keys())
```

```{code-cell} python
print(fcontent["where"])
print(fcontent["how"])
print(fcontent["scan0/moment_3"].keys())
print(fcontent["scan0/moment_3"]["attrs"])
print(fcontent["scan0/moment_3"]["data"].shape)
```

```{code-cell} python
fig = plt.figure(figsize=(10, 10))
da = wrl.georef.create_xarray_dataarray(
    fcontent["scan0/moment_3"]["data"]
).wrl.georef.georeference()
im = da.wrl.vis.plot(fig=fig, crs="cg")
```