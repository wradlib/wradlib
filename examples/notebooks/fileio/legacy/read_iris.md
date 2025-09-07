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

# Vaisala Sigmet IRIS

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

[IRIS](https://www.vaisala.com/en/products/iris-focus-remote-sensing-software) refers to the commercial Vaisala Sigmet **I**nteractive **R**adar **I**nformation **S**ystem. The Vaisala Sigmet Digital Receivers export data in a [well documented](ftp://ftp.sigmet.com/outgoing/manuals/IRIS_Programmers_Manual.pdf) binary format.

The philosophy behind the $\omega radlib$ interface to the IRIS data model is very straightforward: $\omega radlib$ simply translates the complete binary file structure to *one* dictionary and returns this dictionary to the user. Thus, the potential complexity of the stored data is kept and it is left to the user how to proceed with this data. The keys of the output dictionary are strings that correspond to the Sigmet Data Structures.


```{warning}
For radar data in IRIS/Sigmet format the [openradar community](https://openradarscience.org/) published [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/) where xarray-based readers/writers are implemented. That particular code was ported from $\omega radlib$ to xradar. Please refer to xradar for enhancements for polar radar.

From $\omega radlib$ version 1.19 `IRIS/Sigmet` reading code is imported from [xradar](https://github.com/openradar/xradar)-package whenever and wherever necessary.

Please read the more indepth notebook [iris_backend](../backends/iris_backend).
```

Such a file (typical ending: *.RAWXXXX) can be read by:

```{code-cell} python
fpath = "sigmet/cor-main131125105503.RAW2049"
f = wradlib_data.DATASETS.fetch(fpath)
fcontent = wrl.io.read_iris(f)
```

```{code-cell} python
# which keywords can be used to access the content?
print(fcontent.keys())
# print the entire content including values of data and
# metadata of the first sweep
# (numpy arrays will not be entirely printed)
print(fcontent["data"][1].keys())
print()
print(fcontent["data"][1]["ingest_data_hdrs"].keys())
print(fcontent["data"][1]["ingest_data_hdrs"]["DB_DBZ"])
print()
print(fcontent["data"][1]["sweep_data"].keys())
print(fcontent["data"][1]["sweep_data"]["DB_DBZ"])
```

```{code-cell} python
fig = plt.figure(figsize=(10, 10))
swp = fcontent["data"][1]["sweep_data"]
da = wrl.georef.create_xarray_dataarray(
    swp["DB_DBZ"],
).wrl.georef.georeference()
im = da.wrl.vis.plot(fig=fig, crs="cg")
```
