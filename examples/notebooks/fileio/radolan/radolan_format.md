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

# RADOLAN data formats


## RADOLAN binary data format


The RADOLAN binary data file format is described in the RADOLAN Kompositformat. The radolan composite files consists of an ascii header containing all needed information to decode the following binary data block. $\omega radlib$ provides {func}`wradlib.io.read_radolan_composite` to read the data.

The function {func}`wradlib.io.parse_dwd_composite_header` takes care of correctly decoding the ascii header. All available header information is transferred into the metadata dictionary.

```{code-cell} python
import wradlib as wrl
import wradlib_data
import matplotlib.pyplot as plt
import warnings
import io
import tarfile

warnings.filterwarnings("ignore")
try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

```{code-cell} python
# load radolan files
rw_filename = wradlib_data.DATASETS.fetch(
    "radolan/misc/raa01-rw_10000-1408102050-dwd---bin.gz"
)
filehandle = wrl.io.get_radolan_filehandle(rw_filename)
header = wrl.io.read_radolan_header(filehandle)
print(header)
```

```{code-cell} python
attrs = wrl.io.parse_dwd_composite_header(header)
print(attrs)
```

In the following example, the header information of four different composites is extracted.

```{code-cell} python
# load radolan file
filename = "radolan/showcase/raa01-rx_10000-1408102050-dwd---bin.gz"
rx_filename = wradlib_data.DATASETS.fetch(filename)
filename = "radolan/showcase/raa01-ex_10000-1408102050-dwd---bin.gz"
ex_filename = wradlib_data.DATASETS.fetch(filename)
filename = "radolan/showcase/raa01-rw_10000-1408102050-dwd---bin.gz"
rw_filename = wradlib_data.DATASETS.fetch(filename)
filename = "radolan/showcase/raa01-sf_10000-1408102050-dwd---bin.gz"
sf_filename = wradlib_data.DATASETS.fetch(filename)

rxdata, rxattrs = wrl.io.read_radolan_composite(rx_filename)
exdata, exattrs = wrl.io.read_radolan_composite(ex_filename)
rwdata, rwattrs = wrl.io.read_radolan_composite(rw_filename)
sfdata, sfattrs = wrl.io.read_radolan_composite(sf_filename)

# print the available attributes
print("RX Attributes:")
for key, value in rxattrs.items():
    print(key + ":", value)
print("----------------------------------------------------------------")
# print the available attributes
print("EX Attributes:")
for key, value in exattrs.items():
    print(key + ":", value)
print("----------------------------------------------------------------")

# print the available attributes
print("RW Attributes:")
for key, value in rwattrs.items():
    print(key + ":", value)
print("----------------------------------------------------------------")

# print the available attributes
print("SF Attributes:")
for key, value in sfattrs.items():
    print(key + ":", value)
print("----------------------------------------------------------------")
```

## RADOLAN ASCII data format


The ASCII GIS Format is prepended by a limited header and has two flavours as follows:

- RADOLAN / reproc (RADARKLIMATOLOGIE) 2001 – 2019

```
    ncols 900
    nrows 1100
    xllcorner -443462
    yllcorner -4758645
    cellsize 1000
    nodata_value -9999.0
```
    Units: 1.0 mm

- RADOLAN / recent, 2020 – jetzt :
```
    ncols 900
    nrows 900
    xllcorner -523462y
    llcorner -4658645
    cellsize 1000
    NODATA_value -1
```
    Units: 0.1 mm

Product and Datetime need to be extracted from the filename, so extra care has to be taken to not tamper with the filenames.

```{code-cell} python
fname = wradlib_data.DATASETS.fetch("radolan/asc/RW-20221018.tar.gz")
fp = tarfile.open(fname)
names = fp.getnames()
buffer = [io.BytesIO(fp.extractfile(name).read()) for name in names]
for buf, name in zip(buffer, names):
    buf.name = name
ds = wrl.io.open_radolan_mfdataset(buffer)
```

```{code-cell} python
display(ds)
```

```{code-cell} python
ds.RW.plot(col="time", col_wrap=6, vmax=20)
```
