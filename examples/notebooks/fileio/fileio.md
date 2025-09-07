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

# Data Input - Data Output


The binary encoding of many radar products is a major obstacle for many potential radar users. Often, decoder software is not easily available. In case formats are documented, the implementation of decoders is a major programming effort.

```{note}
For radar data in polar format (eg. rays vs. range) the [openradar community](https://openradarscience.org/) published [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/) where xarray-based readers/writers are implemented. That particular code was ported from $\omega radlib$ to xradar. Please refer to xradar for enhancements for polar radar.
```

This section provides a collection of example code snippets to show which data formats $\omega radlib$ can handle and and how to facilitate that.

```{toctree}
:hidden:
:maxdepth: 2
Legacy readers <legacy/legacy_readers>
Xarray readers <backends/xarray_backends>
DWD RADOLAN <radolan/radolan>
GIS Vector Data <gis/vector_data>
GIS Raster Export <gis/raster_data>
```