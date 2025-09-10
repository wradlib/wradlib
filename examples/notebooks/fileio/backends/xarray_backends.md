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

# xarray backends


This section gives an overview on the available xarray backends.



```{warning}
For radar data in polar format the [openradar community](https://openradarscience.org/) published [xradar](https://docs.openradarscience.org/projects/xradar/en/latest/) where xarray-based readers/writers are implemented. That particular code was ported from $\omega radlib$ to xradar. Please refer to xradar for enhancements for polar radar.

Since $\omega radlib$ version 1.19 almost all backend code has been transferred to [xradar](https://github.com/openradar/xradar)-package and is imported from there.
```

```{toctree}
:hidden:
:maxdepth: 1
RADOLAN <radolan_backend>
ODIM <odim_backend>
GAMIC <gamic_backend>
CfRadial1 <cfradial1_backend>
CfRadial2 <cfradial2_backend>
Iris/Sigmet <iris_backend>
Rainbow5 <rainbow_backend>
Furuno SCN/SCNX <furuno_backend>
```