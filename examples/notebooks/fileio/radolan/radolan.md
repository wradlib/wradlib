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

# RADOLAN


RADOLAN is abbreviated from the german **RA**dar-**O**n**L**ine-**AN**eichung, which means Radar-Online-Adjustment.

Using it's [network of 17 weather radar](https://www.dwd.de/SharedDocs/broschueren/DE/presse/wetterradar_pdf.pdf?__blob=publicationFile&v=5) the German Weather Service provides [many products](https://www.dwd.de/DE/leistungen/radolan/produktuebersicht/radolan_produktuebersicht_pdf.pdf?__blob=publicationFile&v=6) for high resolution precipitation analysis and forecast. A comprehensive product list can be found in chapter [RADOLAN Product Showcase](radolan_showcase).

These composite products are distributed in the [RADOLAN Binary Data Format](radolan_format) with an ASCII header. All composites are available in [Polar Stereographic Projection](radolan_grid#Polar-Stereographic-Projection) which will be discussed in the chapter [RADOLAN Grid](radolan_grid).

This notebook tutorial was prepared with material from the [DWD RADOLAN/RADVOR-OP Kompositformat](https://www.dwd.de/DE/leistungen/radolan/radolan_info/radolan_radvor_op_komposit_format_pdf.pdf?__blob=publicationFile&v=5).
We also wish to thank Elmar Weigl, German Weather Service, for providing the extensive set of example data and his valuable information about the RADOLAN products.

```{toctree}
:hidden:
:maxdepth: 1
RADOLAN Quick Start <radolan_quickstart>
RADOLAN Binary Data Format <radolan_format>
RADOLAN Product Showcase <radolan_showcase>
RADOLAN Grid <radolan_grid>
DWD Radar Network <radolan_network>
```