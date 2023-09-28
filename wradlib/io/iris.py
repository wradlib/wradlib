#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
IRIS/Sigmet Data I/O
^^^^^^^^^^^^^^^^^^^^

Reads data from Vaisala's IRIS data formats. Former available code was ported to
`xradar <https://xradar.rtfd.io>`__-package and is imported from there.

IRIS (Vaisala Sigmet Interactive Radar Information System)

See M211318EN-F Programming Guide ftp://ftp.sigmet.com/outgoing/manuals/

Reading sweep data can be skipped by setting `loaddata=False`. By default, the data
is decoded on the fly. Using `rawdata=True` the data will be kept undecoded.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "read_iris",
    "IrisProductFile",
    "IrisCartesianProductFile",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import struct
import warnings
from collections import OrderedDict

import numpy as np
from xradar.io.backends import iris as xiris


class IrisProductFile(xiris.IrisRecordFile):
    """Class for retrieving data from Sigmet IRIS Product files."""

    product_identifier = [
        "CATCH",
        "FCAST",
        "NDOP",
        "SLINE",
        "TDWR",
        "TRACK",
        "VAD",
        "VVP",
        "WARN",
        "WIND",
        "STATUS",
    ]

    def __init__(self, filename, **kwargs):
        """
        Parameters
        ----------
        filename : str
            filename
        """
        super().__init__(filename, **kwargs)

        self.check_product_identifier()
        self._protect_setup = None
        self._data = OrderedDict()
        if self.loaddata:
            self.get_data()

    @property
    def data(self):
        return self._data

    @property
    def protect_setup(self):
        return self._protect_setup

    def get_protect_setup(self):
        protected_setup = self.read_from_record(1024, "uint8")
        protected_regions = OrderedDict()
        for i in range(32):
            region = xiris._unpack_dictionary(
                protected_setup[i * 32 : i * 32 + 32],
                xiris.ONE_PROTECTED_REGION,
                self._rawdata,
            )
            if not region["region_name"].isspace():
                protected_regions[i] = region

        return protected_regions

    def get_results(self, results, num, structure):
        cnt = struct.calcsize(xiris._get_fmt_string(structure))
        for i in range(num):
            dta = self.read_from_record(cnt, "uint8")
            res = xiris._unpack_dictionary(dta, structure, self._rawdata)
            results[i] = res

    def get_data(self):
        """Retrieves cartesian data from file."""
        # set filepointer accordingly
        self.init_record(0)
        self._rh.pos = 640
        if "protect_setup" in self.product_type_dict:
            self._protect_setup = self.get_protect_setup()
        product_end = self.product_hdr["product_end"]
        product_config = self.product_hdr["product_configuration"]
        num_elements = product_end["number_elements"]
        specific_info = product_config["product_specific_info"]

        result = OrderedDict()
        if self.product_type in ["FCAST", "NDOP"]:
            x_size = product_config.get("x_size")
            y_size = product_config.get("y_size")
            z_size = product_config.get("z_size", 1)

            cnt = struct.calcsize(
                xiris._get_fmt_string(self.product_type_dict["result"])
            )
            z = []
            for _zi in range(z_size):
                y = []
                for _yi in range(y_size):
                    x = []
                    for _xi in range(x_size):
                        dta = self.read_from_record(cnt, "uint8")
                        res = xiris._unpack_dictionary(
                            dta, self.product_type_dict["result"], self._rawdata
                        )
                        x.append(res)
                    y.append(x)
                z.append(y)
            result[0] = np.array(z)
        # get vvp num_elements
        elif self.product_type in ["VVP"]:
            num_elements = specific_info["num_intervals"]
            self.get_results(result, num_elements, self.product_type_dict["result"])
        # get wind num_elements
        elif self.product_type in ["WIND"]:
            num_elements = (
                specific_info["num_panel_points"] * specific_info["num_range_points"]
            )
            cnt = struct.calcsize(xiris._get_fmt_string(xiris.VVP_RESULTS))
            dta = self.read_from_record(cnt, "uint8")
            res = xiris._unpack_dictionary(dta, xiris.VVP_RESULTS, self._rawdata)
            result["VVP"] = res
            self.get_results(result, num_elements, self.product_type_dict["result"])
        elif self.product_type in ["STATUS"]:
            self.get_results(result, num_elements, self.product_type_dict["result"])
        else:
            if num_elements:
                self.get_results(result, num_elements, self.product_type_dict["result"])
            else:
                warnings.warn(
                    f"{self.product_type} - No product result array(s) available",
                    RuntimeWarning,
                    stacklevel=3,
                )

        if self._protect_setup is not None:
            result["protect_setup"] = self._protect_setup

        self._data = result


class IrisCartesianProductFile(xiris.IrisRecordFile):
    """Class for retrieving data from Sigmet IRIS Cartesian Product files."""

    product_identifier = [
        "MAX",
        "TOPS",
        "HMAX",
        "BASE",
        "THICK",
        "PPI",
        "RHI",
        "CAPPI",
        "RAINN",
        "RAIN1",
        "CROSS",
        "SHEAR",
        "SRI",
        "RTI",
        "VIL",
        "LAYER",
        "BEAM",
        "MLHGT",
    ]

    def __init__(self, irisfile, **kwargs):
        """
        Parameters
        ----------
        irisfile : str
            filename
        """
        self._origin = kwargs.get("origin", "lower")
        super().__init__(irisfile, **kwargs)
        self.check_product_identifier()
        self._data = OrderedDict()
        if self.loaddata:
            self.get_data()

    @property
    def data(self):
        return self._data

    def fix_ext_header(self, ext):
        prod_conf = self.product_hdr["product_configuration"]
        ext.update({"x_size": ext.pop("x_size", prod_conf.get("x_size"))})
        ext.update({"y_size": ext.pop("y_size", prod_conf.get("y_size"))})
        ext.update({"z_size": ext.pop("z_size", prod_conf.get("z_size", 1))})
        ext.update({"data_type": ext.pop("iris_type", prod_conf.get("data_type"))})

    def get_extended_header(self):
        # hack, get from actual position to end of record
        ext = self.rh.record[self.rh.pos :]
        if len(ext) == 0:
            return False
        # extended header token
        search = [0x00, 0xFF]
        ext = np.where((ext[:-1] == search[0]) & (ext[1:] == search[1]))[0][0]
        extended_header = OrderedDict([("extended_header", xiris.string_dict(ext))])
        ext_str = xiris._unpack_dictionary(
            self.bytes_from_record(ext, 1), extended_header, self._rawdata
        )["extended_header"]
        # skip search bytes
        self.bytes_from_record(2, 1)
        ext_hdr = OrderedDict()
        for d in ext_str.split("\n"):
            kv = d.split("=")
            try:
                ext_hdr[kv[0]] = int(kv[1])
            except ValueError:
                ext_hdr[kv[0]] = kv[1]
            except Exception:
                pass
        self.fix_ext_header(ext_hdr)
        return ext_hdr

    def get_image(self, header):
        """Retrieve cartesian image.

        Parameters
        ----------
        header : dict
            header dictionary

        Returns
        -------
        data : :class:`numpy:numpy.ndarray`
            3D array of cartesian data

        """
        prod = xiris.SIGMET_DATA_TYPES[header.get("data_type")]
        x_size = header.get("x_size")
        y_size = header.get("y_size")
        z_size = header.get("z_size")
        cnt = x_size * y_size * z_size
        data = self.read_from_record(cnt, prod["dtype"])
        data = self.decode_data(data, prod=prod)
        data.shape = (z_size, y_size, x_size)
        if self._origin == "upper":
            data = np.flip(data, axis=1)
        return data

    def get_data(self):
        """Retrieves cartesian data from file."""
        # set filepointer accordingly
        self.init_record(0)
        self.rh.pos = 640

        product_hdr = self.product_hdr
        product_end = product_hdr["product_end"]
        if product_hdr["product_end"]["number_elements"]:
            warnings.warn(
                f"{self.product_type} Not Implemented - Product results "
                "array available \nnot loading dataset",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            self._data[0] = self.get_image(product_hdr["product_configuration"])
            if product_end["extended_product_header_offset"]:
                ext_hdr = OrderedDict()
                i = 0
                ext = self.get_extended_header()
                while ext:
                    ext_hdr[i + 1] = ext
                    i += 1
                    self._data[i] = self.get_image(ext)
                    ext = self.get_extended_header()
                self.product_hdr["extended_header"] = ext_hdr

    def decode_data(self, data, prod):
        """Decode data according given prod-dict.

        Parameters
        ----------
        data : :py:class:`numpy:numpy.ndarray`
            data to decode
        prod : dict
            dictionary holding decoding information

        Returns
        -------
        data : :py:class:`numpy:numpy.ndarray`
            decoded data

        """
        if self._rawdata:
            return data
        kw = {}
        if prod["func"]:
            try:
                kw.update(prod["fkw"])
            except KeyError:
                pass
            if prod["func"] in [xiris.decode_vel, xiris.decode_width, xiris.decode_kdp]:
                wavelength = self.product_hdr["product_end"]["wavelength"]
                if prod["func"] == xiris.decode_kdp:
                    kw.update({"wavelength": wavelength / 100})
                    return prod["func"](data, **kw)

                prf = self.product_hdr["product_end"]["prf"]
                nyquist = wavelength * prf / (10000.0 * 4.0)
                kw.update({"nyquist": nyquist})
            return prod["func"](data.view(prod["dtype"]), **kw)
        else:
            return data


def read_iris(
    filename,
    *,
    loaddata=True,
    rawdata=False,
    debug=False,
    **kwargs,
):
    """Read Iris file and return dictionary.

    Parameters
    ----------
    filename : str or file-like
        Filename of data file or file-like object.
    loaddata : bool or dict
                If true, retrieves whole data section from file.
                If false, retrievs only ingest_data_headers, but no data.
                If dict, retrieves according to given dict::

                    loaddata = {'moment': ['DB_DBZ', 'DB_VEL'],
                                'sweep': [1, 3, 9]}

    rawdata : bool
        If true, returns raw unconverted/undecoded data.
    debug : bool
        If true, print debug messages.

    Returns
    -------
    data : dict
        Ordered Dictionary with data and metadata retrieved from file.

    Examples
    --------
    See :ref:`/notebooks/fileio/legacy/read_iris.ipynb`.
    """
    if not isinstance(filename, str):
        filename = filename.read()

    sid, opener = xiris._check_iris_file(filename)

    if not opener:
        raise TypeError(f"Unknown Iris File or Iris Product Type {sid}")

    irisfile = opener(
        filename, loaddata=loaddata, rawdata=rawdata, debug=debug, **kwargs
    )

    properties = [
        "product_hdr",
        "product_type",
        "ingest_header",
        "ingest_data_header",
        "nrays_expected",
        "sweep",
        "nsweeps",
        "nrays",
        "nbins",
        "data_types",
        "data",
        "raw_product_bhdrs",
        "sweeps",
        "spare_0",
        "gparm",
    ]

    data = OrderedDict()
    for k in properties:
        item = getattr(irisfile, k, None)
        if item:
            data.update({k: item})

    return data
