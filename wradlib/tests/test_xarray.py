import xarray

import wradlib


def test_accessor():
    da = xarray.DataArray()

    print(type(da.wrl.vis.plot))
    print(type(wradlib.vis.VisMethods.plot))
