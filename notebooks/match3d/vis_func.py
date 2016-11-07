import wradlib as wrl
import matplotlib.pyplot as pl
import numpy as np
#import os

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = pl.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0.1, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def comparison(filename):

    outdict = wrl.io.read_generic_netcdf(filename)
    print(outdict.keys())

    nrejgr = outdict['variables']['nrejgr']['data']
    nrejpr = outdict['variables']['nrejpr']['data']
    ntotgr = outdict['variables']['ntotgr']['data']
    ntotpr = outdict['variables']['ntotpr']['data']
    frac1 = nrejpr/ntotpr
    frac2 = nrejgr/ntotgr
    ipair = (frac1 <= 0.1) & (frac2 <= 0.1)
    print(ipair)

    refgr = outdict['variables']['refgr']['data'][ipair]
    x = outdict['variables']['x']['data'][ipair] / 1000.
    y = outdict['variables']['y']['data'][ipair] / 1000.
    ds = outdict['variables']['ds']['data'][ipair] / 1000.
    itilt = outdict['variables']['itilt']['data'][ipair]
    print(refgr.shape, x.shape, y.shape, ds.shape)

    tilt_max = np.argmax(np.bincount(np.asarray(itilt, dtype=np.int64)))
    print(tilt_max)

    ipairm = (itilt == tilt_max)
    refgr = refgr[ipairm]
    x = x[ipairm]
    y = y[ipairm]
    ds = ds[ipairm]

    clabel = 'Reflectivity [dBZ]'
    pl.scatter(x, y, s=ds*10., c=refgr, cmap=discrete_cmap(12, 'jet'), vmin=0, vmax=60)
    pl.gca().set_aspect('equal')
    cbar = pl.colorbar(ticks=np.arange(0, 60, 5), pad=0.05, label=clabel)
    pl.clim(0, 60)
    pl.gca().set_xlim(-150, 150)
    pl.gca().set_ylim(-150, 150)
