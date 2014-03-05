*************************************
automate image generation with sphinx
*************************************

With active version control it is often undesirable to bundle images with the documentation.
This tutorial shows how to generate images on the fly with `sphinx`.

First thing is to add the necessary sphinx extension configuration on your accompanying `conf.py`.

Add `matplotlib.sphinxext.plot_directive` to the extensions section::

    # Add any Sphinx extension module names here, as strings. They can be extensions
    # coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
    extensions = ['sphinx.ext.autodoc',
                  'sphinx.ext.todo',
                  'sphinx.ext.coverage',
                  'sphinx.ext.pngmath',
                  'sphinx.ext.autosummary',
                  'numpydoc',
                  'matplotlib.sphinxext.plot_directive',
                  ]

This was actually the hardest part! Next time you can just write in your documentation::

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        x = np.random.randn(1000)
        plt.hist( x, 20)
        plt.grid()
        plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
        plt.show()

This will lead to a nice arranged histogram plot in the final documentation. See yourself:

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   x = np.random.randn(1000)
   plt.hist( x, 20)
   plt.grid()
   plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
   plt.show()

So, now let's plot some weather radar stuff::

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import wradlib

        # load a polar scan and create range and azimuth arrays accordingly
        data = np.loadtxt('../../../examples/data/polar_dBZ_tur.gz')
        r = np.arange(0, data.shape[1])
        az = np.arange(0, data.shape[0])
        # mask data array for better presentation
        mask_ind = np.where(data <= np.nanmin(data))
        data[mask_ind] = np.nan
        ma = np.ma.array(data, mask=np.isnan(data))
        # the simplest call, plot cg ppi in new window
        # plot simple CG PPI
        wradlib.vis.plot_cg_ppi(ma, refrac=False)
        t = plt.title('Simple CG PPI')
        t.set_y(1.05)
        plt.tight_layout()
        plt.show()

But, take care of the correct path to your example data. It starts from your documentation root directory
(the one where 'conf.py` resides). After all this nice radar ppi will magically compile into your documentation:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib

    # load a polar scan and create range and azimuth arrays accordingly
    # path starts in your documentation root
    data = np.loadtxt('../../examples/data/polar_dBZ_tur.gz')
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    # the simplest call, plot cg ppi in new window
    # plot simple CG PPI
    wradlib.vis.plot_cg_ppi(ma, refrac=False)
    t = plt.title('Simple CG PPI')
    t.set_y(1.05)
    plt.tight_layout()
    plt.show()

More, you have one option not to clutter your documentation files with elaborate plotting code.
Just give the directive a file with your code. One idea is to take one of your example files.
This is as follows::

    .. plot:: pyplots/plot_cg_rhi_example.py

As you can see, there is a folder `pyplots`, which, you may already gess, lives in your documentation root folder.
All in documentation referenced plotting routines have to be in that folder, which may have some other name.
If you like to build on your already available example files, you have to symlink to them.

The above statement looks like this in the final documentation:

.. plot:: pyplots/plot_cg_rhi_example.py

This plot_directive is very convenient if you just want to add plots to your documentation at compile time.
If you want to learn more about matplotlib and documentation, take a look at the `Matplotlib Sampledoc <http://matplotlib.org/sampledoc/extensions.html/>`_.
Have fun!
