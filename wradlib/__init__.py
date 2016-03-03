# adapted from https://github.com/ARM-DOE/pyart/blob/master/pyart/__init__.py
"""
wradlib
=======

"""

# Detect if we're being called as part of wradlib's setup procedure
try:
    __WRADLIB_SETUP__
except NameError:
    __WRADLIB_SETUP__ = False

if __WRADLIB_SETUP__:
    import sys as _sys
    _sys.stderr.write("Running from wradlib source directory.\n")
    del _sys
else:
    # Make sure that deprecation warnings get printed by default
    import warnings as _warnings
    _warnings.simplefilter("always", DeprecationWarning)

     # versioning
    from .version import git_revision as __git_revision__
    from .version import version as __version__

    # import subpackages
    from . import adjust
    from . import atten
    from . import clutter
    from . import comp
    from . import dp
    from . import fill
    from . import georef
    from . import io
    from . import ipol
    from . import qual
    from . import trafo
    from . import util
    from . import verify
    from . import vis
    from . import vpr
    from . import zonalstats
    from . import zr
