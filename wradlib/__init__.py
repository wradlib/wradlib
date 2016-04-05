# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
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
    from .version import git_revision as __git_revision__  # noqa
    from .version import version as __version__  # noqa

    # import subpackages
    from . import adjust  # noqa
    from . import atten  # noqa
    from . import clutter  # noqa
    from . import comp  # noqa
    from . import dp  # noqa
    from . import fill  # noqa
    from . import georef  # noqa
    from . import io  # noqa
    from . import ipol  # noqa
    from . import qual  # noqa
    from . import trafo  # noqa
    from . import util  # noqa
    from . import verify  # noqa
    from . import vis  # noqa
    from . import vpr  # noqa
    from . import zonalstats  # noqa
    from . import zr  # noqa
