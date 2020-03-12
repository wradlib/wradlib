# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib_tests
=============

"""

from . import test_adjust  # noqa
from . import test_atten  # noqa
from . import test_classify  # noqa
from . import test_clutter  # noqa
from . import test_comp  # noqa
from . import test_dp  # noqa
from . import test_georef  # noqa
from . import test_io  # noqa
from . import test_io_odim  # noqa
from . import test_ipol  # noqa
from . import test_qual  # noqa
from . import test_trafo  # noqa
from . import test_util  # noqa
from . import test_verify  # noqa
from . import test_vpr  # noqa
from . import test_zonalstats  # noqa
from . import test_zr  # noqa

__all__ = [s for s in dir() if not s.startswith('_')]
