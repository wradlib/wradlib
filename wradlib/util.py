#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not attributable
to the other modules

.. autosummary::
   :nosignatures:
   :toctree: generated/

   aggregate_in_time
   aggregate_equidistant_tseries
   from_to

"""
import warnings
import functools
import os

# TODO: check this again, removed to suppress deprecation warning in util.render_notebook
# warnings.simplefilter('always', DeprecationWarning)
# warnings.simplefilter('always', FutureWarning)


def deprecated(replacement=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.

    Author: Giampaolo Rodola' <g.rodola [AT] gmail [DOT] com>
    License: MIT

    Parameters
    ----------

    replacement: string
        function name of replacement function

    >>> # Todo: warnings are sent to stderr instead of stdout
    >>> # so they are not seen here
    >>> from wradlib.util import deprecated
    >>> import sys
    >>> sys.stderr = sys.stdout
    >>> @deprecated()
    ... def foo(x):
    ...     return x
    >>> ret = foo(1) #doctest: +ELLIPSIS
    /.../util.py:1: DeprecationWarning: wradlib.util.foo is deprecated
      #!/usr/bin/env python

    >>> def newfun(x):
    ...     return 0
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    >>> ret = foo(1) #doctest: +ELLIPSIS
    /.../util.py:1: DeprecationWarning: wradlib.util.foo is deprecated; use <function newfun at 0x...> instead
      #!/usr/bin/env python

    """

    def outer(fun):
        msg = "%s.%s is deprecated" % (fun.__module__, fun.__name__)
        if replacement is not None:
            msg += "; use %s instead" % replacement

        @functools.wraps(fun)
        def inner(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return fun(*args, **kwargs)

        return inner

    return outer


def apichange_kwarg(ver, par, typ, expar=None, exfunc=None, msg=None):
    """A decorator to generate a DeprectationWarning.

    .. versionadded:: 0.4.1

    This decorator function generates a DeprecationWarning if a given kwarg
    is changed/deprecated in a future version.

    The warning is only issued, when the kwarg is actually used in the function call.

    Parameters
    ----------

    ver : string
        Version from when the changes take effect
    par : string
        Name of parameter which is affected
    typ : Python type
        Data type of parameter which is affected
    expar : string
        Name of parameter to be used in future
    exfunc : function
        Function which can convert from old to new parameter
    msg : string
        additional warning message

    """

    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            para = kwargs.pop(par, None)
            key = par
            if para:
                wmsg = "\nPrevious behaviour of parameter '%s' in function '%s.%s' is deprecated " \
                       "\nand will be changed in version '%s'." % (par, func.__module__, func.__name__, ver)
                if expar:
                    wmsg += "\nUse parameter %s instead." % expar
                if exfunc:
                    wmsg += "\nWrong parameter types will be automatically converted by " \
                            "using %s.%s." % (exfunc.__module__, exfunc.__name__)
                if msg:
                    wmsg += "\n%s" % msg
                if isinstance(para, typ):
                    if exfunc:
                        para = exfunc(para)
                    if expar:
                        key = expar
                    warnings.warn(wmsg, category=DeprecationWarning, stacklevel=2)
                kwargs.update({key: para})
            return func(*args, **kwargs)

        return inner

    return outer


class OptionalModuleStub(object):
    """Stub class for optional imports.

    Objects of this class are instantiated when optional modules are not
    present on the user's machine.
    This allows global imports of optional modules with the code only breaking
    when actual attributes from this module are called.
    """

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        link = 'https://wradlib.github.io/wradlib-docs/latest/gettingstarted.html#optional-dependencies'
        raise AttributeError('Module "' + self.name +
                             '" is not installed.\n\n' +
                             'You tried to access function/module/attribute "' +
                             name + '"\nfrom module "' + self.name + '".\nThis ' +
                             'module is optional right now in wradlib.\n' +
                             'You need to separately install this dependency.\n' +
                             'Please refer to ' + link + '\n' +
                             'for further instructions.')


def import_optional(module):
    """Allowing for lazy loading of optional wradlib modules or dependencies.

    This function removes the need to satisfy all dependencies of wradlib before
    being able to work with it.

    Parameters
    ----------
    module : string
             name of the module

    Returns
    -------
    mod : object
          if module is present, returns the module object, on ImportError
          returns an instance of `OptionalModuleStub` which will raise an
          AttributeError as soon as any attribute is accessed.

    Examples
    --------
    Trying to import a module that exists makes the module available as normal.
    You can even use an alias. You cannot use the '*' notation, or import only
    select functions, but you can simulate most of the standard import syntax
    behavior
    >>> m = import_optional('math')
    >>> m.log10(100)
    2.0

    Trying to import a module that does not exists, does not produce any errors.
    Only when some function is used, the code triggers an error
    >>> m = import_optional('nonexistentmodule')
    >>> m.log10(100)  # doctest +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: Module "nonexistentmodule" is not installed.
    <BLANKLINE>
    You tried to access function/module/attribute "log10"
    from module "nonexistentmodule".
    This module is optional right now in wradlib.
    You need to separately install this dependency.
    Please refer to https://wradlib.github.io/wradlib-docs/latest/gettingstarted.html#optional-dependencies
    for further instructions.
    """
    try:
        mod = __import__(module)
    except ImportError:
        mod = OptionalModuleStub(module)

    return mod


def get_wradlib_data_path():
    wrl_data_path = os.environ.get('WRADLIB_DATA', None)
    if wrl_data_path is None:
        raise EnvironmentError("'WRADLIB_DATA' environment variable not set")
    if not os.path.isdir(wrl_data_path):
        raise EnvironmentError("'WRADLIB_DATA' path '{0}' does not exist".format(wrl_data_path))
    return wrl_data_path


def get_wradlib_data_file(relfile):
    data_file = os.path.join(get_wradlib_data_path(), relfile)
    if not os.path.exists(data_file):
        raise EnvironmentError("WRADLIB_DATA file '{0}' does not exist".format(data_file))
    return data_file


def render_notebook(filename, out=None):
    nbformat = import_optional('nbformat')
    runipy = import_optional('runipy.notebook_runner')

    notebook = nbformat.read(filename, 3)
    r = runipy.notebook_runner.NotebookRunner(notebook, mpl_inline=True)
    r.run_notebook()
    nb_name = os.path.basename(filename)
    if out is None:
        out_path = os.path.join(os.path.dirname(filename), 'rendered')
    else:
        out_path = out
    doc_name = os.path.join(out_path, nb_name)
    nbformat.write(r.nb, doc_name, 3)
    notebook = nbformat.read(doc_name, 3)
    nbformat.write(notebook, doc_name, 4)


if __name__ == '__main__':
    print('wradlib: Calling module <util> as main...')
