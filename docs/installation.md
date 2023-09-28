# Installation

## Anaconda/Conda

In order to run {{wradlib}}, you need to have a Python interpreter installed on your local computer, as well as  [a number of Python packages](#dependencies). We recommend installing [Anaconda](https://www.anaconda.com/download) as it includes Python, numerous required packages, and other useful tools (e.g. [Spyder](https://www.spyder-ide.org/)).

Using Anaconda the installation process is harmonised across platforms. Download and install the latest [Anaconda distribution](https://www.anaconda.com/download) for your specific OS. We recommend using the minimal distributions [Miniconda](https://conda.io/miniconda.html) or [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge)  if you do not want to install a full scientific python stack.

We are constantly performing tests with [conda-forge](https://conda-forge.org/) community channel (for the most recent 3 python versions).

If your Python installation is working, the following command (in a console) should work:

```bash
$ python --version
Python 3.11.0
```

Now you can use the ``conda``/``mamba`` package and environment manager ([conda documentation](https://conda.io/docs/) / [mamba documenation](https://mamba.readthedocs.io/en/latest/)) to setup your {{wradlib}} installation.

#. Add the conda-forge channel, where {{wradlib}} and its dependencies are located. Read more about the community effort [conda-forge](https://conda-forge.org):

```bash
$ conda config --add channels conda-forge
```

#. Use strict channel priority to prevent channel clashes:

```bash
$ conda config --set channel_priority strict
```

#. Create a new environment from scratch:

```bash
$ conda create --name wradlib python=3.11
```

#. Activate the {{wradlib}} environment:

```bash
$ conda activate wradlib
```

#. Install {{wradlib}} and its dependencies:

```bash
(wradlib) $ conda install wradlib
```

Now you have a ``conda`` environment with a working {{wradlib}} installation.

Test the integrity of your {{wradlib}} installation by opening a console window and typing calling the python interpreter:

```bash
$ python
Python 3.11.0 | packaged by conda-forge | (main, Oct 25 2022, 06:24:40) [GCC 10.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

The Python prompt should appear. Then type:

```ipython
>>> import wradlib
>>> wradlib.__version__
'2.0.0'
```

If everything is ok, this will show the running {{wradlib}} version. If the {{wradlib}} package is not found by the interpreter, you will get::

```ipython
>>> import wradlib
ImportError: No module named wradlib
```

Alternatively, you can install the [Dependencies](#bleeding-edge-code), but you have to keep track of {{wradlib}}'s dependencies yourself.


## Bleeding edge code

:::{warning}
The {{wradlib}} version on [PyPI](https://pypi.org/project/wradlib) might lag behind the actual developments. You can use the bleeding edge code from the {{wradlib}} [repository](https://github.com/wradlib/wradlib). Note, however, that you need to make sure yourself that all [Dependencies](#dependencies) are met (see below).
:::

[Download](https://codeload.github.com/wradlib/wradlib/zip/main) the source, unzip, and run:

```bash
$ python -m pip install .
```

Alternatively, you can add the {{wradlib}} directory to your environment variable ``PYTHONPATH``.

## Installing via pip

Although we recommend using a conda Python Environment you can install {{wradlib}} from [PyPi](https://pypi.org/project/wradlib) via ``pip``.

Open a terminal and run:

```bash
$ python -m pip install wradlib
```

Depending on your system you might need to be root (or sudo the above command) for this to work or use `--user` to install into user directory.
``pip`` will then fetch the source distribution from the Python Package Index and run the installation.

Afterwards it will check for any dependencies not met, yet.

Be aware that using ``pip`` we can only look for python-module dependencies.
For example the numpy module itself depends on some other libraries, which need to be present in order for the module to compile properly after being downloaded by ``pip``. We have no control over these dependencies and it is rather hard to give a complete overview.

Therefore we recommend trying to satisfy the dependencies using your favorite package management system.

Installing via ``pip`` tries to install all dependencies, but be sure to have all depending non-python libraries installed. Wheels are not available for all dependencies (eg. GDAL).

## Dependencies

{{wradlib}} was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, {{wradlib}} uses additional libraries, which you might need to install before you can use {{wradlib}} depending on your system and installation procedure.

| Package    | min     | recommended |
|------------|---------| ----------- |
| numpy      | >= 1.9  | >= latest   |
| scipy      | >= 1.0  | >= latest   |
| matplotlib | >= 3    | >= latest   |
| xarray     | >= 0.17 | >= latest   |
| xradar     | >= 0.3  | >= latest   |

You can check whether the required [](#dependencies) are available on your computer by opening a Python console and enter:

```ipython
>>> import <package_name>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'package_name'
```

This will be the response in case the package is not available.

In case the import is successful, you should also check the version number:

```ipython
>>> package_name.__version__
`some version number`
```

The version number should be consistent with the above [](#dependencies).


## Optional Dependencies

Apart from the obligatory [](#optional-dependencies), some dependencies in {{wradlib}} are optional. This is because the installation of these dependencies can be somewhat tedious while many {{wradlib}} users will not need them anyway. In case users use a {{wradlib}} function that requires an optional dependency, and this dependency is not satisfied in the local environment, {{wradlib}} will raise an exception.

As for now, the following dependencies are defined as optional:

| Package    |    min    | recommended |
|------------|-----------|-------------|
| cartopy    | >= 0.22   | >= latest   |
| dask       | >= 2.20   | >= latest   |
| gdal       | >= 3.0    | >= latest   |
| h5py       | >= 3.0.0  | >= latest   |
| h5netcdf   | >= 0.8.0  | >= latest   |
| netCDF4    | >= 1.0    | >= latest   |
| requests   | >= 2.23.0 | >= latest   |
| xmltodict  | >= 0.12   | >= latest   |

The following libraries are used by `netCDF4`, `h5py`/`h5netcdf` and `gdal` packages and should apply to these requirements:

| Library    |    min    | recommended | used by |
|------------|-----------|-------------|---------|
| geos       | >= 3.7.0  | >= latest   | gdal    |
| hdf5       | >= 1.9.0  | >= latest   | h5py    |
| libnetcdf  | >= 4.7.3  | >= latest   | netCDF4 |
| proj       | >= 5.2.0  | >= latest   | gdal    |

**The speedup module**

The speedup module is intended as a collection of Fortran code in order to speed up specific {{wradlib}} function that are critical for performance.
In order to build the speedup module as a shared library, you need to use [f2py](https://numpy.org/doc/stable/f2py/usage.html). ``f2py`` usually ships with numpy and should be available via the command line. To test whether f2py is available on your system, execute ``f2py`` on the system console. Or, alternatively, ``f2py.py``. If it is available, you should get a bunch of help instructions. Now change to the {{wradlib}} module directory and execute on the system console:

```bash
$ f2py.py -c -m speedup speedup.f
```

Now the speedup module should be available.

## Known Issues

Depending on your OS and installation method you may encounter different problems. Here are some guidelines for attacking them.

We strongly recommend using the Anaconda conda package and environment manager (see [](#installation)). Using [conda-forge](https://conda-forge.org) we will maintain the [wradlib-feedstock](https://github.com/conda-forge/wradlib-feedstock) for constant availability of recent {{wradlib}} versions.

If you can't use Anaconda/Miniconda, it is generally a good idea to use your systems package manager to install dependencies. This will also take account for other needed bindings, libs etc.

If you encounter problems installing {{wradlib}}, check on your favorite search engine or create an [issue here](https://github.com/wradlib/wradlib/issues) with details on the problem or open a discussion topic on the [openradar-discourse](https://openradar.discourse.group/).
