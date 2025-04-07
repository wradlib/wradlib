# Contributing to wradlib

The wradlib project encourages everyone to contribute to the developement of wradlib. This paper describes the general guidelines for contributing to wradlib. There is more information on [wradlib's community effort](https://wradlib.org/community) at the project's main website [wradlib.org](https://wradlib.org).

In general wradlib uses GitHub Pull Requests for contribution. Second, the [wradlib developers](CONTRIBUTORS.txt) have GitHub Commit Access to the project's main repository.

## Code of Conduct

Contributors to wradlib are expected to act respectfully toward others in accordance with the [OSGeo Code of Conduct](http://www.osgeo.org/code_of_conduct).

## Contributions and Licensing

All contributions shall comply to the project [license](LICENSE.txt).

## GitHub

Source code, testing, bug tracking as well as docs and wiki are handled via GitHub.

## Documentation

* wradlib's documentation source is contained in [wradlib-docs](https://github.com/wradlib/wradlib-docs) repository, and mainly written in reStructuredText.
* Jupyter Notebooks in [wradlib-notebooks](https://github.com/wradlib/wradlib-notebooks) repository are utilized to extend the documentation with examples and tutorials.
* wradlib uses [Sphinx](http://sphinx-doc.org/) and [nbsphinx](https://github.com/spatialaudio/nbsphinx) to compile docs on [readthedocs.org](https://readthedocs.org/projects/wradlib-docs/).

## Issue tracking

Using wradlib's [GitHub issue tracker](https://github.com/wradlib/wradlib/issues) users can file bug reports or discuss feature requests. Bug reports should contain
used wradlib as well as Python and OS version, error messages and instructions to reproduce the bug.

## Forking wradlib

For this purpose, you need to have a [GitHub Account](https://github.com/signup/free) and [Git](https://git-scm.com/). Then fork wradlib, make changes and create a pull request.

## Guide

This is a practical step by step guide. The commands should work in your operating system terminal. Please adapt it to your needs.

### 1a Install environement using conda

- Install [Anaconda or Miniconda](https://www.anaconda.com/download/)
- Activate conda (or start conda terminal for automatic configuration)
- Create development environement (alongside):
```bash
conda config --add channels conda-forge
conda create -n wradlib-dev
conda activate wradlib-dev
```
- Install wradlib package:
```bash
conda install wradlib
conda install libgdal-hdf5
conda install libgdal-netcdf
```

### 1b Install environement on windows

- Create virtual environement:
``` powershell
winget install -e --id Python.Python.3.13
python -m venv $env:USERPROFILE/.venvs/wradlib
& "$env:UserProfile\.venvs\wradlib\Scripts\activate.ps1"
```
- Install unofficial gdal wheel or build gdal
``` powershell
pip install https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.3.30/gdal-3.10.2-cp313-cp313-win_amd64.whl
```
- Build latest version of gdal:
``` powershell
winget install SWIG
$env:Path += ";C:\Users\egoud\AppData\Local\Microsoft\WinGet\Packages\SWIG.SWIG_Microsoft.Winget.Source_8wekyb3d8bbwe\swigwin-4.2.1"
$VCPKG_PATH = "$HOME/vcpkg"
git clone https://github.com/microsoft/vcpkg.git $VCPKG_PATH
& "$VCPKG_PATH\vcpkg.exe" install gdal:x64-windows-static
python.exe -m pip install --upgrade pip
pip install numpy setuptools

$GDAL_PATH = "$HOME/gdal"
git clone https://github.com/OSGeo/gdal.git $GDAL_PATH
$BUILD_PATH = "$GDAL_PATH/build"
cmake -S $GDAL_PATH -B $BUILD_PATH -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_TOOLCHAIN_FILE="$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake" `
      -DVCPKG_TARGET_TRIPLET="x64-windows-static" `
      -DGDAL_USE_STATIC_LIBS=ON `
      -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" `
      -DGDAL_ENABLE_PYTHON=ON `
      -DGDAL_BUILD_OPTIONAL_DRIVERS=OFF `
      -DOGR_BUILD_OPTIONAL_DRIVERS=OFF `
      -DGDAL_ENABLE_DRIVER_HDF5=ON`
      -DGDAL_ENABLE_DRIVER_NETCDF=ON

      cmake --build $BUILD_PATH --config Release
cmake --install $BUILD_PATH --prefix $env:VIRTUAL_ENV
pip install gdal/swig/python

python -c "import osgeo.gdal; print(osgeo.gdal.VersionInfo())"
```
### 1c Install environement on UNIX variants

- Install build tools and dependencies:

``` bash
sudo apt install build-essential cmake git libgdal-dev python3-devel python3-pip python3-venv swig wget
```

``` bash
sudo dnf install gcc gcc-c++ make cmake git gdal-devel proj-devel netcdf-devel hdf5-devel python3-devel python3-pip python3-virtualenv swig wget
```

- Create virtual environement
``` bash
python3 -m venv $HOME/.venvs/wradlib
source  $HOME/.venvs/wradlib/bin/activate
```

- Build latest version of gdal:
``` bash
pip install --upgrade pip
pip install setuptools numpy

GDAL_VERSION=3.10.2

PREFIX=$VIRTUAL_ENV
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

wget https://download.osgeo.org/gdal/$GDAL_VERSION/gdal-$GDAL_VERSION.tar.gz
tar -xzf gdal-$GDAL_VERSION.tar.gz

cmake ./gdal-$GDAL_VERSION -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc) -C gdal-$GDAL_VERSION
make install -C gdal-$GDAL_VERSION

rm -rf gdal-$GDAL_VERSION gdal-$GDAL_VERSION.tar.gz

gdalinfo --version
python -c "from osgeo import gdal; print(gdal.__version__)"

echo 'export LD_LIBRARY_PATH="$HOME/.venvs/wradlib/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

### 2 Install and test latest version of wradlib

- Clone wradlib repository:
``` bash
git clone https://github.com/wradlib/wradlib.git
cd wradlib
```
- Install all dependencies:
``` bash
pip install -e .[dev,opt]
```
- Run the test suite:
``` bash
python -m pytest -n auto
```
- Please report any errors with details on your system

### 3 Configure repository for development

- Configure automatic key-based authentification (More info for [windows](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement) or [linux](https://wiki.archlinux.org/title/SSH_keys) and [github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account))
```bash
ssh-keygen -t ed25519
ssh-add $HOME/.ssh/id_ed25519
cat $HOME/.ssh/id_ed25519.pub
```

- Configure git user
```bash
git config --global user.name "Edouard Goudenhoofdt"
git config --global user.email "egouden@outlook.com"
```

- Configure fork and upstream
```bash
git remote remove origin
git remote add origin git@github.com/egouden/wradlib.git
git remote add upstream https://github.com/wradlib/wradlib
```

### 4 Work on your changes

- Create a new branch
```bash
git checkout main
git rebase upstream main
git push origin main
git checkout -b my-feature
```
- Make changes using your favorite editor
- Save your changes remotely
```bash
git log
git commit -a -m "commit message as in git log"
git push origin my-feature
```
- Update your branch with latest version of wradlib
```bash
git fetch upstream main
git rebase upstream/main
pip install -e .
```
- Make new changes to existing branch
```bash
git fetch upstream main
git rebase upstream/main
git commit -a --amend --date "$(date)"
git push origin my-feature -f
```
- Test your changes
```bash
python -m pytest -n auto
```
- Clean your code
```bash
black .
ruff check
ruff check --fix
```
- Start a pull request on github
- Use draft mode when working on new changes