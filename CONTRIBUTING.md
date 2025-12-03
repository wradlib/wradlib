# Contribute to wradlib

This is a practical guide for contributing. Check the [documentation](docs/dev_guide.md) for general information.

## Configure local repository

### Install git

```powershell
winget install --id Git.Git
```

```bash
sudo apt install git
```

### Configure git user

```bash
git config --global user.name "John Doe"
git config --global user.email "john.doe@gmail.com"
```

### Generate SSH keys

```bash
ssh-keygen -t ed25519
cat $HOME/.ssh/id_ed25519.pub
```

### Add SSH keys to [GitHub account](https://github.com/settings/keys)

### Configure SSH agent on Windows (run as admin):
```powershell
Start-Process powershell -Verb RunAs
Start-Service ssh-agent
Set-Service -Name ssh-agent -StartupType Automatic
git config --global core.sshCommand "C:\\Windows\\System32\\OpenSSH\\ssh.exe"
Restart-Computer
```

### Add SSH key to agent

```bash
ssh-add $HOME/.ssh/id_ed25519
```

### Fork wradlib on github

### Clone and link

```bash
git clone git@github.com:jdoe/wradlib.git
cd wradlib
git remote add upstream https://github.com/wradlib/wradlib
```

## Install/Update development environment

### Install with conda (Linux, macOS, Windows)

```sh
conda env create -f environment.yml
conda activate wradlib-dev
conda update --all
pip install -e .
```

### Install with package manager and uv (Linux, gdal>=3.9)

```bash
sudo apt update && sudo apt upgrade
sudo apt install build-essential
sudo apt install gdal-bin libgdal-dev proj-data python3 python3-dev
sudo snap install astral-uv
```

```bash
sudo dnf update
sudo dnf install gcc gcc-c++ make automake autoconf libtool
sudo dnf install gdal gdal-devel hdf5-devel netcdf-devel proj proj-data-us proj-devel --setopt=install_weak_deps=False
sudo dnf install python3 python3-devel uv
```

```bash
echo "alias activate='source .venv/bin/activate'" >> $HOME/.bashrc
exit
```

```bash
cd wradlib
uv venv
activate
GDAL_VERSION=$(gdal-config --version)
uv pip install -e .[dev] gdal==$GDAL_VERSION.* --no-binary h5py --no-binary netcdf4 --no-binary pyproj
```

### Install with vcpkg and uv (Windows)

```powershell
winget install -e --id Microsoft.VisualStudio.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools"
cd $HOME
git clone --depth 1 https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
[System.Environment]::SetEnvironmentVariable("Path",$env:Path + ";$HOME\vcpkg",[System.EnvironmentVariableTarget]::User)
exit
```

```powershell
winget install --id Python.PythonInstallManager
pymanager install py3.13
winget install --id astral-sh.uv
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force }
Add-Content $PROFILE "`nfunction activate { . .\.venv\Scripts\Activate.ps1 }"
exit
```

```powershell
vcpkg install gdal[core,geos,hdf5,netcdf]
vcpkg list gdal
```

```powershell
cd $HOME\wradlib
uv venv --python 3.13
Add-Content .venv\Scripts\Activate.ps1 '$env:PATH = "$HOME\vcpkg\installed\x64-windows\bin;" + $env:PATH'
Add-Content .venv\Scripts\Activate.ps1 '$env:INCLUDE = "$HOME\vcpkg\installed\x64-windows\include"'
Add-Content .venv\Scripts\Activate.ps1 '$env:LIB = "$HOME\vcpkg\installed\x64-windows\lib"'
activate
uv pip install -e .[dev] gdal==3.12.0.*
```

```powershell
winget install Amazon.AWSCLI
$ProjDownloadDir = python -c "import pyproj; print(pyproj.datadir.get_user_data_dir())"
aws s3 sync s3://cdn.proj.org $ProjDownloadDir --no-sign-request
```

```powershell
vcpkg upgrade gdal --no-dry-run
uv pip install -e .[dev] gdal==3.12.0.* --upgrade
```

### Install with spack and uv (Linux)

```bash
sudo apt install build-essential gfortran python3
```

```bash
git clone --depth 1 https://github.com/spack/spack.git
echo ". ~/spack/share/spack/setup-env.sh" >> .bashrc
exit
```

```bash
spack env create wradlib-dev
spack env activate wradlib-dev
spack add openblas
spack add hdf5 ~mpi
spack add netcdf-c ~mpi
spack add gdal +geos +hdf5 +netcdf
spack add python@3.14
spack compiler find
spack install
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
GDAL_VERSION=$(gdal-config --version)
uv pip install -e wradlib[dev] gdal==$GDAL_VERSION.* --no-binary h5py --no-binary netcdf4 --no-binary pyproj
```

## Work on a bug fix, enhancement or new feature

### Update local repository
```bash
git checkout main
git fetch upstream main
git rebase upstream/main
git push origin main
```

### Run unit test and check coverage
```bash
pytest -n auto --dist loadfile --verbose --doctest-modules --doctest-plus --durations=15 --cov-report xml --cov=wradlib
diff-cover coverage.xml --compare-branch=main
```

### Run notebooks automatically and manually
```bash
python -m pytest -n auto docs/notebooks/
python -m jupyter notebook docs/render/composition/max_reflectivity.ipynb
```

### Build the documentation
```bash
python -m sphinx build -j auto -v -b html docs/ doc-build
```

### Create a branch
```bash
git checkout main
git checkout -b my-branch
python -m pre_commit run
git commit -a -m "describe your changes as in git log"
git push origin my-branch
```

### Update your branch
```bash
git checkout my-branch
git rebase main
python -m pytest --testmon
python -m pre_commit run
git commit -a --amend --date "$(date)"
git push origin my-branch -f
```

### Submit your branch
- Create a pull request.
- Mark as draft.
- Run all checks.
- Mark as ready for a review.

### Resubmit your branch
- Mark as draft.
- Answer reviewer comments.
- Combine updates in a new commit.
- Keep the final commit message.
- Mark as ready for a review.
