
# Contributing

This is a practical guide for contributing. Please check the [documentation](docs/dev_guide.md) for more general information.

## 1a Install environement using conda
- Install miniconda3 on Windows and start Anaconda Powershell Prompt:
``` powershell
winget install --id=Anaconda.Miniconda3 -e --silent
exit
```
- Create development environement (use "wradlib-dev" to keep wradlib for release):
```powershell
conda config --add channels conda-forge
conda create -n wradlib
conda activate wradlib
```
- Install wradlib package:
```bash
conda install wradlib
```
- Install gdal optional packages:
```bash
conda install libgdal-hdf5
conda install libgdal-netcdf
```

## 1b Install environement using pip on Windows

- Install latest version of python
``` powershell
winget install -e --id Python.Python.3.13
exit
```
- Create virtual environement:
``` powershell
python -m venv $env:USERPROFILE/.venvs/wradlib
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
New-Item -Path $PROFILE -ItemType File -Force
'Set-Alias wradlib "$env:UserProfile\.venvs\wradlib\Scripts\activate.ps1"' | Out-File -Append $PROFILE
exit
```
- Activate and deactivate virtual environement:
``` powershell
wradlib
deactivate
```

- Install build tools
``` powershell
winget install --id Git.Git -e --silent
winget install -e --id Kitware.CMake --silent
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```
- Install gdal using vcpkg
``` powershell
cd $HOME
git clone https://github.com/microsoft/vcpkg.git $VCPKG_PATH
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install gdal[core,geos,hdf5,netcdf]:x64-windows
[System.Environment]::SetEnvironmentVariable("Path", "$env:Path;$HOME\vcpkg\installed\x64-windows\bin", [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable("GDAL_DATA", "$HOME\vcpkg\installed\x64-windows\share\gdal", [System.EnvironmentVariableTarget]::User)
exit
```

- Install gdal python bidings using pip
``` bash
$env:INCLUDE = "$HOME\vcpkg\installed\x64-windows\include"
$env:LIB = "$HOME\vcpkg\installed\x64-windows\lib"
wradlib
pip install numpy setuptools wheel
pip install gdal
```
## 1c Install development environement on UNIX variants

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

GDAL_VERSION=3.10.3

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

## 2 Install wradlib from repository

- Activate virtual environement

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

## 3 Configure local repository for development

- Configure automatic SSH on windows:
```powershell
Start-Service ssh-agent  
Set-Service -Name ssh-agent -StartupType Automatic
git config --global core.sshCommand "C:\\Windows\\System32\\OpenSSH\\ssh.exe"
```

- Configure automatic SSH on linux:
```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ssh-agent.service <<EOF
[Unit]
Description=SSH Agent
After=network.target

[Service]
ExecStart=/usr/bin/ssh-agent -D
Restart=always

[Install]
WantedBy=default.target
EOF

systemctl --user enable ssh-agent
systemctl --user start ssh-agent
```

- Generate SSH key and add it
```bash
ssh-keygen -t ed25519
ssh-add $HOME/.ssh/id_ed25519
```

- Copy the public key to your github account
``` powershell
cat $HOME/.ssh/id_ed25519.pub
```

- Configure git user
```bash
git config --global user.name "Edouard Goudenhoofdt"
git config --global user.email "egouden@outlook.com"
```

- Configure fork with ssh and set upstream
```bash
git remote remove origin
git remote add origin git@github.com:egouden/wradlib.git
git remote add upstream https://github.com/wradlib/wradlib
```

## 4 Work on your changes

- Update your fork
``` bash
git checkout main
git fetch upstream main
git rebase upstream/main
git push origin main
```
- Create a new branch
```bash
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