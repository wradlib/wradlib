# Jupyter Notebooks

## Examples: your entry point to wradlib

The documentation of {{wradlib}} basically consists of two elements: the first element is the {doc}`library reference <reference>` which provides a systematic documentation of each function in wradlib's API. In order to use the library reference properly, you should have an idea what you're actually looking for. This is why new users mostly start with the second element, the [wradlib-notebooks](https://github.com/wradlib/wradlib-notebooks). Here you can look for application examples that are close to what you actually want to achieve, and then use these as a basis for further development. Once you understood an example, you also know what to look for in the {doc}`library reference <reference>`.


## Interactive examples with jupyter notebooks

All {{wradlib}} examples are distributed as [jupyter notebooks](https://jupyter.org) in the [wradlib-notebooks](https://github.com/wradlib/wradlib-notebooks) repository. This way, you can interactively explore various topics. Each notebook is organized in documented code blocks. You can browse through a notebook block by block, inspect the results, and experiment with the code. However, you can also view the rendered notebooks including the example results on the web pages of this section: Each page was directly generated from {{wradlib}}'s example notebooks. This way, you can copy&paste code snippets directly into your applications.

:::{note}
Are you using {{wradlib}} on the [Open Radar Science Shortcourse](https://openradarscience.org/erad2022/)? Then please follow-up there to get your openradar software stack running.

You might also fire-up a jupyter notebook server with a [wradlib docker container](docker.md#jupyter-notebook-server).
:::

Otherwise, you need to make sure to

- {ref}`get the actual notebook files <ref_get_notebooks>`,
- {ref}`get the example data <ref_get_data>`,
- and {ref}`install jupyter <ref_get_jupyter>`

Once these conditions are met, you can open a notebook: Launch a console or bash window in the directory that contains the example notebooks.

If you installed {{wradlib}} into a conda environment (as recommended in {doc}`installation`),
you need to activate that environment first. Let's say you named the environment `wradlib`:

```bash
(base) $ conda activate wradlib
(wradlib) $ jupyter notebook
```

Did you forget the name of the environment that contains wradlib? This way, you get an overview over all environments:

```bash
(base) $ conda info --envs
```

which will give you something like (example output under Windows):

```
# conda environments:
#
base                     /home/kai/miniconda
wradlib                  /home/kai/miniconda/envs/wradlib
```

In both cases, a browser window will open (typically at [http://localhost:8888/tree](http://localhost:8888/tree)) which will show the tree of the directory in which you started the jupyter notebook server. Just open any notebook by clicking. Code cells are executed by hitting ``Shift + Enter`` or by using the toolbar icons. It's pretty much self-explaining, and you'll soon get the hang of it.

(ref_get_notebooks)=
## How can I get the example notebooks?

The notebooks are available in the [wradlib-notebooks](https://github.com/wradlib/wradlib-notebooks) repository. Alternatively, you can [download](https://codeload.github.com/wradlib/wradlib-notebooks/zip/main) the latest ("bleeding edge") notebooks. In both cases, you find everything in the directory ``/notebooks`` (after extracting the downloaded archives).


(ref_get_data)=
## How can I get the example data?

Most notebooks use example data. These data are provided in a separate repository and can be fetched using ``pooch``. All you need to do is install this via pip manually:

```bash
  $ python -m pip install git+https://github.com/wradlib/wradlib-data@pooch
```

Now you need to set an environment variable pointing to a folder where the data should be saved when requested:

- **Under Windows**

  open a console and execute ``setx WRADLIB_DATA <full path to wradlib-data>``.

- **Under Linux**

  You need to set the env variable in your current shell profile. In most cases, this will be loaded from ``~/.bashrc``. Open that file in an editor (e.g. via ``$EDITOR ~/.bashrc``) and add the following line at the end of the file:
  ```bash
  $ export WRADLIB_DATA=/insert/full/path/to/wradlib-data/
  ```

After this procedure, the example notebooks will automagically pull the required files from the data archive.

(ref_get_jupyter)=
## How to install jupyter?

As already pointed out above, you can just look at the rendered notebooks [online docs](notebooks/overview.ipynb). In order to use them interactively, you need to install ``jupyter``. ``jupyter`` can be installed conveniently into conda environments. If you installed {{wradlib}} in a separate *conda environment* (as recommended in {doc}`installation`), you need to install ``jupyter`` in that environment, too:

```bash
(base) $ conda activate wradlib
(wradlib) $ conda install jupyter
```

If you are not sure which conda environments you have, you can check via ``conda info --envs``.

If you did not install {{wradlib}} via conda, you should first check whether ``jupyter`` might already be available on your system (use e.g. ``jupyter --version``). If ``jupyter`` is not available, you should check out the [jupyter docs](https://docs.jupyter.org/en/latest/install.html) for alternative installation options.

## I prefer simple Python scripts instead of notebooks

No problem. If you downloaded the notebooks directly from the wradlib repository, you can easily convert them to Python scripts yourself (but you need to {ref}`install jupyter <ref_get_jupyter>` to do the conversion):

```bash
$ jupyter nbconvert --to script <name of the notebook>
```
