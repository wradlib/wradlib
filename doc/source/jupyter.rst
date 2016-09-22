How to use the tutorials and examples?
======================================

Examples: your entry point to wradlib
------------------------------------

The documentation of :math:`\omega radlib` basically consists of two elements: the first element is the :doc:`library reference <reference>` which provides a systematic documentation of each function in wradlib's API. In order to use the library reference properly, you should have an idea what you're actually looking for. This is why new users mostly start with the second element, the :doc:`tutorials and examples <notebooks>`. Here you can look for application examples that are close to what you actually want to achieve, and than use these as a basis for further development. Once you understood an example, you also know what to look for in the :doc:`library reference <reference>`.   


Interactive examples with jupyter notebooks
-------------------------------------------

All :math:`\omega radlib` examples are distributed as `jupyter notebooks <http://jupyter.org/>`_. This way, you can interactively explore various   topics. Each notebook is organized in documented code blocks. You can browse through a notebook block by block, inspect the results, and experiment with the code. However, you can also view the rendered notebooks including the example results on the web pages of this section: Each page was directly generated from :math:`\omega radlib`'s example notebooks. This way, you can copy&paste code snippets directly into your applications.

.. note:: Are you using :math:`\omega radlib` on the `Open Virtual Machine for Cross-Platform Weather Radar Science <http://openradar.github.io/>`_? Then skip the rest: just log in, run the command ``$ ./start_notebook.sh`` and direct your browser `here <http://127.0.0.1:8888/tree>`_. That's it. 

Otherwise, you need to make sure to

- :ref:`get the actual notebook files <ref_get_notebooks>`,
- :ref:`get the example data <ref_get_data>`,
- and :ref:`install jupyter <ref_get_jupyter>`

Once these conditions are met, you can open a notebook: Launch a console or bash window in the directory that contains the example notebooks. If you installed :math:`\omega radlib` (e.g. via ``conda install wradlib``) into Anaconda's default (root) environment, you just need to execute::

	> jupyter notebook
	
If you installed :math:`\omega radlib` into a conda environment (as recommended :doc:`here <gettingstarted>`), 
you need to activate that environment first. Let's say you named the environment `wradlib`::
	
	> activate wradlib
	> jupyter notebook
	
Did you forget the name of the environment that contains wradlib? This way, you get an overview over all environments:: 

	> conda info --envs
	
which will give you something like (example output under Windows)::

	# conda environments
	wradlib      C:\Anaconda2\envs\wradlib
	root      *  C:\Anaconda2
	
In both cases, a browser window will open (typically at http://localhost:8888/tree) which will show the tree of the directory in which you started the jupyter notebook server. Just open any notebook by clicking. Code cells are executed by hitting ``Shift + Enter`` or by using the toolbar icons. It's pretty much self-explaining, and you'll soon get the hang of it.  


.. _ref_get_notebooks:

How can I get the example notebooks?
------------------------------------

The notebooks are distributed with each :math:`\omega radlib` release on `PyPI <https://pypi.python.org/pypi/wradlib>`_ (beginning with version 0.9.0). Alternatively, you can `download <https://github.com/wradlib/wradlib/archive/master.zip>`_ the latest ("bleeding edge") notebooks from our `GitHub repository <https://github.com/wradlib/wradlib>`_. In both cases, you find everything in the directory ``/notebooks`` (after extracting the downloaded archives).


.. _ref_get_data:

How can I get the example data?
-------------------------------

Most notebooks use example data. These data are provided in a separate repository and you need to "install" them manually. *"Installing"* actually means: Download the data archive `here <https://github.com/wradlib/wradlib-data/archive/master.zip>`_ and extract it into any arbitrary directory. Now you need to set an environment variable pointing to that directory:

**Under Windows**, open a console and execute ``setx WRADLIB_DATA <full path to wradlib-data>``.

**Under Linux**, you need to set the env variable in your current shell profile. In most cases, this will be loaded from ``~/.bashrc``. Open that file in an editor (e.g. via ``$EDITOR ~/.bashrc``) and add the following line at the end of the file::

	export WRADLIB_DATA=/insert/full/path/to/wradlib-data/

After this procedure, the example notebooks will automagically pull the required files from the data archive.
	

.. _ref_get_jupyter:
	
How to install jupyter?
-----------------------

As already pointed out above, you can just look at the rendered notebooks :doc:`online docs <notebooks>`. In order to use them interactively, you need to install ``jupyter``. ``jupyter`` is shipped with `Anaconda's <https://www.continuum.io/downloads>`_ distribution by default. If you installed wradlib into Anaconda's *default (root) environment*, you are already done. If you installed wradlib in a seperate *virtual environment* (as recommended :doc:`here <gettingstarted>`), you need to install ``jupyter`` in that virtual environment, too::

	Under Windows:
	> activate wradlib
	[wradlib] > conda install jupyter
	
	Under Linux/OSX:
	$ source activate wradlib
	[wradlib] $ conda install jupyter

If you are not sure which conda environments you have, you can check via ``conda info --envs``. 

If you did not install :math:`\omega radlib` on top of Anaconda, you should first check whether ``jupyter`` might already be available on your system (use e.g. ``jupyter --version``). If ``jupyter`` is not available, you should check out the `jupyter docs <http://jupyter.readthedocs.io/en/latest/install.html>`_ for alternative installation options.  


I prefer simple Python scripts instead of notebooks 
---------------------------------------------------

No problem. Each :math:`\omega radlib` release from PyPI not only distributes the notebooks, but also the corresponding Python scripts. Just look for ``.py`` files in the ``notebooks`` directory. If you downloaded the notebooks directly from the wradlib repository, you can easily convert them to Python scripts yourself (but you need to :ref:`install jupyter <ref_get_jupyter>` to do the conversion)::

	$ jupyter nbconvert --to script <name of the notebook>
	