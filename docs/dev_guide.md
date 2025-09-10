# Contributing Guide

The wradlib project encourages everyone to contribute to the developement of wradlib. This paper describes the general guidelines for contributing to wradlib. There is more information on [wradlib's community effort](https://wradlib.org/pages/community) at the project's main website [wradlib.org](https://wradlib.org).

In general wradlib uses GitHub Pull Requests for contribution. Second, the [wradlib developers](https://github.com/wradlib/wradlib/blob/main/CONTRIBUTORS.txt) have GitHub Commit Access to the project's main repository.

## Code of Conduct

Contributors to wradlib are expected to act respectfully toward others in accordance with the [OSGeo Code of Conduct](http://www.osgeo.org/code_of_conduct).

## Contributions and Licensing

All contributions shall comply to the project [license](https://github.com/wradlib/wradlib/blob/main/LICENSE.txt).

## GitHub

Source code, testing, bug tracking as well as docs and wiki are handled via GitHub.

## Documentation

* wradlib's documentation source is contained in the [docs-folder](https://github.com/wradlib/wradlib/tree/main/docs) of the wradlib repository, and mainly written in myst markdown.
* Jupyter Notebooks in [examples/notebooks-folder](https://github.com/wradlib/wradlib/tree/main/examples/notebooks) are utilized to extend the documentation with examples and tutorials.
* wradlib uses [Sphinx](http://sphinx-doc.org/) and [myst_nb](https://myst-nb.readthedocs.io/en/latest/) to compile docs on [readthedocs.org](https://readthedocs.org/projects/wradlib/).

## Issue tracking

Using wradlib's [GitHub issue tracker](https://github.com/wradlib/wradlib/issues) users can file bug reports or discuss feature requests. Bug reports should contain
used wradlib as well as Python and OS version, error messages and instructions to reproduce the bug.

## Forking wradlib

For this purpose, you need to have a [GitHub Account](https://github.com/signup/free) and [Git](https://git-scm.com/). Then fork wradlib, make changes and create a pull request.


## Setup

The {doc}`installation` section will provide you with detailed guidance on how to install {{wradlib}} and the required dependencies for different operating systems (MS Windows, Linux, Mac OS). {{wradlib}} {{release}} has been tested with Mambaforge Python. It is available via `conda-forge` channel on linux-64, osx-64 and MS Windows 64 versions.

As a developer, though, you should rather link into {{wradlib}}'s version control. This way, it will be easier for you to track changes and to contribute your changes to {{wradlib}}'s main repository (see next section). Just install [Git](https://git-scm.com), then clone the {{wradlib}} repository to your local system by executing the following command in your shell:

```bash
$ git clone https://github.com/wradlib/wradlib.git
```

## Contributing

Everyone can contribute to the developement of {{wradlib}} by using the Fork and Pull model. For this purpose, you need to set up ``Git`` (see section [Setup](#setup)). Then start a [Pull Request](https://help.github.com/articles/creating-a-pull-request)!

* **Step 1:** [Fork](https://github.com/wradlib/wradlib) your own {{wradlib}} repository from the {{wradlib}} [main repo](https://github.com/wradlib/wradlib).
* **Step 2:** Implement your changes into the forked repository. Test your code.
* **Step 3:** Now you want to feed these changes back into the main {{wradlib}} development branch? Start a [Pull Request](https://help.github.com/articles/creating-a-pull-request)!
* **Step 4:** We will review your changes. On approval, we will merge your fork back into the main {{wradlib}} branch.
* **Step 5:** Now everyone can benefit from your improvements.

A step-by-step tutorial for a pull request can be found in the [GitHub documentation](https://guides.github.com/activities/forking).

Finally, you are welcome to **contribute examples of your own** {{wradlib}} **applications** in [examples/notebooks](https://github.com/wradlib/wradlib/tree/main/examples/notebooks).


## Building documentation

In order to build the documentation you need to satisfy a few more dependencies which are mainly related to ``Sphinx``. These are specified in the [readthedocs_environment.yml](https://github.com/wradlib/wradlib/blob/main/ci/requirements/readthedocs.yml).

Once these requirements are met, you can open a console window within the ``wradlib``-repository and execute ``sphinx-build -v -b html docs/ doc-build``. This will give you the latest documentation under the ``wradlib/doc-build`` directory. Simply open the index.html file to view the documentation.

## Testing

{{wradlib}} uses the [pytest](https://docs.pytest.org/en/stable) framework. New functions should come with corresponding unittests in the ``wradlib/wradlib/tests`` directory. Just have a look at [available tests](https://github.com/wradlib/wradlib/tree/main/wradlib/tests) to get an idea. In addition, examples and docstrings are a good way to combine testing and documentation. In the docstrings, the ``Examples`` section will be tested by our testing framework. This could look like this:

```python
def foo(a):
    """Docstring to be evaluated by doctest

    Examples
    --------
    >>> from wradlib.some_module import foo
    >>> print(foo(3))
    4
    """
    return a + 1
```

## Continuous Integration

We use GitHub Actions for Continuous Integration (CI). CI means, in our case, that each commit pushed to {{wradlib}}'s main repository will trigger different test suites on the CI service. If all tests pass successfully, a new documentation will be built on [https://readthedocs.org](https://readthedocs.org) and published on [https://docs.wradlib.org](https://docs.wradlib.org). In case a new release tag is associated with a commit, a new release will be distributed via [PyPI](https://pypi.org/project/wradlib).
