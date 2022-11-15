# -*- coding: utf-8 -*-
#
# wradlib documentation build configuration file, created by
# sphinx-quickstart on Wed Oct 26 13:48:08 2011.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import datetime as dt
import glob
import os
import subprocess
import warnings

from pybtex.plugin import register_plugin  # noqa
from pybtex.style.formatting.unsrt import Style  # noqa
from pybtex.style.labels.alpha import LabelStyle, _strip_nonalnum  # noqa

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = [
    "substitution",
    "dollarmath",
    "amsmath",
    "colon_fence",
]

myst_heading_anchors = 4


extlinks = {
    "issue": ("https://github.com/wradlib/wradlib/issues/%s", "GH%s"),
    "pull": ("https://github.com/wradlib/wradlib/pull/%s", "PR%s"),
    "at": ("https://github.com/%s", "@%s"),
}

# mathjax_path = ("https://cdn.mathjax.org/mathjax/latest/MathJax.js?"
#                "config=TeX-AMS-MML_HTMLorMML")

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The main toctree document.
master_doc = "index"

# General information about the project.
project = "wradlib"
copyright = "2011-2023, wradlib developers"
author = "Wradlib Community"
url = "https://github.com/wradlib"

# check readthedocs
on_rtd = os.environ.get("READTHEDOCS") == "True"

# processing on readthedocs
if on_rtd:
    # rtd_version will be either 'latest', 'stable' or some tag name which
    # should correspond to the wradlib and wradlib-docs tag
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")
    rtd_version_name = os.environ.get("READTHEDOCS_VERSION_NAME", "latest")
    rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE", "latest")

    print("RTD Version: {}".format(rtd_version))
    print("RTD Version Name: {}".format(rtd_version_name))
    print("RTD Version Type: {}".format(rtd_version_type))

    # latest wradlib commit
    if rtd_version == "latest" or rtd_version_type == "external":
        wradlib_notebooks_branch = "devel"
        wradlib_branch_or_tag = "main"
    # latest tagged commit
    elif rtd_version == "stable":
        tag = subprocess.check_output(
            ["git", "describe", "--abbrev=0", "--tags"], universal_newlines=True
        ).strip()
        wradlib_notebooks_branch = tag
        wradlib_branch_or_tag = tag
    # rely on rtd_version (wradlib-docs tag)
    else:
        wradlib_notebooks_branch = rtd_version
        wradlib_branch_or_tag = rtd_version

    # clone wradlib-notebooks target branch
    repourl = "{0}/wradlib-notebooks.git".format(url)
    reponame = "wradlib-notebooks"
    # first remove any possible left overs
    subprocess.check_call(["rm", "-rf", "wradlib-notebooks"])
    subprocess.check_call(["rm", "-rf", "notebooks"])
    subprocess.check_call(
        ["git", "clone", "-b", wradlib_notebooks_branch, repourl, reponame]
    )
    branch = "origin/{}".format(wradlib_branch_or_tag)
    nb = (
        subprocess.check_output(
            ["git", "--git-dir=wradlib-notebooks/.git", "rev-parse", branch]
        )
        .decode()
        .strip()
    )
    subprocess.check_call(["mv", "wradlib-notebooks/notebooks", "."])
    subprocess.check_call(["rm", "-rf", "wradlib-notebooks"])

    # correct notebook doc-links
    if rtd_version != "latest":
        baseurl = "https://docs.wradlib.org/en/{}"
        search = baseurl.format("latest")
        replace = baseurl.format(rtd_version)
        subprocess.check_call(
            [
                "find notebooks -type f -name '*.ipynb' -exec "
                "sed -i 's%{search}%{replace}%g' {{}} +"
                "".format(search=search, replace=replace)
            ],
            shell=True,
        )

    # install checked out wradlib
    subprocess.check_call(["python", "-m", "pip", "install", "--no-deps", "../."])

# get wradlib modules and create automodule rst-files
import types

# get wradlib version
import wradlib  # noqa

modules = []
for k, v in wradlib.__dict__.items():
    if isinstance(v, types.ModuleType):
        if k not in ["_warnings", "version"]:
            modules.append(k)
            file = open("{0}.rst".format(k), mode="w")
            file.write(".. automodule:: wradlib.{}\n".format(k))
            file.close()

# create API/Library reference md-file
reference = """# Library Reference

This page contains an auto-generated summary of {{wradlib}}'s API. Every submodule is
documented separately.

```{toctree}
:maxdepth: 2
"""

file = open("reference.md", mode="w")
file.write("{}\n".format(reference))
for mod in sorted(modules):
    file.write("{}\n".format(mod))
file.write("```")
file.close()

# get all rst files, do it manually
rst_files = glob.glob("*.rst")
autosummary_generate = rst_files
autoclass_content = "both"

# get version from metadata
from importlib.metadata import version

version = version("wradlib")
release = version

print("RELEASE, VERSION", release, version)

myst_substitutions = {
    "today": dt.datetime.utcnow().strftime("%Y-%m-%d"),
    "release": release,
    "wradlib": "$\\omega radlib$",
}

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# exclude_patterns = []
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "links.rst",
    "**.ipynb_checkpoints",
]

# -- nbsphinx specifics --
# do not execute notebooks ever while building docs
nbsphinx_execute = "never"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# linkcheck options
linkcheck_ignore = ["https://data.apps.fao.org/map/catalog/srv/eng/catalog.search#"]

# -- Options for HTML output --------------------------------------------------


# Tell the theme where the code lives
# adapted from https://github.com/vispy/vispy
def _custom_edit_url(
    github_user,
    github_repo,
    github_version,
    docpath,
    filename,
    default_edit_page_url_template,
):
    """Create custom 'edit' URLs for API modules since they are dynamically generated."""
    if ".rst" in filename:
        if filename.startswith("generated/"):
            # this is a dynamically generated API page, link to actual Python source
            modpath = os.sep.join(
                os.path.splitext(filename)[0].split("/")[-1].split(".")[:-1]
            )
        else:
            modpath = os.path.join("wradlib", os.path.splitext(filename)[0])
        if modpath == "modules":
            # main package listing
            modpath = "wradlib"
        rel_modpath = os.path.join("..", modpath)
        if os.path.isdir(rel_modpath):
            docpath = modpath + "/"
            filename = "__init__.py"
        elif os.path.isfile(rel_modpath + ".py"):
            docpath = os.path.dirname(modpath)
            filename = os.path.basename(modpath) + ".py"
        else:
            warnings.warn(f"Not sure how to generate the API URL for: {filename}")
    return default_edit_page_url_template.format(
        github_user=github_user,
        github_repo=github_repo,
        github_version=github_version,
        docpath=docpath,
        filename=filename,
    )


html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "wradlib",
    "github_repo": "wradlib",
    "github_version": "main",
    "doc_path": "doc",
    "edit_page_url_template": "{{ wradlib_custom_edit_url(github_user, github_repo, github_version, doc_path, file_name, default_edit_page_url_template) }}",
    "default_edit_page_url_template": "https://github.com/{github_user}/{github_repo}/edit/{github_version}/{docpath}/{filename}",
    "wradlib_custom_edit_url": _custom_edit_url,
}

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "repository_url": "https://github.com/wradlib/wradlib",
    "github_url": "https://github.com/wradlib/wradlib",
    "external_links": [
        {
            "url": "https://github.com/wradlib/wradlib-notebooks/blob/main/notebooks/overview.ipynb",
            "name": "Gallery",
        },
    ],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/wradlib",
            "icon": "fas fa-box",
        },
        {
            "name": "Openradar Discourse",
            "url": "https://openradar.discourse.group",
            "icon": "fab fa-discourse",
        },
        {
            "type": "local",
            "name": "OpenRadarScience",
            "url": "https://openradarscience.org",
            "icon": "https://raw.githubusercontent.com/openradar/openradar.github.io/main/_static/openradar_micro.svg",
        },
    ],
    "navbar_align": "right",
    "navbar_end": ["icon-links.html"],
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "repository_branch": "main",
    "path_to_docs": "doc",
}

html_theme = "sphinx_book_theme"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = project

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/wradlib_logo_small.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/favicon.ico"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "gdal": ("https://gdal.org/", None),
}

# -- Napoleon settings for docstring processing -------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "scalar": ":term:`scalar`",
    "sequence": ":term:`sequence`",
    "callable": ":py:func:`callable`",
    "file-like": ":term:`file-like <file-like object>`",
    "array-like": ":term:`array-like <array_like>`",
    "Path": "~~pathlib.Path",
}

bibtex_bibfiles = ["refs.bib", "refs_links.bib"]


class WradlibLabelStyle(LabelStyle):
    # copy from AlphaStyle
    def format_label(self, entry):
        if entry.type == "book" or entry.type == "inbook":
            label = self.author_editor_key_label(entry)
        elif entry.type == "proceedings":
            label = self.editor_key_organization_label(entry)
        elif entry.type == "manual":
            label = self.author_key_organization_label(entry)
        else:
            label = self.author_key_label(entry)
        # add full year comma separated
        if "year" in entry.fields:
            return "{0}, {1}".format(label, entry.fields["year"])
        else:
            return label

    def format_lab_names(self, persons):
        numnames = len(persons)
        person = persons[0]
        result = _strip_nonalnum(person.prelast_names + person.last_names)
        if numnames > 1:
            result += " et al."
        return result


class WradlibStyle(Style):
    default_label_style = "wrl"


register_plugin("pybtex.style.labels", "wrl", WradlibLabelStyle)
register_plugin("pybtex.style.formatting", "wrlstyle", WradlibStyle)

rst_epilog = ""
