---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.17.3
---

```{raw-cell}
:tags: [hide-cell]
This notebook is part of the wradlib documentation: https://docs.wradlib.org.

Copyright (c) wradlib developers.
Distributed under the MIT License. See LICENSE.txt for more info.
```

# A quick start to Python

In order to use $\omega radlib$, you need to be able to use Python.

This notebook will
- give you a glimpse of the Python language,
- confront you with some Python exercises,
- and refer you to further in-depth courses for learning Python.

## Hello interpreted world

Python is an interpreted language just as e.g. R, Matlab and many others. In contrast to a compiled languages such as C, C++ or Fortran, each command is immediately interpreted by the Python interpreter. In this case, executing the following cell produces an immediate output.

```{code-cell} python
print("Hello interpreted world.")
```

## Writing a program in Python

Instead of entering this command interactively in a Python console (or a jupyter cell, as in this case), you can also write a **Python script** (or, in other words, a Python "program").

- Use a text editor to create a file,
- add the line `print("Hello interpreted world.")`,
- save the file under the name `demoscript.py`

(by convention, Python scripts should have the extension `.py`).

Now execute your script from any shell (or the DOS console on Windows):

`$ python demoscript.py`

Note that this will work only if you open the shell in the *same directory* in which you saved your script. Otherwise, you need to provide the full path like e.g.

*Under Linux/OSX:*

`$ python /home/user/path/to/demoscript.py`

*Under Windows:*

`> python e:\path\to\demoscript.py`

## Important data types

Python knows various standard data types such as
- `integer`,
- `float`,
- `bool`,
- and `str`.

If you define a variable, Python typically assigns the *type* of that variable for you.

```{code-cell} python
x1 = 2
print("x1 is of %r" % type(x1))

x2 = 2.3
print("x2 is of %r" % type(x2))

x3 = True
print("x3 is of %r" % type(x3))

x4 = x1 * x2
print("x4 is of %r" % type(x4))

x5 = "Hello interpreted world."
print("x5 is of %r" % type(x5))
```

As you noticed in the above cell, you can inspect the data type of a variable by using function `type()`. Python has a plethora of data types beyond the standard ones shown above. See for example:

```{code-cell} python
print(type(sum))
print(type(type))
```

## Other data types: collections

Further important data types (and specific to Python) are so called `data collections` (or `containers`), namely

- lists (`list`),
- dictionaries (`dict`),
- and tuples (`tuple`).

### Lists

A list is an ordered collection of objects which may have different types.

```{code-cell} python
# a list
l = [1, "Paris", 3.0]
# a list is "mutable" (i.e. you can change the values of its elements)
l[1] = 2
print(l)
print(l[0:2])
```

## Excursus: indexing in Python

You might have noticed that the index `1` refers to the second element of the list.

This is an important rule for all containers in Python: **indexing begins with 0**

*(which is familiar e.g. to C programmers, and new e.g. to R programmers)*

## Dictionaries

- a `dictionary` is basically an efficient table that maps `keys` to `values`.
- it is an unordered container.

There are two ways to define dictionaries - both are equivalent.

```{code-cell} python
# One way to define a dictionary
d1 = {"band": "C", "altitude": 1000.0}

# Another way to define a dictionary
d2 = dict(band="C", altitude=1000.0)

# Both are equivalent
print("d1 equals d2 is a %r statement " % (d1 == d2))
print("The type of d1 is %r" % type(d1))
```

A `dictionary` can be queried using their keywords. You can also iterate over the keys.

```{code-cell} python
print(d1["band"])
print("Printing all keys and values of d1:")
for key in d1.keys():
    print("\t%s: %r" % (key, d1[key]))
```

## Control flow

Loops or iterations as well as conditional statments are key to data processing workflows.

In this section, we will present the most important syntax for control flow.

### "for" loops

```{code-cell} python
# For loops
for i in range(3):
    print(i)
```

**Indentation in Python**

Indentation is a mandatory element of Python syntax - not just a question of style.

Not using indentation will raise an `IndentationError`.

### "while" loops

```{code-cell} python
# While loops (in case the number of iterations is not known a-priori)
i = 0
while i < 3:
    print(i)
    i += 1
```

### Conditional statements

```{code-cell} python
# Conditional statements
a = 5
if a < 10:
    print("a is less than 10.")
elif a == 10:
    print("a equals 10.")
else:
    print("a is greater than 10.")
```

You can also exit a loop before it is finished or skip parts of a specific iteration.

```{code-cell} python
for i in range(3):
    # Exit the loop already if i equals 1
    if i == 1:
        break
    print(i)
```

```{code-cell} python
for i in range(3):
    # Skip the iteration if i equals 1 (but continue the loop)
    if i == 1:
        continue
    print(i)
```

## Importing functions from modules

For most applications, you will need to use `functions` that are available in `modules`. Just think of a module as a collection (or library) of functions. Some are shipped with the Python interpreter ([Python Standard Library](https://docs.python.org/library/)), others (such as wradlib) need to be installed (e.g. from the [Python Package Index](https://pypi.org)) or using the [Anaconda](https://www.anaconda.com/products/individual) Package Manager (`conda`).

In order to use a function from a module, you need to `import` the module. See e.g. how to import the `os` module and use its `getcwd` function (which retruns the path to the current working directory):

```{code-cell} python
import os

mycwd = os.getcwd()
print(mycwd)
```

## Getting help

- In most environments, you can use the *TAB* key for code completion and for listing functions available in a module.

- Use the following cell, enter `os.` and hit the *TAB* key. You can narrow down available functions by typing further characters.

- Once you've selected (or typed) a function name, *jupyter* will show you the help content after hitting *TAB+Shift*.

- Alternatively, you can get help for a function in any Python environment by calling `help` (e.g. `help(os.getcwd)`). In juypter, you can also use the following syntax: `os.getcwd?`

```{code-cell} python
```

## Writing your own functions

In order to organise applications and write reusable code, you will need to write funcions on you own.

This is how you define and call a function (notice again the **indentation**):

```{code-cell} python
# Define the function...
def demofunction(a, b):
    c = a + b
    return c
```

```{code-cell} python
# ...and call the function
demofunction(2, 3)
```

Using `docstrings`, you can document your function as you write it.

```{code-cell} python
def demofunction(a, b):
    """This function computes the sum of a and b.

    Parameters
    ----------
    a : numeric
    b : numeric

    Returns
    -------
    output : numeric (the sum of a and b)

    """
    c = a + b
    return c
```

Now try calling `help(demofunction)`.

```{code-cell} python
```

This should be the output:

```
Help on function demofunction in module __main__:

demofunction(a, b)
    This function computes the sum of a and b

    Parameters
    ----------
    a : numeric
    b : numeric

    Returns
    -------
    output : numeric (the sum of a and b)
```

Functions can have keyword arguments with default values. In the following example, we set a default value for parameter `b`.

```{code-cell} python
def demofunction(a, b=2):
    """Adds 2 to a by default."""
    c = a + b
    return c
```

```{code-cell} python
print(demofunction(3))
print(demofunction(3, b=3))
```

## File input and output

This is the basic way to read from and write to text and binary files (there are a lot of other file formats which have specific interfaces - particularly in the radar world: see [file formats supported by wradlib](../fileio/fileio.ipynb)).

```{code-cell} python
# Open a file for writing
f = open("testfile.txt", "w")  # opens the workfile file
print(type(f))
f.write("This is a test line,\nand another test line.\n")
f.close()
```

```{code-cell} python
# Open a file for reading
f = open("testfile.txt", "r")
s = f.read()
print(s)
f.close()
```

A safer way to handle files is to use Python's `with` statement which makes sure that the file is properly closed even if your code breaks inbetween opening and closing the file.

```{code-cell} python
with open("testfile.txt") as f:
    s = f.read()
    print(s)
```

## The building blocks of scientific Python

With this, we conclude the basic Python intro.

The following sections introduce, we introduce the most important packages for scientific computing in Python. Unlike Matlab, Scilab or R, Python does not come with a pre-bundled set
of modules for scientific computing. Instead, a scientific computing environment can be obtained by combining these building blocks:

* **Python**, a generic and modern computing language, including standard data types and collections, flow control, modules of the standard library, and development tools (automatic testing, documentation generation)

* **IPython**: an interactive **Python shell**, see our article on [how to use notebooks](../../jupyter.md).

* **Numpy**: provides powerful **numerical arrays** objects, and routines to
  manipulate them - see our [NumPy intro article](numpyintro.ipynb)

* **Matplotlib** : visualization and "publication-ready" plots, see related intro [here](mplintro.ipynb)

* **Scipy**: high-level data processing routines - e.g. optimization, regression, interpolation, etc (https://www.scipy.org)

Please also check out further in-depth courses such as
- the [SciPy Lecture Notes](https://scipy-lectures.org/index.html) or
- the [Dive into Python](https://en.wikipedia.org/wiki/Mark_Pilgrim#Dive_into_Python) book.
