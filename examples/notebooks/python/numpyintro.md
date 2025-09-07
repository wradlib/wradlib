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

# NumPy: manipulating numerical data

*NumPy* is the key Python package for creating and manipulating (multi-dimensional) numerical arrays. *NumPy* arrays are also the most important data objects in $\omega radlib$. It has become a convention to import *NumPy* as follows:

```{code-cell} python
import numpy as np
```

## Creating and inspecting NumPy arrays

The `ndarray`, a numerical array, is the most important data type in NumPy.

```{code-cell} python
a = np.array([0, 1, 2, 3])
print(a)
print(type(a))
```

Inspect the `shape` (i.e. the number and size of the dimensions of an array).

```{code-cell} python
print(a.shape)
# This creates a 2-dimensional array
a2 = np.array([[0, 1], [2, 3]])
print(a2.shape)
```

There are various ways to create arrays: from lists (as above), using convenience functions, or from file.

```{code-cell} python
# From lists
a = np.array([0, 1, 2, 3])
print("a looks like:\n%r\n" % a)

# Convenience functions
b = np.ones(shape=(2, 3))
print("b looks like:\n%r\nand has shape %r\n" % (b, b.shape))

c = np.zeros(shape=(2, 1))
print("c looks like:\n%r\nand has shape %r\n" % (c, c.shape))

d = np.arange(2, 10)
print("d looks like:\n%r\nand has shape %r\n" % (d, d.shape))

e = np.linspace(0, 10, 5)
print("e looks like:\n%r\nand has shape %r\n" % (e, e.shape))
```

You can change the shape of an array without changing its size.

```{code-cell} python
a = np.arange(10)
b = np.reshape(a, (2, 5))
print("Array a has shape %r.\nArray b has shape %r" % (a.shape, b.shape))
```

## Indexing and slicing

You can index an `ndarray` in the same way as a `list`:

```{code-cell} python
a = np.arange(10)
print(a)
print(a[0], a[2], a[-1])
```

Just follow your intuition for indexing multi-dimensional arrays:

```{code-cell} python
a = np.diag(np.arange(3))
print(a, end="\n\n")

print("Second row, second column: %r\n" % a[1, 1])

# Setting an array item
a[2, 1] = 10  # third line, second column
print(a, end="\n\n")

# Acessing a full row
print("Second row:\n%r" % a[1])
```

Slicing is just a way to access multiple array items at once:

```{code-cell} python
a = np.arange(10)
print(a, end="\n\n")

print("1st:", a[2:9])
print("2nd:", a[2:])
print("3rd:", a[:5])
print("4th:", a[2:9:3])  # [start:end:step]
print("5th:", a[a > 5])  # using a mask
```

Get further info on NumPy arrays [here](https://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing)!
