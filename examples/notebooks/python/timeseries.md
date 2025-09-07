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

# Dealing with time series


Dealing with radar data typically means implies dealing with time series (of radar records or rain gauge observations). This article gives a brief intro on how to deal with times series and datetimes in Python.


## The datetime module

The datetime module provides a number of types to deal with dates, times, and time intervals.

```{code-cell} python
import datetime as dt
```

There are different ways to create datetime objects.

```{code-cell} python
# This is now (system time)
now = dt.datetime.now()
# Just using the date
birth_van_rossum = dt.datetime(1956, 1, 31)
# Providing both date and time
first_wradlib_commit = dt.datetime(2011, 10, 26, 11, 54, 58)
# Or initialising from a string
erad_2016_begins = dt.datetime.strptime("2016-10-09 09:00:00", "%Y-%m-%d %H:%M:%S")
```

You can compute the difference between two datetime objects.

```{code-cell} python
# Age of Guido van Rossum
age_van_rossum = now - birth_van_rossum
print("This is a %r object.\n" % type(age_van_rossum))
print("It looks like this: %r" % age_van_rossum)
print(
    "and consists of\n\t%d days,\n\t%d seconds,\n\tand %d microseconds.\n"
    % (age_van_rossum.days, age_van_rossum.seconds, age_van_rossum.microseconds)
)
# Age of wradlib
age_wradlib = now - first_wradlib_commit
# Time until (or since) beginning of ERAD 2016 OSS Short course
from_to_erad2016 = now - erad_2016_begins

print("Guido van Rossum is %d seconds old." % age_van_rossum.total_seconds())
print("wradlib's first commit was %d days ago." % age_wradlib.days)
if from_to_erad2016.total_seconds() < 0:
    print(
        "The ERAD 2016 OSS Short course will start in %d days." % -from_to_erad2016.days
    )
else:
    print(
        "The ERAD 2016 OSS Short course took place %d days ago." % from_to_erad2016.days
    )
```

Or you can create a `datetime.timedelta` object yourself
and add/subtract a time interval from/to a `datetime` object.
You can use any of these keywords: `days, seconds, microseconds, milliseconds, minutes, hours, weeks`,
but `datetime.timedelta` will always represent the result in `days, seconds, microseconds`.

```{code-cell} python
# This is an interval of two minutes
print(dt.timedelta(minutes=1, seconds=60))
# And this is, too
print(dt.timedelta(minutes=2))
now = dt.datetime.now()
print("This is now: %s" % now)
print("This is two minutes before: %s" % (now - dt.timedelta(minutes=2)))
```

The default string format of a `datetime` object corresponds to the [isoformat](https://en.wikipedia.org/wiki/ISO_8601). Using the `strftime` function, however, you can control string formatting yourself. The following example shows this feature together with other features we have learned before. The idea is to loop over time and generate corresponding string representations. We also store the `datetime` objects in a list.

```{code-cell} python
start = dt.datetime(2016, 10, 9)
end = dt.datetime(2016, 10, 14)
interval = dt.timedelta(days=1)
dtimes = []
print("These are the ERAD 2016 conference days (incl. short courses):")
while start <= end:
    print(start.strftime("\t%A, %d. %B %Y"))
    dtimes.append(start)
    start += interval
```

[matplotlib](mplintro) generally understands `datetime` objects and tries to make sense of them in plots.

```{code-cell} python
# Instead of %matplotlib inline
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib inline")
except:
    plt.ion()
import numpy as np
```

```{code-cell} python
# Create some dummy data
level = np.linspace(100, 0, len(dtimes))

# And add a time series plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
plt.plot(dtimes, level, "bo", linestyle="dashed")
plt.xlabel("Day of the conference", fontsize=15)
plt.ylabel("Relative attentiveness (%)", fontsize=15)
plt.title(
    "Development of participants' attentiveness during the conference", fontsize=15
)
plt.tick_params(labelsize=12)
```
