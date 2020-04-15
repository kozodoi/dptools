# dptools: data preprocessing functions for Python

---

[![PyPI Latest Release](https://img.shields.io/pypi/v/dptools.svg)](https://pypi.org/project/dptools/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)[![License](https://img.shields.io/pypi/l/pandas.svg)](https://github.com/pandas-dev/pandas/blob/master/LICENSE)
[![Licence](https://img.shields.io/github/license/mashape/apistatus.svg)](http://choosealicense.com/licenses/mit/)
[![Downloads](https://img.shields.io/pypi/dm/pandas)](https://pypi.org/project/dptools/)

---

## Overview

The `dptools` python package provides helper functions to simplify common data preprocessing tasks, including feature engineering, working with missing values, aggregating data and more.

## Installation

The source code is currently hosted on GitHub at: https://github.com/kozodoi/dptools

You can install the package by running:

```
pip install dptools
```

## Dependencies

- [numpy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org/)
- [sklearn](https://scikit-learn.org)

## Features

The package currently features the following functions:
- `add_date_features()`: adds date-based features
- `add_text_features()`: adds basic text-based features 
- `aggregate_data()`: aggregates data by a certain feature
- `fill_missings()`: replaces missings with specific values
- `print_missings()`: counts missing values and prints the results
- `label_encoding()`: performs label encoding on partitioned data
- `print_factor_levels()`: prints levels of categorical features

## Examples

TBA

## Issues, questions

In case you need help or advice on the included data preprocseeing functions or you want to report an issue, please do so in a reproducible example at the corresponding [GitHub page](https://github.com/kozodoi/dptools/issues).