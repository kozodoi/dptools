# dptools: data preprocessing functions for Python

---

[![PyPI Latest Release](https://img.shields.io/pypi/v/dptools.svg)](https://pypi.org/project/dptools/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://pypi.org/project/dptools/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Licence](https://img.shields.io/github/license/mashape/apistatus.svg)](http://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.org/kozodoi/dptools.svg?branch=master)](https://travis-ci.com/kozodoi/dptools)
[![Downloads](https://img.shields.io/pypi/dm/dptools)](https://pypi.org/project/dptools/)

---

## Overview

The `dptools` Python package provides helper functions to simplify common data processing tasks in a data science pipeline, including feature engineering, data aggregation, working with missing values and more.

The package currently encompasses the following functions:
- Feature engineering:
    - `add_date_features()`: create date and time-based features
    - `add_text_features()`: create text-based features (including counts and TF-IDF)
    - `aggregate_data()`: aggregate data and create features based on aggregated statistics
    - `encode_factors()`: perform label or dummy encoding of categorical features
- Data processing:
    - `split_nested_features()`: split features nested in a single column
    - `fill_missings()`: replace missings with specific values
    - `correct_colnames()`: correct column names to be unique and remove foreign symbols
    - `print_missings()`: print information on features with missing values
    - `print_factor_levels()`: print levels of categorical features
- Data cleaning:
    - `find_correlated_features()`: identify features with a high pairwise correlation
    - `find_constant_features()`: identify features with a single unique value
- Import and versioning:
    - `read_csv_with_json()`: read CSV where some columns are in JSON format
    - `save_csv_version()`: save CSV with an automatically assigned version to prevent overwriting


## Installation

The latest stable release of `dptools` can be installed from PyPI:
```
pip install dptools
```

You may also install the development version from Github:
```
pip install git+https://github.com/kozodoi/dptools.git
```

After the installation, you can import the included functions:
```py
from dptools import *
```


## Examples

This section contains a few examples of using functions from `dptools` for different data preprocessing tasks. Please refer to the docstring documentation in the implemented functions for further examples.


### Creating a toy data set

First, let us create a toy data frame to demonstrate the package functionality.

```py
# import dependencies
import pandas as pd
import numpy as np

# create data frame
data = {'age': [27, np.nan, 30, 25, np.nan],
        'height': [170, 168, 173, 177, 165],
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
df = pd.DataFrame(data)
```
| age | height | gender | income |
|---:| ---:| ---:| ---:|   
| 27.0 | 170 | female | high |
| NaN | 168 | male | medium |
| 30.0 | 173 | NaN | low |
| 25.0 | 177 | male | low |
| NaN | 165 | female | no income |


### Aggregating features

```py
# aggregating the data
from dptools import aggregate_data
df_new = aggregate_data(df, group_var = 'gender', num_stats = ['mean', 'max'], fac_stats = 'mode')   
```
| gender | age_mean | age_max | height_mean | height_max | income_mode |
|---:| ---:| ---:| ---:| ---:| ---:|    
| female | 27.0 | 27.0 | 167.5 | 170 | 'high' |
| male | 25.0 | 25.0 | 172.5 | 177 | 'low' |


### Creating text-based features

```py
# creating text-based features
from dptools import add_text_features
df_new = add_text_features(df, text_vars = 'income')
```
| age | height | gender | income_word_count | income_char_count |  income_tfidf_0 | ... | income_tfidf_3 |
|---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| 27.0 | 170 | female | 1 | 4 | 1.0 | ... | 0.0 |
| NaN | 168 | male | 1 | 6 | 0.0 | ... | 1.0 |
| 30.0 | 173 | NaN | 1 | 3 | 0.0 | ... | 0.0 |
| 25.0 | 177 | male | 1 | 3 | 0.0 | ... | 0.0 |
| NaN | 165 | female | 2 | 9 | 0.0 | ... | 0.0 |


### Working with missings

```py
# print statistics on missing values
from dptools import print_missings
print_missings(df)
```
| | Total | Percent |
|---:| ---:| ---:|
| age | 2 | 0.4 |
| gender | 1 | 0.2 |


### Finding correlated features

```py
# displays one correlated feature from each pair
from dptools import find_correlated_features
feats = find_correlated_features(df, cutoff = 0.4, method = 'spearman')
feats
```
> Found 1 correlated features.

> ['age']

### Data versioning

```py
# first call saves df as 'data_v1.csv'
from dptools import save_csv_version
save_csv_version('data.csv', df, index = False)

# second call saves df as 'data_v2.csv' as data_v1.csv already exists
save_csv_version('data.csv', df, index = False)
```


## Dependencies

Installation requires Python 3.7+ and the following packages:
- [numpy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org)
- [sklearn](https://scikit-learn.org)
- [scipy](https://scipy.org)


## Feedback

In case you need help on the included data preprocessing functions or you want to report an issue, please do so at the corresponding [GitHub page](https://github.com/kozodoi/dptools/issues).
