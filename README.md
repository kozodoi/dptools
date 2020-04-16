# dptools: data preprocessing functions for Python

---

[![PyPI Latest Release](https://img.shields.io/pypi/v/dptools.svg)](https://pypi.org/project/dptools/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Licence](https://img.shields.io/github/license/mashape/apistatus.svg)](http://choosealicense.com/licenses/mit/)
[![Downloads](https://img.shields.io/pypi/dm/dptools)](https://pypi.org/project/dptools/)

---

## Overview

The `dptools` python package provides helper functions to simplify common data preprocessing tasks, including feature engineering, working with missing values, aggregating data and more. 

The package currently encompasses the following functions:
- Feature engineering:
    - `add_date_features()`: adds date-based features
    - `add_text_features()`: adds text-based features 
    - `aggregate_data()`: adds aggregation-based features
    - `split_features()`: splits features nested in a single column
- Data processing:
    - `find_constant_features()`: finds features with a single unique value
    - `print_factor_levels()`: prints levels of categorical features
    - `label_encoding()`: performs label encoding on partitioned data
- Working with missings:
    - `print_missings()`: counts missing values and prints the results
    - `fill_missings()`: replaces missings with specific values
- Version control:
    - `save_csv_version()`: saves CSV with an automatically assigned version number to prevent overwriting



## Installation

The latest stable release can be installed from PyPI:
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

This section contains a few examples of using functions from `dptools` for different data preprocessing tasks. Please refer to function docstring documentation for further examples.


### Creating a toy data set

First, let us create a toy data frame to demonstarte the package functionality.

```py
# import dependecies
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
df_new = aggregate_data(df, group_var = 'gender', num_stats = ['min', 'max'], fac_stats = 'mode')   
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


### Version control

```py
# first call saves df as 'data_v1.csv'
from dptools import save_csv_version
save_csv_version('data.csv', df, index = False)

# second call saves df as 'data_v2.csv' as data_v1.csv already exists
save_csv_version('data.csv', df, index = False)
```


## Dependencies

Installation requires Python 3.6+ and the following packages:
- [numpy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org)
- [sklearn](https://scikit-learn.org)
- [scipy](https://scipy.org)


## Feedback

In case you need help on the included data preprocseeing functions or you want to report an issue, please do so at the corresponding [GitHub page](https://github.com/kozodoi/dptools/issues).