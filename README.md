# dptools: data preprocessing functions for Python

---

[![PyPI Latest Release](https://img.shields.io/pypi/v/dptools.svg)](https://pypi.org/project/dptools/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Licence](https://img.shields.io/github/license/mashape/apistatus.svg)](http://choosealicense.com/licenses/mit/)
[![Downloads](https://img.shields.io/pypi/dm/dptools)](https://pypi.org/project/dptools/)

---

## Overview

The `dptools` python package provides helper functions to simplify common data preprocessing tasks, including feature engineering, working with missing values, aggregating data and more.


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

```
from dptools import *
```


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

First, let us create a toy data frame to demonstarte functionality of `dptools`.

```
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
|age | height | gender | income |
|---:| ---:| ---:| ---:|   
| 27.0 | 170 | female | high |
| NaN | 170 | female | high |
| 30.0 | 170 | female | high |
| 25.0 | 170 | female | high |
| NaN | 170 | female | high |

Printing statistics on missing values:
```
from dptools import print_missings
print_missings(df)
```

Aggregating the data:
```
from dptools import aggregate_data
df_new = aggregate_data(df, group_var = 'gender', num_stats = ['mean', 'max'])
```

Creating text-based features:
```
from dptools import add_text_features
df_new = add_text_features(df, string_vars = ['income'])
```

## Dependencies

Dptools supports Python 3.6+. 

Installation requires the following packages:
- [numpy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org)
- [sklearn](https://scikit-learn.org)
- [scipy](https://scipy.org)


## Feedback

In case you need help on the included data preprocseeing functions or you want to report an issue, please do so at the corresponding [GitHub page](https://github.com/kozodoi/dptools/issues).