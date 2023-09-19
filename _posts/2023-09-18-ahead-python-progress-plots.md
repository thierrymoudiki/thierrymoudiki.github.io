---
layout: post
title: "(News from) forecasting in Python with ahead (progress bars and plots)"
description: Univariate and multivariate time series forecasting with uncertainty quantification in Python
date: 2023-09-18
categories: [Python, Forecasting]
---


A _new_ Python version of [`ahead`](https://github.com/Techtonique/ahead_python), `v0.9.0` is now available on [GitHub](https://github.com/Techtonique/ahead_python) and [PyPI](https://pypi.org/project/ahead/). 

`ahead` is a Python and R package for univariate and multivariate time series forecasting, with **uncertainty 
quantification** (in particular, simulation-based uncertainty quantification). 

Here are the new features in `v0.9.0`: 

- progress bars for possibly _long_ calculations: the bootstrap (independent, circular block, moving block)

- plot for `Ridge2Regressor` (a work in progress, still needs to use series names, and display dates correctly, for all classes, not just `Ridge2Regressor`)

Since this implementation is based on the R version, it could take some time to import R packages when using Python's `ahead` **for the first time**. There's something new regarding this situation (well... ha ha): R packages are now installed _on the fly_. Meaning: only when they're required. 

# Example 1

Start by installing `ahead` v0.9.0:

```bash
pip install ahead
```

```Python
import numpy as np
import pandas as pd
from time import time
from ahead import Ridge2Regressor # this is where the R packages are installed (if not available in the environment, and ONLY the 1st time)

url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"


df = pd.read_csv(url)

df.set_index('Month', inplace=True) # only for ice_cream_vs_heater
df.index.rename('date') # only for ice_cream_vs_heater

df = df.pct_change().dropna()
```

```Python
regr1 = Ridge2Regressor(h = 10, date_formatting = "original",
                     type_pi="rvinecopula",
                     margins="empirical",
                     B=50, seed=1)

regr1.forecast(df) # this is where the R packages are installed (if not available in the environment, and ONLY the 1st time)

regr1.plot(0) # dates are missing, + want to use series names
regr1.plot(1)
```

![Ridge2Regressor with R-Vine copula and empirical marginals 1]({{base}}/images/2023-09-18/2023-09-18-image1.png){:class="img-responsive"}

![Ridge2Regressor with R-Vine copula and empirical marginals 2]({{base}}/images/2023-09-18/2023-09-18-image2.png){:class="img-responsive"}

# Example 2

```Python
regr2 = Ridge2Regressor(h = 10, date_formatting = "original",
                     type_pi="movingblockbootstrap",
                     B=50, seed=1)

regr2.forecast(df) # a progress bar is displayed

regr2.plot(0) # dates are missing, + want to use series names
regr2.plot(1)
```

![Ridge2Regressor with moving block bootstrap 1]({{base}}/images/2023-09-18/2023-09-18-image3.png){:class="img-responsive"}

![Ridge2Regressor with moving block bootstrap 2]({{base}}/images/2023-09-18/2023-09-18-image4.png){:class="img-responsive"}

