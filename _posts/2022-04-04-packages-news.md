---
layout: post
title: "News from ESGtoolkit, ycinterextra, and nnetsauce"
description: "News from R packages ESGtoolkit, ycinterextra, and Python/R package nnetsauce"
date: 2022-04-04
categories: [Python, R, Misc]
---

In this post, I introduce new versions of ESGtoolkit, ycinterextra, and nnetsauce. 

- [ESGtoolkit](https://github.com/Techtonique/esgtoolkit) (for R) is a toolkit for **Monte Carlo Simulation** in Finance, Economics, Insurance, Physics, etc. 
- [ycinterextra](https://github.com/Techtonique/ycinterextra) (for R) is used for  **yield curve** interpolation and extrapolation 
- [nnetsauce](https://github.com/Techtonique/nnetsauce) (for Python and R) does supervised Statistical/Machine Learning using **Randomized and Quasi-Randomized _neural_ networks**

**Contents**

Feel free to jump directly to the section that has your interest:

- [1 - ESGtoolkit: ](#ESGtoolkit)
  - [Installing](#installing-esgtoolkit)
  - [News](#news-from-esgtoolkit)
- [2 - ycinterextra: ](#ycinterextra)
  - [Installing](#installing-ycinterextra)
  - [News](#news-from-ycinterextra)
- [3 - nnetsauce: ](#nnetsauce)
  - [Installing](#installing-nnetsauce)
  - [News](#news-from-nnetsauce)


# 1-ESGtoolkit 

ESGtoolkit is no longer available from CRAN (archived). It can be installed from GitHub 
or from [R universe](https://r-universe.dev/search/).

## installing ESGtoolkit

- From Github: 

```r
library(devtools)
devtools::install_github("Techtonique/ESGtoolkit")
```

- From R universe: 

```r
# Enable universe(s) by techtonique
options(repos = c(
  techtonique = 'https://techtonique.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'))

# Install some packages
install.packages('ESGtoolkit')
```

## news from ESGtoolkit

In **version v0.4.0**, spline interpolation (stats::spline) is used for forward 
rates' computation in `esgfwdrates`. New (and only, so far) interpolation options 
are: "fmm", "periodic", "natural", "hyman" 
(type `?stats::spline` in R console for more details on each interpolation method). 

[Here is an example](https://rpubs.com/thierrymoudiki/33287) (in function `simG2plus`, 
and more specifically for `methodyc`, whose possible values are now "fmm", "periodic", 
"natural", "hyman") in which you can see how this new choices will affect the 
simulation results. 

![G2++ simulations]({{base}}/images/2022-04-04/2022-04-04-image1.png){:class="img-responsive"}

# 2-ycinterextra 

ycinterextra is no longer available from CRAN (archived). It can be installed from GitHub 
or from [R universe](https://r-universe.dev/search/).

## installing ycinterextra

- From Github: 

```r
devtools::install_github("Techtonique/ycinterextra")
```

- From R universe: 

```r
# Enable universe(s) by techtonique
options(repos = c(
  techtonique = 'https://techtonique.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'))

# Install some packages
install.packages('ycinterextra')
```

## news from ycinterextra

In **version 0.2.0**

- Rename function `as.list` to a `tolist` doing the same thing. `ycinterextra::as.list` was notably causing bugs in Shiny applications
- [Refactor the code](https://github.com/Techtonique/ycinterextra), to make it more readable

# 3-nnetsauce 

## installing nnetsauce

**Python** version: 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```

- __2nd method__: using `conda` (Linux and macOS only for now)

```bash
conda install -c conda-forge nnetsauce 
```

- __3rd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

**R** version: 

```r
library(devtools)
devtools::install_github("Techtonique/nnetsauce/R-package")
library(nnetsauce)
````

## news from nnetsauce

In **version 0.11.3**: 

- Implementation of a `RandomBagRegressor`; an ensemble of [CustomRegressors](https://techtonique.github.io/nnetsauce/documentation/regressors/#customregressor) 
  in which diversity is achieved by sampling the columns and rows of an input dataset. A Python example can be found [here on GitHub](https://github.com/Techtonique/nnetsauce/blob/master/examples/randombag_regression.py). 
  
- Use of pandas DataFrames in Python, for `MTS` (**a work in progress!**) (see Python example below for details)

```python
# Using a data frame input for forecasting with `MTS`

import nnetsauce as ns
import pandas as pd
from sklearn import linear_model


dataset = {
'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
'series1' : [34, 30, 35.6, 33.3, 38.1],    
'series2' : [4, 5.5, 5.6, 6.3, 5.1],
'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df)

# Adjust Bayesian Ridge and predict
regr5 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr5, lags = 1, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict())

# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))
print(obj_MTS.predict(return_std=True, level=95))
```

