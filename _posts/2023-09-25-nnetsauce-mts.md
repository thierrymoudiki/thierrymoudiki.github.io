---
layout: post
title: "An infinity of time series forecasting models in nnetsauce (Part 2 with uncertainty quantification)"
description: "An infinity of ML-based time series forecasting models with uncertainty quantification"
date: 2023-09-25
categories: [Python, Forecasting, QuasiRandomizedNN]
comments: true
---

**Update 2023-09-27**: `conda` users, `v0.13.0` of nnetsauce [is now available](https://github.com/conda-forge/nnetsauce-feedstock/releases/tag/v0.13.0) for Linux and macOS. For Windows, please use [WSL2](https://t.co/SIS6KsPQ0I).

As [I said a few years ago](https://thierrymoudiki.github.io/blog/2021/03/06/python/r/quasirandomizednn/nnetsauce-mts), this is a family of univariate/multivariate time series forecasting models that I was supposed to present at [R/Finance 2020](https://www.rinfinance.com/) (this post is **100% Python**) in Chicago, IL. But the COVID-19 decided differently. 

The more I thought about it, namely `nnetsauce.MTS` (still doesn't have a more glamorous name), the more I thought **'It's kind of weird...'**. Why? Because in the statistical learning procedure, all the input time series models share the same hyperparameters. Today, I think `nnetsauce.MTS` it's not quite different from a multi-output regression (regression models for predicting multiple responses, based on covariates), and it seems to be working well empirically, as shown below. No grandiose _state-of-the-art_ (**SOTA** for the snobs) claims here, but I think that with the high number of possible model inputs (actually, any regression `Estimator` having `fit` and `predict` methods), you could _cover a lot of space_.

You can read [this post](https://thierrymoudiki.github.io/blog/2021/03/06/python/r/quasirandomizednn/nnetsauce-mts) if you want to understand how it works (but avoid the ugly graph at the end, the ones presented here are hopefully more compelling). [Pull requests](https://github.com/Techtonique/nnetsauce/pulls) and (constructive) [discussions](https://github.com/Techtonique/nnetsauce/discussions) are welcome as usual. 

In the examples presented here, I focus on **uncertainty quantification**: 

- **simulation-based**, using [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) of the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals)
- a **Bayesian** approach, even though 'Bayesianism' is _in hot water_ these days. Its _subjectivity_? I must admit that choosing a prior distribution is quite an _interesting_ (interpret 'interesting' here as you want, I mean both good and bad) experiment. But 'Bayesianism', [Gaussian Processes](http://gaussianprocess.org/gpml/) in particular, works quite well in settings such as [hyperparameters tuning](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt) (I hope the code still works) for example 

[Conformal prediction](https://thierrymoudiki.github.io/blog/2022/10/05/python/explainableml/interpretation-and-PI-for-BCN), the new cool kid on the uncertainty quantification block, will certainly be included in future versions of the tool.  

**Contents**

- 0 - Install and import packages + get data
- 1 - Simulation-based forecasting using Kernel Density Estimation
- 1 - 1 With Ridge regression
- 1 - 2 With Random Forest
- 2 - Bayesian Forecasting
- Appendix

You can also download [this notebook from GitHub](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_250923_nnetsauce_mts_plots.ipynb), which follows the same plan.

# 0 - Install and import packages + get data

**Installing** `nnetsauce` (v0.13.0) with `pip`: 

```bash
pip install nnetsauce
```

Installing `nnetsauce` (v0.13.0) using `conda`:

```bash
conda install -c conda-forge nnetsauce 
```

Installing from [GitHub](https://github.com/Techtonique/nnetsauce):

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

**Import** the packages in Python:

```python
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from time import time
```

**Get data:**

```python
url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

# ice cream vs heater (I don't own the copyright)
df.set_index('Month', inplace=True) 
df.index.rename('date')

df = df.pct_change().dropna()

idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]
```

# 1 - Simulation-based forecasting using Kernel Density Estimation

## 1 - 1 With Ridge regression

```python
regr3 = Ridge()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, #IRL, must be tuned
                  replications=50, kernel='gaussian',
                  seed=24, verbose = 1)
start = time()
obj_MTS3.fit(df_train)
print(f"Elapsed {time()-start} s")
```

```python
res = obj_MTS3.predict(h=15)
print("\n")
print(f" Predictive simulations #10: \n{obj_MTS3.sims_[9]}")
print("\n")
print(f" Predictive simulations #25: \n{obj_MTS3.sims_[24]}")
```

```python
obj_MTS3.plot("heater")
obj_MTS3.plot("ice cream")
```

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image1.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image2.png){:class="img-responsive"}

## 1 - 2 With Random Forest

```python
regr3 = RandomForestRegressor(n_estimators=250)
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, #IRL, must be tuned
                  replications=50, kernel='gaussian',
                  seed=24, verbose = 1)
start = time()
obj_MTS3.fit(df_train)
print(f"Elapsed {time()-start} s")
```

```python
res = obj_MTS3.predict(h=15)
print("\n")
print(f" Predictive simulations #10: \n{obj_MTS3.sims_[9]}")
print("\n")
print(f" Predictive simulations #25: \n{obj_MTS3.sims_[24]}")
```

```python
obj_MTS3.plot("heater")
obj_MTS3.plot("ice cream")
```

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image3.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image4.png){:class="img-responsive"}


# 2 - Bayesian Forecasting

```python
regr4 = BayesianRidge()
obj_MTS4 = ns.MTS(regr4, lags = 3, n_hidden_features=7, #IRL, must be tuned
                  seed=24)
start = time()
obj_MTS4.fit(df_train)
print(f"\n\n Elapsed {time()-start} s")
```

```python
res = obj_MTS4.predict(h=15, return_std=True)
```

```python
obj_MTS4.plot("heater")
obj_MTS4.plot("ice cream")
```

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image5.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2023-09-25/2023-09-25-image6.png){:class="img-responsive"}


# Appendix 

How does this family of time series forecasting models works? 

![image-title-here]({{base}}/images/2021-03-06/2021-03-06-image1.png){:class="img-responsive"}
