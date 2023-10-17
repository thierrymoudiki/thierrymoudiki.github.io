---
layout: post
title: "Version v0.14.0 of nnetsauce for R and Python"
description: "Randomized and Quasi-Randomized 'neural' networks for supervised learning and multivariate time series forecasting"
date: 2023-10-16
categories: [Python, R, Forecasting, QuasiRandomizedNN]
comments: true
---

 Version `v0.14.0` of nnetsauce is now available for R (hopefully a rapid installation) and Python on [GitHub](https://github.com/Techtonique/nnetsauce), [PyPI](https://pypi.org/project/nnetsauce/) and [conda](https://anaconda.org/conda-forge/nnetsauce). It's been mainly tested on Linux and macOS. For Windows users, you can try to install of course, but if it doesn't work, please use [WSL2](https://t.co/SIS6KsPQ0I).

**NEWS**

- update and align as much as possible with `R` version (new plotting function for multivariate time series (`MTS`), `plot.MTS`, is not [`S3`](https://cran.r-project.org/doc/manuals/R-exts.html#Registering-S3-methods), but it’s complicated)

```R
# 0 - install packages ----------------------------------------------------

#utils::install.packages("remotes")
remotes::install_github("Techtonique/nnetsauce/R-package", force = TRUE)

# 1 - ENET simulations ----------------------------------------------------

obj <- nnetsauce::sklearn$linear_model$ElasticNet()
obj2 <- nnetsauce::MTS(obj, 
                       start_input = start(fpp::vn), 
                       frequency_input = frequency(fpp::vn),
                       kernel = "gaussian", replications = 100L)
X <- data.frame(fpp::vn)
obj2$fit(X)
obj2$predict(h = 10L)
typeof(obj2)

par(mfrow=c(2, 2))
plot.MTS(obj2, selected_series = "Sydney")
plot.MTS(obj2, selected_series = "Melbourne")
plot.MTS(obj2, selected_series = "NSW")
plot.MTS(obj2, selected_series = "BrisbaneGC")

# 2 - Bayesian Ridge ----------------------------------------------------

obj <- nnetsauce::sklearn$linear_model$BayesianRidge()
obj2 <- nnetsauce::MTS(obj,
                       start_input = start(fpp::vn), 
                       frequency_input = frequency(fpp::vn))
X <- data.frame(fpp::vn)
obj2$fit(X)
obj2$predict(h = 10L, return_std = TRUE)

par(mfrow=c(2, 2))
plot.MTS(obj2, selected_series = "Sydney")
plot.MTS(obj2, selected_series = "Melbourne")
plot.MTS(obj2, selected_series = "NSW")
plot.MTS(obj2, selected_series = "BrisbaneGC")
```

![image-title-here]({{base}}/images/2023-10-16/2023-10-16-image1.png){:class="img-responsive"}
![image-title-here]({{base}}/images/2023-10-16/2023-10-16-image2.png){:class="img-responsive"}


- colored graphics for `Python` class `MTS` 

```python
# !pip install nnetsauce —upgrade

import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from time import time

url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

# ice cream vs heater (I don't own the copyright)
df.set_index('Month', inplace=True)
df.index.rename('date')

df = df.pct_change().dropna()

idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]

regr3 = Ridge()
obj_MTS3 = ns.MTS(regr3, lags = 4, n_hidden_features=7, #IRL, must be tuned
                  replications=50, kernel='gaussian',
                  seed=24, verbose = 1)
start = time()
obj_MTS3.fit(df_train)
print(f"Elapsed {time()-start} s")

# predict 
obj_MTS3.predict()

obj_MTS3.plot("heater")
obj_MTS3.plot("ice cream")
```

![image-title-here]({{base}}/images/2023-10-16/2023-10-16-image3.png){:class="img-responsive"}
![image-title-here]({{base}}/images/2023-10-16/2023-10-16-image4.png){:class="img-responsive"}
