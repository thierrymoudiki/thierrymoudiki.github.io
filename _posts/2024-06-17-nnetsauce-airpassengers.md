---
layout: post
title: "Forecasting Monthly Airline Passenger Numbers with Quasi-Randomized Neural Networks"
description: "Forecasting Monthly Airline Passenger Numbers with Quasi-Randomized Neural Networks."
date: 2024-06-17
categories: [Python, QuasiRandomizedNN, Forecasting]
comments: true
---

<span>
<a target="_blank" href="https://colab.research.google.com/github/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20240617_nnetsauce_mts.ipynb">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</span>

This post is about forecasting airline passenger numbers with quasi-randomized neural networks, and most specifically using [`nnetsauce`](https://github.com/Techtonique/nnetsauce)'s class `MTS`. `MTS` stands for 'Multivariate Time Series', but `MTS` can also be used for univariate time series as shown in this post. 

The data used here is the famous `AirPassengers` dataset, a time series of monthly totals of international airline passengers from 1949 to 1960. `AirPassengers` has a total number of 144 observations, an upward trend, and a seasonal component. 
It also has a time-varying volatility, which makes it a bit more challenging and interesting to forecast.

# 0 - Install and load packages 

**Install `nnetsauce` command line**: 

```bash
pip install nnetsauce --upgrade --no-cache-dir
```

**Python code for loading the packages**: 

```python
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
```

# 1 - Load `AirPassengers` data

```python
url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/AirPassengers.csv"
df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date)
df.drop(columns=['date'], inplace=True)
df.plot()
```
![xxx]({{base}}/images/2024-06-17/2024-06-17-image1.png){:class="img-responsive"}      

# 2 - Train a quasi-randomized neural network based on Ridge regression

The quasi-randomized neural network used here relies on a Ridge Regression model, and has 5 hidden nodes by default (that makes it a nonlinear model). The `replications` parameter below is the number of predictive simulations. The `kernel` parameter set to 'gaussian' means that a [Gaussian kernel](https://en.wikipedia.org/wiki/Kernel_density_estimation) is used for simulating from the residuals' density. The `lags` parameter is set to 15, which means that the quasi-randomized neural network model is run on the 15 previous values of the time series to predict the next value.

```python
regr = ns.MTS(obj=Ridge(),
              replications=250,
              kernel='gaussian',
              lags=15)
regr.fit(df)              
``` 

# 3 - Forecasting 40 steps ahead

```python
regr.predict(h=40)
```

```python
regr.plot(type_plot="pi")
```
![xxx]({{base}}/images/2024-06-17/2024-06-17-image2.png){:class="img-responsive"}      


```python
regr.plot(type_plot="spaghetti")
```
![xxx]({{base}}/images/2024-06-17/2024-06-17-image3.png){:class="img-responsive"}      

The prediction interval is a bit narrow on this data set, but the model captures the trend, seasonality and time-varying volatility quite well. Do not hesitate to try the model on other time series data sets, and tune the hyperparameters of the model. Another example of univariate time series can be found in the link appearing at the top of this post. Other types of predictions intervals will be available in future versions of `nnetsauce`.