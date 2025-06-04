---
layout: post
title: "Python version of Beyond ARMA-GARCH: leveraging model-agnostic Quasi-Randomized networks and conformal prediction for nonparametric probabilistic stock forecasting (ML-ARCH)"
description: "A flexible hybrid approach to probabilistic stock forecasting that combines machine learning with ARCH effects, offering an alternative to traditional ARMA-GARCH models"
date: 2025-06-03
categories: Python
comments: true
---


# Introduction

Probabilistic ([not point forecasting](https://thierrymoudiki.github.io/blog/2024/12/29/r/stock-forecasting)) stock forecasting is notably useful for **testing trading strategies** or **risk capital valuation**. Because stock prices exhibit a latent stochastic volatility, this type of forecasting methods generally relies on classical  parametric models like [ARMA for the mean and GARCH for volatility](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity). 

**This post offers a flexible hybrid alternative to ARMA-GARCH, combining conformal prediction and machine learning approaches with AutoRegressive Conditional Heteroskedastic (ARCH) effects.**

The model decomposes the time series into two components:

1. Mean component: $$y_t = \mu_t + \sigma_t \varepsilon_t$$
2. Volatility component: $$\sigma_t^2 = f(\varepsilon_{t-1}^2, \varepsilon_{t-2}^2, ...)$$

where:

- $$\mu_t$$ is the conditional mean (modeled using **any forecasting method** from Python package `nnetsauce`)
- $$\sigma_t$$ is the conditional volatility (modeled using **machine learning** from Python package `nnetsauce`)
- $$\varepsilon_t$$ are standardized residuals

The **key innovation** is using any time series model for mean forecast, and machine learning methods + conformal prediction to model the volatility component, allowing for more flexible and potentially more accurate volatility forecasts than traditional GARCH models.

The forecasting process involves:

- Fitting a mean model 
- Modeling the squared residuals using machine learning. For this to work, the residuals from the mean model need to be centered, so that 
  
$$
\mathbb{E}[\epsilon_t^2|F_{t-1}]
$$

(basically a supervised regression of squared residuals on their lags) is a good approximation of the latent conditional volatility

- Conformalizing the standardized residuals for prediction intervals

This new approach combines the interpretability of traditional time series models with the flexibility of machine learning, while maintaining proper uncertainty quantification through conformal prediction.

# Basic Usage in Python package `nnetsauce`

Keep in mind that there's a high number of degrees of freedom (and possible regularization parameters) in this approach, and that tight prediction intervals could be obtained throuch time series cross-validation + minimization of Winkler scores. 

```
!pip install nnetsauce yfinance
```

```
import numpy as np
import yfinance as yf
import nnetsauce as ns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.stattools import kpss
from scipy import stats


# Define the ticker symbol
ticker_symbol = "MSFT"  # Example: Apple Inc.

# Get data for the ticker
ticker_data = yf.Ticker(ticker_symbol)

# Get the historical prices for a specific period (e.g., 1 year)
# You can adjust the start and end dates as needed
history = ticker_data.history(period="1y")

# Extract the 'Close' price series as a numpy array
# You can choose 'Open', 'High', 'Low', or 'Close' depending on your needs
stock_prices = history['Close'].values

print(f"Imported {len(stock_prices)} daily closing prices for {ticker_symbol}")
print("First 5 prices:", stock_prices[:5])
print("Last 5 prices:", stock_prices[-5:])

n_points = len(stock_prices)
h = 20
y = stock_prices[:(n_points-h)] 
y_test = stock_prices[(n_points-h):] 
n = len(y)
level=90
B=1000

mean_model = ns.MTS(GradientBoostingRegressor(random_state=42))
model_sigma = ns.MTS(GradientBoostingRegressor(random_state=42), 
                    lags=2, type_pi="scp2-kde",
                    replications=B)
model_z = ns.MTS(GradientBoostingRegressor(random_state=42), 
                    type_pi="scp2-kde",
                    replications=B)

objMLARCH = ns.MLARCH(model_mean = mean_model,
                      model_sigma = model_sigma, 
                      model_residuals = model_z)

objMLARCH.fit(y)

# Testing if the residuals of mean forecasting model have mean=0 or not at 5%
print(objMLARCH.mean_residuals_wilcoxon_test_)

preds = objMLARCH.predict(h=20, level=level)

mean_f = preds.mean
lower_bound = preds.lower
upper_bound = preds.upper

# Create a time index for the forecasts
forecast_index = np.arange(len(y), len(y) + h)
original_index = np.arange(len(y))

# Plotting
plt.figure(figsize=(12, 6))

# Plot original series
plt.plot(original_index, y, label='Original Series', color='blue')

# Plot mean forecast
plt.plot(forecast_index, mean_f, label='Mean Forecast', 
         color='red', linestyle='--')

# Plot true value
plt.plot(forecast_index, y_test, label='True test value', 
         color='green', linestyle='--')

# Plot prediction intervals
# Use the level from the results dictionary for the label
plt.fill_between(forecast_index, lower_bound, upper_bound, color='orange', 
                 alpha=0.3, label=f'{level}% Prediction Interval')

plt.title('Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

```bash
Imported 250 daily closing prices for MSFT
First 5 prices: [412.90429688 420.78387451 421.28994751 420.62509155 424.61447144]
Last 5 prices: [457.35998535 458.67999268 460.35998535 461.97000122 462.97000122]
100%|██████████| 1/1 [00:00<00:00,  5.16it/s]
100%|██████████| 1/1 [00:00<00:00,  3.36it/s]
100%|██████████| 1/1 [00:00<00:00,  5.65it/s]
WilcoxonResult(statistic=array([12902.]), pvalue=array([0.79136817]))
```

![image-title-here]({{base}}/images/2025-06-03/2025-06-03-image1.png){:class="img-responsive"}    
