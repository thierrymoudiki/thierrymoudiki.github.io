---
layout: post
title: "Beyond ARMA-GARCH: leveraging model-agnostic Machine Learning and conformal prediction for nonparametric probabilistic stock forecasting (ML-ARCH)"
description: "A flexible hybrid approach to probabilistic stock forecasting that combines machine learning with ARCH effects, offering an alternative to traditional ARMA-GARCH models"
date: 2025-06-02
categories: R
comments: true
---


# Introduction

Probabilistic ([not point forecasting](https://thierrymoudiki.github.io/blog/2024/12/29/r/stock-forecasting)) stock forecasting is notably useful for **testing trading strategies** or **risk capital valuation**. Because stock prices exhibit a latent stochastic volatility, this type of forecasting methods relies on classical  parametric models like [ARMA for the mean and GARCH for volatility](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity). 

**This post offers a flexible hybrid alternative to ARMA-GARCH, combining conformal prediction and machine learning approaches with AutoRegressive Conditional Heteroskedastic (ARCH) effects.**

The model decomposes the time series into two components:

1. Mean component: $$y_t = \mu_t + \sigma_t \varepsilon_t$$
2. Volatility component: $$\sigma_t^2 = f(\varepsilon_{t-1}^2, \varepsilon_{t-2}^2, ...)$$

where:

- $\mu_t$ is the conditional mean (modeled using **any forecasting method**)
- $\sigma_t$ is the conditional volatility (modeled using **machine learning**)
- $\varepsilon_t$ are standardized residuals

The **key innovation** is using any time series model for mean forecast, and machine learning methods + conformal prediction to model the volatility component, allowing for more flexible and potentially more accurate volatility forecasts than traditional GARCH models. The function supports various machine learning methods through parameters `fit_func` and `predict_func` as in other `ahead` models, and through the `caret` package.

The forecasting process involves:

- Fitting a mean model (default: `auto.arima`)
- Modeling the squared residuals using machine learning. For this to work, the residuals from the mean model need to be centered, so that 
  
$$
\mathbb{E}[\epsilon_t^2|F_{t-1}]
$$

(basically a supervised regression of squared residuals on their lags) is a good approximation of the latent conditional volatility

- Conformalizing the standardized residuals for prediction intervals

This new approach combines the interpretability of traditional time series models with the flexibility of machine learning, while maintaining proper uncertainty quantification through conformal prediction.

# Basic Usage

Install package: 

```R
options(repos = c(
                    techtonique = "https://r-packages.techtonique.net",
                    CRAN = "https://cloud.r-project.org"
                ))

install.packages("ahead")            
```

Let's start with a simple example using the Google stock price data from the `fpp2` package:

```R
library(forecast)
library(ahead)
library(randomForest)
library(e1071)
library(glmnet)
```

```R
y <- fpp2::goog200

# Default model for volatility (Ridge regression for volatility)
(obj_ridge <- ahead::mlarchf(y, h=20L, B=500L))
```

# Different Machine Learning Methods

The package supports various machine learning methods for volatility modeling. Here are some examples:

```R
# Random Forest
(obj_rf <- ahead::mlarchf(y, fit_func = randomForest::randomForest, 
                     predict_func = predict, h=20L, B=500L))

# Support Vector Machine
(obj_svm <- ahead::mlarchf(y, fit_func = e1071::svm, 
                     predict_func = predict, h=20L, B=500L))

# Elastic Net
(obj_glmnet <- ahead::mlarchf(y, fit_func = glmnet::cv.glmnet, 
                     predict_func = predict, h=20L, B=500L))
```

Let's visualize the forecasts:

```R
par(mfrow=c(1, 2))
plot(obj_ridge, main="Ridge Regression")
plot(obj_rf, main="Random Forest")
```

```R
par(mfrow=c(1, 2))
plot(obj_svm, main="Support Vector Machine")
plot(obj_glmnet, main="Elastic Net")
```

![image-title-here]({{base}}/images/2025-06-02/2025-06-02-image1.png){:class="img-responsive"}    


# Using caret Models

The package also supports models from the `caret` package, which provides access to hundreds of machine learning methods. Here's how to use them:

```R
y <- window(fpp2::goog200, start=100)

# Random Forest via caret
(obj_rf <- ahead::mlarchf(y, ml_method="ranger", h=20L))

# Gradient Boosting via caret
(obj_glmboost <- ahead::mlarchf(y, ml_method="glmboost", h=20L))
```

Visualizing the forecasts:

```R
par(mfrow=c(1, 2))
plot(obj_rf, main="Random Forest (caret)")
plot(obj_glmboost, main="Gradient Boosting (caret)")
```

Looking at the simulation paths:

```R
par(mfrow=c(1, 2))
matplot(obj_rf$sims, type='l', main="RF Simulation Paths")
matplot(obj_glmboost$sims, type='l', main="GBM Simulation Paths")
```

# Customizing Mean and Residual Models

You can also customize both the mean forecasting model and the model for forecasting standardized residuals:

```R
# Using Theta method for both mean and residuals
(obj_svm <- ahead::mlarchf(y, fit_func = e1071::svm, 
                     predict_func = predict, h=20L, 
                     mean_model=forecast::rwf,
                     model_residuals=forecast::thetaf))

(obj_glmnet <- ahead::mlarchf(y, fit_func = glmnet::cv.glmnet, 
                     predict_func = predict, h=20L, 
                     mean_model=forecast::thetaf,
                     model_residuals=forecast::thetaf))
```

```R
par(mfrow=c(1, 2))
plot(obj_svm, main="SVM with Theta")
plot(obj_glmnet, main="Elastic Net with Theta")
```

When using non-ARIMA models for the mean forecast, it's important to check if the residuals are centered and stationary:

```R
# Diagnostic tests for residuals
print(obj_svm$resids_t_test)
## 
##  One Sample t-test
## 
## data:  resids
## t = 1.0148, df = 99, p-value = 0.3127
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.7180739  2.2214961
## sample estimates:
## mean of x 
## 0.7517111
print(obj_svm$resids_kpss_test)
## 
##  KPSS Test for Level Stationarity
## 
## data:  resids
## KPSS Level = 7.5912e-76, Truncation lag parameter = 4, p-value = 0.1
print(obj_glmnet$resids_t_test)
## 
##  One Sample t-test
## 
## data:  resids
## t = 1.0992, df = 100, p-value = 0.2743
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.6460748  2.2513707
## sample estimates:
## mean of x 
## 0.8026479
print(obj_glmnet$resids_kpss_test)
## 
##  KPSS Test for Level Stationarity
## 
## data:  resids
## KPSS Level = 0.26089, Truncation lag parameter = 4, p-value = 0.1
```
