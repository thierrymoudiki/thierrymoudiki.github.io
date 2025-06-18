---
layout: post
title: "Stacked generalization (Machine Learning model stacking) + conformal prediction for forecasting with ahead::mlf"
description: "Examples of use of ahead::mlf for univariate probabilistic time series forecasting"
date: 2025-06-18
categories: R
comments: true
---

# Introduction

In this post, I'll show how to use `ahead::mlf`, a function from [R package ahead](https://docs.techtonique.net/ahead/) that does [Conformalized Forecasting](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series). `ahead::mlf` uses Machine Leaning models (any of them) and time series lags in an autoregressive way, as workhorses for univariate probabilistic time series forecasting. 

This function differs from [`ahead::dynrmf`](https://thierrymoudiki.github.io/blog/2025/04/20/r/tisthemachinelearner-time-series) by  the fact that it doesn't automatically select time series lags lags and by the availability of a simple [stacked generalization](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) functionality. I first discussed a stacked generalization method for forecasting in [this 2018 document](https://github.com/thierrymoudiki/phd-thesis/blob/master/moudiki_thesis.pdf), at page 79. 

In the context of `ahead::mlf`, the algorithm described in [https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) for time series conformal prediction is enriched: **the predictions obtained on the calibration set are used as covariates for test set predictions**. It's important to choose a model that doesn't overfit much here, as there are already a lot of information provided to the calibration set by this algorithm. I choose the [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization). 

**What We'll Cover**

1. Installing the package
2. Run the examples

# 1 - Installing the package

```R
options(repos = c(
                techtonique = "https://r-packages.techtonique.net",
                CRAN = "https://cloud.r-project.org"
            ))            

install.packages("ahead")
install.packages("glmnet")
```

# 2 - Run the examples

At this stage, it's worth trying out other Elastic Net implementations (e.g, just `glmnet::glmnet`), in particular because, due to `glmnet::cv.glmnet`'s implementation, my cross-validation is kind of _backward looking_. Well, use [time series cross-validation](https://docs.techtonique.net/crossvalidation/) to reduce forecasting volatility. 

```R
(res1 <- ahead::mlf(AirPassengers, h=25L, lags=20L, fit_func=glmnet::cv.glmnet, stack=FALSE))
(res2 <- ahead::mlf(AirPassengers, h=25L, lags=20L, fit_func=glmnet::cv.glmnet, stack=TRUE))
(res3 <- ahead::mlf(USAccDeaths, h=25L, lags=20L, fit_func=glmnet::cv.glmnet, stack=TRUE))
(res4 <- ahead::mlf(USAccDeaths, h=25L, lags=20L, fit_func=glmnet::cv.glmnet, stack=FALSE))
```

```R
par(mfrow=c(1, 2))
plot(res1, main="Conformal ML without stacking")
plot(res2, main="Conformal ML with stacking")
```

![image-title-here]({{base}}/images/2025-06-18/2025-06-18-image1.png){:class="img-responsive"}    


```R
par(mfrow=c(1, 2))
plot(res3, main="Conformal ML with stacking")
plot(res4, main="Conformal ML without stacking")
```

    
![image-title-here]({{base}}/images/2025-06-18/2025-06-18-image2.png){:class="img-responsive"}    
    
