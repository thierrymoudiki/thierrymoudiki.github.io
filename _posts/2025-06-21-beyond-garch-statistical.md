---
layout: post
title: "Beyond ARMA-GARCH: leveraging any statistical model for volatility forecasting"
description: "A flexible hybrid approach to probabilistic stock forecasting that combines statistical model with ARCH effects, offering an alternative to traditional ARMA-GARCH models"
date: 2025-06-21
categories: [R, Python]
comments: true
---


**Remark:** There's now a button for copying the code chunks. It's useful; I needed it. 

This post is a sequel of [Beyond ARMA-GARCH: leveraging model-agnostic Machine Learning and conformal prediction for nonparametric probabilistic stock forecasting (ML-ARCH)](https://thierrymoudiki.github.io/blog/2025/06/02/r/beyond-garch) and [Python version of Beyond ARMA-GARCH: leveraging model-agnostic Quasi-Randomized networks and conformal prediction for nonparametric probabilistic stock forecasting (ML-ARCH)](https://thierrymoudiki.github.io/blog/2025/06/03/python/beyond-garch-python). 

The _novelty_ in this post is that: you can use any _statistical_ model (meaning Theta, ARIMA, exponential smoothing, etc.) for volatility forecasting. It will be adapted to the Python version (previous link) in the next few days. Remember that the models used below are not tuned, and that you 
need to tune them to reduce the forecasting uncertainty. 


## Installing the package

```R
options(repos = c(
                techtonique = "https://r-packages.techtonique.net",
                CRAN = "https://cloud.r-project.org"
            ))            

install.packages("ahead")
```

## Theta

```R
# Default model for volatility (Ridge regression for volatility)
(obj_ridge <- ahead::mlarchf(fpp2::goog200, h=20L, B=500L, ml=FALSE, stat_model=forecast::thetaf))
plot(obj_ridge)
```

![image-title-here]({{base}}/images/2025-06-21/2025-06-21-image1.png){:class="img-responsive"}    

## Mean forecast

```R
(obj_ridge <- ahead::mlarchf(fpp2::goog200, h=20L, B=500L, ml=FALSE, stat_model=forecast::meanf))
plot(obj_ridge)
```

![image-title-here]({{base}}/images/2025-06-21/2025-06-21-image2.png){:class="img-responsive"}    

## Auto ARIMA

```R
(obj_ridge <- ahead::mlarchf(fpp2::goog200, h=20L, B=500L, ml=FALSE, stat_model=forecast::auto.arima))
plot(obj_ridge)
```

![image-title-here]({{base}}/images/2025-06-21/2025-06-21-image3.png){:class="img-responsive"}    

## Exponential smoothing

```R
(obj_ridge <- ahead::mlarchf(fpp2::goog200, h=20L, B=500L, ml=FALSE, stat_model=forecast::ets))
plot(obj_ridge)
```

![image-title-here]({{base}}/images/2025-06-21/2025-06-21-image4.png){:class="img-responsive"}    
