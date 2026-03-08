---
layout: post
title: "Explaining Time-Series Forecasts with Exact Shapley Values (ahead::dynrmf with external regressors applied to scenarios)"
description: "Explaining Time-Series Forecasts with Exact Shapley Values (ahead::dynrmf with external regressors applied to macroeconomic scenarios)"
date: 2026-03-08
categories: R
comments: true
---

Shapley values constitute a widely adopted way to attribute the contribution of each feature (explanatory variable) to the prediction of a model. Mostly used in supervised learning, this post illustrates an example of how to use them to explain time-series forecasts, with exact Shapley values, and based on the `ahead::dynrmf` model with external regressors.

The code below uses the `ahead` package to compute exact Shapley values for a time-series forecast. It uses the `ahead::dynrmf_shap` function to compute the Shapley values and the `ahead::plot_dynrmf_shap_waterfall` function to plot them.

First, install the package:

```R
devtools::install_github("Techtonique/ahead")
```

Then, run the following code (applies Shapley values to the `dynrmf` model, for different scenarios). I use the `uschange` dataset (quarterly changes in US macroeconomic variables) from the `fpp2` package. The target time series variable is `Consumption`; the regressors are `Income`, `Savings`, and `Unemployment` (scaled).

```R
library(fpp2); library(ahead); library(e1071); library(misc)
library(ggplot2); library(patchwork)

y       <- fpp2::uschange[, "Consumption"]
xreg    <- scale(fpp2::uschange[, c("Income", "Savings", "Unemployment")])
split   <- misc::splitts(y, split_prob = 0.9)
xreg_train <- window(xreg, start = start(split$training), end = end(split$training))
xreg_test <- window(xreg, start = start(split$testing),  end = end(split$testing))

shap <- ahead::dynrmf_shap(
  y            = split$training,
  xreg_fit     = xreg_train,
  xreg_predict = xreg_test,
  fit_func     = e1071::svm
)

p1 <- ahead::plot_dynrmf_shap_waterfall(shap, title = "Baseline scenario")

xreg_pess <- xreg_test
xreg_pess[,"Income"] <- -1;
xreg_pess[,"Savings"] <- -0.5

shap_pess <- dynrmf_shap(
  y            = split$training,
  xreg_fit     = xreg_train,
  xreg_predict = xreg_pess,
  fit_func     = e1071::svm
)

p2 <- ahead::plot_dynrmf_shap_waterfall(shap_pess, title = "Pessimistic scenario")

xreg_opt  <- xreg_test
xreg_opt[,"Income"]  <-  2;
xreg_opt[,"Savings"]  <-  0.5

shap_opt <- dynrmf_shap(
  y            = split$training,
  xreg_fit     = xreg_train,
  xreg_predict = xreg_opt,
  fit_func     = e1071::svm
)

p3 <- ahead::plot_dynrmf_shap_waterfall(shap_opt, title = "Optimistic scenario")

xreg_ovr  <- xreg_test
xreg_ovr[,"Income"]  <-  2.5;
xreg_ovr[,"Savings"] <-  0.75

shap_ovr <- ahead::dynrmf_shap(
  y            = split$training,
  xreg_fit     = xreg_train,
  xreg_predict = xreg_ovr,
  fit_func     = e1071::svm
)

p4 <- plot_dynrmf_shap_waterfall(shap_ovr, title = "Overly optimistic scenario")

(p1 + p2)/(p3 + p4)
```

![image-title-here]({{base}}/images/2026-03-08/2026-03-08-image1.png){:class="img-responsive"}

One check, which is always a good practice when using Shapley values, is to see if the sum of the Shapley values equals the difference between the prediction and the baseline forecast (the model forecast when every regressor is replaced by its training set column mean). It's the case on the plots above.

It's worth mentioning that exact Shapley values can be computed in this context because there are only a few external regressors. This remains feasible for a small number of regressors (less than 15, which, again, in this context, is not absurd to consider). 