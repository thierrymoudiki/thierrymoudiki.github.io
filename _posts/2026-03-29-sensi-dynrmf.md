---
layout: post
title: "Explaining Time-Series Forecasts with Sensitivity Analysis (ahead::dynrmf and external regressors)"
description: "Explaining Time-Series Forecasts with Sensitivity Analysis (ahead::dynrmf and external regressors)"
date: 2026-03-29
categories: R
comments: true
---

Following [the post on exact Shapley values](https://thierrymoudiki.github.io/blog/2026/03/08/r/exact-shapley-dynrmf) for time series explainability, this post illustrates an example of how to use sensitivity analysis  to explain time-series forecasts, based on the `ahead::dynrmf` model and external regressors. What is **sensitivity analysis** in this context? It's about evaluating the impact of changes in the external regressors on the time-series forecast.


The post uses the [`ahead::dynrmf_sensi`](https://docs.techtonique.net/ahead/reference/dynrmf_sensi.html) function to compute the sensitivities, and the [`ahead::plot_dynrmf_sensitivity`](https://docs.techtonique.net/ahead/reference/plot_dynrmf_sensitivity.html) function to plot the results.


First, install the package:

```R
devtools::install_github("Techtonique/ahead")
```

Then, run the following code: 

```R
# devtools::install_github("Techtonique/ahead")
# install.packages(c("fpp2", "e1071", "patchwork"))

library(ahead)
library(fpp2)
library(patchwork)
library(e1071)

#' # Example 1: US Consumption vs Income
sensitivity_results_auto <- ahead::dynrmf_sensi(
y = fpp2::uschange[, "Consumption"],
xreg = fpp2::uschange[, "Income"],
h = 10
)

plot1 <- ahead::plot_dynrmf_sensitivity(sensitivity_results_auto, 
                           title = "Sensitivity of Consumption to Income (Ridge)",
                           y_label = "Effect (ΔConsumption / ΔIncome)")

#' # Example 1: US Consumption vs Income
sensitivity_results_auto_svm <- ahead::dynrmf_sensi(
  y = fpp2::uschange[, "Consumption"],
  xreg = fpp2::uschange[, "Income"],
  h = 10, 
  fit_func = e1071::svm # additional parameter passed to ahead::dynrmf
)

plot2 <- ahead::plot_dynrmf_sensitivity(sensitivity_results_auto_svm, 
                                        title = "Sensitivity of Consumption to Income (SVM)",
                                        y_label = "Effect (ΔConsumption / ΔIncome)")

 
# Example 2: TV Advertising vs Insurance Quotes
sensitivity_results_tv <- ahead::dynrmf_sensi(
 y = fpp2::insurance[, "Quotes"],
   xreg = fpp2::insurance[, "TV.advert"],
   h = 8
 )

plot3 <- ahead::plot_dynrmf_sensitivity(sensitivity_results_tv,
                           title = "Sensitivity of Insurance Quotes to TV Advertising (Ridge)",
                           y_label = "Effect (ΔQuotes / ΔTV.advert)")

sensitivity_results_tv_svm <- ahead::dynrmf_sensi(
  y = fpp2::insurance[, "Quotes"],
  xreg = fpp2::insurance[, "TV.advert"],
  h = 8, 
  fit_func = e1071::svm # additional parameter passed to ahead::dynrmf
)

plot4 <- ahead::plot_dynrmf_sensitivity(sensitivity_results_tv_svm,
                                        title = "Sensitivity of Insurance Quotes to TV Advertising (SVM)",
                                        y_label = "Effect (ΔQuotes / ΔTV.advert)")

(plot1+plot2)

(plot3+plot4)
```

![image-title-here]({{base}}/images/2026-03-29/2026-03-29-image1.png){:class="img-responsive"}
![image-title-here]({{base}}/images/2026-03-29/2026-03-29-image2.png){:class="img-responsive"}

