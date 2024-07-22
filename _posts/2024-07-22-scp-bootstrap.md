---
layout: post
title: "Forecasting uncertainty: sequential split conformal prediction + Block bootstrap (web app)"
description: "Forecasting uncertainty: sequential split conformal prediction + Block bootstrap (web app)"
date: 2024-07-22
categories: [Python, R]
comments: true
---

This post was first submitted to the [Applied Quantitative Investment Management](https://www.linkedin.com/groups/12877102/) group on LinkedIn. It illustrates a recipe implemented in [Python package nnetsauce](https://thierrymoudiki.github.io/blog/2024/07/03/python/quasirandomizednn/forecasting/nnetsauce-mts-isf2024) for time series forecasting uncertainty quantification (through simulation): **sequential split conformal prediction + block bootstrap** 

Underlying algorithm: 
- Split data into training set, calibration set and test set
- Obtain point forecast on calibration set
- Obtain calibrated residuals = point forecast on calibration set - true observation on calibration set
- Simulate calibrated residuals using block bootstrap
- Obtain Point forecast on test set
- Prediction = Calibrated residuals simulations + point forecast on test set

Interested in experimenting more? Here is [a web app](https://github.com/thierrymoudiki/2024-07-17-scp-block-bootstrap). 

For more details, you can read (under review): [https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series)

![xxx]({{base}}/images/2024-07-22/2024-07-22-image1.png){:class="img-responsive"}      

