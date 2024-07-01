---
layout: post
title: "10 uncertainty quantification methods in nnetsauce forecasting"
description: "10 uncertainty quantification methods in nnetsauce forecasting."
date: 2024-07-01
categories: [Python, QuasiRandomizedNN, Forecasting]
comments: true
---

This week, I released version `0.22.4` of [`nnetsauce`](https://github.com/Techtonique/nnetsauce). [`nnetsauce`](https://github.com/Techtonique/nnetsauce) now contains 10 methods for uncertainty quantification in class `MTS`.

- `gaussian`: simple, fast, but: assumes stationarity of Gaussian in-sample residuals and independence in the multivariate case
- `kde`: based on Kernel Density Estimation of in-sample residuals
- `bootstrap`: based on independent bootstrap of in-sample residuals
- `block-bootstrap`: based on basic block bootstrap of in-sample residuals
- `scp-kde`: Split conformal prediction with Kernel Density Estimation of calibrated residuals
- `scp-bootstrap`: Split conformal prediction with independent bootstrap of calibrated residuals
- `scp-block-bootstrap`: Split conformal prediction with basic block bootstrap of calibrated residuals
- `scp2-kde`: Split conformal prediction with Kernel Density Estimation of standardized calibrated residuals
- `scp2-bootstrap`: Split conformal prediction with independent bootstrap of standardized calibrated residuals
- `scp2-block-bootstrap`: Split conformal prediction with basic block bootstrap of standardized calibrated residuals

The [release](https://github.com/Techtonique/nnetsauce/releases/edit/v0.22.4) is available on Github, Conda and PyPI.

I'll present [`nnetsauce`](https://github.com/Techtonique/nnetsauce)'s  (univariate and multivariate probabilistic) time series forecasting capabilities at the 44th [International Symposium on Forecasting (ISF)](https://isf.forecasters.org/) (ISF) 2024 (Wednesday). 


Next week, I'll release a stable (and documented) version of [`learningmachine`](https://thierrymoudiki.github.io/blog/2024/04/01/python/learningmachine-python). 



