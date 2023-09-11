---
layout: post
title: "Forecasting in Python with ahead"
description: Univariate anc multivariate time series forecasting with uncertainty quantification in Python
date: 2023-09-11
categories: [Python, Forecasting]
---


A _new_ Python version of [`ahead`](https://github.com/Techtonique/ahead_python), is now available on GitHub
and PyPI. Here are the new features: 

- Align with R version (see [https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-070](https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-070) and [https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-080](https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-080)) as much as possible

- moving block bootstrap in `ridge2f`, `basicf`, in addition to circular block bootstrap from 0.6.2

- adjust R-Vine copulas on residuals for `ridge2f` simulation (with empirical and Gaussian marginals)

It still takes forever to be imported (only the first time) when the R packages are not available in the environment. But I'll release a workaround soon. 



