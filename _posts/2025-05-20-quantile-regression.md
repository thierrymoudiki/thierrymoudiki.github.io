---
layout: post
title: "Quantile regression with any regressor -- Examples with RandomForestRegressor, RidgeCV, KNeighborsRegressor"
description: "Examples of use (in R and Python) of package nnetsauce for Quantile regression using any regressor"
date: 2025-05-20
categories: [R, Python]
comments: true
---

**Link to the notebook at the end of this post**

**Quantile regression** is a powerful statistical technique that estimates the conditional quantiles of a response variable, providing a more comprehensive view of the relationship between variables than traditional mean regression. While linear quantile regression is well-established, performing quantile regression with any machine learning regressor is less common but highly valuable.

In this blog post, we'll explore how to perform quantile regression in R and Python using RandomForestRegressor, RidgeCV, and KNeighborsRegressor with the help of [nnetsauce](https://github.com/Techtonique/nnetsauce), a package that extends scikit-learn models with additional functionalities.

## Why Quantile Regression?

Traditional regression models (e.g., linear regression) predict the mean of the dependent variable given the independent variables. However, in many real-world scenarios, we might be interested in:

<ul>
<li> Predicting extreme values (e.g., high or low sales, extreme temperatures). </li>
<li> Assessing uncertainty by estimating prediction intervals. </li>
<li> Handling non-Gaussian distributions where mean regression may be insufficient. </li>
</ul>

Quantile regression allows us to estimate any quantile (e.g., 5th, 50th, 95th percentiles) of the response variable, offering a more robust analysis.

## Quantile Regression with nnetsauce

The nnetsauce package provides a flexible way to perform quantile regression using any scikit-learn regressor. Below, we'll demonstrate how to use it with three different models, in R and Python:

<ul>
<li> RandomForestRegressor </li>
<li> RidgeCV (linear regression with cross-validated regularization) </li>
<li> KNeighborsRegressor </li>
</ul>

{% include 2025-05-20_quantile_regression.html %}

<a target="_blank" href="https://colab.research.google.com/github/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_2025-05-20_quantile_regression.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>