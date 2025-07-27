---
layout: post
title: "Boosting any randomized based learner for regression, classification and univariate/multivariate time series forcasting"
description: "Model-agnostic Boosting of any randomized based learner using Python package cybooster for regression, classification and univariate/multivariate time series forcasting."
date: 2025-07-26
categories: [Python]
comments: true
---

This post/notebook demonstrates the usage of the `cybooster` library for boosting various scikit-learn-like (having `fit` and `predict` methods is enough, for GPU learning see e.g slides 35-38 [https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)) estimators on different datasets. It includes examples of regression and classification and time series forecasting tasks. It's worth mentioning that only regressors are accepted in `cybooster`, no matter the task. 

`cybooster` is a high-performance generic gradient boosting (any based learner can be used) library designed for classification and regression tasks. It is built on Cython (that is, C) for speed and efficiency. This version will also be more GPU friendly, thanks to JAX, making it suitable for large datasets.

In `cybooster`, each base learner is augmented with a randomized neural network (a generalization of [https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) to any base learner), which allows the model to learn complex patterns in the data. The library supports both classification and regression tasks, making it versatile for various machine learning applications.

`cybooster` is born from `mlsauce`, that might be difficult to install on some systems (for now). `cybooster` installation is straightforward. 

<a target="_blank" href="https://colab.research.google.com/github/Techtonique/cybooster/blob/main/cybooster/demo/2025_07_22_cybooster_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>

![image-title-here]({{base}}/images/2025-03-09/2025-03-09-image1.png)

{% include 2025_07_22_cybooster_example.html %}
