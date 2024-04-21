---
layout: post
title: "mlsauce's `v0.13.0`: taking into account inputs heterogeneity through clustering"
description: "Supervised Machine Learning in mlsauce using unsupervised Machine Learning."
date: 2024-04-21
categories: Python
comments: true
---


Last week in #134, I talked about `mlsauce`'s `v0.12.0`, and [`LSBoost`](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) in particular. As shown in the post, it's now possible to obtain prediction intervals for the regression model, notably by employing Split Conformal Prediction. 

Right now (looking for ways to fix it), the best way to install the package, is to use the development version:

```bash
pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```

Now, in `v0.13.0`, it's possible to add explanatory variables' heterogeneity to the mix; through clustering (K-means and Gaussian Mixtures models). This means that, a priori, and in order to assess the conditional expectation of the variable of interest as a function of our covariates, we explicitly tell the model to take into account similarities between individual observations. Some examples of use of this new feature can be found [here](https://github.com/Techtonique/mlsauce/blob/master/examples/adaopt_classifier.py), [here](https://github.com/Techtonique/mlsauce/blob/master/examples/lsboost_classifier.py) and [here](https://github.com/Techtonique/mlsauce/blob/master/examples/lsboost_regressor.py). Keep in mind however: these examples only show that it's possible to overfit the training set (hence reducing the loss function's magnitude) by adding some clusters. The whole model's hyperparameters need to be 'fine-tuned', for example by using  [GPopt](https://thierrymoudiki.github.io/blog/2023/11/05/python/r/adaopt/lsboost/mlsauce_classification).


![](https://en.m.wikipedia.org/wiki/File:Simpsons_paradox_-_animation.gif)

![pres-image]({{base}}/images/2020-11-21/2020-11-21-image6.png){:class="img-responsive"}

 
