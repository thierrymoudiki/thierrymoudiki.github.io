---
layout: post
title: "R version of survivalist: Probabilistic model-agnostic survival analysis using scikit-learn, xgboost, lightgbm (and conformal prediction)" 
description: "Survival analysis is used to predict the time until an event of interest occurs. In this post, I show how to use scikit-learn, xgboost, lightgbm, and conformal prediction for probabilistic survival analysis"
date: 2025-02-12
categories: R
comments: true
---

This post is to be read in conjunction with [https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube) and [https://thierrymoudiki.github.io/blog/2024/12/15/python/agnostic-survival-analysis](https://thierrymoudiki.github.io/blog/2024/12/15/python/agnostic-survival-analysis).

**Survival analysis** is a group of Statistical/Machine Learning (ML) methods for predicting the **time until an event of interest occurs**. Examples of events include: 

- death 
- failure
- recovery
- default
- etc.

And the event of interest can be anything that has a duration: 

- the time until a machine breaks down 
- the time until a customer buys a product 
- the time until a patient dies
- etc. 

The event can be **censored**, meaning that it has'nt occurred for some subjects at the time of analysis. 

In this post, I show how to use `scikit-learn`, `xgboost`, `lightgbm` in R, in  conjuction with Python package [`survivalist`](https://github.com/Techtonique/survivalist) for probabilistic survival analysis. The probabilistic part is based on **conformal prediction and Bayesian inference**, and graphics represent the out-of-sample ML survival function.

![image-title-here]({{base}}/images/2025-02-12/2025-02-12-image1.png)

{% include 2025-02-12-model_agnostic_survival_R.html %}

