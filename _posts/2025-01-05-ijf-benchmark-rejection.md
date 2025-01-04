---
layout: post
title: "Just got a paper on conformal prediction REJECTED by International Journal of Forecasting despite evidence on 30,000 time series (and more). What's going on?"
description: "Extensive benchmark based on 30,000 time series (code provided in this post) and conformal prediction, with R and Python code provided"
date: 2025-01-05
categories: [R, Forecasting, Python, misc]
comments: true
---

Hi everyone, best wishes for 2025!

I just got [this preprint paper](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) rejected by the International Journal of Forecasting (IJF) **despite a benchmark on 30,000 time series (code from the paper in R and Python available in this post; reproducible data and code)**. 

The method, despite being simple (of course, it's always simple after implementing the idea), is performing on par or better than the state of the art (which may certainly frustrate, I get it). For example, lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

So, how do you go from (first round reviews):
- **Reviewer 1**: "I thank the author(s) for the work. It is great to see how [...] Conformal Prediction methods grow[s], especially in challenging tasks such as time series forecasting. I **strongly appreciate** the use of a big data set for the benchmark and that the **code is shared**. I also **appreciate** that 2 forecasting models and 5 conformalization procedures are compared (the one using KDE, the one using surrogate, the one using bootstrap, and the **state of the art** ACI and AgACI)". **Remark**: the initial big data set contained 250 time series, and the final data in the second submission's set contained approximately 30,000 time series (including M3 and M5 competition data).
- **Reviewer 2**: "The paper is **well-written** and **clearly structured**".

To a rejection emphasizing on vanity such as:
- Using "doesn't" instead of "does not": what's so wrong with that?
- Using the term "calibrated residuals": what's so wrong with that?
- Using the term "predictive simulation": what's so wrong with that?

Interested in seeing the method in action more directly? See [this post](https://thierrymoudiki.github.io/blog/2024/11/23/r/generic-conformal-forecast). The post basically contains one-liners to conformalize R packages such as `forecast`. I'm definitely not the type to whine or cry foul (battle-tested sentence), but I'm curious. I'll let you judge for yourself, by executing the Python and R code available in this post.

On a brighter note, a stable version of [www.techtonique.net](https://www.techtonique.net) has now been released. A tutorial on how to use it is available [https://moudiki2.gumroad.com/l/nrhgb](https://moudiki2.gumroad.com/l/nrhgb).


