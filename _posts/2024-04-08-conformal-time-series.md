---
layout: post
title: "Conformalized predictive simulations for univariate time series"
description: "Conformalized predictive simulations for univariate time series on > 250 data sets"
date: 2024-04-08
categories: R
comments: true
---

<span>
<a target="_blank" href="https://colab.research.google.com/github/Techtonique/ahead_python/blob/main/ahead/demo/thierrymoudiki_20240408_conformal_bench.ipynb">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</span>

or 

<span>
<a target="_blank" href="https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Read this preprint"/>
</a>
</span>

**Predictive simulation** of time series data is useful for many applications such as risk management and stress-testing in finance or insurance, climate modeling, and electricity load forecasting.  [This paper](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) proposes a new approach to uncertainty quantification for univariate time series forecasting. This approach adapts [split conformal prediction](https://conformalpredictionintro.github.io/) to sequential data:  after training the model on a _proper training set_, and obtaining an inference of the residuals on a _calibration set_, out-of-sample predictive simulations are obtained through the use of various parametric and semi-parametric simulation methods.  Empirical results on uncertainty quantification scores are presented for more than 250 time series data sets, both real world and synthetic, reproducing a wide range of time series stylized facts. 

![xxx]({{base}}/images/2024-04-08/2024-04-08-image1.png){:class="img-responsive"}      
