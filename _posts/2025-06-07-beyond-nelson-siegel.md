---
layout: post
title: "Beyond Nelson-Siegel and splines: A model-agnostic Machine Learning framework for discount curve calibration, interpolation and extrapolation"
description: "Beyond Nelson-Siegel and splines: A model-agnostic Machine Learning framework for discount curve calibration, interpolation and extrapolation"
date: 2025-06-07
categories: [R, Python]
comments: true
---

This paper introduces a general machine learning framework for yield curve modeling, in which classical parametric models such as Nelson-Siegel and Svensson serve as special cases within a broader class of functional regression approaches. By linearizing the bond pricing/swap valuation equation, I reformulate the estimation of spot rates as a supervised regression problem, where the response variable is derived from observed bond prices and cash flows, and the regressors are constructed as flexible functions of time-to-maturities. I show that this formulation supports a wide range of modeling strategies — including polynomial expansions, Laguerre polynomials, kernel methods, and regularized linearmodels — all within a unified framework that could preserve economic interpretability. This enables not only curve calibration but also static interpolation and extrapolation. By abstracting away from any fixed parametric structure, my framework bridges the gap between traditional yield curve modeling and modern supervised learning, offering a robust, extensible, and data-driven tool for financial applications ranging from asset pricing to regulatory (?) reporting.


[https://www.researchgate.net/publication/392507059_Beyond_Nelson-Siegel_and_splines_A_model-_agnostic_Machine_Learning_framework_for_discount_curve_calibration_interpolation_and_extrapolation](https://www.researchgate.net/publication/392507059_Beyond_Nelson-Siegel_and_splines_A_model-_agnostic_Machine_Learning_framework_for_discount_curve_calibration_interpolation_and_extrapolation)



![image-title-here]({{base}}/images/2025-06-07/2025-06-07-image1.png){:class="img-responsive"}    