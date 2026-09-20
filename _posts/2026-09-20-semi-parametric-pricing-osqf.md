---
layout: post
title: "Semi-parametric option pricing based on underlying's historical data (accepted at the osQF 2026 (ex R/Finance) conference)"
description: "This post is a follow-up to my previous posts on semi-parametric option pricing. A link to the study (accepted for presentation at the osQF 2026 conference) is provided at the end of this post."
date: 2026-09-20
categories: R
comments: true
---

This post is a follow-up to my previous posts on semi-parametric option pricing. A link to the study (accepted for presentation at the osQF 2026 conference) is provided at the end of this post. 

In this study (R code provided), we build an empirical pricing measure for options directly from their underlying's historical dynamics, requiring no market of option prices to calibrate against. Starting from a single observed discounted price series, we filter out linear predictability via an AR(1) fit and preserve the remaining dependence structure through a stationary block bootstrap of the residuals. The resulting empirical distribution is then given a minimal adjustment consistent with no-arbitrage: a single scalar correction, following Duan and Simonato [1998], that enforces the martingale condition required by the Fundamental Theorem of Asset Pricing (FTAP). We are explicit that this construction is not a recovery of "the" risk-neutral measure itself in the usual economic sense, but rather a pricing measure obtained by disturbing the historical dynamics as little as possible, to verify the FTAP. We validate the methodology empirically against the implied volatility surface of DAX index European options quoted July 5, 2002, comparing prices directly rather than implied volatilities. Because the construction requires no option-market input at any stage, it extends naturally to path-dependent payoffs; we illustrate this on arithmetic Asian options. 1 Motivation Standard option pricing practice requires the assumption of a parametric family for the option underlying's dynamics (geometric Brownian motion [Black and Scholes, 1973], stochastic volatility [Heston, 1993], jump-diffusion [Merton, 1976]), and calibrating the parameters of that parametric family to observed option prices. This works well when a liquid market of option prices exists to calibrate against. It does not help when no such market exists, for examples for path-dependent or structured payoffs written on an underlying with no quoted option market (an Asian option on a private index, an embedded option inside an insurance product, a participation certificate). This note develops an alternative construction for option pricing that requires no market prices at all. It builds an empirical pricing measure directly from the historical dynamics of the underlying asset, and imposes no parametric distributional assumption beyond what the data itself exhibits. The construction is then made consistent with the Fundamental Theorem of Asset Pricing (FTAP). We first state the FTAP theorem, then show how it motivates the step of our pricing estimator, before turning to empirical validation.

[https://www.researchgate.net/publication/414513095_Semi-parametric_option_pricing_based_on_underlying's_historical_data](https://www.researchgate.net/publication/414513095_Semi-parametric_option_pricing_based_on_underlying's_historical_data)

I'm now interested in constructive remarks and feedback on the study, that will allow to enrich, improve and robustify the methodology. 

![image-title-here]({{base}}/images/2026-09-20/2026-09-20-image1.png){:class="img-responsive"}

