---
layout: post
title: "Just got a paper on conformal prediction REJECTED by International Journal of Forecasting despite evidence on 30,000 time series (and more). What's going on?"
description: "Extensive benchmark based on 30,000 time series (code provided in this post) and conformal prediction, with R and Python code provided"
date: 2025-01-05
categories: [R, Forecasting, Python, misc]
comments: true
---

Hi everyone, best wishes for 2025!

I just got [this preprint paper](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) rejected by the International Journal of Forecasting (IJF) **despite a benchmark on over 30,000 time series** (code from the paper in R and Python available in this post). The method described in the preprint paper, despite being simple (of course, it's always simple after implementing the idea), is performing on par or much better than the state of the art (which may certainly frustrate some great minds, I get it), as you'll see below.

So, how do you go from first round reviews like:
1. "The comparison with other conformal prediction methods **is performed only on simulated data, not on any real-world data at all**.": **among the 275 time series from the first submission, 240 were real-world data**. See [https://github.com/Techtonique/datasets/blob/main/time_series/univariate/250datasets/250timeseries.txt](https://github.com/Techtonique/datasets/blob/main/time_series/univariate/250datasets/250timeseries.txt) for the list of time series, and [https://github.com/Techtonique/datasets/blob/main/time_series/univariate/250datasets/250datasets_characteristics.R](https://github.com/Techtonique/datasets/blob/main/time_series/univariate/250datasets/250datasets_characteristics.R) for their characteristics. And the proferssor said "at all". 
2. "So the authors should demonstrate the merit of their method on a more standard dataset, such as the M4": see next paragraph for more details. 
3. **Reviewer 1**: "I thank the author(s) for the work. It is great to see how [...] Conformal Prediction methods grow[s], especially in challenging tasks such as time series forecasting. I **strongly appreciate** the use of a big data set for the benchmark and that the **code is shared**. I also **appreciate** that 2 forecasting models and 5 conformalization procedures are compared (the one using KDE, the one using surrogate, the one using bootstrap, and the **state of the art** ACI and AgACI)". **Remark**: the initial big data set contained 250 time series, and the final data in the second submission's set contained over 30,000 time series (including M3 and M5 competition data).
4. **Reviewer 2**: "The paper is **well-written** and **clearly structured**".

To a **rejection** saying things such as:
- About point 2., and after trying to demonstrate the merits on M3 and M5 competition data in submission 2: "The paper shows a **factorial** application of too many variants to **too many datasets** which make it hard to follow". Ok, may look messy, but is there really such a thing as "too many datasets" for proving a point? (No) If so, why not asking to trim down?
- I used "doesn't" instead of "does not": this can be modified at edition. Why are we discussing (along with a Grammarly link I was sent :o) this in such length and over 8 months, as if it really mattered?
- I used the term "calibrated residuals": one reviewer suggested "calibratION residuals" instead while another suggested calling them just "residuals", such a distraction, what do I do? 
- I used the term "predictive simulation": what should I call it then and why does it really matter? I used simulation of future outcomes indeed, see page 12 of [the preprint paper](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) (again, IMHO, a pure distraction)
- "The forecasting community strongly benefits from new approaches and experiments on how to quantify uncertainty, that is why I appreciate the author's contribution." Hmm... Okay then...
- "can only accept a limited number of them, which make **substantial contributions to the science and practice of forecasting**": knock-out punch. 

![gbms]({{base}}/images/2025-01-05/2025-01-05-image1.png){:class="img-responsive"}

I'm definitely not the type to whine or cry foul (this is a battle-tested sentence), but I'm curious. Do you see any coherence in the last 2 paragraphs, or am I losing my mind? (No) One thing that was asked and that I'll address (and I guess it'll make sense) is why not training the models on the whole training set at the end of my conformal prediction algorithm? Because **I want to avoid data leakage**, and I also want to **use the most contemporaneous data for forecasting**.

Interested in seeing the method from the paper in action more directly? 
- See [this other post](https://thierrymoudiki.github.io/blog/2024/11/23/r/generic-conformal-forecast), which basically contains one-liner codes for conformalizing R packages such as `forecast`. 
- See this [interactive dashboard](https://github.com/thierrymoudiki/2024-07-17-scp-block-bootstrap).
- Execute the Python and R code of the paper, available in this post (note: takes looong hours to execute).

On a brighter note, a stable version of [www.techtonique.net](https://www.techtonique.net) has now been released. A tutorial on how to use it is available [https://moudiki2.gumroad.com/l/nrhgb](https://moudiki2.gumroad.com/l/nrhgb).

# Contents of the post

- 1 - M3 competition (R code), 3003 time series
- 2 - M5 competition (Python code), 42840 time series aggregated by item (3049 items)
- 3 - 250 datasets (R code), 240 real-world, 10 synthetic
- 4 - 25 additional synthetic datasets (R code)

Benchmarking errors in the examples below are measured by:
- **Coverage rate**: the percentage of future values that are within the prediction intervals, for 80% and 95% prediction intervals
- **Winkler score**: the length of the prediction intervals, penalized by every time the true future value is outside the interval (see [https://www.otexts.com/fpp3/distaccuracy.html#winkler-score](https://www.otexts.com/fpp3/distaccuracy.html#winkler-score) for more details). The lower the score, the better the method.
  
Plus, `splitconformal` denotes the method described in the paper.

{% include 2025-01-05-conformal-ts-M3-M5-250-25.html %}

# Conclusion

So, based on these extensive experiments against the state of the art (and assuming the implementations of the state of the art methods are correct, which I'm sure they are, see [https://computo.sfds.asso.fr/published-202407-susmann-adaptive-conformal/](https://computo.sfds.asso.fr/published-202407-susmann-adaptive-conformal/), and assuming I'm using them well), **how cool is this contribution to the science of forecasting**? My results are particularly impressive on the 3003 time series from the M3 competition (versus other quite recent conformal prediction methods). On M5 competition data, the method is performing on par with XGBoost, LightGBM, GradientBoostingRegressor quantile regressors. All from 3 prominent package heavyweights on the market. My 2 cents (but I might be wrong): almost nobody likes simplicity in  corporate world, and in academia in particular, because they need to justify, somehow, why something is expensive/why they are funded. A bias that fuels complexity (read: complex="substantial") for the sake of complexity. Man, IT JUST WORKS. And even more than that, as hopefully demonstrated here with extensive benchmarks.


