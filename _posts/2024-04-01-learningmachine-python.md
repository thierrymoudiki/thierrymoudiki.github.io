---
layout: post
title: "learningmachine v1.1.0: prediction intervals around the probability of the event 'a tumor being malignant'"
description: "learningmachine v1.0.0: prediction intervals around the probability of the event 'a tumor being malignant'; using conformal prediction and density estimation"
date: 2024-04-01
categories: Python
comments: true
---

Considering the number of people who read [this post](https://thierrymoudiki.github.io/blog/2024/01/01/r/learningmachine/learningmachine), a lot of you are probably using `learningmachine` `v0.2.3`. Maybe because of the fancy name. Just so you know, `learningmachine` is only doing batch learning at the moment. Stay tuned. 

Well, today, there are good news and bad news. The good news is `learningmachine` is back with `v1.0.0` (Python port coming next week). The "bad" news is: jumping to `v1.0.0` this early means there's a **change in the interface** (that won't change drastically anymore); with a lot of good reasons: 

- **Smaller codebase**: much easier to navigate and maintain, less error-prone 
- **Only 2 classes** in the interface: `Classifier`, `Regressor` with (currently) 7 machine learning `method`s; "bcn" ([Boosted Configuration Networks](https://thierrymoudiki.github.io/blog/2024/02/05/python/gpopt-new2)), "extratrees" (Extremely Randomized Trees), "glmnet" (Elastic Net), "krr" (Kernel Ridge Regression), "ranger" (Random Forest), "ridge" (Automatic Ridge Regression), "xgboost". 
- Every classifier is [regression-based](https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers).

`v0.2.3` remains available [on a branch](https://github.com/Techtonique/learningmachine/tree/v023).

The new features are: 

- Summarizing supervised learning results: **interpretability _via_ sensitivity** of the response to small changes in the explanatory variables + coverage rates for probabilistic predictions
- Uncertainty quantification for both regressors and classifiers (as shown below for classifiers). Right now, only the 'Least Ambiguous set-valued' method (denoted as standard Spit Conformal Prediction [here](https://conformalpredictionintro.github.io/)) is implemented for classifiers, **with a twist** (won't necessarily 
  remain this way): for empty prediction sets, the class with the highest probability is chosen. This _may_ 
  lead to over-conservative prediction sets. 

`learningmachine` is still experimental, probably with some quirks (because achieving this level of abstraction required some effort), with no beautiful documentation, but you can already tinker it and do advanced analysis, as shown below. You may also like [this vignette](https://techtonique.r-universe.dev/learningmachine/doc/getting-started.html) and [this vignette](https://techtonique.r-universe.dev/learningmachine/doc/classifiers-probs.html).

