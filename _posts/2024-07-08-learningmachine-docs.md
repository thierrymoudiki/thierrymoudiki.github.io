---
layout: post
title: "learningmachine v2.0.0: Machine Learning with explanations and uncertainty quantification"
description: "learningmachine v2.0.0: Machine Learning with explanations and uncertainty quantification"
date: 2024-07-08
categories: R
comments: true
---

This is the most stable version of [`learningmachine`](https://github.com/Techtonique/learningmachine) for R: the one you should use. `learningmachine` is a package for Machine Learning that includes **uncertainty quantification for regression and classification** (work in progress), and **explainability through sensitivity analysis**. So far, it offers a unified interface for: 

- `lm`: Linear model
- `bcn`: *Boosted Configuration 'neural' Networks*, see [https://www.researchgate.net/publication/380760578_Boosted_Configuration_neural_Networks_for_supervised_classification](https://www.researchgate.net/publication/380760578_Boosted_Configuration_neural_Networks_for_supervised_classification)
- `extratrees`: Extremely Randomized Trees; see [https://link.springer.com/article/10.1007/s10994-006-6226-1](https://link.springer.com/article/10.1007/s10994-006-6226-1)
- `glmnet`: Elastic Net Regression; see [https://glmnet.stanford.edu/](https://glmnet.stanford.edu/)
- `krr`: Kernel Ridge Regression; see for example [https://www.jstatsoft.org/article/view/v079i03](https://www.jstatsoft.org/article/view/v079i03)
- `ranger`: Random Forest; see [https://www.jstatsoft.org/article/view/v077i01](https://www.jstatsoft.org/article/view/v077i01)
- `ridge`: Ridge regression; see [https://arxiv.org/pdf/1509.09169](https://arxiv.org/pdf/1509.09169)
- `xgboost`: a scalable tree boosting system see [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)

There are only 2 classes `Classifier` and `Regressor`, with methods `fit` and `predict` and `summary`, and all these models can be **enhanced by using a [quasi-randomized layer](https://github.com/Techtonique/nnetsauce)** that basically augments their capacity. The [3 package vignettes](https://github.com/Techtonique/learningmachine/tree/main/vignettes) are a great way to get started. Along with the (work in progress, as I'm struggling a little bit with documenting R6 objects) **documentation**, they'll eventually be available here:

[https://techtonique.r-universe.dev/learningmachine](https://techtonique.r-universe.dev/learningmachine)

There are also unit tests in the `tests` folder [on GitHub](https://github.com/Techtonique/learningmachine). 

![xxx]({{base}}/images/2024-03-25/2024-03-25-image1.png){:class="img-responsive"}      

PS: In [these slides](https://www.researchgate.net/publication/381957724_Probabilistic_Forecasting_with_RandomizedQuasi-Randomized_networks_presentation_at_the_International_Symposium_on_Forecasting_2024), in present probabilistic forecasting with [`nnetsauce`](https://github.com/Techtonique/nnetsauce)