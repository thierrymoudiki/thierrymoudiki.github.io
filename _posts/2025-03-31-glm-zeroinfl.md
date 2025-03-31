---
layout: post
title: "Nonlinear conformalized Generalized Linear Models (GLMs) with R package 'rvfl' (and other models)"
description: "Nonlinear conformalized Generalized Linear Models (GLMs) with R package 'rvfl' (and other models)"
date: 2025-03-31
categories: R
comments: true
---

Start with:

```R
options(repos = c(
    techtonique = "https://r-packages.techtonique.net",
    CRAN = "https://cloud.r-project.org"
))

install.packages(c('pscl', 'randtoolbox', 'MASS', 'randomForest', 'tseries'))

install.packages('rvfl') # that's the package of interest
library(rvfl)
```

(Will be corrected in a future release)

'rvfl' stands for "Random Vector Functional Link", and is a package for nonlinear regression and classification. It implements a type of neural network called a functional link network.

Keep in mind that the models should be tuned in practice. They are not tuned by default in these examples. 

The examples demonstrate how you can obtain conformalized predictions from R models, including Generalized Linear Models (GLMs; here Poisson, Quasi-Poisson, and zero inflated GLMs). More examples can be found in the [package's vignette](https://docs.techtonique.net/rvfl/)(Articles).

{% include 2025-03-31-glm-zero-infl.html %}

