---
layout: post
title: "A lightweight interface to scikit-learn in R Pt.2: probabilistic time series forecasting in conjunction with ahead::dynrmf"
description: "Example of use of tisthemachinelearner; a lightweight interface to scikit-learn in R: probabilistic time series forecasting in conjunction with ahead::dynrmf"
date: 2025-04-20
categories: R
comments: true
---

[tisthemachinelearner](https://docs.techtonique.net/tisthemachinelearner_r) is a  (work in progress) lightweight interface to scikit-learn. Here's an example of use time series forecasting using scikit-learn in conjunction with   `ahead::dynrmf`.

**update 2025-09-04:** The loop on scikit-learn models must be: 

```R
i <- 1
j <- 1
results <- list()
model_list <- tisthemachinelearner::get_model_list()
n <- length(model_list$regressors)
list_names <- list()
pb <- utils::txtProgressBar(min=1, max=n, style=3)
for (j in 1:n) {
  y <- USAccDeaths
  h <- 15L
  model <- model_list$regressors[[j]]
  print(model)
  foo <- function(x, y) tisthemachinelearner::regressor(x=x, y=y, model_name=model)
  fcastf <- function(y, h) ahead::dynrmf(y=y, h=h, fit_func=foo, predict_func=predict)
  fcastf2 <- function(y, h) ahead::conformalize(FUN=ahead::dynrmf, y=y, h=h, 
                                                fit_func=foo, predict_func=predict)
  res <- try(fcastf2(y=y, h=h), silent=TRUE)
  if (!inherits(res, "try-error"))
  {
    print(res)
    results[[i]] <- res
    list_names[[i]] <- model
    i <- i + 1
  } else {
    next
  }
  utils::setTxtProgressBar(pb, j)
  j <- j + 1
}
close(pb)
names(results) <- list_names
```

{% include 2025-04-22-dynrm-sklearn.html %}