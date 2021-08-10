---
layout: post
title: "`crossvalidation` and random search for calibrating support vector machines"
description: "Random search for calibrating support vector machines."
date: 2021-08-06
categories: R
---

# Install and load packages

```R
options(
  repos = c(techtonique = 'https://techtonique.r-universe.dev',
            CRAN = 'https://cloud.r-project.org')
)

install.packages("crossvalidation")

library(crossvalidation)
library(e1071)
```

# Input data

## transforming model response into a factor

```R
y <- as.factor(as.numeric(iris$Species))
```

## explanatory variables

```R
X <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width",
                     "Petal.Length", "Petal.Width")])
```

# Objective -- cross-validation -- function to be maximized

```R
OF <- function(xx) {
  res <- crossvalidation::crossval_ml(
    x = X,
    y = y,
    k = 5,
    repeats = 3,
    p = 0.8,
    fit_func = e1071::svm,
    predict_func = predict,
    packages = "e1071",
    fit_params = list(gamma = xx[1],
                      cost = xx[2])
  )
  # default metric is accuracy
  return(res$mean_training)
}
```

There are many, many ways to maximize this objective function. 

# A naive random search optimization procedure

## simulation of SVM's hyperparameters' matrix

```R
n_points <- 250
set.seed(123)
(hyperparams <- cbind.data.frame(
  gamma = runif(n = n_points,
                min = 0,
                max = 5),
  cost = 10 ^ runif(n = n_points,
                    min = -1,
                    max = 2)
))
```

## accuracies on the set of simulated hyperparameters

```R
scores <- parallel::mclapply(1:n_points,
                             function(i)
                               OF(hyperparams[i,]),
                             mc.cores = parallel::detectCores())
scores <- unlist(scores)
```

For Windows, `future.apply::future_lapply` can be used instead of `parallel::mclapply`.

## 'best' hyperparameters and associated training set score

```R
max_index <- which.max(scores)
xx_best <- hyperparams[max_index,]
print(xx_best)
```
```
      gamma     cost
18 0.2102977 1.101473
```

```R
print(OF(xx_best))
```
```
|===================================================================================| 100%
utilisateur     système      écoulé 
      0.284       0.079       0.365 

[1] 0.9527778
```
