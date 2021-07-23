---
layout: post
title: "`crossvalidation` on R-universe, plus a classification example"
description: Package `crossvalidation` on R-universe is now available from R-universe.
date: 2021-07-23
categories: R
---


I had to rename my R package `crossval` -- generic functions for cross-validation -- to `crossvalidation`, because its name was clashing  with an existing [CRAN](https://cran.r-project.org/) R package's named `crossval`. 
Here is how to install 
`crossvalidation`:

```R
options(repos = c(
  techtonique = 'https://techtonique.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'))

install.packages("crossvalidation")
```

What is the [R-universe](https://ropensci.org/r-universe/) mentioned in the previous code snippet? It is, IMHO, a quite _promising_ CRAN-like repository for storing, sharing and building R packages (for Linux, macOS and Windows). If you want to create your own repository on R-universe, [read this](https://ropensci.org/blog/2021/06/22/setup-runiverse/). 

I've been looking 
for such an infrastructure for some time, and tried [`miniCRAN`](https://cran.r-project.org/web/packages/miniCRAN/index.html) in particular. 
Unfortunately on miniCRAN (which works pretty well for CRAN packages), I haven't been able, so far, to upload/build local packages -- _local_ meaning non-CRAN packages.  Maybe I missed a point on `miniCRAN`'s use, so if you know how to do that, please reach out to me (even though I'll continue to follow R-universe's development)!

Examples of use of `crossvalidation` for __regression__ and __univariate time series__ can be found through the following links (hence, you must __replace `crossval` occurences by `crossvalidation`__): 

- [Grid search cross-validation using crossval](https://thierrymoudiki.github.io/blog/2020/04/10/r/misc/grid-search-crossval) 
- [Linear model, xgboost and randomForest cross-validation using crossval::crossval_ml](https://thierrymoudiki.github.io/blog/2020/04/17/r/misc/crossval-3)
- [Custom errors for cross-validation using crossval::crossval_ml](https://thierrymoudiki.github.io/blog/2020/05/08/r/misc/crossval-custom-errors)
- [Time series cross-validation using crossval](https://thierrymoudiki.github.io/blog/2020/03/27/r/misc/crossval-2)

For __classification__, an example is presented below. 

## Example of use of `crossvalidation` for classification

```R
# Import libraries

library(crossvalidation)
library(randomForest)
```


```R
# Input data 

# Transforming model response into a factor
y <- as.factor(as.numeric(iris$Species))

# Explanatory variables 
X <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")])
```


```R
# 5-fold cross-validation repeated 3 times

# default error metric, when y is a factor: accuracy
crossvalidation::crossval_ml(x = X, y = y, k = 5, repeats = 3,
                             fit_func = randomForest::randomForest, 
                             predict_func = predict,
                             fit_params = list(mtry = 2),
                             packages = "randomForest")

```

```R
## $folds
##         repeat_1  repeat_2  repeat_3
## fold_1 0.9666667 0.9666667 1.0000000
## fold_2 0.9666667 0.9000000 0.9333333
## fold_3 1.0000000 0.9666667 0.9333333
## fold_4 0.9333333 1.0000000 0.9333333
## fold_5 0.9333333 0.9333333 0.9666667
## 
## $mean
## [1] 0.9555556
## 
## $sd
## [1] 0.02999118
## 
## $median
## [1] 0.9666667
```

```R
# We can specify custom error metrics for crossvalidation::crossval_ml
# here, the error rate 

eval_metric <- function (preds, actual)
{
 stopifnot(length(preds) == length(actual))
  res <- 1-mean(preds == actual)
  names(res) <- "error rate"
  return(res)
}

# specify `eval_metric` argument for measuring the error rate
# instead of the (default) accuracy 
crossvalidation::crossval_ml(x = X, y = y, k = 5, repeats = 3,
                             fit_func = randomForest::randomForest, 
                             predict_func = predict,
                             fit_params = list(mtry = 2),
                             packages = "randomForest", 
                             eval_metric=eval_metric)
```

```R
## $folds
##          repeat_1   repeat_2   repeat_3
## fold_1 0.03333333 0.03333333 0.00000000
## fold_2 0.03333333 0.10000000 0.06666667
## fold_3 0.00000000 0.03333333 0.06666667
## fold_4 0.06666667 0.00000000 0.06666667
## fold_5 0.06666667 0.06666667 0.03333333
## 
## $mean
## [1] 0.04444444
## 
## $sd
## [1] 0.02999118
## 
## $median
## [1] 0.03333333
```
