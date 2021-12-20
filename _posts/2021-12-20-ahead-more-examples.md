---
layout: post
title: "Hundreds of Statistical/Machine Learning models for univariate time series, using ahead, ranger, xgboost, and caret"
description: Adjusting hundreds of Statistical/Machine Learning models to univariate time series with ahead, ranger, xgboost, and caret
date: 2021-12-20
categories: [R, Forecasting]
---

Today, we **examine some nontrivial use cases for  [`ahead::dynrmf`](https://techtonique.github.io/ahead/reference/dynrmf.html)**forecasting. Indeed, the examples presented in the [package's README](https://github.com/Techtonique/ahead/blob/main/README.md) work quite smoothly -- for `randomForest::randomForest` and `e1071::svm` -- because: 

- the fitting function can handle matricial inputs (can be called as `fitting_func(x, y)`, also said to have a `x/y` interface), and not only a _formula_ input (can be called as `fitting_func(y ~ ., data=df)`, the _formula_ interface)

- the `predict` functions associated to `randomForest::randomForest` and `e1071::svm` do have a prototype like `predict(object, newx)` or `predict(object, newdata)`, which are both well-understood input formats for `ahead::dynrmf`. 

After reading this post, **you'll know how to adjust hundreds of different Statistical/Machine Learning (ML) models to univariate time series**, and you'll get a better understanding of how `ahead::dynrmf` works. If you're not familiar with package `ahead` yet, you should read the following posts first: 

- [Forecasting with `ahead`](https://thierrymoudiki.github.io/blog/2021/10/15/r/misc/ahead-intro) (R version)

- [Automatic Forecasting with `ahead::dynrmf` and Ridge regression](https://thierrymoudiki.github.io/blog/2021/10/22/r/misc/ahead-ridge) (R version)

- [Forecasting with `ahead` (Python version)](https://thierrymoudiki.github.io/blog/2021/12/13/python/ahead-intro-python)


The demo uses `ahead::dynrmf` in conjunction with R packages: 

- `ranger`: random forests

- `xgboost`: gradient boosted decision trees

- `caret`: functions to streamline the model training process for complex regression problems. 


# Installing package `ahead`

```R
options(repos = c(
    techtonique = 'https://techtonique.r-universe.dev',
    CRAN = 'https://cloud.r-project.org'))
    
install.packages("ahead")
```

# Packages required for the demo

```R
library(ahead)
library(forecast)
library(ranger)
library(xgboost)
library(caret)
library(gbm)
library(ggplot2)
```

# Forecasting using `ahead::dynrmf`'s default parameters 

```R
# ridge ------------------------------------------------------------------

# default, with ridge regression's regularization parameter minimizing GCV
z <- ahead::dynrmf(USAccDeaths, h=15, level=95)
autoplot(z)
```

![image-title-here]({{base}}/images/2021-12-20/2021-12-20-image0.png){:class="img-responsive"}

# Forecasting using `ahead::dynrmf` and `ranger` 

```R
# ranger ------------------------------------------------------------------

fit_func <- function(x, y, ...)
{
  df <- data.frame(y=y, x) # naming of columns is mandatory for `predict`
  ranger::ranger(y ~ ., data=df, ...)
}

predict_func <- function(obj, newx)
{
  colnames(newx) <- paste0("X", 1:ncol(newx)) # mandatory, linked to df in fit_func
  predict(object=obj, data=newx)$predictions # only accepts a named newx
}

z <- ahead::dynrmf(USAccDeaths, h=15, level=95, fit_func = fit_func,
                    fit_params = list(num.trees = 500),
                    predict_func = predict_func)
autoplot(z)

```

![image-title-here]({{base}}/images/2021-12-20/2021-12-20-image1.png){:class="img-responsive"}

# Forecasting using `ahead::dynrmf` and `xgboost`

```R
# xgboost -----------------------------------------------------------------

fit_func <- function(x, y, ...) xgboost::xgboost(data = x, label = y, ...)

z <- ahead::dynrmf(USAccDeaths, h=15, level=95, fit_func = fit_func,
                   fit_params = list(nrounds = 10,
                                     verbose = FALSE),
                   predict_func = predict)
autoplot(z)
```

![image-title-here]({{base}}/images/2021-12-20/2021-12-20-image2.png){:class="img-responsive"}

# Forecasting using `ahead::dynrmf` and `gbm` through `caret`'s unified interface

```R
# caret gbm -----------------------------------------------------------------

# unified interface, with hundreds of regression models
# https://topepo.github.io/caret/available-models.html

fit_func <- function(x, y, ...)
{
  df <- data.frame(y=y, x)

  caret::train(y ~ ., data=df,
               method = "gbm",
               trControl=caret::trainControl(method = "none"), # no cv
               verbose = FALSE,
               tuneGrid=data.frame(...))
}

predict_func <- function(obj, newx)
{
  colnames(newx) <- paste0("X", 1:ncol(newx))
  caret::predict.train(object=obj, newdata=newx, type = "raw")
}

z <- ahead::dynrmf(USAccDeaths, h=15, level=95, fit_func = fit_func,
                   fit_params = list(n.trees=10, shrinkage=0.01,
                                     interaction.depth = 1,
                                     n.minobsinnode = 10),
                   predict_func = predict_func)
autoplot(z)
```

![image-title-here]({{base}}/images/2021-12-20/2021-12-20-image3.png){:class="img-responsive"}

# Forecasting using `ahead::dynrmf` and `glmnet` through `caret`'s unified interface

```R
# caret glmnet -----------------------------------------------------------------

# unified interface, with hundreds of regression models
# https://topepo.github.io/caret/available-models.html

fit_func <- function(x, y, ...)
{
  df <- data.frame(y=y, x)

  caret::train(y ~ ., data=df,
               method = "glmnet",
               trControl=caret::trainControl(method = "none"), # no cv
               verbose = FALSE,
               tuneGrid=data.frame(...))
}

predict_func <- function(obj, newx)
{
  colnames(newx) <- paste0("X", 1:ncol(newx))
  caret::predict.train(object=obj, newdata=newx, type = "raw")
}

z <- ahead::dynrmf(USAccDeaths, h=15, level=95, fit_func = fit_func,
              fit_params = list(alpha=0.5, lambda=0.1),
              predict_func = predict_func)
autoplot(z)
```

![image-title-here]({{base}}/images/2021-12-20/2021-12-20-image4.png){:class="img-responsive"}
