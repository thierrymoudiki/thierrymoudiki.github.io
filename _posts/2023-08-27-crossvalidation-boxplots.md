---
layout: post
title: "Comparing cross-validation results using crossval_ml and boxplots"
description: Cross-validation with xgboost, Random Forest, glmnet, comparing samples distribution
date: 2023-08-27
categories: [R, Misc]
---

# Table of contents 

 - 0 - Install packages + global parameters
 - 1 - Regression example
 - 2 - Classification example

# 0 - Install packages + global parameters

Let's start by installing the main package, [`crossvalidation`](https://github.com/Techtonique/crossvalidation) (version 0.5.0):

- __1st method__: from [R-universe](https://techtonique.r-universe.dev) (where you can also package's long-form descriptions a.k.a vignettes)

In R console:

```R
options(repos = c(
    techtonique = 'https://techtonique.r-universe.dev',
    CRAN = 'https://cloud.r-project.org'))
    
install.packages("crossvalidation")
```

- __2nd method__: from Github

In R console:

```R
remotes::install_github("Techtonique/crossvalidation")
```

When using this package, please note that I'm calling a "validation set", what is usually called a "test set". Because it makes more sense to me (even if I'm the only one in the world doing this). 

Number of folds and repeats for the cross-validation procedure:

```R
(n_folds <- 10)
(repeats <- 5)
```

Loading the other Statistical/Machine Learning packages needed for this post: 

```R
library(glmnet)
library(xgboost)
library(Matrix)
library(randomForest)
library(crossvalidation)
```

# 1 - Regression example

```R
# dataset
 set.seed(123)
 n <- 100 ; p <- 5
 X <- matrix(rnorm(n * p), n, p)
 print(head(X))
 y <- rnorm(n)
 print(head(y))
```

## least squares

```R
# linear model example
(cv_lm <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, 
                                       repeats = repeats,  show_progress = FALSE))
```

## `glmnet`

```R
# glmnet example -----

# fit glmnet, with alpha = 1, lambda = 0.1
(cv_glmnet <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, 
                                           repeats = repeats, 
                                           show_progress = FALSE,
                                           fit_func = glmnet, predict_func = predict,
                                           packages = c("glmnet", "Matrix"), 
                                           fit_params = list(alpha = 0, lambda = 0.01)))
```

## Random Forest

```R
# randomForest example -----

# fit randomForest with mtry = 4
(
  cv_rf <- crossvalidation::crossval_ml(
    x = X,
    y = y,
    k = n_folds,
    repeats = repeats,
    show_progress = FALSE,
    fit_func = randomForest::randomForest,
    predict_func = predict,
    packages = "randomForest",
    fit_params = list(mtry = 4)
  )
)
```

## `xgboost`

```R
# xgboost example -----

# The response and covariates are named 'label' and 'data'
# So, we do this:

f_xgboost <- function(x, y, ...) xgboost::xgboost(data = x, label = y, ...)

# fit xgboost with nrounds = 10

(
  cv_xgboost <-
    crossvalidation::crossval_ml(
      x = X,
      y = y,
      k = n_folds,
      repeats = repeats,
      show_progress = FALSE,
      fit_func = f_xgboost,
      predict_func = predict,
      #packages = "xgboost",
      fit_params = list(nrounds = 10,
                        verbose = FALSE)
    )
)
```

# `glmnet`

```R
# glmnet example -----

# fit glmnet, with alpha = 0.5, lambda = 0.1
 cv_glmnet1 <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, 
                                            repeats = repeats,
                              show_progress = FALSE,
                              fit_func = glmnet, 
                              predict_func = predict.glmnet,
                              packages = c("glmnet", "Matrix"), 
                              fit_params = list(alpha = 0.5, 
                                                lambda = 0.1, 
                                                family = "gaussian"))

# fit glmnet, with alpha = 0, lambda = 0.01

 cv_glmnet2 <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, repeats = repeats, show_progress = FALSE,
 fit_func = glmnet::glmnet, predict_func = predict.glmnet,
 packages = c("glmnet", "Matrix"), fit_params = list(alpha = 0, lambda = 0.01, family = "gaussian"))

 # fit glmnet, with alpha = 0, lambda = 0.01

 cv_glmnet3 <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, repeats = repeats, show_progress = FALSE,
 fit_func = glmnet::glmnet, predict_func = predict.glmnet,
 packages = c("glmnet", "Matrix"), fit_params = list(alpha = 0, lambda = 0.01))

```

## boxplots for regression

```R
(samples <- crossvalidation::create_samples(cv_lm, cv_glmnet1,
                           cv_glmnet2, cv_glmnet3,
                           cv_rf, cv_xgboost,
                           model_names = c("lm", "glmnet1", "glmnet2", 
                                           "glmnet3", "rf", "xgb")))
```

```R
boxplot(samples, main = "RMSE")
```


# 2 - Classification example

```R
data(iris)
```

```R
X <- as.matrix(iris[, 1:4])
print(head(X))
y <- factor(as.numeric(iris$Species))
print(head(y))
```

## `glmnet`

```R
# glmnet example -----

predict_glmnet <- function(object, newx) {
  as.numeric(predict(object = object, 
          newx = newx,
          type = "class"))
}

(cv_glmnet_1 <- crossvalidation::crossval_ml(x = X, 
                                             y = as.integer(iris$Species), 
                                             k = n_folds, repeats = repeats, show_progress = FALSE,
 fit_func = glmnet, predict_func = predict_glmnet,
 packages = c("glmnet", "Matrix"), fit_params = list(alpha = 0.5, lambda = 0.1, family = "multinomial"))) # better to use `nlambda`


(cv_glmnet_2 <- crossvalidation::crossval_ml(x = X, 
                                             y = as.integer(iris$Species), 
                                             k = n_folds, repeats = repeats, show_progress = FALSE,
 fit_func = glmnet::glmnet, predict_func = predict_glmnet,
 packages = c("glmnet", "Matrix"), fit_params = list(alpha = 0, lambda = 0.01, family = "multinomial")))

(cv_glmnet_3 <- crossvalidation::crossval_ml(x = X, y = as.integer(iris$Species) , k = n_folds, repeats = repeats, show_progress = FALSE, 
 fit_func = glmnet::glmnet, predict_func = predict_glmnet,
 packages = c("glmnet", "Matrix"), fit_params = list(alpha = 1, lambda = 0.01, family = "multinomial")))
```

## Random Forest

```R
# randomForest example -----

# fit randomForest with mtry = 4
(
  cv_rf <- crossvalidation::crossval_ml(
    x = X,
    y = y,
    k = n_folds,
    repeats = repeats,
    show_progress = FALSE,
    fit_func = randomForest::randomForest,
    predict_func = predict,
    #packages = "randomForest",
    fit_params = list(mtry = 2L)
  )
)
```

## xgboost

```R
y <- as.integer(iris$Species) - 1

print(y)
```

```R
# xgboost example -----

# fit xgboost with nrounds = 10

f_xgboost <- function(x, y, ...) {
  #xgb_train = xgb.DMatrix(data=x, label=y)
  xgboost::xgboost(data = x, label = y, ...)
} 

(cv_xgboost <- crossvalidation::crossval_ml(x = X, y = y, k = n_folds, repeats = repeats,  fit_func = f_xgboost, predict_func = predict,
                             packages = "xgboost", 
                             show_progress = FALSE,
                             fit_params = list(nrounds = 50L,
                                               verbose = FALSE,
                                               params = list(max_depth = 3L,
                                               eta = 0.1,  
                                               subsample = 0.8,
                                               colsample_bytree = 0.8,
                                               objective = "multi:softmax", 
                                               num_class = 3L))))
```

## boxplots for classification

```R
(samples <- crossvalidation::create_samples(cv_rf, cv_glmnet_1,
                                            cv_glmnet_2, cv_glmnet_3, 
                                            cv_xgboost, 
                                            model_names = c("rf", "glmnet1", "glmnet2", 
                                           "glmnet3", "xgb")))
```

```R
boxplot(samples, main = "Accuracy")
abline(h = 1, col = "red", lty = 2, lwd = 2)
```
