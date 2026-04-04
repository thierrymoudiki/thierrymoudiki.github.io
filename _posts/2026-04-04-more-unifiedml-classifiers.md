---
layout: post
title: "One interface, (Almost) Every Classifier: unifiedml v0.2.1"
description: "A new version of `unifiedml` is out; available on CRAN. `unifiedml` is an effort to offer a unified interface to R's machine learning models."
date: 2026-04-04
categories: R
comments: true
---

A new version of `unifiedml` is out; available on CRAN. `unifiedml` is an effort to offer a unified interface to R's machine learning models. 

The main change in this version `0.2.1` is the removal of `type` (of prediction) from `predict`, and the use of`...` instead, which is more generic and flexible. 

**This post contains advanced examples of use of `unifiedml` for classification**, with `ranger` and `xgboost`. More examples have been added to [the package vignettes](https://cloud.r-project.org/web/packages/unifiedml/vignettes/unifiedml-vignette.html) too.  


```R
install.packages("unifiedml")
```


```R
install.packages(c("ranger"))
```


```R
library("unifiedml")
```

    Loading required package: doParallel
    
    Loading required package: foreach
    
    Loading required package: iterators
    
    Loading required package: parallel
    
    Loading required package: R6
    

# 1 - `ranger` example

```R
library(ranger)
```


```R


# 2 - 'ranger' classification ---------------------------

# -------------------------------
# S3 wrapper for ranger
# -------------------------------

# Fit function remains the same
my_ranger <- function(x, y, ...) {
  if (!is.data.frame(x)) x <- as.data.frame(x)
  y <- as.factor(y)
  colnames(x) <- paste0("X", seq_len(ncol(x)))
  df <- data.frame(y = y, x)
  fit <- ranger::ranger(y ~ ., data = df, probability = TRUE, ...)
  structure(list(fit = fit), class = "my_ranger")
}

# Predict only with newdata
predict.my_ranger <- function(object, newdata = NULL, newx = NULL, ...) {
  if (!is.null(newx)) newdata <- newx
  if (is.null(newdata)) stop("No data provided for prediction")
#  misc::debug_print(newx)
#  misc::debug_print(newdata)
  if (is.matrix(newdata)) newdata <- as.data.frame(newdata)
#  misc::debug_print(newdata)
  # Unconditionally rename to match training
  colnames(newdata) <- paste0("X", seq_len(ncol(newdata)))
#  misc::debug_print(newdata)
  preds <- predict(object$fit, data = newdata)$predictions
#  misc::debug_print(newdata)
  if (is.matrix(preds) && ncol(preds) == 2) {
    lvls <- colnames(preds)
    return(ifelse(preds[, 2] > 0.5, lvls[2], lvls[1]))
  }

  preds
}

# Print method
print.my_ranger <- function(x, ...) {
  cat("my_ranger model\n")
  print(x$fit)
}

# -------------------------------
# Example: Iris binary classification
# -------------------------------

set.seed(123)
iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
X_binary <- iris_binary[, 1:4]
y_binary <- as.factor(as.character(iris_binary$Species))

# Train/test split
train_idx <- sample(seq_len(nrow(X_binary)), size = 0.7 * nrow(X_binary))
X_train <- X_binary[train_idx, ]
y_train <- y_binary[train_idx]
X_test <- X_binary[-train_idx, ]
y_test <- y_binary[-train_idx]

# Initialize and fit model
# Initialize model
mod <- Model$new(my_ranger)

# Fit on training data only
mod$fit(X_train, y_train, num.trees = 150L)

# Predict on test set
preds <- mod$predict(X_test)

# Evaluate
table(Predicted = preds, True =y_test)
mean(preds == y_test)  # Accuracy



# 5-fold cross-validation on training set
cv_scores <- cross_val_score(
  mod,
  X_train,
  y_train,
  num.trees = 150L,
  cv = 5L
)

cv_scores
mean(cv_scores)  # average CV accuracy

```


                True
    Predicted    setosa versicolor
      setosa         15          0
      versicolor      0         15



1


      |======================================================================| 100%



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li></ol>




1

# 2 - `xgboost` example

```R
library(xgboost)

my_xgboost <- function(x, y, ...) {
  
  # Convert to matrix safely
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  
  # Handle factors
  if (is.factor(y)) {
    y <- as.numeric(y) - 1
  }
  
  fit <- xgboost::xgboost(
    data = x,
    label = y,
    ...
  )
  
  structure(list(fit = fit), class = "my_xgboost")
}

predict.my_xgboost <- function(object, newdata, ...) {
  
  # Ensure matrix
  newdata <- as.matrix(newdata)
  
  preds <- predict(object$fit, newdata)
  
  # If binary classification → convert probs to class
  if (!is.null(object$fit$params$objective) &&
      grepl("binary", object$fit$params$objective)) {
    
    return(ifelse(preds > 0.5, 1, 0))
  }
  
  preds
}

predict.my_xgboost <- function(object, newdata = NULL, newx = NULL, ...) {
  
  # Accept both conventions
  if (!is.null(newx)) {
    newdata <- newx
  }
  
  newdata <- as.matrix(newdata)
  
  preds <- predict(object$fit, newdata)
  
  # Binary classification → class labels
  if (!is.null(object$fit$params$objective) &&
      grepl("binary", object$fit$params$objective)) {
    
    return(ifelse(preds > 0.5, 1, 0))
  }
  
  preds
}

print.my_xgboost <- function(x, ...) {
  cat("my_xgboost model\n")
  print(x$fit)
}


set.seed(123)  # for reproducibility

# Binary subset
iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
X_binary <- as.matrix(iris_binary[, 1:4])
y_binary <- as.factor(as.character(iris_binary$Species))

# Split indices: 70% train, 30% test
train_idx <- sample(seq_len(nrow(X_binary)), size = 0.7 * nrow(X_binary))
X_train <- X_binary[train_idx, ]
y_train <- y_binary[train_idx]
X_test <- X_binary[-train_idx, ]
y_test <- y_binary[-train_idx]

# Initialize model
mod <- Model$new(my_xgboost)

# Fit on training data only
mod$fit(X_train, y_train, nrounds = 50, objective = "binary:logistic")

# Predict on test set
preds <- mod$predict(X_test)

# Evaluate
table(Predicted = preds, True =y_test)
mean(preds == y_test)  # Accuracy



# 5-fold cross-validation on training set
cv_scores <- cross_val_score(
  mod, 
  X_train, 
  y_train, 
  nrounds = 50, 
  objective = "binary:logistic", 
  cv = 5L
)

cv_scores
mean(cv_scores)  # average CV accuracy
```

![image-title-here]({{base}}/images/2026-04-04/2026-04-04-image1.png){:class="img-responsive"}


