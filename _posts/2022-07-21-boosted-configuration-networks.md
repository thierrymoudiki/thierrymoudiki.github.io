---
layout: post
title: "Boosted Configuration (_neural_) Networks for classification"
description: "Some examples of Boosted Configuration Networks for classification"
date: 2022-07-21
categories: [R, Misc]
---

A few years ago in 2018, I discussed Boosted Configuration (_neural_) Networks (BCN for multivariate time series forecasting) [in this document](https://www.researchgate.net/publication/332291211_Forecasting_multivariate_time_series_with_boosted_configuration_networks). Unlike [Stochastic Configuration Networks](https://arxiv.org/pdf/1702.03180.pdf) from which they are inspired, BCNs aren't **randomized**. Rather, they are closer to Gradient Boosting Machines and Matching Pursuit algorithms; with base learners being single-layered feedforward _neural_ networks -- that are actually optimized at each iteration of the algorithm. 

The mathematician that you are has certainly been 
asking himself questions about the convexity of the problem at line 4, algorithm 1 (in 
[the document](https://www.researchgate.net/publication/332291211_Forecasting_multivariate_time_series_with_boosted_configuration_networks)). As of July 2022, there are unfortunately no answers to that question. BCNs works well __empirically__, as we'll see, and finding the maximum at line 4 of the algorithm is achieved, by default, with R's `stats::nlminb`. Other derivative-free optimizers are available in [R package `bcn`](https://techtonique.r-universe.dev/ui#package:bcn). 


As it will be shown in this document, BCNs can be used __for classification__. For this purpose, and as implemented in [R package `bcn`](https://techtonique.r-universe.dev/ui#package:bcn), the response (variable to be explained) containing the classes is one-hot encoded as a matrix of probabilities equal to 0 or 1. Then, the classification technique dealing with a one-hot encoded response matrix is similar to the one presented [in this  post](https://thierrymoudiki.github.io/blog/2021/09/26/python/quasirandomizednn/classification-using-regression).

6 _toy_ datasets are used for this basic demo of R package `bcn`: Iris, Wine, Ionosphere, Wisconsin Breast, Digits, Penguins. 
For each dataset, hyperparameter tuning has already been done. Repeated 5-fold cross-validation was carried out on 80% of the data, for each dataset, and the accuracy reported in the table below is calculated on the remaining 20% of the data. BCN results are compared to  [Random Forest](https://cran.r-project.org/web/packages/randomForest/)'s (with default parameters), in order to verify that BCN results are not absurd -- it's not a competition between Random Forest and BCN here. 

In the examples, you'll also notice that BCN's adjustment to a dataset can be relatively slow when the number of explanatory variables is high (>30, see the `digits` dataset example, initially with 64 covariates). This is because of line 4, algorithm 1, too: an optimization problem in a high dimensional space is repeated at each iteration of the algorithm.

The future for [R package `bcn`](https://github.com/Techtonique/bcn) (in no particular order)? 

- Implement BCN for regression (a continuous response)
- Improve the speed of execution for high dimensional problems
- Implement a Python version 


| Dataset      | BCN test set Accuracy | Random Forest test set accuracy |
|--------------|:-----:|-----------:|
| iris |  **100%** |        93.33% |
| Wine |  **97.22%** |      94.44% |
| Ionosphere |  90.14% |      **95.77%** |
| Breast cancer |  **99.12%** |        94.73% |
| Digits |  97.5% |        **98.61%** |
| Penguins |  **100%** |        **100%** |



**Content**

- [0 - Installing/Loading Packages](#installing-and-loading-packages)
- [1 - iris dataset](#iris-dataset)
- [2 - wine dataset](#wine-dataset)
- [3 - Ionosphere dataset](#ionosphere-dataset)
- [4 - Breast Cancer dataset](#breast-cancer-dataset)
- [5 - digits dataset](#digits-dataset)
- [6 - Palmer Penguins dataset](#penguins-dataset)



# 0 - Installing and loading packages

Installing `bcn` From Github:

```R
devtools::install_github("Techtonique/bcn")

# Browse the bcn manual pages
help(package = 'bcn')
```

Installing `bcn` from R universe:

```R
# Enable repository from techtonique
options(repos = c(
  techtonique = 'https://techtonique.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'))
  
# Download and install bcn in R
install.packages('bcn')

# Browse the bcn manual pages
help(package = 'bcn')
````

Loading packages: 

```R
library(bcn) # Boosted Configuration networks (only for classification, for now)
library(mlbench) # Machine Learning Benchmark Problems
library(caret)
library(randomForest)
library(pROC)
```


# 1 - iris dataset 

```R
data("iris")
```

```R
head(iris)

dim(iris)

set.seed(1234)
train_idx <- sample(nrow(iris), 0.8 * nrow(iris))
X_train <- as.matrix(iris[train_idx, -ncol(iris)])
X_test <- as.matrix(iris[-train_idx, -ncol(iris)])
y_train <- iris$Species[train_idx]
y_test <- iris$Species[-train_idx]
```

```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 10L, nu = 0.335855,
                    lam = 10**0.7837525, r = 1 - 10**(-5.470031), tol = 10**-7,
                    activation = "tanh", type_optim = "nlminb", show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```

```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```


# 2-  wine dataset 


```R
data(wine)
```

```R
head(wine)

dim(wine)

set.seed(1234)
train_idx <- sample(nrow(wine), 0.8 * nrow(wine))
X_train <- as.matrix(wine[train_idx, -ncol(wine)])
X_test <- as.matrix(wine[-train_idx, -ncol(wine)])
y_train <- as.factor(wine$target[train_idx])
y_test <- as.factor(wine$target[-train_idx])
```

```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 6L, nu = 0.8715725,
                    lam = 10**0.2143678, r = 1 - 10**(-6.1072786),
                    tol = 10**-4.9605713, show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```


```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```

# 3 - Ionosphere dataset 

```R
data("Ionosphere")
```

```R
head(Ionosphere)

dim(Ionosphere)

Ionosphere$V1 <- as.numeric(Ionosphere$V1)
Ionosphere$V2 <- NULL
set.seed(1234)
train_idx <- sample(nrow(Ionosphere), 0.8 * nrow(Ionosphere))
X_train <- as.matrix(Ionosphere[train_idx, -ncol(Ionosphere)])
X_test <- as.matrix(Ionosphere[-train_idx, -ncol(Ionosphere)])
y_train <- as.factor(Ionosphere$Class[train_idx])
y_test <- as.factor(Ionosphere$Class[-train_idx])
```

```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train,
                     y = y_train, B = 50L,
                     nu = 0.5182606,
                     lam = 10**1.323274,
                     r = 1 - 10**(-6.694688),
                     col_sample = 0.7956659,
                     tol = 10**-7,
                     verbose=FALSE,
                     show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```


```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```

```R
roc_obj <- pROC::roc(as.numeric(y_test), as.numeric(preds))
pROC::auc(roc_obj)
```

```R
roc_obj_rf <- pROC::roc(as.numeric(y_test), as.numeric(predict(rf, newdata=as.matrix(X_test))))
pROC::auc(roc_obj_rf)
```


# 4 - breast cancer dataset 

```R
data("breast_cancer")
```

```R
head(breast_cancer)

dim(breast_cancer)

set.seed(1234)
train_idx <- sample(nrow(breast_cancer), 0.8 * nrow(breast_cancer))
X_train <- as.matrix(breast_cancer[train_idx, -ncol(breast_cancer)])
X_test <- as.matrix(breast_cancer[-train_idx, -ncol(breast_cancer)])
y_train <- as.factor(breast_cancer$target[train_idx])
y_test <- as.factor(breast_cancer$target[-train_idx])
```


```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 31L, nu = 0.4412851,
                    lam = 10**-0.2439358, r = 1 - 10**(-7), col_sample = 0.5, tol = 10**-2, show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```


```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```

```R
roc_obj <- pROC::roc(as.numeric(y_test), as.numeric(preds))
pROC::auc(roc_obj)
```

```R
roc_obj_rf <- pROC::roc(as.numeric(y_test), as.numeric(predict(rf, newdata=as.matrix(X_test))))
pROC::auc(roc_obj_rf)
```

# 5 - digits dataset 

```R
data("digits")
```

```R
head(digits)

dim(digits)

set.seed(1234)
train_idx <- sample(nrow(digits), 0.8 * nrow(digits))
X_train <- as.matrix(digits[train_idx, -ncol(digits)])
X_test <- as.matrix(digits[-train_idx, -ncol(digits)])
y_train <- as.factor(digits$target[train_idx])
X_train <- X_train[, -caret::nearZeroVar(X_train)]
y_test <- as.factor(digits$target[-train_idx])
X_test <- X_test[, colnames(X_train)]
```

```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train,
                    y = y_train, B = 50L,
                    nu = 0.6549268,
                    lam = 10**0.4635435,
                    r = 1 - 10**(-7),
                    col_sample = 0.8928518,
                    tol = 10**-5.483609,
                    verbose=FALSE,
                    show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```


```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```

# 6 - Penguins dataset

```R
data("penguins")
```

```R
penguins_ <- as.data.frame(penguins)

replacement <- median(penguins$bill_length_mm, na.rm = TRUE)
penguins_$bill_length_mm[is.na(penguins$bill_length_mm)] <- replacement

replacement <- median(penguins$bill_depth_mm, na.rm = TRUE)
penguins_$bill_depth_mm[is.na(penguins$bill_depth_mm)] <- replacement

replacement <- median(penguins$flipper_length_mm, na.rm = TRUE)
penguins_$flipper_length_mm[is.na(penguins$flipper_length_mm)] <- replacement

replacement <- median(penguins$body_mass_g, na.rm = TRUE)
penguins_$body_mass_g[is.na(penguins$body_mass_g)] <- replacement

# replacing NA's by the most frequent occurence
penguins_$sex[is.na(penguins$sex)] <- "male" # most frequent

print(summary(penguins_))
print(sum(is.na(penguins_)))

# one-hot encoding for covariates
penguins_mat <- model.matrix(species ~., data=penguins_)[,-1]
penguins_mat <- cbind(penguins_$species, penguins_mat)
penguins_mat <- as.data.frame(penguins_mat)
colnames(penguins_mat)[1] <- "species"

print(head(penguins_mat))
print(tail(penguins_mat))

y <- as.integer(penguins_mat$species)
X <- as.matrix(penguins_mat[,2:ncol(penguins_mat)])

n <- nrow(X)
p <- ncol(X)

set.seed(1234)
index_train <- sample(1:n, size=floor(0.8*n))
X_train <- X[index_train, ]
y_train <- factor(y[index_train])
X_test <- X[-index_train, ]
y_test <- factor(y[-index_train])
```

```R
ptm <- proc.time()
fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 23, nu = 0.470043,
                    lam = 10**-0.05766029, r = 1 - 10**(-7.905866), tol = 10**-7, 
                    show_progress = FALSE)
cat("Elapsed: ", (proc.time() - ptm)[3])
```

```R
plot(fit_obj$errors_norm, type='l')
```


```R
preds <- predict(fit_obj, newx = X_test)

mean(preds == y_test)

table(y_test, preds)
```

```R
rf <- randomForest::randomForest(x = X_train, y = y_train)
mean(predict(rf, newdata=as.matrix(X_test)) == y_test)
```

```R
print(head(predict(fit_obj, newx = X_test, type='probs')))
print(head(predict(rf, newdata=as.matrix(X_test), type='prob')))
```
