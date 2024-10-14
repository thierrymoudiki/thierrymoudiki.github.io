---
layout: post
title: "Gradient-Boosting anything (alert: high performance): Part2, R version"
description: "Gradient boosting with any regression algorithm in Python package mlsauce. Part2, R version"
date: 2024-10-14
categories: R
comments: true
---

[Last week](https://thierrymoudiki.github.io/blog/2024/10/06/python/r/genericboosting), I presented a functionality from Python package `mlsauce` that allows gradient boosting of any regression algorithm. This post is about the [R version](https://github.com/Techtonique/mlsauce_r). 

I think (?) I finally wrapped my head around the process of creating an R package from a Python package systematically, using `reticulate`. By default when _onload_ ing, `reticulate` creates a Python virtual environment in the working directory (should ask). Then [you need to tell R](https://github.com/Techtonique/mlsauce_r/blob/main/R/zzz.R) where to find the Python packages: in that virtual environment. 

Keep in mind that there are many layers here: Cython, C, Python, R, and the R package interface, so it **may** not work on your machine. I only tested it on Linux Ubuntu 20.04. Also, every model presented below is using its default hyperparameters...

```R
devtools::install_github("Techtonique/mlsauce_r")
```

```R
library(mlsauce)

# 1 ---- MASS::Aids2 data set: Australian AIDS Survival Data

 X <- model.matrix(status ~ ., data=MASS::Aids2)[,-1]
 y <- as.integer(MASS::Aids2$status) - 1 
 
 n <- dim(X)[1]
 p <- dim(X)[2]
 
 set.seed(213)
 train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
 test_index <- -train_index
 
 X_train <- as.matrix(X[train_index, ])
 y_train <- as.integer(y[train_index])
 X_test <- as.matrix(X[test_index, ])
 y_test <- as.integer(y[test_index])
 
 obj <- LazyBoostingClassifier(verbose=0, ignore_warnings=TRUE,
                               custom_metric=NULL, preprocess=FALSE, 
                               random_state=42L)
 
 res <- obj$fit(X_train, X_test, y_train, y_test)
 
 print(res[[1]])

# 2 ---- MASS::bacteria data set

 X <- model.matrix(y ~ ., data=MASS::bacteria)[,-1]
 y <- as.integer(MASS::bacteria$y) - 1 
 
 n <- dim(X)[1]
 p <- dim(X)[2]
 
 set.seed(213)
 train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
 test_index <- -train_index
 
 X_train <- as.matrix(X[train_index, ])
 y_train <- as.integer(y[train_index])
 X_test <- as.matrix(X[test_index, ])
 y_test <- as.integer(y[test_index])
 
 obj <- LazyBoostingClassifier(verbose=0, ignore_warnings=TRUE,
                               custom_metric=NULL, preprocess=FALSE, 
                               random_state=42L)
 
 res <- obj$fit(X_train, X_test, y_train, y_test)
 
 print(res[[1]])

# 3 - MASS::VA: Veteran's Administration Lung Cancer Trial -----

X <- model.matrix(status ~ ., data=MASS::VA)[,-2]
y <- as.integer(MASS::VA$status)

n <- dim(X)[1]
p <- dim(X)[2]

set.seed(213)
train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
test_index <- -train_index

X_train <- as.matrix(X[train_index, ])
y_train <- as.integer(y[train_index])
X_test <- as.matrix(X[test_index, ])
y_test <- as.integer(y[test_index])

obj <- LazyBoostingClassifier(verbose=0, ignore_warnings=TRUE,
                              custom_metric=NULL, preprocess=FALSE, 
                              random_state=42L)

res <- obj$fit(X_train, X_test, y_train, y_test)

print(res[[1]])



#  4 ---- iris data set
 data(iris)
 
 X <- as.matrix(iris[, 1:4])
 y <- as.integer(iris[, 5]) - 1L
 
 n <- dim(X)[1]
 p <- dim(X)[2]
 
 set.seed(2134)
 train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
 test_index <- -train_index
 
 X_train <- as.matrix(X[train_index, ])
 y_train <- as.integer(y[train_index])
 X_test <- as.matrix(X[test_index, ])
 y_test <- as.integer(y[test_index])
 
 obj <- LazyBoostingClassifier(verbose=0, ignore_warnings=TRUE,
                               custom_metric=NULL, preprocess=FALSE, 
                               random_state=42L)
 
 res <- obj$fit(X_train, X_test, y_train, y_test)
 
 print(res[[1]])
```

```R
                                                       Accuracy Balanced Accuracy   ROC AUC  F1 Score   Time Taken
GenericBooster(ExtraTreeRegressor)                    0.9945184         0.9937213 0.9937213 0.9945155  0.214021206
RandomForestClassifier                                0.9937353         0.9927233 0.9927233 0.9937308  0.241777658
GenericBooster(DecisionTreeRegressor)                 0.9937353         0.9927233 0.9927233 0.9937308  0.402045250
XGBClassifier                                         0.9929522         0.9917253 0.9917253 0.9929459  0.500556469
GenericBooster(LinearRegression)                      0.9146437         0.9279997 0.9279997 0.9155734  1.003419161
GenericBooster(Ridge)                                 0.9146437         0.9279997 0.9279997 0.9155734  0.171954870
GenericBooster(TransformedTargetRegressor)            0.9146437         0.9279997 0.9279997 0.9155734  1.493569136
GenericBooster(RidgeCV)                               0.9138606         0.9273553 0.9273553 0.9148027  1.519951582
GenericBooster(Lars)                                  0.8942835         0.9034663 0.9034663 0.8953262  0.559155703
GenericBooster(KNeighborsRegressor)                   0.8480814         0.8343275 0.8343275 0.8469727  3.606536388
GenericBooster(MultiTask(BayesianRidge))              0.8269381         0.8427488 0.8427488 0.8289683  5.020956278
GenericBooster(MultiTask(SGDRegressor))               0.8245889         0.8447062 0.8447062 0.8265735  0.838425159
GenericBooster(MultiTask(TweedieRegressor))           0.7916993         0.8218884 0.8218884 0.7930700  0.855145216
GenericBooster(MultiTask(LinearSVR))                  0.7760376         0.7817689 0.7817689 0.7784222 19.247414351
GenericBooster(MultiTask(PassiveAggressiveRegressor)) 0.6108066         0.5747268 0.5747268 0.6013961  0.807721853
GenericBooster(DummyRegressor)                        0.6076742         0.5000000 0.5000000 0.4593816  0.006204605
GenericBooster(ElasticNet)                            0.6076742         0.5000000 0.5000000 0.4593816  0.008437157
GenericBooster(LassoLars)                             0.6076742         0.5000000 0.5000000 0.4593816  0.007481813
GenericBooster(MultiTaskLasso)                        0.6076742         0.5000000 0.5000000 0.4593816  0.007218838
GenericBooster(MultiTaskElasticNet)                   0.6076742         0.5000000 0.5000000 0.4593816  0.007399082
GenericBooster(Lasso)                                 0.6076742         0.5000000 0.5000000 0.4593816  0.008338690
GenericBooster(MultiTask(QuantileRegressor))          0.6076742         0.5000000 0.5000000 0.4593816 23.718370676
```

```R
                                                      Accuracy Balanced Accuracy   ROC AUC  F1 Score   Time Taken
GenericBooster(Ridge)                                     0.83         0.6936322 0.6936322 0.8242597  0.142935991
RandomForestClassifier                                    0.82         0.6068876 0.6068876 0.7930897  0.123536110
GenericBooster(DecisionTreeRegressor)                     0.81         0.6208577 0.6208577 0.7924833  0.141850471
GenericBooster(ElasticNet)                                0.81         0.5000000 0.5000000 0.7249724  0.006238937
GenericBooster(DummyRegressor)                            0.81         0.5000000 0.5000000 0.7249724  0.004231691
GenericBooster(MultiTask(QuantileRegressor))              0.81         0.5000000 0.5000000 0.7249724  1.892220974
GenericBooster(KNeighborsRegressor)                       0.81         0.6007147 0.6007147 0.7855172  0.314696789
GenericBooster(LassoLars)                                 0.81         0.5000000 0.5000000 0.7249724  0.005181074
GenericBooster(Lasso)                                     0.81         0.5000000 0.5000000 0.7249724  0.006288290
GenericBooster(MultiTaskElasticNet)                       0.81         0.5000000 0.5000000 0.7249724  0.005995750
GenericBooster(LinearRegression)                          0.81         0.7014295 0.7014295 0.8118458  8.247184992
GenericBooster(MultiTaskLasso)                            0.81         0.5000000 0.5000000 0.7249724  0.005704165
GenericBooster(TransformedTargetRegressor)                0.81         0.7014295 0.7014295 0.8118458 11.520040274
GenericBooster(RidgeCV)                                   0.80         0.6146849 0.6146849 0.7848214 18.733512878
GenericBooster(ExtraTreeRegressor)                        0.79         0.6286550 0.6286550 0.7829091  0.128053665
GenericBooster(MultiTask(SGDRegressor))                   0.77         0.6364522 0.6364522 0.7722344  0.497604609
GenericBooster(MultiTask(LinearSVR))                      0.77         0.5760234 0.5760234 0.7560189  3.430291653
GenericBooster(MultiTask(TweedieRegressor))               0.76         0.6504224 0.6504224 0.7683906  0.830221891
GenericBooster(MultiTask(BayesianRidge))                  0.75         0.6241066 0.6241066 0.7567879 33.510189772
XGBClassifier                                             0.74         0.5575049 0.5575049 0.7343631  0.398538351
GenericBooster(MultiTask(PassiveAggressiveRegressor))     0.69         0.5669266 0.5669266 0.7071111  0.511162758
```

```R
                                                       Accuracy Balanced Accuracy   ROC AUC  F1 Score  Time Taken
GenericBooster(ElasticNet)                            0.9523810         0.5000000 0.5000000 0.9291521 0.005884886
GenericBooster(DummyRegressor)                        0.9523810         0.5000000 0.5000000 0.9291521 0.004161835
GenericBooster(MultiTaskElasticNet)                   0.9523810         0.5000000 0.5000000 0.9291521 0.005287170
GenericBooster(MultiTaskLasso)                        0.9523810         0.5000000 0.5000000 0.9291521 0.005177975
GenericBooster(Lasso)                                 0.9523810         0.5000000 0.5000000 0.9291521 0.005835295
GenericBooster(LassoLars)                             0.9523810         0.5000000 0.5000000 0.9291521 0.004950762
GenericBooster(MultiTask(QuantileRegressor))          0.9523810         0.5000000 0.5000000 0.9291521 0.860675573
GenericBooster(MultiTask(LinearSVR))                  0.9523810         0.5000000 0.5000000 0.9291521 0.375921965
GenericBooster(RidgeCV)                               0.9206349         0.4833333 0.4833333 0.9130264 0.134217024
RandomForestClassifier                                0.8888889         0.4666667 0.4666667 0.8963585 0.098402023
GenericBooster(Ridge)                                 0.8730159         0.4583333 0.4583333 0.8878128 0.102342844
GenericBooster(TransformedTargetRegressor)            0.8730159         0.4583333 0.4583333 0.8878128 0.142889023
GenericBooster(LinearRegression)                      0.8730159         0.4583333 0.4583333 0.8878128 0.091339111
XGBClassifier                                         0.8571429         0.4500000 0.4500000 0.8791209 0.246236563
GenericBooster(Lars)                                  0.8571429         0.4500000 0.4500000 0.8791209 0.350574017
GenericBooster(ExtraTreeRegressor)                    0.8412698         0.4416667 0.4416667 0.8702791 0.086171627
GenericBooster(DecisionTreeRegressor)                 0.8095238         0.4250000 0.4250000 0.8521303 0.094847202
GenericBooster(KNeighborsRegressor)                   0.7142857         0.3750000 0.3750000 0.7936508 0.179689169
GenericBooster(MultiTask(TweedieRegressor))           0.7142857         0.3750000 0.3750000 0.7936508 0.727254391
GenericBooster(MultiTask(BayesianRidge))              0.6825397         0.3583333 0.3583333 0.7726864 0.787358522
GenericBooster(MultiTask(SGDRegressor))               0.6190476         0.3250000 0.3250000 0.7282913 0.488824844
GenericBooster(MultiTask(PassiveAggressiveRegressor)) 0.2857143         0.1500000 0.1500000 0.4232804 0.458680630
```

```R
                                                       Accuracy Balanced Accuracy ROC AUC  F1 Score  Time Taken
GenericBooster(RidgeCV)                               1.0000000         1.0000000    <NA> 1.0000000 0.100842953
GenericBooster(LinearRegression)                      1.0000000         1.0000000    <NA> 1.0000000 0.082373857
GenericBooster(TransformedTargetRegressor)            1.0000000         1.0000000    <NA> 1.0000000 0.134787083
GenericBooster(Ridge)                                 0.9848485         0.9814815    <NA> 0.9847932 0.100898504
RandomForestClassifier                                0.9848485         0.9855072    <NA> 0.9848849 0.107151031
XGBClassifier                                         0.9696970         0.9669887    <NA> 0.9696970 0.030355215
GenericBooster(ExtraTreeRegressor)                    0.9696970         0.9710145    <NA> 0.9698057 0.094514847
GenericBooster(DecisionTreeRegressor)                 0.9696970         0.9629630    <NA> 0.9694370 0.112530708
GenericBooster(KNeighborsRegressor)                   0.9242424         0.9194847    <NA> 0.9244244 0.198357344
GenericBooster(MultiTask(SGDRegressor))               0.8636364         0.8574879    <NA> 0.8641242 0.648634434
GenericBooster(MultiTask(TweedieRegressor))           0.8636364         0.8574879    <NA> 0.8641242 1.208353758
GenericBooster(MultiTask(PassiveAggressiveRegressor)) 0.6969697         0.7020934    <NA> 0.6593600 0.788725376
GenericBooster(MultiTask(LinearSVR))                  0.6969697         0.7101449    <NA> 0.6345321 1.032293797
GenericBooster(MultiTask(BayesianRidge))              0.6666667         0.6811594    <NA> 0.5771073 1.168911695
GenericBooster(MultiTask(QuantileRegressor))          0.3787879         0.3333333    <NA> 0.2081252 1.453683376
GenericBooster(MultiTaskElasticNet)                   0.3333333         0.3913043    <NA> 0.2259820 0.026070118
GenericBooster(Lars)                                  0.2878788         0.2850564    <NA> 0.2899037 0.424385309
GenericBooster(DummyRegressor)                        0.2727273         0.3333333    <NA> 0.1168831 0.003482342
GenericBooster(ElasticNet)                            0.2727273         0.3333333    <NA> 0.1168831 0.005678177
GenericBooster(MultiTaskLasso)                        0.2727273         0.3333333    <NA> 0.1168831 0.005366087
GenericBooster(Lasso)                                 0.2727273         0.3333333    <NA> 0.1168831 0.006078720
GenericBooster(LassoLars)                             0.2727273         0.3333333    <NA> 0.1168831 0.004939079
```

If you want use one of the models, type `?mlsauce::GradientBoostingClassifier` 
or `?mlsauce::GradientBoostingRegressor` in the console.


![xxx]({{base}}/images/2024-10-14/2024-10-14-image1.png){:class="img-responsive"}  