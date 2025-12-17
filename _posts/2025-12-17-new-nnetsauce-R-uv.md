---
layout: post
title: "Finally figured out a way to port python packages to R using uv and reticulate: example with nnetsauce"
description: "How to use nnetsauce in R"
date: 2025-12-17
categories: [R, Python]
comments: true
---

In this post, we will explore how to use nnetsauce in R. The updated code is available on GitHub: [Techntonique/nnetsauce_r](https://github.com/Techtonique/nnetsauce_r). 

## Install 

```bash
# pip install uv # if necessary
uv venv venv
source venv/bin/activate
uv pip install pip nnetsauce
```

**From GitHub**

```bash
install.packages("remotes")
remotes::install_github("Techtonique/nnetsauce_r")
```

## Examples

### Classification

```R
library(nnetsauce)

 set.seed(123)
 X <- as.matrix(iris[, 1:4])
 y <- as.integer(iris$Species) - 1L

 (index_train <- base::sample.int(n = nrow(X),
                                  size = floor(0.8*nrow(X)),
                                  replace = FALSE))
 X_train <- X[index_train, ]
 y_train <- y[index_train]
 X_test <- X[-index_train, ]
 y_test <- y[-index_train]

 obj <- nnetsauce::GLMClassifier(venv_path = "../venv")
 obj$fit(X_train, y_train)
 print(obj$score(X_test, y_test))
```

### Regression

```R
 library(datasets)

 n <- 20 ; p <- 5
 X <- matrix(rnorm(n * p), n, p) # no intercept!
 y <- rnorm(n)
 
 sklearn <- nnetsauce::get_sklearn(venv_path = "../venv")
 obj <- sklearn$tree$DecisionTreeRegressor()
 obj2 <- nnetsauce::RandomBagRegressor(obj, venv_path = "../venv")
 obj2$fit(X[1:12,], y[1:12])
 print(sqrt(mean((obj2$predict(X[13:20, ]) - y[13:20])**2)))
```


# AutoML

### Regression

```R
X <- MASS::Boston[,-14] # dataset has an ethical problem
y <- MASS::Boston$medv

set.seed(13)
(index_train <- base::sample.int(n = nrow(X),
                                 size = floor(0.8*nrow(X)),
                                 replace = FALSE))
X_train <- X[index_train, ]
y_train <- y[index_train]
X_test <- X[-index_train, ]
y_test <- y[-index_train]

obj <- LazyRegressor(venv_path = "../venv")
(res <- obj$fit(X_train, X_test, y_train, y_test))
```

```bash
##                                                 Adjusted R-Squared   R-Squared
## DeepCustomRegressor(ExtraTreesRegressor)                 0.9057805  0.91790778
## DeepCustomRegressor(BaggingRegressor)                    0.8634080  0.88098916
## DeepCustomRegressor(RandomForestRegressor)               0.8511052  0.87026992
## DeepCustomRegressor(MLPRegressor)                        0.8499194  0.86923669
## RandomForestRegressor                                    0.8448451  0.86481553
## DeepCustomRegressor(AdaBoostRegressor)                   0.8005664  0.82623607
## DeepCustomRegressor(LinearRegression)                    0.7472819  0.77981001
## DeepCustomRegressor(TransformedTargetRegressor)          0.7472819  0.77981001
## DeepCustomRegressor(LassoLarsIC)                         0.7472695  0.77979918
## DeepCustomRegressor(RidgeCV)                             0.7471410  0.77968718
## DeepCustomRegressor(LassoLarsCV)                         0.7429694  0.77605256
## DeepCustomRegressor(LassoCV)                             0.7428514  0.77594970
## DeepCustomRegressor(Ridge)                               0.7420801  0.77527772
## DeepCustomRegressor(HuberRegressor)                      0.7395445  0.77306849
## DeepCustomRegressor(ElasticNetCV)                        0.7327859  0.76717982
## DeepCustomRegressor(BayesianRidge)                       0.7314119  0.76598263
## DeepCustomRegressor(KNeighborsRegressor)                 0.7154223  0.75205113
## DeepCustomRegressor(SGDRegressor)                        0.7078489  0.74545254
## DeepCustomRegressor(LarsCV)                              0.7019229  0.74028925
## DeepCustomRegressor(LinearSVR)                           0.6968942  0.73590783
## DeepCustomRegressor(Lasso)                               0.6535395  0.69813345
## DeepCustomRegressor(LassoLars)                           0.6535340  0.69812867
## DeepCustomRegressor(ExtraTreeRegressor)                  0.6211290  0.66989454
## DeepCustomRegressor(PassiveAggressiveRegressor)          0.6086479  0.65901998
## DeepCustomRegressor(DecisionTreeRegressor)               0.6065707  0.65721013
## DeepCustomRegressor(ElasticNet)                          0.6049283  0.65577912
## DeepCustomRegressor(TweedieRegressor)                    0.5796672  0.63376940
## DeepCustomRegressor(RANSACRegressor)                     0.4054278  0.48195691
## DeepCustomRegressor(DummyRegressor)                     -0.1830116 -0.03074282
## DeepCustomRegressor(QuantileRegressor)                  -0.2675167 -0.10437103
## DeepCustomRegressor(Lars)                               -3.0377540 -2.51804307
##                                                      RMSE Time Taken
## DeepCustomRegressor(ExtraTreesRegressor)         2.711916 1.81884480
## DeepCustomRegressor(BaggingRegressor)            3.265265 0.43161893
## DeepCustomRegressor(RandomForestRegressor)       3.409146 2.49213696
## DeepCustomRegressor(MLPRegressor)                3.422695 2.50529003
## RandomForestRegressor                            3.480075 1.14117718
## DeepCustomRegressor(AdaBoostRegressor)           3.945527 0.88237572
## DeepCustomRegressor(LinearRegression)            4.441442 0.10591102
## DeepCustomRegressor(TransformedTargetRegressor)  4.441442 0.09575891
## DeepCustomRegressor(LassoLarsIC)                 4.441551 0.15964413
## DeepCustomRegressor(RidgeCV)                     4.442681 0.09690094
## DeepCustomRegressor(LassoLarsCV)                 4.479178 0.28806591
## DeepCustomRegressor(LassoCV)                     4.480206 0.94069099
## DeepCustomRegressor(Ridge)                       4.486920 0.08716798
## DeepCustomRegressor(HuberRegressor)              4.508921 0.30278468
## DeepCustomRegressor(ElasticNetCV)                4.567048 1.00011015
## DeepCustomRegressor(BayesianRidge)               4.578775 0.16070414
## DeepCustomRegressor(KNeighborsRegressor)         4.713096 0.14384484
## DeepCustomRegressor(SGDRegressor)                4.775398 0.10862613
## DeepCustomRegressor(LarsCV)                      4.823588 0.26737404
## DeepCustomRegressor(LinearSVR)                   4.864106 0.11280203
## DeepCustomRegressor(Lasso)                       5.200352 0.11331511
## DeepCustomRegressor(LassoLars)                   5.200393 0.17438006
## DeepCustomRegressor(ExtraTreeRegressor)          5.438155 0.23284817
## DeepCustomRegressor(PassiveAggressiveRegressor)  5.527003 0.13693190
## DeepCustomRegressor(DecisionTreeRegressor)       5.541652 0.35019112
## DeepCustomRegressor(ElasticNet)                  5.553207 0.12111807
## DeepCustomRegressor(TweedieRegressor)            5.727994 0.10166979
## DeepCustomRegressor(RANSACRegressor)             6.812526 1.06024408
## DeepCustomRegressor(DummyRegressor)              9.609491 0.18867731
## DeepCustomRegressor(QuantileRegressor)           9.946785 0.46191096
## DeepCustomRegressor(Lars)                       17.753165 0.15315008
```

### Time series

```R
set.seed(123)
X <- matrix(rnorm(300), 100, 3)

(index_train <- base::sample.int(n = nrow(X),
                                 size = floor(0.8*nrow(X)),
                                 replace = FALSE))
X_train <- data.frame(X[index_train, ])
X_test <- data.frame(X[-index_train, ])

obj <- LazyMTS(venv_path = "../venv")

(res <- obj$fit(X_train, X_test))
```

```bash
##                                      RMSE       MAE       MPL Time Taken
## MTS(BayesianRidge)              0.9693231 0.7284321 0.3642161 0.41069007
## MTS(ElasticNetCV)               0.9857480 0.7467560 0.3733780 2.63610816
## MTS(LassoCV)                    0.9899658 0.7509848 0.3754924 1.73795819
## MTS(DummyRegressor)             0.9907240 0.7519778 0.3759889 0.10515714
## MTS(ElasticNet)                 0.9907240 0.7519778 0.3759889 0.13000989
## MTS(Lasso)                      0.9907240 0.7519778 0.3759889 0.21233797
## MTS(LassoLarsCV)                0.9907240 0.7519778 0.3759889 0.85480905
## MTS(LassoLars)                  0.9907240 0.7519778 0.3759889 0.19775724
## MTS(LarsCV)                     0.9956436 0.7448679 0.3724340 1.20561934
## MTS(QuantileRegressor)          0.9966559 0.7577475 0.3788738 0.34696984
## MTS(RandomForestRegressor)      1.0050021 0.7788496 0.3894248 6.10781717
## VAR                             1.0110766 0.7739979 0.3869989 0.03306627
## MTS(MLPRegressor)               1.0155043 0.7619689 0.3809844 0.64326286
## MTS(KNeighborsRegressor)        1.0204875 0.7723135 0.3861567 0.22889900
## MTS(AdaBoostRegressor)          1.0255662 0.7926204 0.3963102 2.82807398
## MTS(ExtraTreesRegressor)        1.0429340 0.7754142 0.3877071 6.00860429
## MTS(BaggingRegressor)           1.0568736 0.8238503 0.4119252 0.75960994
## MTS(TweedieRegressor)           1.0676467 0.8112648 0.4056324 0.45418596
## MTS(RidgeCV)                    1.2321289 0.9610130 0.4805065 0.16912007
## MTS(ExtraTreeRegressor)         1.2463809 0.9800255 0.4900127 0.28515887
## VECM                            1.3765455 1.1327719 0.5663860 0.02011490
## MTS(SGDRegressor)               1.4109004 1.1323320 0.5661660 0.15479517
## MTS(DecisionTreeRegressor)      1.4451682 1.1283590 0.5641795 0.19908309
## MTS(Ridge)                      1.6345720 1.3245349 0.6622674 0.18555593
## MTS(PassiveAggressiveRegressor) 1.7938410 1.4594719 0.7297360 0.17754388
## MTS(TransformedTargetRegressor) 2.0591218 1.6801762 0.8400881 0.36497688
## MTS(LinearRegression)           2.0591218 1.6801762 0.8400881 0.18544602
## MTS(HuberRegressor)             2.0779168 1.6827919 0.8413959 1.01927090
## MTS(LinearSVR)                  2.0784001 1.6831479 0.8415739 0.17197490
```

    
![image-title-here]({{base}}/images/2025-12-17/2025-12-17-new-nnetsauce-R-uv.png){:class="img-responsive"}
    

