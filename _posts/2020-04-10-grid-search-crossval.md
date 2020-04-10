---
layout: post
title: "Grid search cross-validation using crossval"
description: Cross-validation using a grid of hyperparameters and R package crossval
date: 2020-04-10
categories: [R, Misc]
---

`crossval` is an R package which contains generic functions for cross-validation. [Two weeks ago]({% post_url 2020-03-27-crossval-2 %}), I presented an example of time series cross-validation based on `crossval`. This week's post is about cross-validation on a grid of hyperparameters. `glmnet` is used as statistical learning model for the demo, but it could be any other package of your choice. 

## Installing and loading the packages

Installing `crossval` from GitHub (in R console):

```r
devtools::install_github("thierrymoudiki/crossval")
```

Loading packages:

```r
library(glmnet)
library(crossval)
```


## Load `mtcars` dataset

```r
data("mtcars")
df <- mtcars[, c(1, 2, 3, 4, 6, 11)]
summary(df)
```


## Create response and explanatory variables from `mtcars` dataset

```r
X <- as.matrix(df[, -1]) # explanatory variables
y <- df$mpg # response
```

![image-title-here]({{base}}/images/2020-04-10/2020-04-10-image1.png){:class="img-responsive"}


## Grid of hyperparameters for `glmnet`

```r
tuning_grid <- expand.grid(alpha = c(0, 0.5, 1),
                           lambda = c(0.01, 0.1, 1))
n_params <- nrow(tuning_grid)
print(tuning_grid)
```
```
##   alpha lambda
## 1   0.0   0.01
## 2   0.5   0.01
## 3   1.0   0.01
## 4   0.0   0.10
## 5   0.5   0.10
## 6   1.0   0.10
## 7   0.0   1.00
## 8   0.5   1.00
## 9   1.0   1.00
```

## Grid search cross-validation

- list of cross-validation results
- 5-fold cross-validation (`k`)
- repeated 3 times (`repeats`)
- cross-validation of 80% of the data (`p`)
- validation on the remaining 20%

```r
cv_results <- lapply(1:n_params,
                     function(i)
                       crossval::crossval_ml(
                         x = X,
                         y = y,
                         k = 5,
                         repeats = 3,
                         p = 0.8,
                         fit_func = glmnet::glmnet,
                         predict_func = predict.glmnet,
                         packages = c("glmnet", "Matrix"),
                         fit_params = list(alpha = tuning_grid[i, "alpha"],
                                           lambda = tuning_grid[i, "lambda"])
                       ))
names(cv_results) <- paste0("params_set", 1:n_params)
```

## Print grid search results

[__Remarks__]({% post_url 2020-02-14-git-github %}) are welcome.

```r
print(cv_results)
```
```
## $params_set1
## $params_set1$folds
##                    repeat_1  repeat_2 repeat_3
## fold_training_1   2.7116571 3.4204585 2.970296
## fold_validation_1 1.1310676 2.1443185 2.038922
## fold_training_2   1.7335414 1.0317404 3.740119
## fold_validation_2 1.6528925 1.5592805 0.905873
## fold_training_3   2.9526843 4.4059576 3.063401
## fold_validation_3 2.4348686 0.9470344 1.227135
## fold_training_4   4.3206047 3.7097429 3.252773
## fold_validation_4 0.8305158 1.7408722 1.793542
## fold_training_5   1.3484699 1.9396528 1.322698
## fold_validation_5 1.4838844 1.8029411 1.288075
## 
## $params_set1$mean_training
## [1] 2.79492
## 
## $params_set1$mean_validation
## [1] 1.532082
## 
## $params_set1$sd_training
## [1] 1.089414
## 
## $params_set1$sd_validation
## [1] 0.4773198
## 
## $params_set1$median_training
## [1] 2.970296
## 
## $params_set1$median_validation
## [1] 1.559281
## 
## 
## $params_set2
## $params_set2$folds
##                    repeat_1  repeat_2  repeat_3
## fold_training_1   2.6942232 3.4034435 2.9509207
## fold_validation_1 1.1283288 2.1212168 2.0922106
## fold_training_2   1.7071382 1.0236950 3.7337454
## fold_validation_2 1.6183054 1.5395739 0.9289049
## fold_training_3   2.9572493 4.3913568 3.0347475
## fold_validation_3 2.4458845 0.9670278 1.2149724
## fold_training_4   4.3683721 3.6924562 3.2383772
## fold_validation_4 0.8786582 1.7168194 1.7714379
## fold_training_5   1.3451998 1.9338217 1.3169037
## fold_validation_5 1.4832682 1.7971002 1.2850981
## 
## $params_set2$mean_training
## [1] 2.78611
## 
## $params_set2$mean_validation
## [1] 1.532587
## 
## $params_set2$sd_training
## [1] 1.093607
## 
## $params_set2$sd_validation
## [1] 0.470716
## 
## $params_set2$median_training
## [1] 2.957249
## 
## $params_set2$median_validation
## [1] 1.539574
## 
## 
## $params_set3
## $params_set3$folds
##                    repeat_1 repeat_2  repeat_3
## fold_training_1   2.6762742 3.385479 2.9318273
## fold_validation_1 1.1267161 2.094206 2.1505220
## fold_training_2   1.6851155 1.017365 3.7272127
## fold_validation_2 1.5972579 1.519639 0.9543918
## fold_training_3   2.9614653 4.376096 3.0052024
## fold_validation_3 2.4567021 0.989157 1.2033089
## fold_training_4   4.4107761 3.674386 3.2273064
## fold_validation_4 0.9223574 1.691938 1.7506447
## fold_training_5   1.3421543 1.928040 1.3124042
## fold_validation_5 1.4833113 1.791768 1.2834694
## 
## $params_set3$mean_training
## [1] 2.777407
## 
## $params_set3$mean_validation
## [1] 1.534359
## 
## $params_set3$sd_training
## [1] 1.096777
## 
## $params_set3$sd_validation
## [1] 0.4656268
## 
## $params_set3$median_training
## [1] 2.961465
## 
## $params_set3$median_validation
## [1] 1.519639
## 
## 
## $params_set4
## $params_set4$folds
##                   repeat_1  repeat_2 repeat_3
## fold_training_1   2.582406 3.2605565 2.864273
## fold_validation_1 1.168777 1.9268152 2.078255
## fold_training_2   1.650031 0.8984717 3.686839
## fold_validation_2 1.482450 1.4721014 1.017757
## fold_training_3   2.708588 4.2802020 2.939830
## fold_validation_3 2.235334 1.0667584 1.204211
## fold_training_4   4.466894 3.5879682 3.081803
## fold_validation_4 1.004771 1.5646289 1.570059
## fold_training_5   1.326640 1.9144285 1.380771
## fold_validation_5 1.509964 1.7885983 1.335888
## 
## $params_set4$mean_training
## [1] 2.708647
## 
## $params_set4$mean_validation
## [1] 1.495091
## 
## $params_set4$sd_training
## [1] 1.086611
## 
## $params_set4$sd_validation
## [1] 0.3817352
## 
## $params_set4$median_training
## [1] 2.864273
## 
## $params_set4$median_validation
## [1] 1.48245
## 
## 
## $params_set5
## $params_set5$folds
##                    repeat_1  repeat_2 repeat_3
## fold_training_1   2.5706795 3.1103749 2.803793
## fold_validation_1 1.2068333 1.7463001 2.075065
## fold_training_2   1.5011386 0.8148667 3.688756
## fold_validation_2 1.3987359 1.3301017 1.001877
## fold_training_3   2.7010045 4.2517543 2.747419
## fold_validation_3 2.2688901 1.1557910 1.178889
## fold_training_4   4.4448265 3.4750530 3.016761
## fold_validation_4 0.9854596 1.4860992 1.437257
## fold_training_5   1.3487225 1.8742370 1.295727
## fold_validation_5 1.5372721 1.7800714 1.311683
## 
## $params_set5$mean_training
## [1] 2.643008
## 
## $params_set5$mean_validation
## [1] 1.460022
## 
## $params_set5$sd_training
## [1] 1.093856
## 
## $params_set5$sd_validation
## [1] 0.3720159
## 
## $params_set5$median_training
## [1] 2.747419
## 
## $params_set5$median_validation
## [1] 1.398736
## 
## 
## $params_set6
## $params_set6$folds
##                    repeat_1  repeat_2  repeat_3
## fold_training_1   2.5990228 2.9781640 2.7493269
## fold_validation_1 1.2121089 1.5799814 2.0848477
## fold_training_2   1.4225076 0.7604152 3.6906583
## fold_validation_2 1.4216262 1.2543630 0.9884764
## fold_training_3   2.7409312 4.2492745 2.7175981
## fold_validation_3 2.3240529 1.1598608 1.1607340
## fold_training_4   4.4339739 3.4654770 3.0117350
## fold_validation_4 0.9800525 1.4991208 1.4168583
## fold_training_5   1.3765304 1.8415788 1.3257447
## fold_validation_5 1.5496021 1.8006454 1.3220442
## 
## $params_set6$mean_training
## [1] 2.624196
## 
## $params_set6$mean_validation
## [1] 1.450292
## 
## $params_set6$sd_training
## [1] 1.097017
## 
## $params_set6$sd_validation
## [1] 0.3811666
## 
## $params_set6$median_training
## [1] 2.740931
## 
## $params_set6$median_validation
## [1] 1.416858
## 
## 
## $params_set7
## $params_set7$folds
##                   repeat_1 repeat_2 repeat_3
## fold_training_1   2.698210 2.885301 2.455576
## fold_validation_1 1.551401 1.704756 1.716643
## fold_training_2   1.783057 1.028166 3.528652
## fold_validation_2 1.688929 1.457255 1.192856
## fold_training_3   2.635762 3.951937 2.764754
## fold_validation_3 2.325906 1.361088 1.478338
## fold_training_4   4.383367 3.622788 2.966129
## fold_validation_4 1.262743 1.758041 1.628874
## fold_training_5   1.520805 1.968637 1.384429
## fold_validation_5 1.747330 2.061490 1.586987
## 
## $params_set7$mean_training
## [1] 2.638505
## 
## $params_set7$mean_validation
## [1] 1.634842
## 
## $params_set7$sd_training
## [1] 0.9764259
## 
## $params_set7$sd_validation
## [1] 0.2898281
## 
## $params_set7$median_training
## [1] 2.69821
## 
## $params_set7$median_validation
## [1] 1.628874
## 
## 
## $params_set8
## $params_set8$folds
##                   repeat_1 repeat_2 repeat_3
## fold_training_1   2.966475 2.806465 1.737976
## fold_validation_1 1.692210 1.804410 1.498461
## fold_training_2   1.392634 1.104673 3.578175
## fold_validation_2 2.068470 1.499582 1.163872
## fold_training_3   2.684285 3.930335 2.611488
## fold_validation_3 2.543810 1.441189 1.498748
## fold_training_4   4.269152 3.760451 3.327202
## fold_validation_4 1.381628 2.067037 2.049550
## fold_training_5   1.771081 2.323059 1.777073
## fold_validation_5 1.946920 2.405880 1.787871
## 
## $params_set8$mean_training
## [1] 2.669368
## 
## $params_set8$mean_validation
## [1] 1.789976
## 
## $params_set8$sd_training
## [1] 0.9775324
## 
## $params_set8$sd_validation
## [1] 0.3908084
## 
## $params_set8$median_training
## [1] 2.684285
## 
## $params_set8$median_validation
## [1] 1.787871
## 
## 
## $params_set9
## $params_set9$folds
##                   repeat_1 repeat_2 repeat_3
## fold_training_1   3.254495 2.789325 1.094701
## fold_validation_1 1.878893 1.938494 1.581836
## fold_training_2   1.546198 1.179068 3.647095
## fold_validation_2 2.495703 1.623228 1.219294
## fold_training_3   2.478050 3.922693 2.414970
## fold_validation_3 2.594489 1.592061 1.575155
## fold_training_4   4.171757 3.904126 3.695321
## fold_validation_4 1.754051 2.395155 2.455114
## fold_training_5   2.053809 2.660005 2.127650
## fold_validation_5 2.177259 2.741885 2.012704
## 
## $params_set9$mean_training
## [1] 2.729284
## 
## $params_set9$mean_validation
## [1] 2.002355
## 
## $params_set9$sd_training
## [1] 1.014183
## 
## $params_set9$sd_validation
## [1] 0.454853
## 
## $params_set9$median_training
## [1] 2.660005
## 
## $params_set9$median_validation
## [1] 1.938494
```

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!




