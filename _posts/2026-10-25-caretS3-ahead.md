---
layout: post
title: "caretS3 and ahead for Tabular and Time Series Machine Learning"
description: "Exploring the integration of caretS3 and ahead for tabular and time series machine learning tasks."
date: 2026-10-25
categories: R 
comments: true
---

### Packages

```R
#install.packages("remotes")
#install.packages("glmnet")
#install.packages("randomForest")
```

```R
remotes::install_github("thierrymoudiki/caretS3")
```

### `caretS3::caretS3` Tabular Machine Learning


```R
 data(mtcars)

 # Prepare data
 X_reg <- mtcars[, -1]  # All except mpg
 y_reg <- mtcars$mpg     # Target variable

 # Split into train/test
 set.seed(123)
 idx_reg <- sample(nrow(X_reg), 0.7 * nrow(X_reg))
 X_train <- X_reg[idx_reg, ]
 y_train <- y_reg[idx_reg]
 X_test <- X_reg[-idx_reg, ]
 y_test <- y_reg[-idx_reg]

 mod <- caretS3::caretS3(X_train, y_train, method = "glmnet")
 print(predict(mod, X_test))
```

    Registered S3 method overwritten by 'future':
      method               from      
      all.equal.connection parallelly

              Mazda RX4       Mazda RX4 Wag      Hornet 4 Drive             Valiant 
               22.88234            22.41295            21.12871            20.55533 
             Merc 450SE          Merc 450SL Lincoln Continental       Toyota Corona 
               14.37278            15.58185            10.76004            24.86612 
             Camaro Z28    Pontiac Firebird 
               13.55088            15.78322           Mazda RX4       Mazda RX4 Wag      Hornet 4 Drive             Valiant 
               20.09445            20.05342            19.60069            19.34515 
             Merc 450SE          Merc 450SL Lincoln Continental       Toyota Corona 
               15.35807            15.32211            12.98877            24.25831 
             Camaro Z28    Pontiac Firebird 
               15.33895            16.01282 

### `ahead::dynrmf` time series forecasting


```R
par(mfrow=c(2, 2))
plot(ahead::dynrmf(AirPassengers, h=20, fit_func = caretS3::caretS3, fit_params = list(method="glmnet", alpha=0, lambda=0.1)))
plot(ahead::dynrmf(AirPassengers, h=20, fit_func = caretS3::caretS3, fit_params = list(method="glmnet", alpha=0.5, lambda=0.1)))
plot(ahead::dynrmf(AirPassengers, h=20, fit_func = caretS3::caretS3, fit_params = list(method="glmnet", alpha=1, lambda=0.1)))
plot(ahead::dynrmf(AirPassengers, h=20, fit_func = caretS3::caretS3, fit_params = list(method="glm")))
```
    
![image-title-here]({{base}}/images/2026-10-25/2026-10-25-caretS3-ahead_6_0.png){:class="img-responsive"}
    

