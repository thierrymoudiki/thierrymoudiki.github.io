---
layout: post
title: "Forecasting the Economy"
description: "Forecasting the Economy."
date: 2024-05-27
categories: [R]
comments: true
---

Keep in mind that there's no hyperparameter tuning in these examples. Hyperparameter tuning must be used in practice. Looking for `reticulate` and `rpy2` experts to discuss speedups for this R package (port from the stable Python version) installation and loading.


```python
%load_ext rpy2.ipython
```

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython



```python
!pip install nnetsauce
```

```r
%%R

# 0 - packages -----
utils::install.packages(c("reticulate",
                          "remotes",
                          "forecast",
                          "fpp2"))
remotes::install_github("Techtonique/nnetsauce_r") # slow here

library("reticulate")
library("nnetsauce")
library("fpp2")
```


```r
%%R

# 1 - data -----
set.seed(123)
X <- fpp2::uschange
idx_train <- 1:floor(0.8*nrow(X))
X_train <- X[idx_train, ]
X_test <- X[-idx_train, ]

```


```r
%%R

# 2 - model fitting ---
obj_MTS <- nnetsauce::MTS(sklearn$linear_model$BayesianRidge(),
                          lags = 1L) # use a Bayesian model for uncertainty quantification
obj_DeepMTS <- nnetsauce::DeepMTS(sklearn$linear_model$ElasticNet(),
                                  lags = 1L,
                                  replications=100L,
                                  kernel='gaussian') # use Kernel density for uncertainty quantification
obj_MTS$fit(X_train)
obj_DeepMTS$fit(X_train)
```


```r
%%R

# 3 - model predictions ---
preds_MTS <- obj_MTS$predict(h = nrow(X_test),
                      level = 95,
                      return_std = TRUE)
preds_DeepMTS <- obj_DeepMTS$predict(h=nrow(X_test),
                        level = 95)

```

    100%|██████████| 100/100 [00:00<00:00, 3510.91it/s]
    100%|██████████| 100/100 [00:00<00:00, 5638.11it/s]



```r
%%R

# 4 - Graph ---
par(mfrow=c(2, 4))
for (series_id in c(2, 3, 4, 5))
{
  plot(1:nrow(X_test), X_test[, series_id],
       main = paste0("MTS (Bayesian) -- \n", colnames(fpp2::uschange)[series_id]),
       type='l', ylim = c(min(preds_MTS$lower[, series_id]),
                          max(preds_MTS$upper[, series_id])))
  lines(preds_MTS$lower[, series_id], col="blue", lwd=2)
  lines(preds_MTS$upper[, series_id], col="blue", lwd=2)
  lines(preds_MTS$mean[, series_id], col="red", lwd=2)
}
for (series_id in c(2, 3, 4, 5))
{
  plot(1:nrow(X_test), X_test[, series_id],
       main = paste0("DeepMTS (KDE) -- \n", colnames(fpp2::uschange)[series_id]),
       type='l', ylim = c(min(preds_DeepMTS$lower[, series_id]),
                          max(preds_DeepMTS$upper[, series_id])))
  lines(preds_DeepMTS$lower[, series_id], col="blue", lwd=2)
  lines(preds_DeepMTS$upper[, series_id], col="blue", lwd=2)
  lines(preds_DeepMTS$mean[, series_id], col="red", lwd=2)
}
```

![pres-image]({{base}}/images/2024-05-27/2024-05-27-image1.png){:class="img-responsive"}        

In this figure, KDE stands for Kernel Density Estimation. Prediction intervals are depicted as a blue line, and mean forecast as a red line. The true value is depicted as a black line. Again, keep in mind that every model is used with its default hyperparameters, and hyperparameters' tuning will give a different result.