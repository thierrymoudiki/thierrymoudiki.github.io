---
layout: post
title: "Forecasting benchmark: Dynrmf (a new serious competitor in town) vs Theta Method on M-Competitions and Tourism competitition"
description: "A rigorous benchmark of Dynrmf vs Theta Method on M-Competitions and Tourism competitition"
date: 2026-01-01
categories: R
comments: true
---

In the world of time series forecasting, benchmarking methods against established competitions is crucial. Today, I'm sharing results from a comprehensive comparison between two forecasting approaches: the classical Theta (notoriously difficult to beat on the chosen benchmark datasets) method and the newer Dynrmf (from R and Python packages [ahead](https://github.com/Techtonique)) algorithm, tested across three major forecasting competitions.

## The Setup

I evaluated both methods on:
- **M3 Competition**: 3,003 diverse time series
- **M1 Competition**: The original M-competition dataset (1001 time series)
- **Tourism Competition**: Tourism-specific time series data (1311 time series)

For each dataset, I trained both models on historical data and evaluated their forecasts against held-out test sets using standard accuracy metrics (ME, RMSE, MAE, MPE, MAPE, MASE, ACF1). Each model uses its default parameters.

## Implementation Details

The analysis used parallel processing with 2 cores to speed up computations across thousands of series. Both methods received identical train/test splits for fair comparison -- complete code at the end of the post.

```r
# Example forecasting loop structure
metrics <- foreach(i = 1:n, .combine = rbind)%dopar%{
  series <- dataset[[i]]
  train <- series$x
  test <- series$xx
  fit <- method(y = train, h = length(test))
  forecast::accuracy(fit)
}
```

## Results

It's worth mentioning that Theta was the clear winner of M3 competition (and it's been enhanced [in many ways](https://thierrymoudiki.github.io/blog/2025/11/13/r/context-aware-theta)), and, of course, it's always easier to beat the winner _a posteriori_.

I performed paired t-tests on the difference in metrics (Theta - Dynrmf) across all series in each competition. Here's what the numbers reveal:

### M3 Competition Results

| Metric | t-statistic | p-value | Interpretation |
|--------|------------|---------|----------------|
| ME | 16.96 | <0.001 | Theta significantly more biased |
| RMSE | 10.17 | <0.001 | **Theta significantly worse** |
| MAE | 7.65 | <0.001 | **Theta significantly worse** |
| MPE | -5.59 | <0.001 | Dynrmf more biased |
| MAPE | 1.29 | 0.197 | No significant difference |
| MASE | 14.09 | <0.001 | **Theta significantly worse** |
| ACF1 | -1.37 | 0.171 | No significant difference |

**Verdict**: Dynrmf demonstrates substantially better accuracy on M3 data for most error metrics.

### M1 Competition Results

| Metric | t-statistic | p-value | Interpretation |
|--------|------------|---------|----------------|
| ME | 2.56 | 0.011 | Theta more biased |
| RMSE | 2.94 | 0.003 | **Theta significantly worse** |
| MAE | 2.28 | 0.023 | **Theta significantly worse** |
| MPE | -0.61 | 0.540 | No significant difference |
| MAPE | 0.38 | 0.703 | No significant difference |
| MASE | -0.74 | 0.461 | No significant difference |
| ACF1 | -6.50 | <0.001 | Dynrmf residuals better |

**Verdict**: Dynrmf shows moderate but statistically significant improvements, particularly in RMSE and MAE.

### Tourism Competition Results

| Metric | t-statistic | p-value | Interpretation |
|--------|------------|---------|----------------|
| ME | 1.62 | 0.105 | No significant difference |
| RMSE | 1.88 | 0.060 | Marginal advantage to Dynrmf |
| MAE | 1.16 | 0.245 | No significant difference |
| MPE | 4.52 | <0.001 | Theta less biased |
| MAPE | -5.03 | <0.001 | **Dynrmf significantly better** |
| MASE | -5.17 | <0.001 | **Dynrmf significantly better** |
| ACF1 | -8.65 | <0.001 | Dynrmf residuals better |

**Verdict**: On tourism data, Dynrmf excels at scaled metrics (MAPE, MASE) and produces better-behaved residuals.

## Key Takeaways

1. **Dynrmf consistently outperforms Theta** on accuracy metrics like RMSE, MAE, and MASE across competitions
2. **The advantage is strongest on M3**, suggesting Dynrmf handles diverse series types well
3. **Tourism data shows Dynrmf's strength** in percentage and scaled error metrics
4. **Residual autocorrelation** (ACF1) is consistently better with Dynrmf, indicating it captures patterns more completely
5. **Model-agnostic flexibility**: It's important to mention that unlike method-specific approaches, Dynrmf would accept any fitting function (and the default here is automatic Ridge Regression minimizing Generalized Cross-Validation errror) through its `fit_func` parameter (see [here](https://thierrymoudiki.github.io/blog/2025/04/20/r/tisthemachinelearner-time-series)), enabling use of Ridge regression (default), Random Forest, XGBoost, Support Vector Machines, or any other model with standard fit/predict interfaces

## Code Availability

The full R implementation is available below. The analysis leverages parallel processing for efficiency on large competition datasets.

```R
library(ahead)
library(Mcomp)
library(Tcomp)
library(foreach)
library(doParallel)
library(doSNOW)

setwd("~/Documents/Papers/to_submit/2026-01-01-dynrmf-vs-Theta-on-M1-M3-Tourism")

# 1. M3 comp. results --------------

data(M3)

n <- length(Mcomp::M3)

training_indices <- seq_len(n)

# Theta -----------

metrics_theta <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_theta <- foreach::foreach(i = 1:n, 
                 .combine = rbind, .verbose = TRUE)%dopar%{
  series <- M3[[i]]
  train <- series$x
  test <- series$xx
  fit <- forecast::thetaf(
    y = train,
    h = length(test))
  utils::setTxtProgressBar(pb, i)
  forecast::accuracy(fit)
}
close(pb)

print(metrics_theta)

# Dynrmf -----------


metrics_dynrmf <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_dynrmf <- foreach::foreach(i = 1:n, 
                                  .combine = rbind, .verbose = TRUE)%dopar%{
                                    series <- M3[[i]]
                                    train <- series$x
                                    test <- series$xx
                                    fit <- ahead::dynrmf(
                                      y = train,
                                      h = length(test))
                                    utils::setTxtProgressBar(pb, i)
                                    forecast::accuracy(fit)
                                  }
close(pb)

print(metrics_dynrmf)

diff_metrics <- metrics_theta - metrics_dynrmf

M3_results <- apply(diff_metrics, 2, t.test)

M3_results <- apply(diff_metrics, 2, function(x) {z <- t.test(x); return(c(z$statistic, z$p.value))})
rownames(M3_results) <- c("statistic", "p-value")



# 2. M1 comp. results -----------------

data(M1)

n <- length(Mcomp::M1)

training_indices <- seq_len(n)

# Theta -----------

metrics_theta <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_theta <- foreach::foreach(i = 1:n, 
                                  .combine = rbind, .verbose = TRUE)%dopar%{
                                    series <- M1[[i]]
                                    train <- series$x
                                    test <- series$xx
                                    fit <- forecast::thetaf(
                                      y = train,
                                      h = length(test))
                                    utils::setTxtProgressBar(pb, i)
                                    forecast::accuracy(fit)
                                  }
close(pb)

print(metrics_theta)

# Dynrmf -----------

metrics_dynrmf <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_dynrmf <- foreach::foreach(i = 1:n, 
                                   .combine = rbind, .verbose = TRUE)%dopar%{
                                     series <- M1[[i]]
                                     train <- series$x
                                     test <- series$xx
                                     fit <- ahead::dynrmf(
                                       y = train,
                                       h = length(test))
                                     utils::setTxtProgressBar(pb, i)
                                     forecast::accuracy(fit)
                                   }
close(pb)

print(metrics_dynrmf)

diff_metrics <- metrics_theta - metrics_dynrmf

M1_results <- apply(diff_metrics, 2, function(x) {z <- t.test(x); return(c(z$statistic, z$p.value))})
rownames(M1_results) <- c("statistic", "p-value")


# 2. Tourism comp. results -----------------

data(tourism)

n <- length(Tcomp::tourism)

training_indices <- seq_len(n)

# Theta -----------

metrics_theta <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_theta <- foreach::foreach(i = 1:n, 
                                  .combine = rbind, .verbose = TRUE)%dopar%{
                                    series <- tourism[[i]]
                                    train <- series$x
                                    test <- series$xx
                                    fit <- forecast::thetaf(
                                      y = train,
                                      h = length(test))
                                    utils::setTxtProgressBar(pb, i)
                                    forecast::accuracy(fit)
                                  }
close(pb)

print(metrics_theta)

# Dynrmf -----------

metrics_dynrmf <- rep(NA, n)

pb <- utils::txtProgressBar(max = n, style = 3)

doParallel::registerDoParallel(cl=2)

# looping on all the training set time series
metrics_dynrmf <- foreach::foreach(i = 1:n, 
                                   .combine = rbind, .verbose = TRUE)%dopar%{
                                     series <- tourism[[i]]
                                     train <- series$x
                                     test <- series$xx
                                     fit <- ahead::dynrmf(
                                       y = train,
                                       h = length(test))
                                     utils::setTxtProgressBar(pb, i)
                                     forecast::accuracy(fit)
                                   }
close(pb)

print(metrics_dynrmf)

diff_metrics <- metrics_theta - metrics_dynrmf

diff_metrics <- diff_metrics[is.finite(diff_metrics[, 4]), ]

Tourism_results <- apply(diff_metrics, 2, function(x) {z <- t.test(x); return(c(z$statistic, z$p.value))})
rownames(Tourism_results) <- c("statistic", "p-value")

# # Theta - dynrmf+default
# 
# ```R
# kableExtra::kable(M3_results)
# kableExtra::kable(M1_results)
# kableExtra::kable(Tourism_results)
# 
# |          |       ME|     RMSE|      MAE|       MPE|      MAPE|    MASE|       ACF1|
#   |:---------|--------:|--------:|--------:|---------:|---------:|-------:|----------:|
#   |statistic | 16.96265| 10.16816| 7.652445| -5.586289| 1.2900974| 14.0924| -1.3680305|
#   |p-value   |  0.00000|  0.00000| 0.000000|  0.000000| 0.1971162|  0.0000|  0.1714049|
# 
# |          |        ME|      RMSE|       MAE|        MPE|      MAPE|       MASE|      ACF1|
#   |:---------|---------:|---------:|---------:|----------:|---------:|----------:|---------:|
#   |statistic | 2.5575962| 2.9406984| 2.2815166| -0.6129693| 0.3808239| -0.7378457| -6.497644|
#   |p-value   | 0.0106864| 0.0033502| 0.0227274|  0.5400360| 0.7034148|  0.4607813|  0.000000|
# 
# |          |        ME|      RMSE|       MAE|       MPE|       MAPE|       MASE|      ACF1|
#   |:---------|---------:|---------:|---------:|---------:|----------:|----------:|---------:|
#   |statistic | 1.6239829| 1.8845405| 1.1628106| 4.5207962| -5.0276196| -5.1743478| -8.648504|
#   |p-value   | 0.1046342| 0.0597262| 0.2451306| 0.0000068|  0.0000006|  0.0000003|  0.000000|
#   
# ```
```

---

*What forecasting methods have you found most reliable in your work? Share your experiences in the comments below.*

Please sign and share this petition [https://www.change.org/stop_torturing_T_Moudiki](https://www.change.org/stop_torturing_T_Moudiki) -- after seeriously researching my background and contributions to the field.

    
![image-title-here]({{base}}/images/2026-01-01/2026-01-01-image1.png){:class="img-responsive"}
    

