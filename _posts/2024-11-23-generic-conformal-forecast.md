---
layout: post
title: "Unified interface and conformal prediction (calibrated prediction intervals) for R package forecast (and affiliates)"
description: "Unified interface and calibrated prediction intervals for R package forecast, with multiple examples"
date: 2024-11-23
categories: R
comments: true
---

In the popular R package [`forecast`](https://github.com/robjhyndman/forecast), there are 2 different types of  interfaces: 

- A _direct_ interface for functions like `forecast::thetaf` doing fitting and inference simultaneously 

```R
nile.fcast <- forecast::thetaf(Nile)
plot(nile.fcast)
```

- An interface for fitting first and then forecasting, like `forecast::ets`, where you need to use, in addition `forecast::forecast`:

```R
fit <- forecast::ets(USAccDeaths)
plot(forecast::forecast(fit))
```

In this post, I describe how to obtain probabilistic forecasts from R package `forecast` -- and packages that follow a similar philosophy such as forecastHybrid, [ahead](https://github.com/Techtonique/ahead), etc. --, by  using a unified interface (`ahead::genericforecast`). Then, I present `ahead::conformalize`, a function that allows to obtain forecasts using the method described in [Conformalized predictive simulations for univariate time series](https://www.researchgate.net/publication/379643443_Conformalized_predictive_simulations_for_univariate_time_series) (more details can be found in [these slides](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)). 

# 0 - Packages

```{r}
utils::install.packages(c("remotes", "e1071", "forecast", "glmnet"))
remotes::install_github("Techtonique/ahead")
```

```{r}
library(ahead)
library(forecast)
```

```{r}
y <- fdeaths #AirPassengers #Nile #mdeaths #fdeaths #USAccDeaths
h <- 25L
```

# 1 - Generic forecaster (unified interface)

## 1 - 1 - Using default parameters

```{r}
par(mfrow=c(2, 2))
plot(ahead::genericforecast(FUN=forecast::thetaf, y, h))
plot(ahead::genericforecast(FUN=forecast::meanf, y, h))
plot(ahead::genericforecast(FUN=forecast::rwf, y, h))
plot(ahead::genericforecast(FUN=forecast::ets, y, h))

par(mfrow=c(2, 2))
plot(ahead::genericforecast(FUN=forecast::tbats, y, h))
plot(ahead::genericforecast(FUN=HoltWinters, y, h))
plot(ahead::genericforecast(FUN=forecast::Arima, y, h))
plot(ahead::genericforecast(FUN=ahead::dynrmf, y, h))
```

![xxx]({{base}}/images/2024-11-23/2024-11-23-image1.png){:class="img-responsive"} 

![xxx]({{base}}/images/2024-11-23/2024-11-23-image2.png){:class="img-responsive"}          

## 1 - 2 - Using additional parameters

```{r}
par(mfrow=c(2, 2))
plot(ahead::genericforecast(FUN=ahead::dynrmf, y=y, h=h, 
                            fit_func=e1071::svm, predict_func=predict))
plot(ahead::genericforecast(FUN=ahead::dynrmf, y=y, h=h, 
                            fit_func=glmnet::cv.glmnet, predict_func=predict))
plot(ahead::genericforecast(FUN=forecast::tbats, y=y, h=h, 
                            use.box.cox = TRUE, use.trend=FALSE))
plot(ahead::genericforecast(FUN=forecast::rwf, 
                            y=y, h=h, lambda=1.1))
```

![xxx]({{base}}/images/2024-11-23/2024-11-23-image3.png){:class="img-responsive"}          

# 2 - Conformal prediction

## 2 - 1 - Using default parameters

```{r}
y <- USAccDeaths

par(mfrow=c(3, 2))
obj <- ahead::conformalize(FUN=forecast::thetaf, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::meanf, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::rwf, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::ets, y, h); plot(obj)

par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN=forecast::auto.arima, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::tbats, y, h); plot(obj)
obj <- ahead::conformalize(FUN=HoltWinters, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::Arima, y, h); plot(obj)
```

![xxx]({{base}}/images/2024-11-23/2024-11-23-image4.png){:class="img-responsive"}          

![xxx]({{base}}/images/2024-11-23/2024-11-23-image5.png){:class="img-responsive"}          

## 2 - 2 - Using additional parameters

```{r}
y <- AirPassengers

par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN=forecast::thetaf, y, h); plot(obj)
obj <- ahead::conformalize(FUN=forecast::rwf, y=y, h=h, drift=TRUE); plot(obj)
obj <- ahead::conformalize(FUN=HoltWinters, y=y, h=h, seasonal = "mult"); plot(obj)
obj <- ahead::conformalize(FUN=ahead::dynrmf, y=y, h=h, fit_func=glmnet::cv.glmnet, predict_func=predict); plot(obj)

```

![xxx]({{base}}/images/2024-11-23/2024-11-23-image6.png){:class="img-responsive"}          

## 2 - 3 - Using other simulation methods (conformal prediction-based)

```{r}
y <- fdeaths

par(mfrow=c(3, 2))
obj <- ahead::conformalize(FUN=forecast::thetaf, y=y, h=h, method="block-bootstrap"); plot(obj)
obj <- ahead::conformalize(FUN=forecast::rwf, y=y, h=h, drift=TRUE, method="bootstrap"); plot(obj)
obj <- ahead::conformalize(FUN=forecast::ets, y, h, method="kde"); plot(obj)
obj <- ahead::conformalize(FUN=forecast::tbats, y=y, h=h, method="surrogate"); plot(obj)
obj <- ahead::conformalize(FUN=HoltWinters, y=y, h=h, seasonal = "mult", method="block-bootstrap"); plot(obj)
obj <- ahead::conformalize(FUN=ahead::dynrmf, y=y, h=h, fit_func=glmnet::cv.glmnet, 
                           predict_func=predict, method="surrogate"); plot(obj)
```

![xxx]({{base}}/images/2024-11-23/2024-11-23-image7.png){:class="img-responsive"}          