---
layout: post
title: "Forecasting lung disease progression"
description: Forecasting lung disease progression.
date: 2020-10-02
categories: [Misc, R]
---


In [OSIC Pulmonary Fibrosis Progression competition](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)
on Kaggle, participants are tasked to determine the likelihood of recovery (**prognosis**) of several 
patients affected by a lung disease. For each patient, the maximum volume of air they can exhale after a maximum inhalation (**FVC**, Forced Vital Capacity) is measured over the weeks, for approximately 1-2 years of time. 

In addition, we have the following information about these people:

- A __chest computer scan__ obtained at time `Week=0`
- Their __age__
- Their __sex__
- Their __smoking status__: currently smokes, ex-smoker, never smoked 

The challenge is to __assess the lung function's health by forecasting the FVC__ (I'm not asking myself here, if it's the good or bad way to do that). What I like about this competition, is that there are __many ways to approach it__. Here's a non-exhaustive list:  

1.One way could be to __construct a Statistical/Machine Learning (ML) model on the whole dataset__, and study the (conditional) distribution of the FVC, knowing the scan, age, 
sex, and smoking status. In this first approach we consider that disease evolution can be generalized 
among categories of patients sharing the same patterns. A [Bayesian](https://thierrymoudiki.github.io/blog/2019/10/18/quasirandomizednn/nnetsauce-prediction-intervals) ML model could capture the uncertainty around predictions, or we 
could use a more or less sophisticated [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) procedure for the same purpose. Or, even, consider that ML model residuals are irregularly spaced time series. 


2.Another way, the _quick and dirty_ one I'll present here, __considers each patient's case individually__. Age, sex, smoking status and 
the chest scan are not used, but the of measurement is. If we are only interested in forecasting the **FVC**, the approach will be fine. But if we want to understand how 
each one of the factors we previously described [influence the FVC](https://techtonique.github.io/teller/index.html), either individually or in conjunction, then the first approach is better. 

# 0 - Functions

These are the functions that I use in the analysis. The first one extracts a patient's information from the whole database, based on his/her identifier. The second one fits a [smoothing spline](https://en.wikipedia.org/wiki/Smoothing_spline) to a patient's data, and forecasts his/her FVC.

**get patient data**
```{r}

suppressPackageStartupMessages(library(dplyr))

# 0 - 1 get patient data -----
get_patient_data <- function(id, train)
{
  df <- dplyr::select(dplyr::filter(train, Patient == id), c(Weeks, FVC))
  df$log_Weeks <- log(13 + df$Weeks) # the relative timing of FVC measurements (varies widely)
  df$log_FVC <- log(df$FVC) # transformed response variable
  df$Patient <- id
  return(df)
}
```


**fit and forecast FVC**
```{r}
# 0 - 2 fit, predict and plot -----
fit_predict <- function(df, plot_=TRUE)
{
  min_week <- 13
  n <- nrow(df)

  test_seq_week <- seq(-12, 133)
  log_test_seq_week <- log(min_week + test_seq_week)
  
    
    # Fit a smoothing spline, using Leave-one-out cross-validation for regularization
    fit_obj <- stats::smooth.spline(x = df$log_Weeks,
                                    y = df$log_FVC,
                                    cv = TRUE)

    resids <- residuals(fit_obj)
    mean_resids <- mean(resids)
    conf <- max(exp(sd(resids)), 70) # https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/overview/evaluation

    preds <- predict(fit_obj, x=log_test_seq_week)

    res <- list(Weeks_pred = test_seq_week, FVC_pred = exp(preds$y))
    conf_sqrt_n <- conf/sqrt(n)
    ubound <- res$FVC_pred + mean_resids + 1.96*conf_sqrt_n # strong hypothesis
    lbound  <- res$FVC_pred + mean_resids - 1.96*conf_sqrt_n


  if (plot_)
  {
    leg.txt <- c("Measured FVC", "Interpolated/Extrapolated FVC", "95% Confidence interval bound")
    
    plot(df$Weeks, df$FVC, col="blue", type="l", lwd=3,
         xlim = c(-12, 133), 
         ylim = c(min(min(lbound), min(df$FVC)),
                  max(max(ubound), max(df$FVC)) ),
         xlab = "Week", ylab = "FVC",
         main = paste0("Patient: ", df$Patient[1]))
    lines(res$Weeks_pred, res$FVC_pred)
    lines(res$Weeks_pred, ubound, lty=2, col="red")
    lines(res$Weeks_pred, lbound, lty=2, col="red")
    abline(v = max(df$Weeks), lty=2)
    legend("bottomright", legend = leg.txt, 
           lwd=c(3, 1, 1), lty=c(1, 1, 2), 
           col = c("blue", "black", "red"))
  }

  return(invisible(list(res = res,
              conf = rep(conf, length(res$FVC_pred)),
              mean = res$FVC_pred,
              ubound = ubound,
              lbound = lbound,
              resids = resids)))
}
```

# 1 - Import the whole dataset

```{r}
# Training set data
train <- read.csv("~/Documents/Kaggle/OSIC_August2020/train.csv")
```

```{r}
# Training set snippet
print(head(train))
print(tail(train))
```
![new-techtonique-website]({{base}}/images/2020-10-02/2020-10-02-image5.png){:class="img-responsive"}


# 2 - Predict FVC for a few patients (4)


```{r}
# Four patient ids are selected
ids <- c("ID00421637202311550012437", "ID00422637202311677017371",
         "ID00426637202313170790466", "ID00248637202266698862378")

#par(mfrow=c(2, 2))
for(i in 1:length(ids))
{
  # Extract patient's data based on his/her ID
  (df <- get_patient_data(id=ids[i], train))
  # Obtain FVC forecasts, with 95% confidence interval
  # warnings when repeated measures in the same week
  suppressWarnings(fit_predict(df, plot_=TRUE))
}
```

![new-techtonique-website]({{base}}/images/2020-10-02/2020-10-02-image1.png){:class="img-responsive"}

![new-techtonique-website]({{base}}/images/2020-10-02/2020-10-02-image2.png){:class="img-responsive"}

![new-techtonique-website]({{base}}/images/2020-10-02/2020-10-02-image3.png){:class="img-responsive"}

![new-techtonique-website]({{base}}/images/2020-10-02/2020-10-02-image4.png){:class="img-responsive"}



For a _quick and dirty_ baseline model, this one seems to produce quite coherent forecasts, which could be used for decision making. Of 
course, validation data (unseen by the model) could reveal a whole different truth. 
