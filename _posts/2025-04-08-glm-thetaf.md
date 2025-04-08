---
layout: post
title: "Extending the Theta forecasting method to GLMs and attention"
description: "Python and R examples of use of the Theta model for GLMs and attention (context)"
date: 2025-04-08
categories: [R, Python]
comments: true
---


In the new version (`v0.18.0`) of the [`ahead` package](https://r-packages.techtonique.net), I have extended the `forecast::thetaf` function to support Generalized Linear Models (GLMs) and added an [attention mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning)). 

Attention is widely used in current neural networks (because they tend to forget; blame it on the gradients :) ) to focus on specific parts of the input data when making predictions. 

In this case, it helps the model to learn which parts of the time series are more important for forecasting, by using weighted averages of the past observations.

More on this later **in a paper**. A link to a notebook containing Python and R examples is provided at the end of this post.

# 1 - R version 

Start with:

```R
options(repos = c(
    techtonique = "https://r-packages.techtonique.net",
    CRAN = "https://cloud.r-project.org"
))
install.packages("ahead")
```
# 1 - 1 - USAccDeaths 

```R
library(forecast)
library(ahead)

# glm.nb
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::glm.nb, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::glm.nb, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# glm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=stats::glm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=stats::glm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# rlm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::rlm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::rlm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# lqs
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::lqs, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=MASS::lqs, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# lm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=stats::lm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=stats::lm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# gam
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=gam::gam, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=gam::gam, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# rq
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=quantreg::rq, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(USAccDeaths, h=25L, fit_func=quantreg::rq, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")
```

# 1 - 2 - AirPassengers

```R
# glm.nb
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::glm.nb, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::glm.nb, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# glm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=stats::glm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=stats::glm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# rlm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::rlm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::rlm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# lm
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=stats::lm, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=stats::lm, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# lqs
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::lqs, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=MASS::lqs, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# gam
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=gam::gam, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=gam::gam, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")

# rq
par(mfrow=c(2,1))
obj1 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=quantreg::rq, attention = TRUE, type_pi = "conformal-split", method = "adj"))
plot(obj1, main="With attention")
obj2 <- suppressWarnings(ahead::glmthetaf(AirPassengers, h=25L, fit_func=quantreg::rq, attention = FALSE, type_pi = "conformal-split", method = "adj"))
plot(obj2, main="Without attention")
```

# 2 - Python version 

```python
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri

# Import required R packages
ahead = importr('ahead')
mass = importr('MASS')
base = importr('base')
stats = importr('stats')

# Get the data and fit the model
with localconverter(ro.default_converter):
    # Get AirPassengers data
    data = r('USAccDeaths')
    data = np.array(data)
    
    # Fit the model
    fit = r('''
        suppressWarnings(
            ahead::glmthetaf(
                USAccDeaths, 
                h=25L, 
                fit_func=MASS::glm.nb, 
                attention=TRUE, 
                type_pi="conformal-split", 
                method="adj"
            )
        )
    ''')
    
    # Extract predictions and intervals
    forecasts = np.array(fit.rx2('mean'))
    lower = np.array(fit.rx2('lower'))
    upper = np.array(fit.rx2('upper'))

# Create time indices
time_train = np.arange(len(data))
time_test = np.arange(len(data), len(data) + len(forecasts))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(time_train, data, 'b-', label='Observed', alpha=0.7)

# Plot forecasts and prediction intervals
plt.plot(time_test, forecasts, 'r--', label='Forecast')
plt.fill_between(time_test, lower, upper, 
                 color='r', alpha=0.2, 
                 label='95% Prediction Interval')

# Customize the plot
plt.title('USAccDeaths Forecast with Attention')
plt.xlabel('Time')
plt.ylabel('-')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
```

![image-title-here]({{base}}/images/2025-04-08/2025-04-08-image1.png)

<a target="_blank" href="https://colab.research.google.com/github/thierrymoudiki/notebooks/blob/main/Python/2025_04_08_theta_attention_Tourism2010.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>