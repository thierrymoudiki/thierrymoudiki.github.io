---
layout: post
title: "A lightweight interface to scikit-learn in R: Bayesian and Conformal prediction"
description: "Example of use of tisthemachinelearner; a lightweight interface to scikit-learn in R: Bayesian and Conformal prediction"
date: 2025-04-21
categories: R
comments: true
---

[tisthemachinelearner](https://docs.techtonique.net/tisthemachinelearner_r) is a  (work in progress) lightweight interface to scikit-learn. Here's an example of use based on Bayesian and Conformal prediction.

If it doesn't work directly, try to [install from source](https://r-packages.techtonique.net/). 

# Install packages 

```R
options(repos = c(
    techtonique = "https://r-packages.techtonique.net",
    CRAN = "https://cloud.r-project.org"
))
```

```R
install.packages("tisthemachinelearner")
install.packages("tseries")
library("tisthemachinelearner")
library("tseries")
```

# Import data

```R
library(MASS)
data(Boston)

set.seed(1243)
train_idx <- sample(nrow(Boston), size = floor(0.8 * nrow(Boston)))
train_data <- Boston[train_idx, ]
test_data <- Boston[-train_idx, -14]  # -14 removes 'medv' (target variable)

X_train <- as.matrix(Boston[train_idx, -14])
X_test <- as.matrix(Boston[-train_idx, -14])
y_train <- Boston$medv[train_idx]
y_test <- Boston$medv[-train_idx]
```

# Fit model

```R
# R6 interface
model <- tisthemachinelearner::Regressor$new(model_name = "BayesianRidge")
start <- proc.time()[3]
model$fit(X_train, y_train)
end <- proc.time()[3]
cat("Time taken:", end - start, "seconds\n")

start <- proc.time()[3]
preds_bayesian <- model$predict(X_test, method="bayesian")
end <- proc.time()[3]
cat("Time taken:", end - start, "seconds\n")
#print(preds_bayesian)

model <- tisthemachinelearner::Regressor$new(model_name = "RidgeCV")
start <- proc.time()[3]
model$fit(X_train, y_train)
end <- proc.time()[3]
cat("Time taken:", end - start, "seconds\n")

start <- proc.time()[3]
preds_conformal <- model$predict(X_test, method="surrogate")
end <- proc.time()[3]
cat("Time taken:", end - start, "seconds\n")
#print(preds_conformal)
```

# Coverage rate as a function of level = 95 

```R
> mean((preds_bayesian[, "lwr"] <= y_test)*(preds_bayesian[, "upr"] >= y_test))*100
[1] 99.01961
> mean((preds_conformal[, "lwr"] <= y_test)*(preds_conformal[, "upr"] >= y_test))*100
[1] 95.09804
> 
```

# Plot results

```R
par(mfrow=c(1, 2))

x_ordered <- order(y_test)
# Ridge Model Plot
plot(y_test, preds_bayesian[, "fit"],
     main = "Ridge Model Predictions",
     xlab = "Actual MPG", ylab="Predicted MPG", 
     ylim = c(min(preds_bayesian[, "lwr"]), 
              max(preds_bayesian[, "upr"])))
# Add shaded prediction intervals
polygon(c(y_test[x_ordered], rev(y_test[x_ordered])),
        c(preds_bayesian[, "lwr"][x_ordered], 
          rev(preds_bayesian[, "upr"][x_ordered])),
        col=rgb(0, 0, 1, 0.2), border=NA)
points(y_test, preds_bayesian[, "fit"])  # Replot points over shading
abline(0, 1, col="red", lty=2)  # Add diagonal line

x_ordered <- order(y_test)
# Ridge Model Plot
plot(y_test, preds_conformal[, "fit"],
     main = "Ridge Model Predictions",
     xlab = "Actual MPG", ylab="Predicted MPG", 
     ylim = c(min(preds_conformal[, "lwr"]), 
              max(preds_conformal[, "upr"])))
# Add shaded prediction intervals
polygon(c(y_test[x_ordered], rev(y_test[x_ordered])),
        c(preds_conformal[, "lwr"][x_ordered], 
          rev(preds_conformal[, "upr"][x_ordered])),
        col=rgb(0, 0, 1, 0.2), border=NA)
points(y_test, preds_conformal[, "fit"])  # Replot points over shading
abline(0, 1, col="red", lty=2)  # Add diagonal line
```

![image-title-here]({{base}}/images/2025-04-21/2025-04-21-image1.png)


