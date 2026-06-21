---
layout: post
title: "Using scikit-learn models in R easily with the tisthemachinelearner package"
description: "Using the tisthemachinelearner R package to train scikit-learn models and compute prediction intervals"
date: 2026-06-21
categories: R
comments: true
---

This post is about the [`tisthemachinelearner` R package](https://github.com/Techtonique/tisthemachinelearner_r/tree/main), that allows to use scikit-learn models in R. It is a wrapper around the [tisthemachinelearner Python package](https://github.com/Techtonique/tisthemachinelearner/tree/main). Prediction intervals can be computed using either split conformal prediction, surrogate methods or the bootstrap. 

First, you need to create a virtual environment and install the required packages in it. In R, you can use `system()` to run shell commands. At the command line: 

```bash
# pip install uv # if necessary
uv venv venv
source venv/bin/activate
uv pip install pip scikit-learn
```

In R, try:  

```R
# create venv
system("uv venv venv")
# install directly into that venv
system("venv/bin/uv pip install pip scikit-learn")
```

Now, the code below shows how to use `tisthemachinelearner` to train a Gradient Boosting Regressor on the `mtcars` dataset and compute prediction intervals using both the split conformal and surrogate methods. **Remark**: the package is also available from the [R-universe](https://techtonique.r-universe.dev/builds). 

```R
#install.packages("remotes")
#remotes::install_github("Techtonique/tisthemachinelearner_r")
#install.packages("tseries")

library(tisthemachinelearner)
library(tseries)

# Data
data(mtcars)

# Features and target
X <- subset(mtcars,
            select = c(cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb))

y <- mtcars$mpg

# Split features and target
X <- as.matrix(mtcars[, -1])  # all columns except mpg
y <- mtcars[, 1]              # mpg column

# Create train/test split
set.seed(123)
train_idx <- sample(nrow(mtcars), size = floor(0.7 * nrow(mtcars)))
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Train a Gradient Boosting Regressor
gbr_01 <- Regressor$new(
  model_name = "GradientBoostingRegressor",
  learning_rate = 0.1,
  n_estimators = 200L,
  max_depth = 3L,
  random_state = 123L,
  venv_path = "./venv" # this is crucial: the path to the virtual environment where scikit-learn is installed
)

gbr_01$fit(as.matrix(X_train), as.numeric(y_train))

pred_01 <- gbr_01$predict(as.matrix(X_test))

# Training RMSE
rmse_01 <- sqrt(mean((pred_01 - y_test)^2))

cat("RMSE (learning_rate = 0.1):", rmse_01, "\n")

gbr_001 <- Regressor$new(
  model_name = "GradientBoostingRegressor",
  learning_rate = 0.01,
  n_estimators = 200L,
  max_depth = 3L,
  random_state = 123L,
  venv_path = "./venv"
)

gbr_001$fit(as.matrix(X_train), as.numeric(y_train))

pred_001 <- gbr_001$predict(as.matrix(X_test))

rmse_001 <- sqrt(mean((pred_001 - y_test)^2))

cat("RMSE (learning_rate = 0.01):", rmse_001, "\n")

(pred_001_scp <- gbr_001$predict(as.matrix(X_test), method = "splitconformal"))
(pred_001_surr <- gbr_001$predict(as.matrix(X_test), method = "surrogate"))


# Test index
idx <- seq_along(y_test)

# Convert outputs to data frames
scp <- as.data.frame(pred_001_scp)
surr <- as.data.frame(pred_001_surr)

# --- Plot setup ---
plot(idx, y_test,
     pch = 19,
     col = "black",
     ylim = range(c(scp$lwr, scp$upr, surr$lwr, surr$upr, y_test)),
     xlab = "Test sample index",
     ylab = "MPG",
     main = "Prediction intervals: Split conformal vs Surrogate")

# Split conformal intervals
for (i in idx) {
  segments(i, scp$lwr[i], i, scp$upr[i], col = "blue", lwd = 2)
}

# Surrogate intervals (slightly shifted for visibility)
for (i in idx) {
  segments(i, surr$lwr[i], i, surr$upr[i], col = "red", lwd = 2, lty = 2)
}

# Point predictions (split conformal)
points(idx, scp$fit, col = "blue", pch = 16)

# Point predictions (surrogate)
points(idx, surr$fit, col = "red", pch = 17)

# True values
points(idx, y_test, col = "black", pch = 19)

legend("bottomleft",
       legend = c("True values", "Split conformal", "Surrogate"),
       col = c("black", "blue", "red"),
       pch = c(19, 16, 17),
       lty = c(NA, 1, 2),
       bty = "n")
```

![image-title-here]({{base}}/images/2026-06-21/2026-06-21-image1.png){:class="img-responsive"}


