---
layout: post
title: "Conformalize (improved prediction intervals and simulations) any R Machine Learning model with misc::conformalize" 
description: "A new function in the misc package allows you to perform conformal prediction with any R machine learning model. Conformal prediction improves prediction intervals' coverage rate thanks to held-out set cross-validation errors."
date: 2025-03-25
categories: R
comments: true
---

In the new version of [`misc`](https://r-packages.techtonique.net/packages), we introduce a `conformalize` function (work in progress, along with `predict` and `simulate` S3 methods), which allows you to perform conformal prediction with any R machine learning model. Conformal prediction improves prediction intervals' coverage rate thanks to held-out set cross-validation errors. 


```R
options(repos = c(techtonique = "https://r-packages.techtonique.net",
                  CRAN = "https://cloud.r-project.org"))

install.packages("misc")
```

## Example: Conformal Prediction with Out-of-Sample Coverage

In this example, we demonstrate how to use the `misc::conformalize` function to perform conformal prediction and calculate the out-of-sample coverage rate.

### Simulated Data

We will generate a simple dataset for demonstration purposes.

```R
set.seed(123)
n <- 200
x <- matrix(runif(n * 2), ncol = 2)
y <- 3 * x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.5)
data <- data.frame(x1 = x[, 1], x2 = x[, 2], y = y)
```

### Fit Conformal Model

Now, we'll use a linear model (`lm`) as the `fit_func` and its corresponding `predict` function as the `predict_func`.

```R
library(misc)
library(stats)

# Define fit and predict functions
fit_func <- function(formula, data, ...) lm(formula, data = data, ...)
predict_func <- function(fit, newdata, ...) predict(fit, newdata = newdata, ...)

# Apply conformalize
conformal_model <- misc::conformalize(
  formula = y ~ x1 + x2,
  data = data,
  fit_func = fit_func,
  predict_func = predict_func,
  split_ratio = 0.8,
  seed = 123
)
```

### Generate Predictions and Prediction Intervals

We will use the `predict` method to generate predictions and calculate prediction intervals.

```R
# New data for prediction
new_data <- data.frame(x1 = runif(50), x2 = runif(50))

# Predict with split conformal method
predictions <- predict(
  conformal_model,
  newdata = new_data,
  level = 0.95,
  method = "split"
)

head(predictions)
```

```R
##         fit        lwr      upr
## 1 1.6023773  0.5217324 2.683022
## 2 2.4634938  1.3828489 3.544139
## 3 0.6216433 -0.4590017 1.702288
## 4 0.9257140 -0.1549310 2.006359
## 5 2.0106565  0.9300115 3.091301
## 6 0.7427247 -0.3379203 1.823370
```

```R
head(simulate(conformal_model, newdata = new_data, method = "kde")[,1:10])
```

```R
         [,1]      [,2]      [,3]        [,4]      [,5]       [,6]      [,7]     [,8]
[1,]  1.0061613 1.4378707 1.5956107  0.82351501 2.7246968 0.73219187 2.0356222 1.695236
[2,]  1.8008609 2.7971134 1.2861305  2.58125871 3.4363754 1.86727777 1.2363179 3.012968
[3,]  0.7061998 1.0880965 0.7643145 -0.01608328 0.6978976 0.08354196 1.2873470 1.644337
[4,]  1.6744387 2.1808671 1.3589588  0.71969680 0.6200716 0.41251896 0.2132685 1.143104
[5,]  2.5601307 2.6514539 0.9412205  1.71331614 1.8461498 2.50201601 1.6053888 2.244651
[6,] -0.1938776 0.6363327 0.6612391  0.95181269 2.2220346 1.91485674 1.5329600 1.151063
          [,9]     [,10]
[1,] 0.9646508 1.8197675
[2,] 2.8635302 2.1993619
[3,] 1.0299818 0.3076988
[4,] 1.7906682 0.7944157
[5,] 2.3608802 2.0952129
[6,] 0.5367075 1.9065546
```

### Calculate Out-of-Sample Coverage Rate

The coverage rate is the proportion of true values that fall within the prediction intervals.

```R
# Simulate true values for the new data
true_y <- 3 * new_data$x1 + 2 * new_data$x2 + rnorm(50, sd = 0.5)

# Check if true values fall within the prediction intervals
coverage <- mean(true_y >= predictions[, "lwr"] & true_y <= predictions[, "upr"])

cat("Out-of-sample coverage rate:", coverage)
```

```R
## Out-of-sample coverage rate: 0.98
```

### Results

- The prediction intervals are calculated using the split conformal method.
- The out-of-sample coverage rate is displayed, which should be close to the specified confidence level (e.g., 0.95).

## Example: Conformal Prediction with the `MASS::Boston` Dataset

In this example, we use the `MASS::Boston` dataset to demonstrate conformal prediction.

### Load the Data

We will use the `MASS` package to access the `Boston` dataset.

```R
library(MASS)

# Load the Boston dataset
data(Boston)

# Inspect the dataset
head(Boston)
```

```R
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21
##   medv
## 1 24.0
## 2 21.6
## 3 34.7
## 4 33.4
## 5 36.2
## 6 28.7
```

### Split the Data

We will split the data into training and test sets to ensure they are disjoint.

```R
set.seed(123)
n <- nrow(Boston)
train_indices <- sample(seq_len(n), size = floor(0.8 * n))
train_data <- Boston[train_indices, ]
test_data <- Boston[-train_indices, ]
```

### Fit Conformal Model 1

```R
# Define fit and predict functions
fit_func <- function(formula, data, ...) MASS::rlm(formula, data = data, ...)
predict_func <- function(fit, newdata, ...) predict(fit, newdata, ...)

# Apply conformalize using the training data
conformal_model_boston <- misc::conformalize(
  formula = medv ~ .,
  data = train_data,
  fit_func = fit_func,
  predict_func = predict_func,
  seed = 123
)
```

### Generate Predictions and Prediction Intervals 1

We will use the `predict.conformalize` method to generate predictions and calculate prediction intervals for the test set.

```R
# Predict with split conformal method on the test data
predictions_boston <- predict(
  conformal_model_boston,
  newdata = test_data,
  level = 0.95,
  method = "split"
)

head(predictions_boston)
```

```R
##         fit       lwr      upr
## 1  29.92942 20.263283 39.59556
## 15 19.30837  9.642229 28.97451
## 17 20.71124 11.045100 30.37738
## 19 14.86650  5.200365 24.53264
## 28 14.79883  5.132688 24.46497
## 37 20.98752 11.321382 30.65366
```

### Calculate Out-of-Sample Coverage Rate 1

The coverage rate is the proportion of true values in the test set that fall within the prediction intervals.

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston[, "lwr"] & true_y_boston <= predictions_boston[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9509804
```

### Fit Conformal Model 2


```R
# Define fit and predict functions
fit_func <- function(formula, data, ...) stats::glm(formula, data = data, ...)
predict_func <- function(fit, newdata, ...) predict(fit, newdata, ...)

# Apply conformalize using the training data
conformal_model_boston <- misc::conformalize(
  formula = medv ~ .,
  data = train_data,
  fit_func = fit_func,
  predict_func = predict_func,
  seed = 123
)
```

### Generate Predictions and Prediction Intervals 2

We will use the `predict.conformalize` method to generate predictions and calculate prediction intervals for the test set.

```R
# Predict with split conformal method on the test data
predictions_boston <- predict(
  conformal_model_boston,
  newdata = test_data,
  level = 0.95,
  method = "split"
)

head(predictions_boston)
```

```R
# Predict with split conformal method on the test data
predictions_boston2 <- predict(
  conformal_model_boston,
  newdata = test_data,
  level = 0.95,
  method = "kde"
)

head(predictions_boston2)
```

```R
# Predict with split conformal method on the test data
predictions_boston3 <- predict(
  conformal_model_boston,
  newdata = test_data,
  level = 0.95,
  method = "surrogate"
)

head(predictions_boston3)
```

```R
# Predict with split conformal method on the test data
predictions_boston4 <- predict(
  conformal_model_boston,
  newdata = test_data,
  level = 0.95,
  method = "bootstrap"
)

head(predictions_boston4)
```

### Fit Conformal Model 2


```R
# Define fit and predict functions
fit_func <- function(formula, data, ...) ranger::ranger(formula, data = data)
predict_func <- function(fit, newdata, ...) predict(fit, newdata)$predictions

# Apply conformalize using the training data
conformal_model_boston_rf <- misc::conformalize(
  formula = medv ~ .,
  data = train_data,
  fit_func = fit_func,
  predict_func = predict_func,
  seed = 123
)

# Predict with split conformal method on the test data
predictions_boston_rf <- predict(
  conformal_model_boston_rf,
  newdata = test_data,
  predict_func = predict_func,
  level = 0.95,
  method = "kde"
)

head(predictions_boston_rf)
```

```R
##           fit       lwr      upr
## [1,] 27.03134 21.991838 32.43038
## [2,] 19.20299 13.542260 25.05314
## [3,] 21.34472 17.000993 30.77696
## [4,] 18.77455 12.341589 25.88818
## [5,] 15.60764  9.157478 21.48264
## [6,] 21.31355 14.591954 29.75374
```

```R
# Create a data frame for plotting
plot_data <- data.frame(
  Observation = seq_len(nrow(test_data)),
  TrueValue = test_data$medv,
  LowerBound = predictions_boston_rf[, "lwr"],
  UpperBound = predictions_boston_rf[, "upr"]
)
```

```R
# Sort data by observation for proper plotting
plot_data <- plot_data[order(plot_data$Observation), ]

# Plot the true values
plot(
  plot_data$Observation, plot_data$TrueValue,
  pch = 16, col = "blue", cex = 0.7,
  xlab = "Observation", ylab = "Value",
  main = "Prediction Intervals vs True Values"
)

# Add the prediction intervals using polygon
polygon(
  c(plot_data$Observation, rev(plot_data$Observation)),
  c(plot_data$LowerBound, rev(plot_data$UpperBound)),
  col = rgb(1, 0, 0, 0.2), border = NA
)

# Add points for true values again to overlay on the polygon
points(
  plot_data$Observation, plot_data$TrueValue,
  pch = 16, col = "blue", cex = 0.7
)
```

![image-title-here]({{base}}/images/2025-03-24/2025-03-24-image1.png)

### Calculate Out-of-Sample Coverage Rate 2

The coverage rate is the proportion of true values in the test set that fall within the prediction intervals.

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston[, "lwr"] & true_y_boston <= predictions_boston[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9411765
```

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston2[, "lwr"] & true_y_boston <= predictions_boston2[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9607843
```

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston3[, "lwr"] & true_y_boston <= predictions_boston3[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9705882
```

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston4[, "lwr"] & true_y_boston <= predictions_boston4[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9607843
```

```R
# True values for the test set
true_y_boston <- test_data$medv

# Check if true values fall within the prediction intervals
coverage_boston <- mean(true_y_boston >= predictions_boston_rf[, "lwr"] & true_y_boston <= predictions_boston_rf[, "upr"])

cat("Out-of-sample coverage rate for Boston dataset:", coverage_boston)
```

```R
## Out-of-sample coverage rate for Boston dataset: 0.9215686
```

### Results

- The prediction intervals are calculated using the split conformal method.
- The out-of-sample coverage rate is displayed, which should be close to the specified confidence level (e.g., 0.95).


