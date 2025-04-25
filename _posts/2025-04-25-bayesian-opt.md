---
layout: post
title: "'Bayesian' optimization of hyperparameters in a R machine learning model using the bayesianrvfl package"
description: "Example of use of bayesianrvfl package for Bayesian optimization of hyperparameters in a machine learning model"
date: 2025-04-25
categories: R
comments: true
---


In this post, I will demonstrate how to use the `bayesianrvfl` package for ['Bayesian'  optimization](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_Optimization) of hyperparameters in a machine learning model. We will use the `Sonar` dataset from the `mlbench` package and optimize hyperparameters for an XGBoost model.

The surrogate model used for [Bayesian optimization](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_Optimization) is a Non-Bayesian Gaussian Random Vector Functional Link (RVFL) network (instead of a Gaussian Process) (see [Chapter 6](https://www.researchgate.net/publication/328954526_Interest_rates_modeling_for_insurance_interpolation_extrapolation_and_forecasting)), whose number of nodes in the hidden layer and volatility of residuals are chosen by using **maximum likelihood estimation** (MLE). This surrogate model is trained on 10 results of the objective function evaluations, and an Expected Improvement acquisition function is used to determine the next point to sample in the hyperparameter space.

```R
library("bayesianrvfl")
library("mlbench")
```

```R
data(Sonar)
```


```R
library(caret)
set.seed(998)
inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training <- Sonar[ inTraining,]
testing  <- Sonar[-inTraining,]
```

```R
objective <- function(xx) {
  fitControl <- trainControl(method = "cv", 
                           number = 3,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

  set.seed(825)
  model <- train(Class ~ ., data = training, 
                method = "xgbTree", 
                trControl = fitControl, 
                verbose = FALSE, 
                tuneGrid = data.frame(max_depth = floor(xx[1]),
                                      eta = xx[2],
                                      subsample = xx[3], 
                                      nrounds = floor(xx[5]),
                                      gamma = 0,
                                      colsample_bytree = xx[4],
                                      min_child_weight = 1),
                metric = "ROC")
  
  # Return the ROC value (higher is better)
  return(-getTrainPerf(model)$TrainROC)
}
```

```R
(res_rvfl <- bayesianrvfl::bayes_opt(objective, # objective function
          lower = c(1L, 0.001, 0.7, 0.7, 100L), # lower bound for search
          upper = c(8L, 0.1, 1, 1, 250L), # upper bound for search
          type_acq = "ei", # type of acquisition function
          nb_init = 10L, # number of points in initial design
          nb_iter = 40L, # number of iterations of the algo
          surrogate_model = "rvfl")) # surrogate model
```

# out-of-sample prediction

```R
xx <- res_rvfl$best_param

fitControl <- trainControl(method = "none", 
                           classProbs = TRUE)

  set.seed(825)
  model <- train(Class ~ ., data = training, 
                method = "xgbTree", 
                trControl = fitControl, 
                verbose = FALSE, 
                tuneGrid = data.frame(max_depth = floor(xx[1]),
                                      eta = xx[2],
                                      subsample = xx[3], 
                                      nrounds = floor(xx[5]),
                                      gamma = 0,
                                      colsample_bytree = xx[4],
                                      min_child_weight = 1),
                metric = "ROC")

preds <- predict(model, newdata = testing)
```

```R
caret::confusionMatrix(data = preds, reference = testing$Class)
```

```R
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  M  R
##          M 22  4
##          R  5 20
##                                          
##                Accuracy : 0.8235         
##                  95% CI : (0.6913, 0.916)
##     No Information Rate : 0.5294         
##     P-Value [Acc > NIR] : 1.117e-05      
##                                          
##                   Kappa : 0.6467         
##                                          
##  Mcnemar's Test P-Value : 1              
##                                          
##             Sensitivity : 0.8148         
##             Specificity : 0.8333         
##          Pos Pred Value : 0.8462         
##          Neg Pred Value : 0.8000         
##              Prevalence : 0.5294         
##          Detection Rate : 0.4314         
##    Detection Prevalence : 0.5098         
##       Balanced Accuracy : 0.8241         
##                                          
##        'Positive' Class : M              
## 
```

```{r fig.width=6.5}
# Get probability predictions for the whole test set
probs <- predict(model, newdata = testing, type = "prob")

# Create calibration curve data
create_calibration_data <- function(probs, actual, n_bins = 10) {
  # Convert actual to numeric (0/1)
  actual_numeric <- as.numeric(actual == levels(actual)[2])
  
  # Create bins based on predicted probabilities
  bins <- cut(probs[,2], breaks = seq(0, 1, length.out = n_bins + 1), 
              include.lowest = TRUE)
  
  # Calculate mean predicted probability and actual outcome for each bin
  cal_data <- data.frame(
    bin_mid = tapply(probs[,2], bins, mean),
    actual_freq = tapply(actual_numeric, bins, mean),
    n_samples = tapply(actual_numeric, bins, length)
  )
  
  cal_data$bin <- 1:nrow(cal_data)
  return(na.omit(cal_data))
}

# Generate calibration data
cal_data <- create_calibration_data(probs, testing$Class)

# Plot calibration curve
library(ggplot2)
ggplot(cal_data, aes(x = bin_mid, y = actual_freq)) +
  geom_point(aes(size = n_samples)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  geom_line() +
  xlim(0,1) + ylim(0,1) +
  labs(x = "Predicted Probability",
       y = "Observed Frequency",
       size = "Number of\nSamples",
       title = "Calibration Curve for XGBoost Model") +
  theme_minimal()

# Calculate calibration metrics
brier_score <- mean((probs[,2] - as.numeric(testing$Class == levels(testing$Class)[2]))^2)
cat("Brier Score:", round(brier_score, 4), "\n")
```

![image-title-here]({{base}}/images/2025-04-25/2025-04-25-image1.png)

```R
## Brier Score: 0.1268
```
