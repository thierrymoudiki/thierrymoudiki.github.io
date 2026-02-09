---
layout: post
title: "R version of Python package survivalist, for model-agnostic  survival analysis"
description: "Example of use of Python package survivalist, for model-agnostic survival analysis"
date: 2026-02-09
categories: [R, Python]
comments: true
---

In this post, I present an example of use of R package `survivalist`. This package that works with [`uv`](https://docs.astral.sh/uv/) is a port of the Python package with the same name, so it follows exactly the same API, except for `.`s become `$`s in R. 

`survivalist` does model-agnostic survival analysis, i.e survival analysis with any supervised learning model which and `fit` + `predict`.

Blog posts containing Python examples: 

- [https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube)
- [https://thierrymoudiki.github.io/blog/2025/02/12/r/R-agnostic-survival-analysis](https://thierrymoudiki.github.io/blog/2025/02/12/r/R-agnostic-survival-analysis) (this was an R version, not packaged)
- [https://thierrymoudiki.github.io/blog/2024/12/15/python/agnostic-survival-analysis](https://thierrymoudiki.github.io/blog/2024/12/15/python/agnostic-survival-analysis)


Now the R version. 

## Install

Here you need to create a virtual environment, and install Python packages inside of it: 

- A the command line: 

```bash
# pip install uv # if necessary
uv venv venv
source venv/bin/activate
uv pip install pip survivalist
```

- In R console: 

```R
install.packages("remotes")
remotes::install_github("Techtonique/survivalist_r")
```

## Example 

This must be in the same folder as the virtual environment. 

```R
library(survivalist)
library(reticulate)
library(ggplot2)

# Initialize Python modules
survivalist <- survivalist::get_survivalist(venv_path = "./venv")
sklearn <- survivalist::get_sklearn(venv_path = "./venv")
pd <- survivalist::get_pandas(venv_path = "./venv")

# Aliases for commonly used functions/classes (see Python docs:https://docs.techtonique.net/survivalist/survivalist.html)
# And replace "." by "$"
load_whas500 <- survivalist$datasets$load_whas500
PIBoost <- survivalist$ensemble$PIComponentwiseGenGradientBoostingSurvivalAnalysis
Ridge <- sklearn$linear_model$Ridge
LinearRegression <- sklearn$linear_model$LinearRegression
train_test_split <- sklearn$model_selection$train_test_split

# One-hot encoding function
encode_categorical_columns <- function(df, categorical_columns = NULL) {
  if (is.null(categorical_columns)) {
    categorical_columns <- names(df)[sapply(df, is.character)]
  }
  py_to_r(pd$get_dummies(df, columns = categorical_columns))
}

# Split data in Python to avoid repeated conversion
# Load data
data <- load_whas500()
X <- py_to_r(data[[1]])
y <- py_to_r(data[[2]])

# Split into training and test sets
split_data <- train_test_split(
  reticulate::r_to_py(X),
  reticulate::r_to_py(y),
  test_size = 0.2,
  random_state = 42L
)
X_train <- split_data[[1]] # Keep X_train as a Python object
X_test <- split_data[[2]]  # Keep X_test as a Python object
y_train <- split_data[[3]] # Keep y_train as a Python object
y_test <- split_data[[4]]  # Keep y_test as a Python object

# Fit survival model
estimator <- PIBoost(regr = Ridge(), # could be any model here (sklearn-like)
                     type_pi = "bootstrap") 
estimator$fit(X_train, y_train)

# Predict survival functions for an individual
# Use X_test as a Python object
surv_funcs <- estimator$predict_survival_function(X_test$iloc[0:1, ]) # Indexing in Python starts from 0

# Extract survival function values
times <- surv_funcs$mean[[1]]$x
surv_prob <- surv_funcs$mean[[1]]$y
lower_bound <- surv_funcs$lower[[1]]$y
upper_bound <- surv_funcs$upper[[1]]$y

plot(
  times,
  surv_prob,
  type = "s",
  ylim = c(0, 1),
  xlab = "Time",
  ylab = "Survival Probability",
  main = "Survival Function for \n GradientBoosting(Ridge)"
)
polygon(c(times, rev(times)),
        c(lower_bound, rev(upper_bound)),
        col = "pink",
        border = NA)
lines(times, surv_prob, type = "s", lwd = 2)
```

![image-title-here]({{base}}/images/2026-02-09/2026-02-09-image1.png){:class="img-responsive"}
