---
layout: post
title: "tisthemachinelearner: New Workflow with uv for R and Python Integration of scikit-learn"
description: "Discover the updated workflow of tisthemachinelearner, now utilizing uv for seamless R and Python scikit-learn integration."
date: 2025-12-01
categories: [R, Python]
comments: true
---

A quick reminder of the previous post:  
👉 [https://thierrymoudiki.github.io/blog/2025/02/17/python/r/tisthemllearner](https://thierrymoudiki.github.io/blog/2025/02/17/python/r/tisthemllearner)

[tisthemachinelearner](https://github.com/Techtonique/tisthemachinelearner_r) is an R ([and Python](https://github.com/Techtonique/tisthemachinelearner)) package that provides a lightweight interface (with approx. 2 classes, hence facilitating benchmarks e.g) to the popular Python machine learning library **scikit-learn**. The package allows R users to leverage the power of scikit-learn models directly from R, using both **S3** and **R6** object-oriented programming styles.

Since then, **tisthemachinelearner** has evolved with a cleaner and more predictable workflow for connecting **R** to **Python scikit-learn**, using both **S3** and **R6** interfaces. It's now using a dedicated virtual environment manager called **uv** to handle Python dependencies seamlessly. Faster setup, less hassle!

uv is a lightweight and extremely fast tool to create and manage isolated Python environments. It simplifies the process of setting up the necessary Python environment for R packages that depend on Python libraries. Another advantage here, is that I know exactly what is installed in the environment, making it easier to debug potential issues.

## 1. Command line

```bash
# pip install uv # if necessary
uv venv venv
source venv/bin/activate
uv pip install pip scikit-learn
```

This creates an isolated Python environment containing the correct dependencies for the R interface to use.

---

## 2. Use it from R

```R
install.packages("devtools")
devtools::install_github("Techtonique/tisthemachinelearner_r")

library(tisthemachinelearner)

# Load data
data(mtcars)
head(mtcars)

# Split features and target
X <- as.matrix(mtcars[, -1])  # all columns except mpg
y <- mtcars[, 1]              # mpg column

# Create train/test split
set.seed(42)
train_idx <- sample(nrow(mtcars), size = floor(0.8 * nrow(mtcars)))
X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]

# --- R6 interface ---
model <- Regressor$new(model_name = "LinearRegression")
model$fit(X_train, y_train)
preds <- model$predict(X_test)
print(preds)

# --- S3 interface ---
model <- regressor(X_train, y_train, model_name = "LinearRegression")
preds <- predict(model, X_test)
print(preds)
```

![image-title-here]({{base}}/images/2025-02-17/2025-02-17-image1.png){:class="img-responsive"}

