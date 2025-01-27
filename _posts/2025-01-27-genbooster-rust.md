---
layout: post
title: "Gradient-Boosting and Boostrap aggregating anything (alert: high performance): Part5, easier install and Rust backend"
description: "Gradient-Boosting and Boostrap aggregating anything (alert: high performance): Part5, easier install and Rust backend"
date: 2025-01-27
categories: [Python, R]
comments: true
---

I recently released the [`genbooster`](https://github.com/Techtonique/genbooster) Python package (usable from R), a [Gradient-Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and [Boostrap aggregating](https://en.wikipedia.org/wiki/Bootstrap_aggregating) implementations that uses a Rust backend. Any base learner in the ensembles use randomized features as a form of feature engineering. The package was downloaded 3000 times in 5 days, so I guess it's somehow useful.

The last version of my generic Gradient Boosting algorithm was implemented in Python package mlsauce (see #172, #169, #166, #165), but the package can be difficult to install on some systems. If you're using Windows, for example, you may want to use the Windows Subsystem for Linux (WSL). Installing directly from PyPI is also nearly impossible, and it needs to be installed directly from GitHub.

This post is a quick overview of `genbooster`, in Python and R. It was an occasion to "learn"/try the Rust programming language, and I'm happy with the result; a stable package that's easy to install. However, I wasn't blown away ( hey Rust guys ;)) by the speed, which is roughly equivalent to Cython (that is, using C under the hood). 

![Bootstrap aggregating, source: Wikipedia]({{base}}/images/2025-01-27/2025-01-27-image1.png){:class="img-responsive"}

# Python version

{% include 2025-01-27-genbooster-rust.html %}

# R version

```bash
!mkdir -p ~/.virtualenvs
!python3 -m venv ~/.virtualenvs/r-reticulate
!source ~/.virtualenvs/r-reticulate/bin/activate
!pip install numpy pandas matplotlib scikit-learn tqdm genbooster
```

```R
utils::install.packages("reticulate")
```

```R
library(reticulate)

# Use a virtual environment or conda environment to manage Python dependencies
use_virtualenv("r-reticulate", required = TRUE)

# Import required Python libraries
np <- import("numpy")
pd <- import("pandas")
plt <- import("matplotlib.pyplot")
sklearn <- import("sklearn")
tqdm <- import("tqdm")
time <- import("time")

# Import specific modules from sklearn
BoosterRegressor <- import("genbooster.genboosterregressor", convert = FALSE)$BoosterRegressor
BoosterClassifier <- import("genbooster.genboosterclassifier", convert = FALSE)$BoosterClassifier
RandomBagRegressor <- import("genbooster.randombagregressor", convert = FALSE)$RandomBagRegressor
RandomBagClassifier <- import("genbooster.randombagclassifier", convert = FALSE)$RandomBagClassifier
ExtraTreeRegressor <- import("sklearn.tree")$ExtraTreeRegressor
mean_squared_error <- import("sklearn.metrics")$mean_squared_error
train_test_split <- import("sklearn.model_selection")$train_test_split


# Load Boston dataset
url <- "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv"
df <- read.csv(url)

# Split dataset into features and target
y <- df[["target"]]
X <- df[colnames(df)[c(-which(colnames(df) == "target"), -which(colnames(df) == "training_index"))]]

# Split into training and testing sets
set.seed(123)
index_train <- sample.int(nrow(df), nrow(df) * 0.8, replace = FALSE)
X_train <- X[index_train, ]
X_test <- X[-index_train,]
y_train <- y[index_train]
y_test <- y[-index_train]

# BoosterRegressor on Boston dataset
regr <- BoosterRegressor(base_estimator = ExtraTreeRegressor())
start <- time$time()
regr$fit(X_train, y_train)
end <- time$time()
cat(sprintf("Time taken: %.2f seconds\n", end - start))
rmse <- np$sqrt(mean_squared_error(y_test, regr$predict(X_test)))
cat(sprintf("BoosterRegressor RMSE: %.2f\n", rmse))

# RandomBagRegressor on Boston dataset
regr <- RandomBagRegressor(base_estimator = ExtraTreeRegressor())
start <- time$time()
regr$fit(X_train, y_train)
end <- time$time()
cat(sprintf("Time taken: %.2f seconds\n", end - start))
rmse <- np$sqrt(mean_squared_error(y_test, regr$predict(X_test)))
cat(sprintf("RandomBagRegressor RMSE: %.2f\n", rmse))


X <- as.matrix(iris[, 1:4])
y <- as.numeric(iris[, 5]) - 1
# Split into training and testing sets
set.seed(123)
index_train <- sample.int(nrow(iris), nrow(iris) * 0.8, replace = FALSE)
X_train <- X[index_train, ]
X_test <- X[-index_train,]
y_train <- y[index_train]
y_test <- y[-index_train]

regr <- BoosterClassifier(base_estimator = ExtraTreeRegressor())
start <- time$time()
regr$fit(X_train, y_train)
end <- time$time()
cat(sprintf("Time taken: %.2f seconds\n", end - start))
accuracy <- mean(y_test == as.numeric(regr$predict(X_test)))
cat(sprintf("BoosterClassifier accuracy: %.2f\n", accuracy))

regr <- RandomBagClassifier(base_estimator = ExtraTreeRegressor())
start <- time$time()
regr$fit(X_train, y_train)
end <- time$time()
cat(sprintf("Time taken: %.2f seconds\n", end - start))
accuracy <- mean(y_test == as.numeric(regr$predict(X_test)))
cat(sprintf("RandomBagClassifier accuracy: %.2f\n", accuracy))
```

```R
Time taken: 0.39 seconds
BoosterRegressor RMSE: 3.49
Time taken: 0.44 seconds
RandomBagRegressor RMSE: 4.06
Time taken: 0.28 seconds
BoosterClassifier accuracy: 0.97
Time taken: 0.37 seconds
RandomBagClassifier accuracy: 0.97
```