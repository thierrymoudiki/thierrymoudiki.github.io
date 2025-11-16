---
layout: post
title: "unifiedml: A Unified Machine Learning Interface for R, is now on CRAN + Discussion about AI replacing humans"
description: "unifiedml is now available on CRAN. This package provides a consistent interface for machine learning in R, making it easier to work with different ML algorithms using a unified API."
date: 2025-11-16
categories: R
comments: true
---

`unifiedml` is now available on CRAN. Just because. I wanted to see what was new :) 

We had an interesting discussion about the process of package publishing, and AI _replacing_ human beings (the type of discussion that I'll have with you when I'll be out of [this mysterious jail](https://www.change.org/stop_torturing_T_Moudiki)). 

Here is what **I had to say about it**: 

```bash
>      > I see it has changed :) Just a question for my personal culture:
>     Is this going to be done by AI someday? Because it could/should.
>
>     You want to be replaced by AI?


On 11.11.2025 09:23, Thierry Moudiki wrote:
> The question of whether one would want to be replaced by AI or not is
> too narrow (in general and in particular in this context).
>
> To me, personally, it's like asking someone, back in the day, whether
> he/she wants to use the wheel or not.
>
> It's already much better than us in many mundane tasks. And it could be
> way, way more objective in this particular context.

On 11.12.2025 00:09, Thierry Moudiki wrote:
I said "AI" to follow up on your argument but this has nothing to do with text models. Not everything has to be AI.

In order to validate or push such a thing in prod, you typically need GitHub Actions and a set of objective rules (e.g what's a critical warning, or a critical note, if there's such a thing...). It's not even AI (see e.g https://github.com/Techtonique/techtonique-r-pkgs). 

Typically, since submitting this package to your highest authority, I've shipped ~5 packages to PyPI (within minutes, and they also check if the code is malicious, or it will be removed) without being bothered by gatekeeping ("this comma is missing", "oh english grammar requires a full stop there"). And this is not an insult, it's food for thought (if you're humble enough to accept it), when Python is eating R at breakfast everyday. 

R seems to be stuck somewhere in a very distant past with discussions like this. 
```

That's the comment of a _foolish_ guy. If that is _foolishness_, I want to be _foolish_ forever.

The package, `unifiedml`, provides a consistent interface for machine learning in R, making it easier to work with different ML algorithms using a unified API.

# Why `unifiedml`?

R has an incredibly rich ecosystem of machine learning packages, but each comes with its own syntax and conventions. `unifiedml` is an effort to bridge this gap by providing:

- Extremely lightweight (see for yourself: [https://github.com/Techtonique/unifiedml/blob/main/R/model.R](https://github.com/Techtonique/unifiedml/blob/main/R/model.R)) and consistent API across different ML algorithms
- Automatic task detection (regression vs classification)
- Built-in cross-validation with appropriate metrics
- Model interpretation tools including feature importance and partial dependence plots
- Seamless integration with existing R packages (once installed) like glmnet, randomForest, and more

# Installation

```R
install.packages("unifiedml", repos = "https://cran.r-project.org/")

library(unifiedml)
```

# Quick Start: Regression Example

Here's how easy it is to build a regression model:

```R
library(glmnet)
data(mtcars)

# Prepare data
X <- as.matrix(mtcars[, -1])
y <- mtcars$mpg  # numeric → automatic regression

# Fit model
mod <- Model$new(glmnet::glmnet)
mod$fit(X, y, alpha = 0, lambda = 0.1)

# Make predictions
predictions <- mod$predict(X)

# Get model summary with feature importance
mod$summary()

# Visualize partial dependence
mod$plot(feature = 1)

# Cross-validation (automatically uses RMSE for regression)
cv_scores <- cross_val_score(mod, X, y, cv = 5)
cat("Mean RMSE:", mean(cv_scores), "\n")
```

# Quick Start: Classification Example

The same intuitive API works for classification:

```R
library(randomForest)

data(iris)

# Prepare data
X <- as.matrix(iris[, 1:4])
y <- iris$Species  # factor → automatic classification

# Fit model
mod <- Model$new(randomForest::randomForest)
mod$fit(X, y, ntree = 100)

# Make predictions
predictions <- mod$predict(X)

# Cross-validation (automatically uses accuracy for classification)
cv_scores <- cross_val_score(mod, X, y, cv = 5)
cat("Mean Accuracy:", mean(cv_scores), "\n")
```

**Key Features**

1. Consistent Interface
Whether you're using glmnet, randomForest, xgboost, or any other compatible algorithm, the interface remains the same. This makes it easy to:

- Switch between algorithms
- Compare model performance
- Build reproducible workflows

2. Automatic Task Detection
No need to specify whether you're doing regression or classification—unifiedml automatically detects this based on your target variable type.

3. Model Interpretation (in progress)
Built-in tools for understanding your models:
Feature importance rankings
Partial dependence plots
Comprehensive model summaries


# Note

`unifiedml` is still very young, so there _might_ be some rough edges. The package will mature over time as I refine the lightweight API (see for yourself: [https://github.com/Techtonique/unifiedml/blob/main/R/model.R](https://github.com/Techtonique/unifiedml/blob/main/R/model.R)), expand functionality, and incorporate user feedback. 

Feedback Welcome. 

![image-title-here]({{base}}/images/2025-11-16/2025-11-16-image1.png){:class="img-responsive"}

