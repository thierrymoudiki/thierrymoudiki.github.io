---
layout: post
title: "Classification using linear regression"
description: Classification as linear regression of an Indicator Matrix, using nnetsauce.
date: 2021-09-26
categories: Python, QuasiRandomizedNN
---

In this post, I illustrate _classification using linear regression_, as implemented in Python/R package `nnetsauce`, and more precisely, in `nnetsauce`'s  `MultitaskClassifier`. If you're not interested in reading about the model description, you can jump directly to the 2nd section, "Two examples in Python". In addition, the [source code](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/multitask/multitaskClassifier.py) is relatively self-explanatory.

# Model description

Chapter 4 of [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) (ESL), at section _4.2 Linear Regression of an Indicator Matrix_, describes _classification using linear regression_ pretty well. Let $K \in \mathbb{N}$ be the number of classes and $y \in \mathbb{N}^n$ with values in $\lbrace 1, \ldots, K \rbrace$ be the variable to be explained. An __indicator response__ matrix $\textbf{Y} \in \mathbb{N}^{n \times K }$, containing only 0’s and 1’s, can be obtained from $y$. Each row  of $\textbf{Y}$ shall contain a single 1 -- in the column corresponding to the class where the example belongs, and 0's elsewhere.

Now, let $\textbf{X} \in \mathbb{R}^{n \times p }$ be the set of explanatory variables for $y$ and $\textbf{Y}$, with examples in rows, and characteristics in columns. ESL applies $K$ least squares  models to $\textbf{X}$, for each column of $\textbf{Y}$. The regression's predicted values can be interpreted as _raw_ estimates of probabilities, because the least squares' solution is a conditional expectation. And for $G$, a random variable describing the class, we have: 

$$
\mathbb{E} \left[ \mathbb{1}_{ G = k } | X = x \right] = \mathbb{P} \left[ G = k | X = x \right]
$$

The difference between `nnetsauce`'s `MultitaskClassifier` and the model described in ESL is:

- Any model possessing methods `fit` and `predict` can be used in lieu of a linear regression of $\textbf{Y}$ on $\textbf{X}$
- the set of covariates include the original covariates, $\textbf{X}$, __plus nonlinear transformations__ of  $\textbf{X}$, $h(\textbf{X})$, as done in [Quasi-Randomized Networks](https://thierrymoudiki.github.io/blog/index.html#QuasiRandomizedNN). Having $h(\textbf{X})$ as additional explanatory variables  enhances the models' flexibility; the __model is no longer linear__. 

- If for each $k \in \lbrace 1, \ldots, K \rbrace$, $\hat{f}_k(x)$ is the regression's predicted value for class $k$ and an observation characterized by $x$, `nnetsauce`'s `MultitaskClassifier`  obtains _probabilities_ that an observation characterized by $x$ belongs to class $k$ as: 

$$
\hat{p}_k(x) := \frac{expit \left( \hat{f}_k(x) \right)}{\sum_{i=1}^K expit \left( \hat{f}_k(x) \right)}
$$

Where we have $expit := \frac{1}{1 + exp(-x)}$. $x \mapsto expit(x)$ is strictly increasing, hence it preserves the ordering of _linear_ regression's predictions. $x \mapsto expit(x)$ is also  bounded in $[0, 1]$, which helps in avoiding overflows. I divide $expit \left( \hat{f}_k(x) \right)$ by $\sum_{i=1}^K expit \left( \hat{f}_k(x) \right)$, so that the _probabilities_ add up to 1. And to finish, the class predicted for an example characterized by $x$ is: 

$$
argmax_{k \in \lbrace 1, \ldots, K \rbrace} \hat{p}_k(x)
$$

# Two examples in Python

Currently, installing `nnetsauce` from Pypi doesn't work -- and I'm working on fixing it. However, you can install `nnetsauce` from GitHub as follows: 

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

Import the packages required for the 2 examples. 

```python
import nnetsauce as ns
import numpy as np
from sklearn.datasets import  load_wine, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
```

__1. Classification of iris dataset:__

```python
dataset = load_iris()
Z = dataset.data
t = dataset.target

# training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=143)

# Linear Regression is used here
regr3 = LinearRegression()

# `n_hidden_features` makes the model nonlinear
# `n_clusters` takes into account heterogeneity
fit_obj3 = ns.MultitaskClassifier(regr3, n_hidden_features=5, 
                                  n_clusters=2, type_clust="gmm")

# Adjust the model
start = time()
fit_obj3.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

# Classification report
start = time()
preds = fit_obj3.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))
```

```python
Elapsed 0.021012067794799805
Elapsed 0.0010943412780761719
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        12
           1       1.00      1.00      1.00         5
           2       1.00      1.00      1.00        13

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

__2. Classification of wine dataset:__

```python
dataset = load_wine()
Z = dataset.data
t = dataset.target

# training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=143)

# Linear Regression is used here 
regr4 = LinearRegression()

# `n_hidden_features` makes the model nonlinear
# `n_clusters` takes into account heterogeneity
fit_obj4 = ns.MultitaskClassifier(regr4, n_hidden_features=5, 
                                  n_clusters=2, type_clust="gmm")

# Adjust the model
start = time()
fit_obj4.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

# Classification report
start = time()
preds = fit_obj4.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))
```

```python
Elapsed 0.019229650497436523
Elapsed 0.001451253890991211
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      1.00      1.00        11
           2       1.00      1.00      1.00         9

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```

