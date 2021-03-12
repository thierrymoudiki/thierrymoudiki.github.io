---
layout: post
title: "Explaining xgboost predictions with the teller"
description: "Explaining xgboost predictions (like R's linear models) with the teller "
date: 2021-03-12
categories: [Python, ExplainableML]
---

Nowadays, explaining the decisions of Statistical/Machine learning (ML) algorithms is 
becoming a must, and also, mainstream. In healthcare for example, ML explainers could help in understanding how _black-box_ -- but accurate -- ML prognosis about patients are formed.  

One way to obtain these explanations (here is [another way](https://thierrymoudiki.github.io/blog/2020/11/06/explainableml/r/misc/xai-krr-surrogate) that I introduced in a previous post, based on Kernel Ridge 
regression), is to use [the teller](https://techtonique.github.io/teller/). The teller computes explanatory variables's effects by using [finite differences](https://en.wikipedia.org/wiki/Finite_difference). In this post, in particular, the teller is utilized to explain the popular **xgboost**'s  predictions on the Boston dataset.


The Boston dataset contains the following columns: 

+ crim: per capita crime rate by town.

+ zn: proportion of residential land zoned for lots over 25,000 sq.ft.

+ indus: proportion of non-retail business acres per town.

+ chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

+ nox: nitrogen oxides concentration (parts per 10 million).

+ rm: average number of rooms per dwelling.

+ age: proportion of owner-occupied units built prior to 1940.

+ dis: weighted mean of distances to five Boston employment centres.

+ rad: index of accessibility to radial highways.

+ tax: full-value property-tax rate per \$10,000.

+ ptratio: pupil-teacher ratio by town.

+ lstat: lower status of the population (percent).

+ medv: median value of owner-occupied homes in \$1000s.

Our objective is **understand how xgboost's predictions of `medv`, are influenced by the other explanatory variables**. 


# Installing packages `teller` and `xgboost`

```bash
!pip install the-teller --upgrade
```

```bash
!pip install xgboost --upgrade
```


# Applying the teller's `Explainer` to xgboost predictions

We start by importing the packages and dataset useful for the demo: 

```python
import teller as tr
import pandas as pd
import numpy as np  
import xgboost as xgb    

from sklearn import datasets, linear_model
from sklearn.datasets import load_boston
from sklearn import datasets
from sklearn.model_selection import train_test_split
from time import time


# import data
boston = datasets.load_boston()
X = np.delete(boston.data, 11, 1)
y = boston.target
col_names = np.append(np.delete(boston.feature_names, 11), 'MEDV')
```

The dataset is **splitted into a training set and a test set**, then xgboost is 
adjusted to the training set: 

```python 
# training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1233)

# fitting xgboost to the training set 
regr = xgb.XGBRegressor(max_depth = 4, n_estimators = 100).fit(X_train, y_train)
```

The teller's `Explainer` is now used in order to: understand how xgboost's predictions of `medv` are influenced by the explanatory variables.

```python
start = time()

# creating an Explainer for the fitted object `regr`
expr = tr.Explainer(obj=regr)

# confidence int. and tests on covariates' effects (Jackknife)
expr.fit(X_test, y_test, X_names=col_names[:-1], y_name=col_names[-1], method="ci")

# summary of results
expr.summary()

# timing
print(f"\n Elapsed: {time()-start}")
```

![image-title-here]({{base}}/images/2021-03-12/2021-03-12-image1.png){:class="img-responsive"}

The variables with the most impactful effect on `medv` are `nox` and `rm` which is an acceptable observation: an increasing number of rooms drives the price higher, whereas pollution, an increase in nitrogen oxides concentration (as long as the information is well-known by people in the city) drives a decrease in home prices.  

In order to obtain the 95% confidence intervals presented in the output, [Jackknife resampling](https://en.wikipedia.org/wiki/Jackknife_resampling) is employed. If the confidence interval 
does not contain 0, then the average effect is significantly different from 0, and the hypothesis 
that it's equal to 0 is rejected with a 5% risk of being wrong. 

The little stars on the right indicate how significant is the Student test (not robust, but still carrying some useful information) "average effect of covariate x = 0" versus the contrary, with a 5% risk of being wrong. 

Armed with this, we could say that in this context, in this particular setting, removing RAD, ZN, CHAS, CRIM is suggested by `xgboost` and the teller. 
