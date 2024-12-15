---
layout: post
title: "survivalist: Probabilistic model-agnostic survival analysis using scikit-learn, glmnet, xgboost, lightgbm, pytorch, keras, nnetsauce and mlsauce" 
description: "Survival analysis is used to predict the time until an event of interest occurs. In this post, I show how to use scikit-learn, glmnet, xgboost, pytorch, keras, and nnetsauce for probabilistic survival analysis"
date: 2024-12-15
categories: Python
comments: true
---

**Survival analysis** is a group of Statistical/Machine Learning (ML) methods for predicting the **time until an event of interest occurs**. Examples of events include: 

- death 
- failure
- recovery
- default
- etc.

And the event of interest can be anything that has a duration: 

- the time until a machine breaks down 
- the time until a customer buys a product 
- the time until a patient dies
- etc. 

The event can be **censored**, meaning that it has'nt occurred for some subjects at the time of analysis. 

In this post, I show how to use `scikit-learn`, `glmnet`, `xgboost`, `lightgbm`, `pytorch`, `keras`, `nnetsauce` and `mlsauce` in conjuction with Python package [`survivalist`](https://github.com/Techtonique/survivalist) for probabilistic survival analysis. The probabilistic part is based on **conformal prediction and Bayesian inference**, and graphics represent the out-of-sample ML survival function vs Empirical Kaplan-Meier survival function (with confidence intervals).

A link to the corresponding notebook can be found at the end of this post.

# Contents

- [Contents](#contents)
- [0 - Installation](#0---installation)
- [1 - using `scikit-learn` with conformal prediction](#1---using-scikit-learn-with-conformal-prediction)
- [2 - using `nnetsauce`](#2---using-nnetsauce)
  - [2 - 1 with conformal prediction](#2---1-with-conformal-prediction)
  - [2 - 2 with Bayesian Inference](#2---2-with-bayesian-inference)
- [3 - using `glmnet`](#3---using-glmnet)
- [4 - using `pytorch`](#4---using-pytorch)
- [5 - Using keras (through `scikeras`)](#5---using-keras-through-scikeras)
- [6 - using `xgboost`](#6---using-xgboost)
- [7 - using `lightgbm`](#7---using-lightgbm)
- [8 - using Generic Boosting (`mlsauce`)](#8---using-generic-boosting-mlsauce)


# 0 - Installation

```python
!pip uninstall -y survivalist
```

```python
!pip install survivalist --upgrade --no-cache-dir
```

```python
!pip install glmnetforpython --verbose --upgrade --no-cache-dir
```

```python
!pip install nnetsauce --verbose --upgrade --no-cache-dir
```

```python
!pip install scikeras
```

```python
!pip install xgboost --upgrade --no-cache-dir
!pip install lightgbm --upgrade --no-cache-dir
```

```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```

```python
import numpy as np
```

```python
import pandas as pd

def _encode_categorical_columns(df, categorical_columns=None):
    """
    Automatically identifies categorical columns and applies one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with mixed continuous and categorical variables.
    - categorical_columns (list): Optional list of column names to treat as categorical.

    Returns:
    - pd.DataFrame: A new DataFrame with one-hot encoded categorical columns.
    """
    # Automatically identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply one-hot encoding to the identified categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Convert boolean columns to integer (0 and 1)
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

    return df_encoded

```


```python
import matplotlib.pyplot as plt
import nnetsauce as ns
import glmnetforpython as glmnet
from survivalist.nonparametric import kaplan_meier_estimator
from survivalist.datasets import load_whas500, load_gbsg2, load_veterans_lung_cancer
from survivalist.ensemble import ComponentwiseGenGradientBoostingSurvivalAnalysis
from survivalist.custom import SurvivalCustom
from survivalist.custom import PISurvivalCustom
from survivalist.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from survivalist.ensemble import PIComponentwiseGenGradientBoostingSurvivalAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from time import time

import matplotlib.pyplot as plt
import nnetsauce as ns
import numpy as np
from survivalist.datasets import load_whas500, load_veterans_lung_cancer, load_gbsg2
from survivalist.custom import SurvivalCustom
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from survivalist.metrics import brier_score, integrated_brier_score
from time import time

import pandas as pd
```

# 1 - using `scikit-learn` with conformal prediction


```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="bootstrap")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:09<00:00, 10.54it/s]
    100%|██████████| 100/100 [00:10<00:00,  9.61it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_13_2.png){:class="img-responsive"}          
    

```python
X, y = load_gbsg2()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])


for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:02<00:00, 42.51it/s]
    100%|██████████| 100/100 [00:02<00:00, 42.67it/s]



    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_14_2.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:02<00:00, 44.45it/s]
    100%|██████████| 100/100 [00:02<00:00, 35.99it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_15_2.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="ecdf")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:01<00:00, 51.44it/s]
    100%|██████████| 100/100 [00:01<00:00, 51.30it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_16_2.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")


estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:01<00:00, 51.20it/s]
    100%|██████████| 100/100 [00:01<00:00, 51.52it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_17_2.png){:class="img-responsive"}          
    




# 2 - using `nnetsauce`

## 2 - 1 with conformal prediction


```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(RidgeCV()),
                                                               type_pi="bootstrap")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:42<00:00,  2.35it/s]
    100%|██████████| 100/100 [00:40<00:00,  2.46it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_21_2.png){:class="img-responsive"}          
    



```python
from pickle import Pickler
X, y = load_whas500()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=ns.CustomRegressor(RidgeCV()), type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```
    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_22_1.png){:class="img-responsive"}          
    



```python
X, y = load_gbsg2()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(RidgeCV()),
                                                               type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])


for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:22<00:00,  4.48it/s]
    100%|██████████| 100/100 [00:21<00:00,  4.71it/s]

    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_23_2.png){:class="img-responsive"}          
    


```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(RidgeCV()),
                                                               type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:17<00:00,  5.88it/s]
    100%|██████████| 100/100 [00:18<00:00,  5.42it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_24_2.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=ns.CustomRegressor(RandomForestRegressor()),
                             type_pi="bootstrap")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```
    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_25_1.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(RidgeCV()),
                                                               type_pi="ecdf")


estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:16<00:00,  5.95it/s]
    100%|██████████| 100/100 [00:17<00:00,  5.79it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_26_2.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=ns.CustomRegressor(RandomForestRegressor()),
                             type_pi="ecdf")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```
    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_27_1.png){:class="img-responsive"}          
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(RidgeCV()),
                                                               type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:27<00:00,  3.69it/s]
    100%|██████████| 100/100 [00:16<00:00,  6.04it/s]


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_28_2.png){:class="img-responsive"}          
    


## 2 - 2 with Bayesian Inference


```python


def encode_categorical_columns(df, categorical_columns=None):
    """
    Automatically identifies categorical columns and applies one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with mixed continuous and categorical variables.
    - categorical_columns (list): Optional list of column names to treat as categorical.

    Returns:
    - pd.DataFrame: A new DataFrame with one-hot encoded categorical columns.
    """
    # Automatically identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply one-hot encoding to the identified categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Convert boolean columns to integer (0 and 1)
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

    return df_encoded


X, y = load_veterans_lung_cancer()
X = encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=42)

print("\n\n BayesianRidge ------------------")

estimator = SurvivalCustom(regr=ns.CustomRegressor(BayesianRidge()))
estimator2 = SurvivalCustom(regr=ns.CustomRegressor(GaussianProcessRegressor()))
estimator3 = SurvivalCustom(regr=ns.CustomRegressor(ARDRegression()))

start = time()
estimator.fit(X_train, y_train)
print("Time to fit BayesianRidge: ", time() - start)
start = time()
estimator2.fit(X_train, y_train)
print("Time to fit GaussianProcessRegressor: ", time() - start)
start = time()
estimator3.fit(X_train, y_train)
print("Time to fit ARDRegression: ", time() - start)


surv_funcs = estimator.predict_survival_function(X_test.iloc[0:1,:], return_std=True)
surv_funcs2 = estimator2.predict_survival_function(X_test.iloc[0:1,:], return_std=True)
surv_funcs3 = estimator3.predict_survival_function(X_test.iloc[0:1,:], return_std=True)
```

    
    
     BayesianRidge ------------------
    Time to fit BayesianRidge:  0.17850041389465332
    Time to fit GaussianProcessRegressor:  0.4104886054992676
    Time to fit ARDRegression:  0.5052413940429688



```python
event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")

```


```python
for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_32_0.png){:class="img-responsive"}          
    



```python
for fn in surv_funcs2.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs2.lower[0].y, surv_funcs2.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_33_0.png){:class="img-responsive"}          
    



```python
for fn in surv_funcs3.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs3.lower[0].y, surv_funcs3.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_34_0.png){:class="img-responsive"}          
    


# 3 - using `glmnet`


```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=ns.CustomRegressor(glmnet.GLMNet(lambdau=1000)),
                             type_pi="bootstrap")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_36_0.png){:class="img-responsive"}          
    



```python
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(glmnet.GLMNet(lambdau=1000)),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:36<00:00,  2.71it/s]
    100%|██████████| 100/100 [00:40<00:00,  2.49it/s]



    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_37_1.png){:class="img-responsive"}          
    


# 4 - using `pytorch`


```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=1, hidden_sizes=(64, 32), activation=nn.ReLU,
                 learning_rate=0.001, max_epochs=100, batch_size=32, random_state=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

    def _build_model(self):
        layers = []
        input_dim = self.input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation())
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def fit(self, X, y, sample_weight=None):

        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        X, y = self._prepare_data(X, y)
        self._build_model()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = self._prepare_data(X)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).squeeze()
        return predictions.numpy()

    def _prepare_data(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float32)
            return X, y
        return X
```


```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=42)
# Convert X_train and X_test to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=MLPRegressorTorch(input_size=X_train.shape[1]+1,
                                                    hidden_sizes=(20, 20, 20),
                                                    max_epochs=200,
                                                    random_state=42),
                             type_pi="ecdf")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_40_0.png){:class="img-responsive"}          
    



```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=42)
# Convert X_train and X_test to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=MLPRegressorTorch(input_size=X_train.shape[1]+1,
                                                    hidden_sizes=(20, 20, 20),
                                                    max_epochs=200,
                                                    random_state=42),
                             type_pi="ecdf")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_41_0.png){:class="img-responsive"}          
    


# 5 - Using keras (through `scikeras`)


```python
import keras
import keras.models
from scikeras.wrappers import KerasRegressor


def get_reg(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    return model

reg = KerasRegressor(
    model=get_reg,
    loss="mse",
    metrics=[keras.metrics.R2Score],
    hidden_layer_sizes=(20, 20, 20),
    dropout=0.1,
    verbose=0,
    random_state=123
)
```


```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=reg,
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```
    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_44_1.png){:class="img-responsive"}          
    

```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=reg,
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_45_0.png){:class="img-responsive"}          
    


# 6 - using `xgboost`


```python

```


```python
import xgboost as xgb

X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=xgb.XGBRegressor(),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_48_0.png){:class="img-responsive"}          
    



```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=xgb.XGBRegressor(),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_49_0.png){:class="img-responsive"}          
    


# 7 - using `lightgbm`


```python

```


```python
import lightgbm as lgb

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=lgb.LGBMRegressor(verbose=-1,
                                                    random_state=42),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_52_0.png){:class="img-responsive"}          
    



```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=lgb.LGBMRegressor(verbose=-1,
                                                    random_state=42),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_53_0.png){:class="img-responsive"}          
    


# 8 - using Generic Boosting (`mlsauce`)


```python
import mlsauce as ms
```


```python
regr_ridge = ms.GenericBoostingRegressor(ms.RidgeRegressor(reg_lambda=1e3),
                                         verbose=0)

```


```python

```


```python
# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=regr_ridge,
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_58_0.png){:class="img-responsive"}          
    



```python
X, y = load_whas500()
X = _encode_categorical_columns(X)

# Create a new structured array with Survival_in_days as float32
new_dtype = [('Status', '?'), ('Survival_in_days', '<f4')]
y_converted = np.array(y.tolist(), dtype=new_dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y_converted,
                                                    test_size=0.2,
                                                    random_state=4)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=regr_ridge,
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

![png]({{base}}/images/2024-12-15/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce_59_0.png){:class="img-responsive"}          


<a target="_blank" href="https://colab.research.google.com/github/Techtonique/survivalist/blob/main/survivalist/demo/2024-12-15-survivalist-sklearn-pytorch-keras-nnetsauce.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
