---
layout: post
title: "mlsauce's `v0.12.0`: prediction intervals for LSBoostRegressor"
description: "Conformalized predictive simulations for LSBoostRegressor, a gradient boosting algorithm for penalized nonlinear least squares."
date: 2024-04-15
categories: Python
comments: true
---


Many of you (> 2600 reads so far) are reading [this document](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) on LSBoost, a gradient boosting algorithm for penalized nonlinear least squares. This **never** ceases to amaze me, because this document is quite... empty :)  


mlsauce's `v0.12.0` includes **prediction intervals** for the `LSBoostRegressor` in particular. These prediction intervals are obtained through the use of split conformal prediction (SCP, so far) with, also, the possibility of using SCP-bootstrap or SCP-kernel density estimation for simulation. 


```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose # this is the preferred way
```

```python
import subprocess
import sys
import matplotlib.pyplot as plt
import warnings

import mlsauce as ms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics

# ridge

print("\n")
print("ridge -----")
print("\n")


dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


dataset = load_diabetes()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")



# lasso

print("\n")
print("lasso -----")
print("\n")


dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50,
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


dataset = load_diabetes()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", reg_lambda=0.002,
                          col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(n_estimators=10, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, reg_lambda=0.003, dropout=0.4,
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


obj = ms.LSBoostRegressor(n_estimators=10, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, reg_lambda=0.001, dropout=0.4,
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True,
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


```

    
    
    ridge -----
    
    
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:04<00:00, 23.89it/s]


    4.20000147819519


    100%|██████████| 100/100 [00:03<00:00, 27.94it/s]


    4.357612133026123
    splitconformal coverage 1: 0.9505813953488372
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': 'bootstrap', 'verbose': 1}


    100%|██████████| 100/100 [00:04<00:00, 22.02it/s]


    4.560582876205444


    100%|██████████| 100/100 [00:02<00:00, 44.89it/s]
    100%|██████████| 50/50 [00:00<00:00, 28606.63it/s]


    2.8877038955688477
    splitconformal bootstrap coverage 1: 0.9590600775193798
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': 'kde', 'verbose': 1}


    100%|██████████| 100/100 [00:04<00:00, 20.70it/s]


    4.869342803955078


    100%|██████████| 100/100 [00:02<00:00, 34.91it/s]
    100%|██████████| 50/50 [00:00<00:00, 212.32it/s]


    3.8961992263793945
    splitconformal kde coverage 1: 0.9626937984496124
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:00<00:00, 260.57it/s]


    0.41154003143310547


    100%|██████████| 100/100 [00:00<00:00, 366.96it/s]


    0.32917046546936035
    splitconformal coverage 2: 0.9550561797752809
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': 'bootstrap', 'verbose': 1}


    100%|██████████| 100/100 [00:00<00:00, 299.65it/s]


    0.35674047470092773


    100%|██████████| 100/100 [00:00<00:00, 252.19it/s]
    100%|██████████| 50/50 [00:00<00:00, 89507.13it/s]


    0.4990723133087158
    splitconformal bootstrap coverage 2: 0.9662921348314607
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': 'kde', 'verbose': 1}


    100%|██████████| 100/100 [00:00<00:00, 199.36it/s]


    0.5178115367889404


    100%|██████████| 100/100 [00:00<00:00, 218.56it/s]
    100%|██████████| 50/50 [00:00<00:00, 238.12it/s]


    0.7703864574432373
    splitconformal kde coverage 2: 0.9662921348314607
    
    
    lasso -----
    
    
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 50, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 50/50 [00:01<00:00, 39.21it/s]


    1.3068976402282715


    100%|██████████| 50/50 [00:00<00:00, 95.31it/s]


    0.6802136898040771
    splitconformal coverage 3: 0.9510658914728682
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 50, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': 'bootstrap', 'verbose': 1}


    100%|██████████| 50/50 [00:00<00:00, 54.06it/s]


    0.9406630992889404


    100%|██████████| 50/50 [00:00<00:00, 102.88it/s]
    100%|██████████| 50/50 [00:00<00:00, 18774.86it/s]


    0.6879117488861084
    splitconformal bootstrap coverage 3: 0.9605135658914729
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 50, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': 'kde', 'verbose': 1}


    100%|██████████| 50/50 [00:01<00:00, 31.35it/s]


    1.624236822128296


    100%|██████████| 50/50 [00:00<00:00, 54.72it/s]
    100%|██████████| 50/50 [00:00<00:00, 315.17it/s]


    1.4067130088806152
    splitconformal kde coverage 3: 0.9631782945736435
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 50, 'n_hidden_features': 5, 'reg_lambda': 0.002, 'replications': None, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 50/50 [00:00<00:00, 131.96it/s]


    0.3877689838409424


    100%|██████████| 50/50 [00:00<00:00, 197.36it/s]


    0.27906036376953125
    splitconformal coverage 4: 0.9550561797752809
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0.4, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 10, 'n_hidden_features': 5, 'reg_lambda': 0.003, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': 'bootstrap', 'verbose': 1}


    100%|██████████| 10/10 [00:00<00:00, 193.79it/s]


    0.06406068801879883


    100%|██████████| 10/10 [00:00<00:00, 172.80it/s]
    100%|██████████| 50/50 [00:00<00:00, 81190.55it/s]


    0.10322904586791992
    splitconformal bootstrap coverage 4: 0.9662921348314607
    {'activation': 'relu', 'backend': 'cpu', 'col_sample': 0.9, 'direct_link': 1, 'dropout': 0.4, 'kernel': None, 'learning_rate': 0.1, 'n_estimators': 10, 'n_hidden_features': 5, 'reg_lambda': 0.001, 'replications': 50, 'row_sample': 0.9, 'seed': 123, 'solver': 'lasso', 'tolerance': 0.0001, 'type_pi': 'kde', 'verbose': 1}


    100%|██████████| 10/10 [00:00<00:00, 306.63it/s]


    0.05145859718322754


    100%|██████████| 10/10 [00:00<00:00, 228.20it/s]
    100%|██████████| 50/50 [00:00<00:00, 1266.03it/s]

    0.128190279006958
    splitconformal kde coverage 4: 0.9550561797752809


    



```python
warnings.filterwarnings('ignore')

split_color = 'green'
split_color2 = 'orange'
local_color = 'gray'

def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color="",
              method_name="",
              title=""):

    fig = plt.figure()

    plt.plot(x, y, 'k.', alpha=.3, markersize=10,
             fillstyle='full', label=u'Test set observations')

    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_u, y_l[::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' Prediction interval')

    if pred is not None:
        plt.plot(x, pred, 'k--', lw=2, alpha=0.9,
                 label=u'Predicted value')

    #plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.show()
```


```python
max_idx = 50
plot_func(x = range(max_idx),
          y = y_test[0:max_idx],
          y_u = preds.upper[0:max_idx],
          y_l = preds.lower[0:max_idx],
          pred = preds.mean[0:max_idx],
          shade_color=split_color2,
          title = f"LSBoostRegressor ({max_idx} first points in test set)")
```

![xxx]({{base}}/images/2024-04-15/2024-04-15-image1.png){:class="img-responsive"}          

