---
layout: post
title: "New nnetsauce version with CustomBackPropRegressor (CustomRegressor with Backpropagation) and ElasticNet2Regressor (Ridge2 with ElasticNet regularization)"
description: "New nnetsauce version with CustomBackPropRegressor and ElasticNet2Regressor: python examples included."
date: 2025-07-15
categories: [Python]
comments: true
---

A new version of `nnetsauce` is out, with 2 new classes: 

-  `CustomBackPropRegressor` (see [https://docs.techtonique.net/nnetsauce/nnetsauce.html#CustomBackPropRegressor](https://docs.techtonique.net/nnetsauce/nnetsauce.html#CustomBackPropRegressor)) 
  
-  `ElasticNet2Regressor` (see [https://docs.techtonique.net/nnetsauce/nnetsauce.html#ElasticNet2Regressor](https://docs.techtonique.net/nnetsauce/nnetsauce.html#ElasticNet2Regressor))

`CustomBackPropRegressor` is based on `CustomRegressor` and allows for a backpropagation of the hidden layer parameters. 

`ElasticNet2Regressor` is a nonlinear Elastic Net regression model is an enhanced elastic net regression model with dual regularization paths, and additional nonlinear features supporting both CPU and GPU/TPU acceleration via JAX. It implements a mixed L1/L2 regularization approach with separate parameters for direct and hidden layer connections. More details can be found in the [https://www.researchgate.net/publication/393711332_Introducing_nnetsauce's_ElasticNet2Regressor_an_enhanced_elastic_net_regression_model_with_dual_regularization_paths_and_additional_nonlinear_features](https://www.researchgate.net/publication/393711332_Introducing_nnetsauce's_ElasticNet2Regressor_an_enhanced_elastic_net_regression_model_with_dual_regularization_paths_and_additional_nonlinear_features).

# CustomBackPropRegressor 

```python
import nnetsauce as ns 
import numpy as np 
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import Ridge
from time import time 

load_datasets = [load_diabetes(), fetch_california_housing()]

datasets_names = ["diabetes", "housing"]

for i, data in enumerate(load_datasets):
    
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    regr = ns.CustomBackPropRegressor(base_model=Ridge(), 
                                      type_grad="finitediff")

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")

    regr = ns.CustomBackPropRegressor(base_model=Ridge(), 
                                      type_grad="finitediff", 
                                      type_loss="quantile")

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")
```

# ElasticNet2Regressor

```python
import nnetsauce as ns 
import numpy as np 
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing
from time import time 

load_datasets = [load_diabetes(), fetch_california_housing()]

datasets_names = ["diabetes", "housing"]

for i, data in enumerate(load_datasets):
    
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    regr = ns.ElasticNet2Regressor(solver="lbfgs")

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")

    regr = ns.ElasticNet2Regressor(solver="cd")

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")
```

![image-title-here]({{base}}/images/2025-07-15/2025-07-15-image1.png){:class="img-responsive"}