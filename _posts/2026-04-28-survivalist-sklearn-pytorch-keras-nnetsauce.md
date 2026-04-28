---
layout: post
title: "Survival analysis with sklearn, glmnet, keras, pytorch, lightgbm, xgboost, nnetsauce, mlsauce Part 2"
date: 2026-04-28
categories: Python
comments: true
---

This is Part 2 of the 2024-12-15 post, with up-to-date packages.

# Survival analysis with sklearn, glmnet, keras, pytorch, lightgbm, xgboost, nnetsauce, mlsauce

<a target="_blank" href="https://colab.research.google.com/github/thierrymoudiki/2026-04-26-survival_benchmark/blob/main/2026_04_26_survivalist_sklearn_pytorch_keras_nnetsauce.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>


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
!pip install mlsauce --verbose
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

    /usr/local/lib/python3.12/dist-packages/sklearn/utils/deprecation.py:71: FutureWarning: Class PassiveAggressiveRegressor is deprecated; this is deprecated in version 1.8 and will be removed in 1.10. Use `SGDRegressor(loss='epsilon_insensitive', penalty=None, learning_rate='pa1', eta0 = 1.0)` instead.
      warnings.warn(msg, category=FutureWarning)


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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:11<00:00,  8.90it/s]
    100%|██████████| 100/100 [00:04<00:00, 20.60it/s]


    Time to fit PIRandomForestRegressor:  16.17304039001465



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_13_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_13_3.png){:class="img-responsive"}
    






```python
X, y = load_gbsg2()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])


for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:02<00:00, 34.45it/s]
    100%|██████████| 100/100 [00:02<00:00, 46.93it/s]


    Time to fit PIRandomForestRegressor:  5.094623327255249



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_15_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_15_3.png){:class="img-responsive"}
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:01<00:00, 59.56it/s]
    100%|██████████| 100/100 [00:01<00:00, 58.55it/s]


    Time to fit PIRandomForestRegressor:  3.4469754695892334



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_16_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_16_3.png){:class="img-responsive"}
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="ecdf")

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:01<00:00, 59.25it/s]
    100%|██████████| 100/100 [00:02<00:00, 49.04it/s]


    Time to fit PIRandomForestRegressor:  3.786191940307617



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_17_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_17_3.png){:class="img-responsive"}
    



```python
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=RidgeCV(), type_pi="kde")

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:02<00:00, 46.29it/s]
    100%|██████████| 100/100 [00:01<00:00, 58.42it/s]


    Time to fit PIRandomForestRegressor:  3.9259252548217773



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_18_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_18_3.png){:class="img-responsive"}
    




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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:36<00:00,  2.75it/s]
    100%|██████████| 100/100 [00:51<00:00,  1.96it/s]


    Time to fit PIRandomForestRegressor:  87.83672404289246



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_22_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_22_3.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    Time to fit PIRandomForestRegressor:  0.6711692810058594



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_23_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_23_2.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])


for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:21<00:00,  4.68it/s]
    100%|██████████| 100/100 [00:19<00:00,  5.17it/s]


    Time to fit PIRandomForestRegressor:  41.195496559143066



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_24_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_24_3.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:15<00:00,  6.58it/s]
    100%|██████████| 100/100 [00:16<00:00,  6.13it/s]


    Time to fit PIRandomForestRegressor:  31.901167631149292



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_25_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_25_3.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    Time to fit PIRandomForestRegressor:  3.839787006378174



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_26_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_26_2.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:16<00:00,  5.93it/s]
    100%|██████████| 100/100 [00:15<00:00,  6.35it/s]


    Time to fit PIRandomForestRegressor:  33.034319162368774



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_27_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_27_3.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    Time to fit PIRandomForestRegressor:  4.180232763290405



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_28_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_28_2.png){:class="img-responsive"}
    



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

start = time()
estimator.fit(X_train, y_train)
print("Time to fit PIRandomForestRegressor: ", time() - start)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:16<00:00,  6.19it/s]
    100%|██████████| 100/100 [00:15<00:00,  6.26it/s]


    Time to fit PIRandomForestRegressor:  32.65510678291321



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_29_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_29_3.png){:class="img-responsive"}
    


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


surv_funcs = estimator.predict_survival_function(X_test.iloc[0:2,:], return_std=True)
surv_funcs2 = estimator2.predict_survival_function(X_test.iloc[0:2,:], return_std=True)
surv_funcs3 = estimator3.predict_survival_function(X_test.iloc[0:2,:], return_std=True)
```

    
    
     BayesianRidge ------------------
    Time to fit BayesianRidge:  0.34748315811157227
    Time to fit GaussianProcessRegressor:  0.5891196727752686
    Time to fit ARDRegression:  0.6887409687042236



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


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_33_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_33_1.png){:class="img-responsive"}
    



```python
for fn in surv_funcs2.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs2.lower[0].y, surv_funcs2.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_34_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_34_1.png){:class="img-responsive"}
    



```python
for fn in surv_funcs3.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs3.lower[0].y, surv_funcs3.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_35_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_35_1.png){:class="img-responsive"}
    


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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_37_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_37_1.png){:class="img-responsive"}
    



```python
estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr=ns.CustomRegressor(glmnet.GLMNet(lambdau=1000)),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    100%|██████████| 100/100 [00:36<00:00,  2.75it/s]
    100%|██████████| 100/100 [00:39<00:00,  2.56it/s]



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_38_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_38_2.png){:class="img-responsive"}
    





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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_42_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_42_1.png){:class="img-responsive"}
    



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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_43_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_43_1.png){:class="img-responsive"}
    


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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```

    WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7a2a4afe5760> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7a2a4afe5760> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:5 out of the last 9 calls to <function TensorFlowTrainer._make_function.<locals>.multi_step_on_iterator at 0x7a2a4ab86840> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:6 out of the last 11 calls to <function TensorFlowTrainer._make_function.<locals>.multi_step_on_iterator at 0x7a2a4a9f6b60> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_46_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_46_2.png){:class="img-responsive"}
    



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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_47_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_47_1.png){:class="img-responsive"}
    


# 6 - using `xgboost`





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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_50_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_50_1.png){:class="img-responsive"}
    



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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_51_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_51_1.png){:class="img-responsive"}
    


# 7 - using `lightgbm`





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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_54_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_54_1.png){:class="img-responsive"}
    



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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_55_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_55_1.png){:class="img-responsive"}
    


# 8 - using Generic Boosting (`mlsauce`)


```python
import mlsauce as ms
```


```python
regr_ridge = ms.GenericBoostingRegressor(ms.RidgeRegressor(reg_lambda=1e3),
                                         verbose=0)

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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_60_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_60_1.png){:class="img-responsive"}
    



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

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()
```


    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_61_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-04-28/2026-04-28-survivalist-sklearn-pytorch-keras-nnetsauce_61_1.png){:class="img-responsive"}
    

