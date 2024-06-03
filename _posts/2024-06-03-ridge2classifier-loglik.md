---
layout: post
title: "Recognizing handwritten digits with Ridge2Classifier"
description: "Recognizing handwritten digits with Ridge2Classifier and log-likelihood loss."
date: 2024-06-03
categories: [Python, QuasiRandomizedNN]
comments: true
---

This post is about `Ridge2Classifier`, a classifier that I presented 5 years ago in [this document](https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks?_sg%5B0%5D=NPKnuErQR9Y_2rArlBo1jVTWO-_Z_cnAoGCBezmSnEOgtzg18-jIPIWqWVZnbSvMQBUwDxG6k_HdssAbGVaOXheKOr4eiAoJCCF05EMT.djK2qEK_BkajhqKGRK2UgnEEQw-674gr8r7T_EDclTn7a3qVWQA9R8_OtxemjKjmOaBPP_FDdO8QV2xglL7CjA&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InByb2ZpbGUiLCJwYWdlIjoicHJvZmlsZSIsInByZXZpb3VzUGFnZSI6InByb2ZpbGUiLCJwb3NpdGlvbiI6InBhZ2VDb250ZW50In19). It's now possible to choose starting values of the (likelihood) optimization algorithm which are solutions from least squares regression. Not always better,
but can be seen as a new hyperparameter. Also, `Ridge2Classifier` used to fail miserably on digits data sets but now, with `nnetsauce`'s maturity, `Ridge2Classifier` is doing much better on this type of data, as demonstrated below.

# 0 - Install and load packages


```python
!pip install nnetsauce
```


```python
!pip install GPopt
```


```python
import GPopt as gp
import nnetsauce as ns
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from time import time
```

# 1 - Cross-validation and hyperparameter tuning


```python
def ridge2_cv(X_train, y_train,
              lambda1 = 0.1,
              lambda2 = 0.1,
              n_hidden_features=5,
              n_clusters=5,
              dropout = 0.8,
              solver="L-BFGS-B"): # 'solver' is the optimization algorithm

  estimator  = ns.Ridge2Classifier(lambda1 = lambda1,
                                   lambda2 = lambda2,
                                   n_hidden_features=n_hidden_features,
                                   n_clusters=n_clusters,
                                   dropout = dropout,
                                   solver=solver)

  return -cross_val_score(estimator, X_train, y_train,
                          scoring='accuracy',
                          cv=5, n_jobs=None,
                          verbose=0).mean()

def optimize_ridge2(X_train, y_train, solver="L-BFGS-B"):
  # objective function for hyperparams tuning
  def crossval_objective(x):
    return ridge2_cv(X_train=X_train,
                  y_train=y_train,
                  lambda1 = 10**x[0],
                  lambda2 = 10**x[1],
                  n_hidden_features=int(x[2]),
                  n_clusters=int(x[3]),
                  dropout = x[4],
                  solver = solver)
  gp_opt = gp.GPOpt(objective_func=crossval_objective,
                    lower_bound = np.array([ -10, -10,   3, 2, 0.6]),
                    upper_bound = np.array([  10,  10, 100, 5,   1]),
                    params_names=["lambda1", "lambda2", "n_hidden_features", "n_clusters", "dropout"],
                    n_init=10, n_iter=90, seed=3137)
  return gp_opt.optimize(verbose=2, abs_tol=1e-3)

```


```python

```


```python
dataset = load_digits()
X = dataset.data
y = dataset.target

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=3137)

# hyperparams tuning
res_opt1 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B")
print(res_opt1)

# hyperparams tuning with different starting values for the optimization algorithm
res_opt2 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B-lstsq")
print(res_opt2)
```


```python
res_opt1.best_params["lambda1"] = 10**(res_opt1.best_params["lambda1"])
res_opt1.best_params["lambda2"] = 10**(res_opt1.best_params["lambda2"])
res_opt1.best_params["n_hidden_features"] = int(res_opt1.best_params["n_hidden_features"])
res_opt1.best_params["n_clusters"] = int(res_opt1.best_params["n_clusters"])
print(res_opt1.best_params)

res_opt2.best_params["lambda1"] = 10**(res_opt2.best_params["lambda1"])
res_opt2.best_params["lambda2"] = 10**(res_opt2.best_params["lambda2"])
res_opt2.best_params["n_hidden_features"] = int(res_opt2.best_params["n_hidden_features"])
res_opt2.best_params["n_clusters"] = int(res_opt2.best_params["n_clusters"])
print(res_opt2.best_params)
```

    {'lambda1': 5.243297406977503e-10, 'lambda2': 1.2433817601870388e-05, 'n_hidden_features': 14, 'n_clusters': 2, 'dropout': 0.94100341796875}
    {'lambda1': 1.747558169384434e-08, 'lambda2': 1360.0188315151736, 'n_hidden_features': 14, 'n_clusters': 2, 'dropout': 0.7794189453125}


# 2 - Out-of-sample scores


```python
from time import time


clf1 = ns.Ridge2Classifier(**res_opt1.best_params,
                          solver="L-BFGS-B")
start = time()
clf1.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(clf1.score(X_test, y_test))


clf2 = ns.Ridge2Classifier(**res_opt2.best_params,
                          solver="L-BFGS-B-lstsq")
start = time()
clf2.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(clf2.score(X_test, y_test))
```

    Elapsed: 2.6086528301239014
    0.9138888888888889
    Elapsed: 1.2307183742523193
    0.9416666666666667



```python
# confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = clf2.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.arange(0, 10), yticklabels=np.arange(0, 10))
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()
```


![xxx]({{base}}/images/2024-06-03/2024-06-03-image1.png){:class="img-responsive"}      

