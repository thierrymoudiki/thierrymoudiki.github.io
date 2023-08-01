---
layout: post
title: "Hyperparameters tuning with GPopt"
description: "LSBoostClassifier's hyperparameters tuning (with GPopt)"
date: 2021-06-11
categories: [Python, Misc]
---

Update: 2023-08-01

Statistical/Machine learning models can have multiple _hyperparameters_ 
that control their performance (out-of-sample accuracy, area under the curve, Root Mean Squared Error, etc.). In this post, in order to determine these hyperparameters for mlsauce's `LSBoostClassifier` (on the [wine dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)), cross-validation is used along with a [Bayesian optimizer, GPopt]({% post_url 2021-04-16-gpopt %}). The _best_ set of hyperparameters is the one that __maximizes 5-fold cross-validation accuracy__. 

__Installing packages__

```bash
!pip install mlsauce GPopt numpy sklearn 
```

__Import packages__

```python
import GPopt as gp 
import mlsauce as ms
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from time import time
```

__Define the objective function to be minimized__

```python
def lsboost_cv(X_train, y_train, learning_rate=0.1, 
               n_hidden_features=5, reg_lambda=0.1, 
               dropout=0, tolerance=1e-4,                 
               seed=123):

  estimator = ms.LSBoostClassifier(n_estimators=100, 
                                   learning_rate=learning_rate,
                                   n_hidden_features=np.int(n_hidden_features), 
                                   reg_lambda=reg_lambda,
                                   dropout=dropout,
                                   tolerance=tolerance,
                                   seed=seed, verbose=0)

  return -cross_val_score(estimator, X_train, y_train,
                          scoring='accuracy', cv=5, n_jobs=-1).mean()

```

__Define the optimizer (based on GPopt)__

```python
def optimize_lsboost(X_train, y_train):

  def crossval_objective(x):

    return lsboost_cv(            
      X_train=X_train, 
      y_train=y_train,
      learning_rate=x[0],
      n_hidden_features=np.int(x[1]), 
      reg_lambda=x[2], 
      dropout=x[3],        
      tolerance=x[4])

  gp_opt = gp.GPOpt(objective_func=crossval_objective, 
                      lower_bound = np.array([0.001, 5, 1e-2, 0, 0]), 
                      upper_bound = np.array([0.4, 250, 1e4, 0.7, 1e-1]),
                      n_init=10, n_iter=190, seed=123)    
  return {'parameters': gp_opt.optimize(verbose=2, abs_tol=1e-2), 'opt_object':  gp_opt}
```

__Using the wine dataset__

```python
wine = load_wine()
X = wine.data
y = wine.target
# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=15029)
```

__Hyperparameters optimization__

```python
res = optimize_lsboost(X_train, y_train)
print(res)
```

```bash
{'parameters': (array([2.36980835e-01, 6.82434082e+00, 3.72193011e+03, 4.01013184e-01,
       3.09204102e-02]), -0.9652709359605911), 'opt_object': <GPopt.GPOpt.GPOpt.GPOpt object at 0x7ab7fc3b5ab0>}
```

Cross-validation average accuracy is equal to 96.53%. 

__Test set accuracy__

```python 
parameters = res["parameters"]

start = time()
estimator_wine = ms.LSBoostClassifier(n_estimators=100,
                                   learning_rate=parameters[0][0],
                                   n_hidden_features=np.int(parameters[0][1]), 
                                   reg_lambda=parameters[0][2], 
                                   dropout=parameters[0][3],
                                   tolerance=parameters[0][4],
                                   seed=123, verbose=1).fit(X_train, y_train)

print(f"\n\n Test set accuracy: {estimator_wine.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
```

```bash
 45%|████▌     | 45/100 [00:00<00:00, 668.38it/s]



 Test set accuracy: 0.9722222222222222

 Elapsed: 0.08791041374206543
```

Due to `LSBoostClassifier`'s `tolerance` hyperparameter (equal to 0.05940552 here), the learning 
procedure is stopped early, and only 8 iterations of the classifier are necessary 
to obtain a high accuracy. 
