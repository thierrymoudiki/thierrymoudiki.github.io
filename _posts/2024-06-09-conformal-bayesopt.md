---
layout: post
title: "Automated hyperparameter tuning using any conformalized surrogate"
description: "Automated hyperparameter tuning for Machine Learning using any conformalized surrogate."
date: 2024-06-09
categories: [Python, QuasiRandomizedNN]
comments: true
---

[Bayesian optimization](https://thierrymoudiki.github.io/blog/2024/02/05/python/gpopt-new2)(BO) is widely used for Machine Learning hyperparameter tuning. BO relies mainly on a probabilistic model of the objective function (generally a Gaussian process model that approximates the objective function) called the **surrogate** and improved sequentially, and an **acquisition function** that allows to select the next point to evaluate in the sequential optimization procedure. 

In this post, I will show how to use conformalized surrogates to tune hyperparameters of machine learning models. In this context, instead of using the posterior closed-form distribution of a Gaussian Process, any conformalized surrogate can be used for a probabilistic approximation of the objective function. And since there's no closed-form expression of the acquisition function (here the **Expected Improvement** over the current optimum), **a monte-carlo approximation of the (expectation) acquisition function is used, based on simulations of the conformalized surrogate**. The simulation approach is similar to the one used in [this post](https://thierrymoudiki.github.io/blog/2024/04/07/r/conformal-time-series), except, the sequential ordering doesn't matter here.

# 0 - Install and load packages


```python
!pip install nnetsauce
```


```python
!pip install git+https://github.com/Techtonique/GPopt.git --upgrade --no-cache-dir
```


```python
import GPopt as gp
import nnetsauce as ns
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
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
              solver="L-BFGS-B"):

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

def optimize_ridge2(X_train, y_train, solver="L-BFGS-B",
                    surrogate="rf"):
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
  if surrogate == "rf":
    gp_opt = gp.GPOpt(objective_func=crossval_objective,
                      lower_bound = np.array([ -10, -10,   3, 2, 0.6]),
                      upper_bound = np.array([  10,  10, 100, 5,   1]),
                      surrogate_obj = ns.CustomRegressor(obj=RandomForestRegressor(), # it's a conformalized quasi-randomized network
                                                        replications=250, # number of simulations for evaluating the expected improvement
                                                        type_pi="kde"), # Kernel Density Estimation is used for simulation
                      acquisition="ei", # expected improvement by simulation
                      params_names=["lambda1", "lambda2", "n_hidden_features", "n_clusters", "dropout"],
                      n_init=10, n_iter=90, seed=3137)
  elif surrogate == "enet":
    gp_opt = gp.GPOpt(objective_func=crossval_objective,
                      lower_bound = np.array([ -10, -10,   3, 2, 0.6]),
                      upper_bound = np.array([  10,  10, 100, 5,   1]),
                      surrogate_obj = ns.CustomRegressor(obj=ElasticNetCV(), # the model is nonlinear, it's a conformalized quasi-randomized network
                                                        replications=250, # number of simulations for evaluating the expected improvement
                                                        type_pi="kde"), # Kernel Density Estimation is used for simulation
                      acquisition="ei", # expected improvement by simulation
                      params_names=["lambda1", "lambda2", "n_hidden_features", "n_clusters", "dropout"],
                      n_init=10, n_iter=90, seed=3137)

  return gp_opt.optimize(method = "mc", verbose=2, abs_tol=1e-3) # monte carlo computation of expected improvement

```


```python

```


```python
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=3137)

# hyperparams tuning, surrogate = conformalized Random Forest
res_opt1 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B", surrogate = "rf")
print(res_opt1)

# hyperparams tuning with different starting values for the optimization algorithm, surrogate = conformalized Random Forest
res_opt2 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B-lstsq", surrogate = "rf")
print(res_opt2)

# hyperparams tuning, surrogate = conformalized ElasticNet
res_opt3 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B", surrogate = "enet")
print(res_opt3)

# hyperparams tuning with different starting values for the optimization algorithm, surrogate = conformalized ElasticNet
res_opt4 = optimize_ridge2(X_train, y_train, solver="L-BFGS-B-lstsq", surrogate = "enet")
print(res_opt4)
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

res_opt3.best_params["lambda1"] = 10**(res_opt3.best_params["lambda1"])
res_opt3.best_params["lambda2"] = 10**(res_opt3.best_params["lambda2"])
res_opt3.best_params["n_hidden_features"] = int(res_opt3.best_params["n_hidden_features"])
res_opt3.best_params["n_clusters"] = int(res_opt3.best_params["n_clusters"])
print(res_opt3.best_params)

res_opt4.best_params["lambda1"] = 10**(res_opt4.best_params["lambda1"])
res_opt4.best_params["lambda2"] = 10**(res_opt4.best_params["lambda2"])
res_opt4.best_params["n_hidden_features"] = int(res_opt4.best_params["n_hidden_features"])
res_opt4.best_params["n_clusters"] = int(res_opt4.best_params["n_clusters"])
print(res_opt4.best_params)
```

    {'lambda1': 0.2143160456513889, 'lambda2': 99.32768474363539, 'n_hidden_features': 3, 'n_clusters': 4, 'dropout': 0.80830078125}
    {'lambda1': 1.19372502075462e-10, 'lambda2': 0.0003873778332245682, 'n_hidden_features': 5, 'n_clusters': 3, 'dropout': 0.8306396484375}
    {'lambda1': 0.03853145684685379, 'lambda2': 0.0020254361391973223, 'n_hidden_features': 91, 'n_clusters': 4, 'dropout': 0.75242919921875}
    {'lambda1': 1.19372502075462e-10, 'lambda2': 0.0003873778332245682, 'n_hidden_features': 5, 'n_clusters': 3, 'dropout': 0.8306396484375}


# 2 - Out-of-sample scores


```python
from time import time


clf1 = ns.Ridge2Classifier(**res_opt1.best_params,
                          solver="L-BFGS-B")
start = time()
clf1.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(f"Test set accuracy: {clf1.score(X_test, y_test)}")


clf2 = ns.Ridge2Classifier(**res_opt2.best_params,
                          solver="L-BFGS-B-lstsq")
start = time()
clf2.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(f"Test set accuracy: {clf2.score(X_test, y_test)}")

clf3 = ns.Ridge2Classifier(**res_opt3.best_params,
                          solver="L-BFGS-B")
start = time()
clf3.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(f"Test set accuracy: {clf3.score(X_test, y_test)}")


clf4 = ns.Ridge2Classifier(**res_opt4.best_params,
                          solver="L-BFGS-B-lstsq")
start = time()
clf4.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
print(f"Test set accuracy: {clf4.score(X_test, y_test)}")
```

    Elapsed: 1.5195319652557373
    Test set accuracy: 0.9736842105263158
    Elapsed: 1.8859667778015137
    Test set accuracy: 0.9736842105263158
    Elapsed: 0.5796549320220947
    Test set accuracy: 0.9736842105263158
    Elapsed: 0.6930491924285889
    Test set accuracy: 0.9736842105263158



```python
# confusion matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = clf2.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.arange(0, 2), yticklabels=np.arange(0, 2))
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()
```

![xxx]({{base}}/images/2024-06-09/2024-06-09-image1.png){:class="img-responsive"}      
