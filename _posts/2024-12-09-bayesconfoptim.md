---
layout: post
title: "Model-agnostic 'Bayesian' optimization (for hyperparameter tuning) using conformalized surrogates in GPopt"
description: "'Bayesian' optimization is used for hyperparameter tuning. In this post, I show how any surrogate can be used 
for this purpose, thanks to Conformal Prediction, GPopt and nnetsauce"
date: 2024-12-09
categories: Python
comments: true
---

**Bayesian optimization (BO)** is a popular (and clever, and elegant, and beautiful, and efficient) optimization method for hyperparameter tuning in Machine Learning and Deep Learning. BO is based on the use of a surrogate model that approximates the objective function (the function to be minimized) in a probabilistic way. It optimizes a cheaper [_acquisition function_](https://www.researchgate.net/publication/332292006_Online_Bayesian_Quasi-Random_functional_link_networks_application_to_the_optimization_of_black_box_functions) that  allows to select the next point to evaluate.

The most common surrogate model in BO is the [Gaussian process](https://gaussianprocess.org/gpml/) regressor, a Bayesian  model with a Gaussian prior, and the most common acquisition function is the Expected Improvement (EI). The idea of EI is to select the next point to evaluate based on the expected improvement relative to the current best point.

**Conformal Prediction** is a framework allowing, among other things, to make supervised learning predictions with prediction intervals. For more details on Bayesian optimization and Conformal Prediction, see the following references:
- [Bayesian optimization book](https://bayesoptbook.com/)
- [Conformal Prediction](https://arxiv.org/pdf/2107.07511)

In this post, I'll show how to use conformalized surrogates for optimization, thanks to [GPopt](https://github.com/Techtonique/GPopt) and [nnetsauce](https://github.com/Techtonique/nnetsauce). With this approach, any surrogate model can be used for optimization, and there's no more constraint on the choice of a prior (Gaussian, Laplace, etc.). The acquisition function is the lower confidence bound (LCB) of the conformalized surrogate model.

A future post will show how to use conformalized surrogates for Machine Learning and Deep Learning hyperparameter tuning.

```bash
pip install GPopt nnetsauce
```

```python
import GPopt as gp
import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess


# Six-Hump Camel Function (Objective function, to be minimized)
def six_hump_camel(x):
    """
    Six-Hump Camel Function:
    - Global minima located at:
      (0.0898, -0.7126),
      (-0.0898, 0.7126)
    - Function value at the minima: f(x) = -1.0316
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3


# Minimize the objective function using scipy.optimize.minimize
def minimize_function(func, bounds, x0=None):
    if x0 is None:
        x0 = np.mean(bounds, axis=1)  # Start at the center of the bounds
    result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
    return result

# GPopt for Bayesian optimization
gp_opt = gp.GPOpt(objective_func=six_hump_camel,
                   lower_bound = np.array([-3, -2]),
                   upper_bound = np.array([3, 2]),
                   acquisition="ucb",
                   method="splitconformal",
                   surrogate_obj=ns.CustomRegressor(RidgeCV()), # Any surrogate model can be used, thanks to nnetsauce
                   n_init=10,
                   n_iter=190,
                   seed=432)

print(f"gp_opt.method: {gp_opt.method}")

res = gp_opt.optimize(verbose=2, ucb_tol=1e-6)

print(f"\n\n result: {res}")

display(res.best_params)
display(res.best_score)

result = minimize_function(func, bounds_tuples)
display(results[objective].x)
display(results[objective].fun)

# Concatenate the parameters and scores in a DataFrame
combined_array = np.concatenate((np.asarray(gp_opt.parameters),
                                 np.asarray(gp_opt.scores)[:, np.newaxis]),
                                 axis=1)
df = pd.DataFrame(combined_array, columns=["x", "y", "z"])


# Extract x, y, and z values from the DataFrame
x = df["x"].values
y = df["y"].values
z = df["z"].values

# Range of x and y values
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Data for the line plot
iterations = np.arange(len(gp_opt.max_acq))  # Generate x-axis values
acq_values = gp_opt.max_acq

# Apply LOESS smoothing
smoothed_values = lowess(acq_values, iterations, frac=0.2)  # Adjust `frac` for smoothness

# Create a figure and specify the number of subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Adjust figsize for layout

# Plot the first graph on the first subplot
axes[0].plot(gp_opt.max_acq)
axes[0].plot(smoothed_values[:, 0], smoothed_values[:, 1],
             color='red', label='LOESS Smooth',
             linewidth=2)
axes[0].set_title('Optimization Progress')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Acquisition Value')

contour = axes[1].tricontourf(x, y, z, levels=100, cmap='terrain')
fig.colorbar(contour, ax=axes[1], label='z')  # Add colorbar to the second subplot
axes[1].scatter(res.best_params[0], res.best_params[1],
                color='green', label='Minimum Found (Conformal Optimization)')
axes[1].scatter(result.x[0], result.x[1],
                color='red', marker='x', label='L-BFGS-B Minimum')
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].set_title('Contour Plot of df')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend()
plt.tight_layout()
plt.show()
```


![xxx]({{base}}/images/2024-12-09/2024-12-09-image1.gif){:class="img-responsive"}          