---
layout: post
title: "Model-agnostic 'Bayesian' optimization (for hyperparameter tuning) using conformalized surrogates in GPopt"
description: "'Bayesian' optimization is used for hyperparameter tuning. In this post, I show how any surrogate can be used 
for this purpose, thanks to Conformal Prediction, GPopt and nnetsauce"
date: 2024-12-09
categories: Python
comments: true
---

**Disclaimer:** Updated on 2025-06-28

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
```


```python
import matplotlib.pyplot as plt
import numpy as np
# Generate a grid of points in the input space
x = np.linspace(-3, 3, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the objective function at each point in the grid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = six_hump_camel([X[i, j], Y[i, j]])

# Plot the contour map
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Objective function value')
plt.title('Contour plot of the Six-Hump Camel function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

![xxx]({{base}}/images/2024-12-09/2024-12-09-image1.png){:class="img-responsive"}              



```python
from sklearn.utils import all_estimators
from tqdm import tqdm

# Get all available scikit-learn estimators
estimators = all_estimators(type_filter='regressor')

results = []

# Loop through all regressors
for name, RegressorClass in tqdm(estimators):
    try:
        # Instantiate the regressor (you might need to handle potential exceptions or required parameters)
        regressor = RegressorClass()
        print(f"\n Successfully instantiated regressor: {name} ----------")
        # GPopt for Bayesian optimization
        gp_opt = gp.GPOpt(objective_func=six_hump_camel,
                          lower_bound = np.array([-3, -2]),
                          upper_bound = np.array([3, 2]),
                          acquisition="ucb",
                          method="splitconformal",
                          surrogate_obj=ns.PredictionInterval(regressor), # Any surrogate model can be used, thanks to nnetsauce
                          n_init=10,
                          n_iter=190,
                          seed=432)
        print(f"gp_opt.method: {gp_opt.method}")
        res = gp_opt.optimize(verbose=1, ucb_tol=1e-6)
        print(f"\n\n result: {res}")
        display(res.best_params)
        display(res.best_score)
        results.append((name, res))

    except Exception as e:
        print(f"Could not instantiate regressor {name}: {e}")

```

```python
import pandas as pd

results_df = pd.DataFrame(columns=['Regressor', 'Best Params', 'Best Score'])

for name, res in results:
    best_params = res.best_params
    best_score = res.best_score
    results_df = pd.concat([results_df, pd.DataFrame({'Regressor': [name], 'Best Params': [best_params], 'Best Score': [best_score]})], ignore_index=True)

results_df.sort_values(by='Best Score', ascending=True, inplace=True)
results_df.reset_index(drop=True, inplace=True)

results_df.style.format({'Best Score': "{:.5f}"})
```




<style type="text/css">
</style>
<table id="T_a09a0" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a09a0_level0_col0" class="col_heading level0 col0" >Regressor</th>
      <th id="T_a09a0_level0_col1" class="col_heading level0 col1" >Best Params</th>
      <th id="T_a09a0_level0_col2" class="col_heading level0 col2" >Best Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a09a0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a09a0_row0_col0" class="data row0 col0" >BaggingRegressor</td>
      <td id="T_a09a0_row0_col1" class="data row0 col1" >[ 0.09649658 -0.71691895]</td>
      <td id="T_a09a0_row0_col2" class="data row0 col2" >-1.03133</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a09a0_row1_col0" class="data row1 col0" >GaussianProcessRegressor</td>
      <td id="T_a09a0_row1_col1" class="data row1 col1" >[ 0.09649658 -0.71691895]</td>
      <td id="T_a09a0_row1_col2" class="data row1 col2" >-1.03133</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a09a0_row2_col0" class="data row2 col0" >NuSVR</td>
      <td id="T_a09a0_row2_col1" class="data row2 col1" >[ 0.09649658 -0.71691895]</td>
      <td id="T_a09a0_row2_col2" class="data row2 col2" >-1.03133</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a09a0_row3_col0" class="data row3 col0" >SVR</td>
      <td id="T_a09a0_row3_col1" class="data row3 col1" >[ 0.09649658 -0.71691895]</td>
      <td id="T_a09a0_row3_col2" class="data row3 col2" >-1.03133</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a09a0_row4_col0" class="data row4 col0" >MLPRegressor</td>
      <td id="T_a09a0_row4_col1" class="data row4 col1" >[-0.09155273  0.69482422]</td>
      <td id="T_a09a0_row4_col2" class="data row4 col2" >-1.02905</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_a09a0_row5_col0" class="data row5 col0" >GradientBoostingRegressor</td>
      <td id="T_a09a0_row5_col1" class="data row5 col1" >[ 0.04907227 -0.71142578]</td>
      <td id="T_a09a0_row5_col2" class="data row5 col2" >-1.02514</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_a09a0_row6_col0" class="data row6 col0" >KNeighborsRegressor</td>
      <td id="T_a09a0_row6_col1" class="data row6 col1" >[ 0.08203125 -0.6640625 ]</td>
      <td id="T_a09a0_row6_col2" class="data row6 col2" >-1.01372</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_a09a0_row7_col0" class="data row7 col0" >ExtraTreeRegressor</td>
      <td id="T_a09a0_row7_col1" class="data row7 col1" >[ 0.08203125 -0.6640625 ]</td>
      <td id="T_a09a0_row7_col2" class="data row7 col2" >-1.01372</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_a09a0_row8_col0" class="data row8 col0" >RandomForestRegressor</td>
      <td id="T_a09a0_row8_col1" class="data row8 col1" >[ 0.08203125 -0.6640625 ]</td>
      <td id="T_a09a0_row8_col2" class="data row8 col2" >-1.01372</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_a09a0_row9_col0" class="data row9 col0" >DecisionTreeRegressor</td>
      <td id="T_a09a0_row9_col1" class="data row9 col1" >[ 0.08203125 -0.6640625 ]</td>
      <td id="T_a09a0_row9_col2" class="data row9 col2" >-1.01372</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_a09a0_row10_col0" class="data row10 col0" >HistGradientBoostingRegressor</td>
      <td id="T_a09a0_row10_col1" class="data row10 col1" >[-0.00732422 -0.72167969]</td>
      <td id="T_a09a0_row10_col2" class="data row10 col2" >-0.99277</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_a09a0_row11_col0" class="data row11 col0" >AdaBoostRegressor</td>
      <td id="T_a09a0_row11_col1" class="data row11 col1" >[ 0.09375 -0.8125 ]</td>
      <td id="T_a09a0_row11_col2" class="data row11 col2" >-0.93858</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_a09a0_row12_col0" class="data row12 col0" >ExtraTreesRegressor</td>
      <td id="T_a09a0_row12_col1" class="data row12 col1" >[-0.05877686 -0.66418457]</td>
      <td id="T_a09a0_row12_col2" class="data row12 col2" >-0.93331</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_a09a0_row13_col0" class="data row13 col0" >ElasticNet</td>
      <td id="T_a09a0_row13_col1" class="data row13 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row13_col2" class="data row13 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_a09a0_row14_col0" class="data row14 col0" >ARDRegression</td>
      <td id="T_a09a0_row14_col1" class="data row14 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row14_col2" class="data row14 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_a09a0_row15_col0" class="data row15 col0" >ElasticNetCV</td>
      <td id="T_a09a0_row15_col1" class="data row15 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row15_col2" class="data row15 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_a09a0_row16_col0" class="data row16 col0" >KernelRidge</td>
      <td id="T_a09a0_row16_col1" class="data row16 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row16_col2" class="data row16 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_a09a0_row17_col0" class="data row17 col0" >HuberRegressor</td>
      <td id="T_a09a0_row17_col1" class="data row17 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row17_col2" class="data row17 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_a09a0_row18_col0" class="data row18 col0" >Lars</td>
      <td id="T_a09a0_row18_col1" class="data row18 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row18_col2" class="data row18 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_a09a0_row19_col0" class="data row19 col0" >LarsCV</td>
      <td id="T_a09a0_row19_col1" class="data row19 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row19_col2" class="data row19 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_a09a0_row20_col0" class="data row20 col0" >LassoLars</td>
      <td id="T_a09a0_row20_col1" class="data row20 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row20_col2" class="data row20 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_a09a0_row21_col0" class="data row21 col0" >LassoLarsCV</td>
      <td id="T_a09a0_row21_col1" class="data row21 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row21_col2" class="data row21 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_a09a0_row22_col0" class="data row22 col0" >Lasso</td>
      <td id="T_a09a0_row22_col1" class="data row22 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row22_col2" class="data row22 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_a09a0_row23_col0" class="data row23 col0" >LassoCV</td>
      <td id="T_a09a0_row23_col1" class="data row23 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row23_col2" class="data row23 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_a09a0_row24_col0" class="data row24 col0" >LinearRegression</td>
      <td id="T_a09a0_row24_col1" class="data row24 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row24_col2" class="data row24 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_a09a0_row25_col0" class="data row25 col0" >LassoLarsIC</td>
      <td id="T_a09a0_row25_col1" class="data row25 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row25_col2" class="data row25 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_a09a0_row26_col0" class="data row26 col0" >LinearSVR</td>
      <td id="T_a09a0_row26_col1" class="data row26 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row26_col2" class="data row26 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_a09a0_row27_col0" class="data row27 col0" >OrthogonalMatchingPursuit</td>
      <td id="T_a09a0_row27_col1" class="data row27 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row27_col2" class="data row27 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_a09a0_row28_col0" class="data row28 col0" >OrthogonalMatchingPursuitCV</td>
      <td id="T_a09a0_row28_col1" class="data row28 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row28_col2" class="data row28 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_a09a0_row29_col0" class="data row29 col0" >PLSRegression</td>
      <td id="T_a09a0_row29_col1" class="data row29 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row29_col2" class="data row29 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_a09a0_row30_col0" class="data row30 col0" >DummyRegressor</td>
      <td id="T_a09a0_row30_col1" class="data row30 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row30_col2" class="data row30 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_a09a0_row31_col0" class="data row31 col0" >BayesianRidge</td>
      <td id="T_a09a0_row31_col1" class="data row31 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row31_col2" class="data row31 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_a09a0_row32_col0" class="data row32 col0" >QuantileRegressor</td>
      <td id="T_a09a0_row32_col1" class="data row32 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row32_col2" class="data row32 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_a09a0_row33_col0" class="data row33 col0" >PassiveAggressiveRegressor</td>
      <td id="T_a09a0_row33_col1" class="data row33 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row33_col2" class="data row33 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_a09a0_row34_col0" class="data row34 col0" >RadiusNeighborsRegressor</td>
      <td id="T_a09a0_row34_col1" class="data row34 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row34_col2" class="data row34 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_a09a0_row35_col0" class="data row35 col0" >RANSACRegressor</td>
      <td id="T_a09a0_row35_col1" class="data row35 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row35_col2" class="data row35 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_a09a0_row36_col0" class="data row36 col0" >Ridge</td>
      <td id="T_a09a0_row36_col1" class="data row36 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row36_col2" class="data row36 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_a09a0_row37_col0" class="data row37 col0" >RidgeCV</td>
      <td id="T_a09a0_row37_col1" class="data row37 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row37_col2" class="data row37 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_a09a0_row38_col0" class="data row38 col0" >SGDRegressor</td>
      <td id="T_a09a0_row38_col1" class="data row38 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row38_col2" class="data row38 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_a09a0_row39_col0" class="data row39 col0" >TheilSenRegressor</td>
      <td id="T_a09a0_row39_col1" class="data row39 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row39_col2" class="data row39 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_a09a0_row40_col0" class="data row40 col0" >TransformedTargetRegressor</td>
      <td id="T_a09a0_row40_col1" class="data row40 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row40_col2" class="data row40 col2" >-0.92451</td>
    </tr>
    <tr>
      <th id="T_a09a0_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_a09a0_row41_col0" class="data row41 col0" >TweedieRegressor</td>
      <td id="T_a09a0_row41_col1" class="data row41 col1" >[-0.06650758 -0.66453519]</td>
      <td id="T_a09a0_row41_col2" class="data row41 col2" >-0.92451</td>
    </tr>
  </tbody>
</table>





```python
# Michalewicz Function
def michalewicz(x, m=10):
    """
    Michalewicz Function (for n=2 dimensions):
    """
    return -sum(np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x))


import matplotlib.pyplot as plt
import numpy as np
# Generate a grid of points in the input space
x = np.linspace(0, 2, 100)
y = np.linspace(np.pi, 2, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the objective function at each point in the grid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = michalewicz([X[i, j], Y[i, j]])

# Plot the contour map
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Objective function value')
plt.title('Contour plot of the Six-Hump Camel function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

```

![xxx]({{base}}/images/2024-12-09/2024-12-09-image2.png){:class="img-responsive"                  


```python
from sklearn.utils import all_estimators
from tqdm import tqdm

# Get all available scikit-learn estimators
estimators = all_estimators(type_filter='regressor')

results = []

# Loop through all regressors
for name, RegressorClass in tqdm(estimators):
    try:
        # Instantiate the regressor (you might need to handle potential exceptions or required parameters)
        regressor = RegressorClass()
        print(f"\n Successfully instantiated regressor: {name} ----------")
        # GPopt for Bayesian optimization
        gp_opt = gp.GPOpt(objective_func=michalewicz,
                          lower_bound = np.array([0, np.pi]),
                          upper_bound = np.array([2, 2]),
                          acquisition="ucb",
                          method="splitconformal",
                          surrogate_obj=ns.PredictionInterval(regressor), # Any surrogate model can be used, thanks to nnetsauce
                          n_init=10,
                          n_iter=190,
                          seed=432)
        print(f"gp_opt.method: {gp_opt.method}")
        res = gp_opt.optimize(verbose=1, ucb_tol=1e-6)
        print(f"\n\n result: {res}")
        display(res.best_params)
        display(res.best_score)
        results.append((name, res))

    except Exception as e:
        print(f"Could not instantiate regressor {name}: {e}")

```


```python
import pandas as pd

results_df = pd.DataFrame(columns=['Regressor', 'Best Params', 'Best Score'])

for name, res in results:
    best_params = res.best_params
    best_score = res.best_score
    results_df = pd.concat([results_df, pd.DataFrame({'Regressor': [name], 'Best Params': [best_params], 'Best Score': [best_score]})], ignore_index=True)

results_df.sort_values(by='Best Score', ascending=True, inplace=True)
results_df.reset_index(drop=True, inplace=True)

results_df.style.format({'Best Score': "{:.5f}"})
```




<style type="text/css">
</style>
<table id="T_b8a9d" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b8a9d_level0_col0" class="col_heading level0 col0" >Regressor</th>
      <th id="T_b8a9d_level0_col1" class="col_heading level0 col1" >Best Params</th>
      <th id="T_b8a9d_level0_col2" class="col_heading level0 col2" >Best Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b8a9d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b8a9d_row0_col0" class="data row0 col0" >BaggingRegressor</td>
      <td id="T_b8a9d_row0_col1" class="data row0 col1" >[1.9989624  2.71631734]</td>
      <td id="T_b8a9d_row0_col2" class="data row0 col2" >-0.77895</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b8a9d_row1_col0" class="data row1 col0" >GradientBoostingRegressor</td>
      <td id="T_b8a9d_row1_col1" class="data row1 col1" >[1.9989624  2.71631734]</td>
      <td id="T_b8a9d_row1_col2" class="data row1 col2" >-0.77895</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b8a9d_row2_col0" class="data row2 col0" >GaussianProcessRegressor</td>
      <td id="T_b8a9d_row2_col1" class="data row2 col1" >[1.9989624  2.71631734]</td>
      <td id="T_b8a9d_row2_col2" class="data row2 col2" >-0.77895</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b8a9d_row3_col0" class="data row3 col0" >AdaBoostRegressor</td>
      <td id="T_b8a9d_row3_col1" class="data row3 col1" >[1.99511719 2.70736381]</td>
      <td id="T_b8a9d_row3_col2" class="data row3 col2" >-0.76882</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b8a9d_row4_col0" class="data row4 col0" >MLPRegressor</td>
      <td id="T_b8a9d_row4_col1" class="data row4 col1" >[1.99978638 2.68494514]</td>
      <td id="T_b8a9d_row4_col2" class="data row4 col2" >-0.74841</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b8a9d_row5_col0" class="data row5 col0" >RandomForestRegressor</td>
      <td id="T_b8a9d_row5_col1" class="data row5 col1" >[1.99978638 2.68494514]</td>
      <td id="T_b8a9d_row5_col2" class="data row5 col2" >-0.74841</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b8a9d_row6_col0" class="data row6 col0" >ExtraTreesRegressor</td>
      <td id="T_b8a9d_row6_col1" class="data row6 col1" >[1.97668457 2.67872644]</td>
      <td id="T_b8a9d_row6_col2" class="data row6 col2" >-0.67143</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b8a9d_row7_col0" class="data row7 col0" >ExtraTreeRegressor</td>
      <td id="T_b8a9d_row7_col1" class="data row7 col1" >[1.9453125  2.68227998]</td>
      <td id="T_b8a9d_row7_col2" class="data row7 col2" >-0.60804</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b8a9d_row8_col0" class="data row8 col0" >HuberRegressor</td>
      <td id="T_b8a9d_row8_col1" class="data row8 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row8_col2" class="data row8 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b8a9d_row9_col0" class="data row9 col0" >KNeighborsRegressor</td>
      <td id="T_b8a9d_row9_col1" class="data row9 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row9_col2" class="data row9 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_b8a9d_row10_col0" class="data row10 col0" >KernelRidge</td>
      <td id="T_b8a9d_row10_col1" class="data row10 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row10_col2" class="data row10 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_b8a9d_row11_col0" class="data row11 col0" >ElasticNetCV</td>
      <td id="T_b8a9d_row11_col1" class="data row11 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row11_col2" class="data row11 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_b8a9d_row12_col0" class="data row12 col0" >LarsCV</td>
      <td id="T_b8a9d_row12_col1" class="data row12 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row12_col2" class="data row12 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_b8a9d_row13_col0" class="data row13 col0" >LassoCV</td>
      <td id="T_b8a9d_row13_col1" class="data row13 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row13_col2" class="data row13 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_b8a9d_row14_col0" class="data row14 col0" >Lars</td>
      <td id="T_b8a9d_row14_col1" class="data row14 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row14_col2" class="data row14 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_b8a9d_row15_col0" class="data row15 col0" >ARDRegression</td>
      <td id="T_b8a9d_row15_col1" class="data row15 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row15_col2" class="data row15 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_b8a9d_row16_col0" class="data row16 col0" >OrthogonalMatchingPursuitCV</td>
      <td id="T_b8a9d_row16_col1" class="data row16 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row16_col2" class="data row16 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_b8a9d_row17_col0" class="data row17 col0" >PLSRegression</td>
      <td id="T_b8a9d_row17_col1" class="data row17 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row17_col2" class="data row17 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_b8a9d_row18_col0" class="data row18 col0" >NuSVR</td>
      <td id="T_b8a9d_row18_col1" class="data row18 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row18_col2" class="data row18 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_b8a9d_row19_col0" class="data row19 col0" >OrthogonalMatchingPursuit</td>
      <td id="T_b8a9d_row19_col1" class="data row19 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row19_col2" class="data row19 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_b8a9d_row20_col0" class="data row20 col0" >LinearRegression</td>
      <td id="T_b8a9d_row20_col1" class="data row20 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row20_col2" class="data row20 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_b8a9d_row21_col0" class="data row21 col0" >LassoLarsIC</td>
      <td id="T_b8a9d_row21_col1" class="data row21 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row21_col2" class="data row21 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_b8a9d_row22_col0" class="data row22 col0" >LinearSVR</td>
      <td id="T_b8a9d_row22_col1" class="data row22 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row22_col2" class="data row22 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_b8a9d_row23_col0" class="data row23 col0" >LassoLarsCV</td>
      <td id="T_b8a9d_row23_col1" class="data row23 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row23_col2" class="data row23 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_b8a9d_row24_col0" class="data row24 col0" >PassiveAggressiveRegressor</td>
      <td id="T_b8a9d_row24_col1" class="data row24 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row24_col2" class="data row24 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_b8a9d_row25_col0" class="data row25 col0" >QuantileRegressor</td>
      <td id="T_b8a9d_row25_col1" class="data row25 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row25_col2" class="data row25 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_b8a9d_row26_col0" class="data row26 col0" >SGDRegressor</td>
      <td id="T_b8a9d_row26_col1" class="data row26 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row26_col2" class="data row26 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_b8a9d_row27_col0" class="data row27 col0" >RidgeCV</td>
      <td id="T_b8a9d_row27_col1" class="data row27 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row27_col2" class="data row27 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_b8a9d_row28_col0" class="data row28 col0" >Ridge</td>
      <td id="T_b8a9d_row28_col1" class="data row28 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row28_col2" class="data row28 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_b8a9d_row29_col0" class="data row29 col0" >RadiusNeighborsRegressor</td>
      <td id="T_b8a9d_row29_col1" class="data row29 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row29_col2" class="data row29 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_b8a9d_row30_col0" class="data row30 col0" >RANSACRegressor</td>
      <td id="T_b8a9d_row30_col1" class="data row30 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row30_col2" class="data row30 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_b8a9d_row31_col0" class="data row31 col0" >BayesianRidge</td>
      <td id="T_b8a9d_row31_col1" class="data row31 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row31_col2" class="data row31 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_b8a9d_row32_col0" class="data row32 col0" >TweedieRegressor</td>
      <td id="T_b8a9d_row32_col1" class="data row32 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row32_col2" class="data row32 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_b8a9d_row33_col0" class="data row33 col0" >TransformedTargetRegressor</td>
      <td id="T_b8a9d_row33_col1" class="data row33 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row33_col2" class="data row33 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_b8a9d_row34_col0" class="data row34 col0" >TheilSenRegressor</td>
      <td id="T_b8a9d_row34_col1" class="data row34 col1" >[1.93724655 2.67858092]</td>
      <td id="T_b8a9d_row34_col2" class="data row34 col2" >-0.58092</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_b8a9d_row35_col0" class="data row35 col0" >DecisionTreeRegressor</td>
      <td id="T_b8a9d_row35_col1" class="data row35 col1" >[1.8515625  2.73579214]</td>
      <td id="T_b8a9d_row35_col2" class="data row35 col2" >-0.47178</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_b8a9d_row36_col0" class="data row36 col0" >SVR</td>
      <td id="T_b8a9d_row36_col1" class="data row36 col1" >[0.76176453 2.71127445]</td>
      <td id="T_b8a9d_row36_col2" class="data row36 col2" >-0.41275</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_b8a9d_row37_col0" class="data row37 col0" >DummyRegressor</td>
      <td id="T_b8a9d_row37_col1" class="data row37 col1" >[0.75       2.71349541]</td>
      <td id="T_b8a9d_row37_col2" class="data row37 col2" >-0.41257</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_b8a9d_row38_col0" class="data row38 col0" >ElasticNet</td>
      <td id="T_b8a9d_row38_col1" class="data row38 col1" >[0.75       2.71349541]</td>
      <td id="T_b8a9d_row38_col2" class="data row38 col2" >-0.41257</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_b8a9d_row39_col0" class="data row39 col0" >HistGradientBoostingRegressor</td>
      <td id="T_b8a9d_row39_col1" class="data row39 col1" >[0.75       2.71349541]</td>
      <td id="T_b8a9d_row39_col2" class="data row39 col2" >-0.41257</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_b8a9d_row40_col0" class="data row40 col0" >Lasso</td>
      <td id="T_b8a9d_row40_col1" class="data row40 col1" >[0.75       2.71349541]</td>
      <td id="T_b8a9d_row40_col2" class="data row40 col2" >-0.41257</td>
    </tr>
    <tr>
      <th id="T_b8a9d_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_b8a9d_row41_col0" class="data row41 col0" >LassoLars</td>
      <td id="T_b8a9d_row41_col1" class="data row41 col1" >[0.75       2.71349541]</td>
      <td id="T_b8a9d_row41_col2" class="data row41 col2" >-0.41257</td>
    </tr>
  </tbody>
</table>


![xxx]({{base}}/images/2024-12-09/2024-12-09-image1.gif){:class="img-responsive"}          