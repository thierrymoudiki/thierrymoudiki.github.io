---
layout: post
title: "Natively Interpretable Boosting"
description: "This notebook works with a booster, cybooster, that is interpretable by construction instead of relying on post hoc methods like SHAP or LIME."
date: 2026-07-12
categories: Python
comments: true
---


# Natively Interpretable Boosting

Gradient boosting machines are accurate but usually opaque: to explain a prediction you bolt on a *post hoc* method (SHAP, LIME, permutation importance) that approximates the model after the fact. This notebook works with a booster (`cybooster`) that is interpretable *by construction* instead.

Each boosting round fits a randomized-neural-network "weak learner": a random projection of a column-subsample of the features through a nonlinear activation (ReLU, tanh, sigmoid, ...), optionally concatenated with the raw features via a direct link, with a linear model (e.g. Ridge) fit on the current residuals. Because every hidden unit's transformation is known in closed form, so is its derivative -- which means feature attributions don't need to be estimated numerically:

- **Sensitivities** (`dF/dx`) -- the pointwise gradient of the ensemble's output with respect to each input, summed analytically across boosting rounds.
- **Integrated Gradients** -- exact, closed-form, baseline-relative attributions (no numerical quadrature), obtained via the secant/mean-value form of each activation so that attributions sum exactly to `F(x) - F(baseline)`.



```python
!pip install cybooster
```


```python
"""
examples.py
============

Self-contained worked examples for the closed-form Integrated Gradients
methodology; `cybooster.SkBoosterRegressor` already exposes:

    .fit(X, y)                                   -> self
    .predict(X)                                  -> np.ndarray
    .get_sensitivities(X, columns=...)           -> DataFrame  (pointwise dF/dx_j, heterogenous)
    .get_feature_importances(X, columns=...)     -> DataFrame  (mean |sensitivity|)
    .get_summary(X, columns=...)                 -> DataFrame  (mean/std/CI/p-value)
    .get_integrated_gradients(X, X_baseline=None, columns=...) -> DataFrame  (closed-form IG)
    .plot_importance(X, columns=..., kind=...)               -> Figure
    .plot_beeswarm(X, columns=..., kind=...)                 -> Figure
    .plot_heterogeneity(X, columns=..., kind=..., smooth=..., frac=..., n_boot=..., ci=...) -> Figure

Two worked examples:
  1. Diabetes (sklearn built-in, no network dependency) -- the main case.
  2. Boston housing (classic benchmark, GitHub CSV mirror) -- kept for
     continuity with prior work, with an explicit caveat on its `b`
     variable (see the docstring of `run_boston`).

Each example: fits the model, runs a completeness check on the closed-form
IG (sum of attributions should equal F(x) - F(baseline) to numerical
precision), and produces three figures: global importance, beeswarm
(attribution + heterogeneity), and LOWESS-smoothed local heterogeneity for
the top features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.datasets import load_diabetes

from cybooster import SkBoosterRegressor

OUTDIR = "."  # change if you want figures written elsewhere


# ----------------------------------------------------------------------------
# shared fit / diagnostics / figure-generation routine
# ----------------------------------------------------------------------------
def run_example(X, y, cols, name, top_k_features=3,
                 n_estimators=100, learning_rate=0.1, n_hidden_features=5,
                 activation="relu", reg_alpha=0.1, test_size=0.25, seed=123):
    """Fit, sanity-check, and plot one dataset. Returns the fitted model
    and the test split, in case further ad hoc inspection is wanted."""

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    model = SkBoosterRegressor(
        obj=Ridge(alpha=reg_alpha, fit_intercept=False),
        n_estimators=n_estimators, learning_rate=learning_rate,
        n_hidden_features=n_hidden_features, activation=activation,
        col_sample=1.0, row_sample=1.0, direct_link=1, verbose=0, seed=seed,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print(f"\n\n\n=== {name} ===")
    print(f"\n\n Test R2:  {r2_score(yte, pred):.4f}")
    print(f"\n\n Test MAE: {mean_absolute_error(yte, pred):.4f}")
    sensitivities = model.get_sensitivities(Xte, columns=cols)
    print(f"\n\n Get sensitivities: \n {sensitivities}")
    print(f"\n\n Probability of sensitivity > 0: \n { np.mean(sensitivities > 0, axis=0) }")
    print(f"\n\n Get importance: \n {model.get_feature_importances(Xte, columns=cols)}")
    print(f"\n\n Get summary: \n {model.get_summary(Xte, columns=cols)}")

    # --- closed-form Integrated Gradients + completeness check ---
    ig = model.get_integrated_gradients(Xte, columns=cols)
    baseline = np.tile(model.fit_obj_["xm"], (Xte.shape[0], 1))
    baseline_pred = model.predict(baseline)
    completeness_err = np.max(np.abs(ig.sum(axis=1).values - (pred - baseline_pred)))
    print(f"\n IG completeness max abs error: {completeness_err:.2e}  (should be ~machine eps)")

    # --- global importance (mean |IG|) ---
    print("\n\n Mean |IG| per feature:")
    print(ig.abs().mean().sort_values(ascending=False).round(3))

    # --- figures ---
    fig_bee = model.plot_beeswarm(Xte, columns=cols, kind="ig")
    fig_bee.savefig(f"{OUTDIR}/{name}_beeswarm.png", bbox_inches="tight")

    fig_imp = model.plot_importance(Xte, columns=cols, kind="ig")
    fig_imp.savefig(f"{OUTDIR}/{name}_importance.png", bbox_inches="tight")

    fig_het = model.plot_heterogeneity(Xte, columns=cols, kind="ig",
                                        top_k=top_k_features, smooth=True,
                                        frac=0.4, n_boot=200, ci=0.90)
    fig_het.savefig(f"{OUTDIR}/{name}_heterogeneity.png", bbox_inches="tight")

    print(f"Saved: {name}_importance.png, {name}_beeswarm.png, {name}_heterogeneity.png")
    return model, (Xtr, Xte, ytr, yte)


# ----------------------------------------------------------------------------
# example 1: diabetes (main case -- no network dependency, no ethical baggage)
# ----------------------------------------------------------------------------
def run_diabetes():
    data = load_diabetes(as_frame=True)
    X = data.data.values.astype(np.float64)
    y = data.target.values.astype(np.float64)
    cols = data.data.columns.tolist()
    return run_example(X, y, cols, name="diabetes", top_k_features=3)


# ----------------------------------------------------------------------------
# example 2: Boston housing (continuity case)
#
# Kept here for comparability with earlier results in this line of work, not
# as an endorsement of the dataset. The `b` column is 1000*(Bk-0.63)^2, a
# variable constructed around an assumption that racial composition affects
# property values non-monotonically; scikit-learn removed `load_boston` over
# this and related issues (see Carlisle, "Racist data destruction?", 2019).
# Results below should be read purely as a demonstration of the attribution
# method, not as a claim about the housing market.
# ----------------------------------------------------------------------------
def run_boston():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    df["chas"] = df["chas"].astype(int)
    y = df["medv"].values.astype(np.float64)
    X = df.drop(columns=["medv"]).values.astype(np.float64)
    cols = df.drop(columns=["medv"]).columns.tolist()
    return run_example(X, y, cols, name="boston", top_k_features=3)


if __name__ == "__main__":
    run_diabetes()
    run_boston()
```

    
    
    
    === diabetes ===
    
    
     Test R2:  0.5246
    
    
     Test MAE: 39.6518


    100%|██████████| 100/100 [00:01<00:00, 95.28it/s]


    
    
     Get sensitivities: 
            age     sex    bmi     bp      s1     s2     s3     s4     s5     s6
    0   140.25 -135.75 599.50 360.89 -904.42 426.03 197.63 322.09 659.40  98.09
    1    29.26 -114.81 492.63 410.05 -725.05 397.51 -41.97 248.65 706.47 -29.27
    2   142.60 -130.74 605.98 363.81 -904.33 426.11 203.03 324.27 660.64 102.55
    3   142.60 -130.74 605.98 363.81 -904.33 426.11 203.03 324.27 660.64 102.55
    4   134.84 -129.38 551.47 362.20 -906.51 391.45 154.09 318.14 612.25  73.00
    ..     ...     ...    ...    ...     ...    ...    ...    ...    ...    ...
    106 204.97 -220.27 546.19 492.93 -584.90 337.61 -43.53 229.42 760.56 -39.01
    107 219.03  -92.42 682.84 357.55 -748.84 368.26 267.01 410.04 718.87  37.63
    108 142.60 -130.74 605.98 363.81 -904.33 426.11 203.03 324.27 660.64 102.55
    109   5.61 -332.33 459.52 385.59 -596.72 355.77 -85.85 133.13 668.86 -90.40
    110 276.88 -172.84 632.56 611.12 -555.77 457.64 163.86 492.19 817.12 192.99
    
    [111 rows x 10 columns]
    
    
     Probability of sensitivity > 0: 
     age   0.90
    sex   0.00
    bmi   1.00
    bp    1.00
    s1    0.00
    s2    1.00
    s3    0.46
    s4    1.00
    s5    1.00
    s6    0.45
    dtype: float64


    100%|██████████| 100/100 [00:01<00:00, 94.89it/s]


    
    
     Get importance: 
         age    sex    bmi     bp     s1     s2     s3     s4     s5    s6
    0 78.31 246.15 523.84 383.15 728.83 380.95 131.86 220.51 663.86 94.07
    
    
     Get summary: 
            Mean  Std. Dev.     Min     Max  Median    SE  Lower CI  Upper CI  \
    s5   663.86      47.07  464.44  817.12  668.53  4.47    655.00    672.71   
    bmi  523.84      87.28  236.09  700.71  492.63  8.28    507.42    540.25   
    bp   383.15      46.52  256.74  611.12  387.71  4.42    374.40    391.90   
    s2   380.95      56.86  153.32  517.01  359.42  5.40    370.25    391.65   
    s4   220.51     107.11   10.34  492.19  168.27 10.17    200.36    240.66   
    age   72.64      79.56  -86.64  276.88   29.26  7.55     57.67     87.60   
    s3    32.75     143.40 -196.34  267.01  -63.61 13.61      5.78     59.73   
    s6    -6.13      98.95 -164.75  211.34  -77.04  9.39    -24.74     12.48   
    sex -246.15     105.55 -457.02  -62.21 -296.30 10.02   -266.01   -226.30   
    s1  -728.83     141.80 -944.88 -532.40 -672.97 13.46   -755.50   -702.15   
    
         t-statistic  p-value Signif. Code  
    s5        148.59     0.00          ***  
    bmi        63.23     0.00          ***  
    bp         86.77     0.00          ***  
    s2         70.58     0.00          ***  
    s4         21.69     0.00          ***  
    age         9.62     0.00          ***  
    s3          2.41     0.02            *  
    s6         -0.65     0.52            -  
    sex       -24.57     0.00          ***  
    s1        -54.15     0.00          ***  
    
     IG completeness max abs error: 9.24e-14  (should be ~machine eps)
    
    
     Mean |IG| per feature:
    s1    26.54
    s5    25.83
    bmi   21.63
    bp    15.38
    s2    13.81
    sex   11.56
    s4     7.71
    s3     5.05
    s6     3.53
    age    3.07
    dtype: float64
    Saved: diabetes_importance.png, diabetes_beeswarm.png, diabetes_heterogeneity.png
    
    
    
    === boston ===
    
    
     Test R2:  0.7982
    
    
     Test MAE: 2.5683


    100%|██████████| 100/100 [00:00<00:00, 120.65it/s]


    
    
     Get sensitivities: 
          crim    zn  indus  chas    nox    rm   age   dis  rad   tax  ptratio  \
    0    0.01 -0.04   0.02 -4.88 -10.01  9.30 -0.06 -1.23 0.43 -0.02    -0.77   
    1    0.22  0.09   0.26  4.14 -13.17 10.85 -0.01  0.04 0.67 -0.01    -0.24   
    2   -0.14  0.03   0.00  6.38 -33.67 -1.01  0.01 -2.28 0.19 -0.00    -1.20   
    3    0.01 -0.04   0.02 -4.88 -10.01  9.30 -0.06 -1.23 0.43 -0.02    -0.77   
    4   -0.14  0.03   0.00  6.47 -33.57 -0.98  0.01 -2.27 0.19 -0.00    -1.20   
    ..    ...   ...    ...   ...    ...   ...   ...   ...  ...   ...      ...   
    122 -0.14  0.03  -0.00  5.97 -33.86 -1.12  0.01 -2.29 0.19 -0.00    -1.18   
    123  0.01 -0.04   0.02 -4.88 -10.01  9.30 -0.06 -1.23 0.43 -0.02    -0.77   
    124  0.01 -0.04   0.02 -4.88 -10.01  9.30 -0.06 -1.23 0.43 -0.02    -0.77   
    125  0.01 -0.04   0.02 -4.88 -10.01  9.30 -0.06 -1.23 0.43 -0.02    -0.77   
    126 -0.23 -0.01  -0.05  4.39 -24.72  5.06 -0.05 -1.12 0.28 -0.01    -1.01   
    
           b  lstat  
    0   0.01  -0.06  
    1   0.02  -0.20  
    2   0.01  -0.81  
    3   0.01  -0.06  
    4   0.01  -0.81  
    ..   ...    ...  
    122 0.01  -0.82  
    123 0.01  -0.06  
    124 0.01  -0.06  
    125 0.01  -0.06  
    126 0.00  -0.51  
    
    [127 rows x 13 columns]
    
    
     Probability of sensitivity > 0: 
     crim      0.56
    zn        0.48
    indus     0.76
    chas      0.51
    nox       0.05
    rm        0.67
    age       0.42
    dis       0.02
    rad       1.00
    tax       0.04
    ptratio   0.06
    b         0.94
    lstat     0.05
    dtype: float64


    100%|██████████| 100/100 [00:00<00:00, 119.59it/s]


    
    
     Get importance: 
        crim   zn  indus  chas   nox   rm  age  dis  rad  tax  ptratio    b  lstat
    0  0.10 0.04   0.06  4.81 18.27 6.04 0.04 1.54 0.36 0.01     0.88 0.01   0.35
    
    
     Get summary: 
               Mean  Std. Dev.    Min   Max  Median   SE  Lower CI  Upper CI  \
    rm        5.27       4.81  -2.05 11.37    7.85 0.43      4.43      6.12   
    chas      0.52       5.13  -5.69 11.17    1.00 0.46     -0.38      1.42   
    rad       0.36       0.14   0.09  0.72    0.43 0.01      0.33      0.38   
    indus     0.04       0.11  -0.23  0.56    0.02 0.01      0.02      0.06   
    b         0.01       0.01  -0.02  0.04    0.01 0.00      0.01      0.01   
    zn        0.00       0.05  -0.06  0.17   -0.01 0.00     -0.01      0.01   
    tax      -0.01       0.01  -0.03  0.02   -0.01 0.00     -0.01     -0.01   
    age      -0.03       0.05  -0.14  0.16   -0.04 0.00     -0.03     -0.02   
    crim     -0.03       0.13  -0.29  0.38    0.01 0.01     -0.05     -0.01   
    lstat    -0.34       0.35  -0.85  0.27   -0.16 0.03     -0.40     -0.28   
    ptratio  -0.83       0.42  -1.68  0.79   -0.78 0.04     -0.91     -0.76   
    dis      -1.53       0.69  -2.64  0.48   -1.25 0.06     -1.65     -1.41   
    nox     -17.78      12.03 -37.83 10.03  -10.79 1.07    -19.90    -15.67   
    
             t-statistic  p-value Signif. Code  
    rm             12.36     0.00          ***  
    chas            1.14     0.26            -  
    rad            27.96     0.00          ***  
    indus           3.83     0.00          ***  
    b              13.15     0.00          ***  
    zn              0.37     0.71            -  
    tax           -13.81     0.00          ***  
    age            -6.36     0.00          ***  
    crim           -2.42     0.02            *  
    lstat         -11.08     0.00          ***  
    ptratio       -22.16     0.00          ***  
    dis           -25.11     0.00          ***  
    nox           -16.66     0.00          ***  
    
     IG completeness max abs error: 1.78e-14  (should be ~machine eps)
    
    
     Mean |IG| per feature:
    rm        2.80
    dis       2.53
    lstat     2.51
    rad       2.35
    nox       1.75
    ptratio   1.49
    tax       1.32
    age       1.12
    zn        0.67
    chas      0.61
    crim      0.60
    b         0.55
    indus     0.28
    dtype: float64
    Saved: boston_importance.png, boston_beeswarm.png, boston_heterogeneity.png



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_9.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_10.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_11.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_12.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_13.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-07-12/2026-07-12-Natively-Interpretable-Boosting_2_14.png){:class="img-responsive"}
    


## Takeaways

The completeness check above confirms the core claim: summing the closed-form Integrated Gradients across features reproduces `F(x) - F(baseline)` to numerical precision, with no sampling or quadrature error. That's the payoff of building interpretability into the weak learners themselves -- every attribution is exact and essentially free to compute, rather than an approximation layered on top of a black box.

The importance, beeswarm, and heterogeneity plots turn those attributions into something readable: which features matter on average, how their effects vary across observations, and whether that variation follows a discernible shape (via the LOWESS trend). Together __they give a boosted ensemble the same kind of transparency usually reserved for much simpler models, without giving up predictive performance__.

Natural next steps: try other activations (`tanh`, `sigmoid`, `relu6`) and compare how attribution shapes change, or swap in a different base learner (`Lasso`, `ElasticNet`) for the boosting rounds. Tune the model. 
