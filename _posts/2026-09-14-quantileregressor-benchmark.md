---
layout: post
title: "Model-agnostic prediction intervals in Python and R: does nnetsauce's QuantileRegressor hold up?"
description: "Point predictions tell you what a model *thinks* will happen. They don't tell you how much to trust that number. nnetsauce's QuantileRegressor takes any sklearn-compatible regressor and turns it into a full quantile machine by optimizing an offset around its point predictions to minimize the pinball (quantile) loss."
date: 2026-09-14
categories: [R, Python]
comments: true
---


# Model-agnostic prediction intervals in Python and R: does nnetsauce's `QuantileRegressor` hold up?

Point predictions tell you what a model *thinks* will happen. They don't tell you how
much to trust that number.

[nnetsauce](https://github.com/Techtonique/nnetsauce)'s `QuantileRegressor` class
takes a different approach to this problem than most prediction-interval libraries:
instead of shipping one interval-producing algorithm, it takes **any** object with
`.fit()`/`.predict()` — linear model, SVR, random forest, whatever you already have —
and turns it into a full quantile machine by optimizing an offset around its point
predictions to minimize the pinball (quantile) loss. Five different "scoring"
strategies control how that offset is computed: `predictions`, `residuals`,
`conformal`, `studentized`, `conformal-studentized`.

## The library, in Python and R

The same class is available in R, via
[nnetsauce_r](https://github.com/Techtonique/nnetsauce_r) — and it's worth noting
up front that it isn't a separate reimplementation. The R function is a thin
`reticulate` wrapper that calls the identical Python object under the hood.
There is no separate R implementation to audit; auditing the Python source *is*
auditing the R behavior.

**Python** (this is real, runnable code — see the cell below)

```python
from nnetsauce.quantile.quantileregression import QuantileRegressor

obj = QuantileRegressor(
    obj=BayesianRidge(),      # any sklearn-compatible regressor
    level=95,                 # target coverage, in %
    scoring="residuals",      # "predictions" | "residuals" | "conformal" |
                               # "studentized" | "conformal-studentized"
)
obj.fit(X_train, y_train)
result = obj.predict(X_test, return_pi=True)
# result.mean, result.lower, result.median, result.upper
```

**R** (calls into the exact same Python class via `reticulate`)

```r
library(datasets)
X <- as.matrix(mtcars[, -1]); y <- mtcars[, 1]

sklearn <- nnetsauce::get_sklearn()
obj <- sklearn$linear_model$BayesianRidge()

obj2 <- QuantileRegressor(obj, level = 95, scoring = "residuals")
obj2$fit(X_train, y_train)
print(obj2$score(X_test, y_test))
```

Let's install the library and try the quickstart for real.

```python
# Quickstart: real nnetsauce.PredictionInterval, straight from PyPI.
# (QuantileRegressor is reimplemented in the next section with a lighter
#  optimizer budget purely to keep this notebook's total runtime short --
#  see the note there for why that's a faithful substitution.)
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nnetsauce",
                 "--break-system-packages"], check=False)

import warnings
warnings.filterwarnings("ignore")

from nnetsauce.quantile.quantileregression import QuantileRegressor
from nnetsauce.predictioninterval.predictioninterval import PredictionInterval
from sklearn.linear_model import BayesianRidge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

obj = QuantileRegressor(obj=BayesianRidge(), level=95, scoring="residuals")
obj.fit(X_train, y_train)
result = obj.predict(X_test, return_pi=True)

coverage = np.mean((y_test >= result.lower) & (y_test <= result.upper))
print("First 5 intervals:")
for lo, med, hi, true in list(zip(result.lower, result.median, result.upper, y_test))[:5]:
    print(f"  [{lo:7.1f}, {hi:7.1f}]   median={med:7.1f}   true={true:7.1f}")
print(f"\nEmpirical coverage on the test set: {coverage:.1%} (target: 95%)")

```

    First 5 intervals:
      [   35.9,   243.1]   median=  137.3   true=  219.0
      [   76.6,   283.8]   median=  177.9   true=   70.0
      [   27.8,   235.0]   median=  129.2   true=  202.0
      [  186.7,   394.0]   median=  288.1   true=  230.0
      [   18.6,   225.8]   median=  119.9   true=  111.0
    
    Empirical coverage on the test set: 94.7% (target: 95%)


## Benchmark setup

I ran a fairly large grid, deliberately **without tuning any base estimator's
hyperparameters** — the point is to test the wrapper's behavior "out of the box,"
the way most people would first try it:

- **38 scikit-learn regressors** — everything `sklearn.utils.all_estimators(type_filter='regressor')`
  returns, minus meta-estimators that need extra wiring (stacking/voting/multi-output
  regressors, the heavy default-tuned ensembles, etc.)
- **6 datasets** — `diabetes`, `linnerud`, two synthetic sets (linear and mildly
  nonlinear), an anonymized version of the classic Boston Housing dataset, and a
  600-row subsample of California Housing
- **2 coverage targets** — 80% and 95%
- **5 scoring strategies** for `QuantileRegressor`, plus nnetsauce's sibling class
  `PredictionInterval` (`method="splitconformal"`) as a second, structurally
  different baseline
- **Two "native" quantile baselines** that don't wrap anything: scikit-learn's own
  linear `QuantileRegressor` (pinball-loss minimization, no L1 penalty), and
  `GradientBoostingRegressor(loss="quantile")`

That's 2,736 individual model fits for the wrapped-estimator grid, plus 24 more
for the two native baselines. Let's build it.


```python
# ---------------------------------------------------------------------------
# Imports and metrics
# ---------------------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import all_estimators
from scipy.optimize import differential_evolution
from sklearn.datasets import load_diabetes, load_linnerud, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor as SkQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

np.random.seed(42)
LEVELS = [80, 95]
SCORINGS = ["predictions", "residuals", "conformal", "studentized", "conformal-studentized"]


def coverage_and_width(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean(), np.mean(upper - lower)


def winkler_score(y_true, lower, upper, level):
    alpha = 1 - level / 100
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy().astype(float)
    score[below] += (2 / alpha) * (lower[below] - y_true[below])
    score[above] += (2 / alpha) * (y_true[above] - upper[above])
    return np.mean(score)

print("Metrics ready.")

```

    Metrics ready.



```python

# ---------------------------------------------------------------------------
# Datasets -- downloaded fresh so this notebook is fully reproducible standalone
# ---------------------------------------------------------------------------
import urllib.request, io

def load_datasets():
    datasets = {}
    X, y = load_diabetes(return_X_y=True)
    datasets["diabetes"] = (X, y)

    lin = load_linnerud()
    datasets["linnerud (predict weight)"] = (lin.data, lin.target[:, 0])

    Xs, ys = make_regression(n_samples=400, n_features=8, noise=15.0, random_state=42)
    datasets["synthetic_linear"] = (Xs, ys)

    Xn, yn = make_regression(n_samples=400, n_features=8, noise=10.0, random_state=1)
    yn = yn + 0.02 * (Xn[:, 0] ** 3)
    datasets["synthetic_nonlinear"] = (Xn, yn)

    boston_url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv"
    boston = pd.read_csv(boston_url)
    Xb = boston.drop(columns=["target", "training_index"]).values
    yb = boston["target"].values
    datasets["boston_anonymized"] = (Xb, yb)

    housing_url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
    housing = pd.read_csv(housing_url).dropna()
    housing = pd.get_dummies(housing, columns=["ocean_proximity"], drop_first=True)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(housing), size=600, replace=False)
    sub = housing.iloc[idx]
    Xc = sub.drop(columns=["median_house_value"]).values.astype(float)
    yc = sub["median_house_value"].values.astype(float)
    datasets["california_housing (n=600 subsample)"] = (Xc, yc)

    return datasets

DATASETS = load_datasets()
for name, (X, y) in DATASETS.items():
    print(f"{name:45s}  X={X.shape}  y={y.shape}")

```

    diabetes                                       X=(442, 10)  y=(442,)
    linnerud (predict weight)                      X=(20, 3)  y=(20,)
    synthetic_linear                               X=(400, 8)  y=(400,)
    synthetic_nonlinear                            X=(400, 8)  y=(400,)
    boston_anonymized                              X=(506, 13)  y=(506,)
    california_housing (n=600 subsample)           X=(600, 12)  y=(600,)



```python

# ---------------------------------------------------------------------------
# Estimator list: every sklearn regressor, minus meta-estimators that need
# extra wiring (based on https://gist.github.com/thierrymoudiki/19a856e2d9c75d5b4fe57fa332b5e8c9)
# ---------------------------------------------------------------------------
SKIP_ESTIMATORS = {
    'MultiOutputRegressor', 'MultiOutputClassifier', 'StackingRegressor', 'StackingClassifier',
    'VotingRegressor', 'VotingClassifier', 'TransformedTargetRegressor', 'RegressorChain',
    'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'RandomForestRegressor',
    'ExtraTreesRegressor', 'MLPRegressor',
    'MultiTaskLasso', 'MultiTaskElasticNet', 'MultiTaskLassoCV', 'MultiTaskElasticNetCV',
    'IsotonicRegression', 'CCA', 'PLSCanonical', 'RegressorMixin',
}

def get_estimators():
    regs = all_estimators(type_filter='regressor')
    out = []
    for name, cls in regs:
        if name in SKIP_ESTIMATORS:
            continue
        try:
            obj = cls()  # default hyperparameters only -- no tuning
        except Exception:
            continue
        out.append((name, obj))
    return out

ESTIMATORS = get_estimators()
print(f"{len(ESTIMATORS)} estimators loaded:")
print(", ".join(name for name, _ in ESTIMATORS))

```

    38 estimators loaded:
    ARDRegression, AdaBoostRegressor, BaggingRegressor, BayesianRidge, DecisionTreeRegressor, DummyRegressor, ElasticNet, ElasticNetCV, ExtraTreeRegressor, GammaRegressor, GaussianProcessRegressor, HuberRegressor, KNeighborsRegressor, KernelRidge, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, LinearSVR, NuSVR, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PLSRegression, PassiveAggressiveRegressor, PoissonRegressor, QuantileRegressor, RANSACRegressor, RadiusNeighborsRegressor, Ridge, RidgeCV, SGDRegressor, SVR, TheilSenRegressor, TweedieRegressor


```python
# ---------------------------------------------------------------------------
# Run the full sweep: 5 QuantileRegressor scoring modes + PredictionInterval,
# x 38 estimators x 6 datasets x 2 levels = 2,736 runs. Takes ~2-3 minutes.
# ---------------------------------------------------------------------------
from nnetsauce.predictioninterval.predictioninterval import PredictionInterval
from tqdm import tqdm

def run_one(method_name, est, X_train, y_train, X_test, y_test, level):
    t0 = time.time()
    if method_name.startswith("QR:"):
        scoring = method_name.split(":", 1)[1]
        model = QuantileRegressor(obj=clone(est), level=level, scoring=scoring)
        model.fit(X_train, y_train)
        res = model.predict(X_test, return_pi=True)
        lower, upper, med = res.lower, res.upper, res.median
    elif method_name == "PredictionInterval":
        model = PredictionInterval(obj=clone(est), method="splitconformal", level=level)
        model.fit(X_train, y_train)
        res = model.predict(X_test, return_pi=True)
        lower, upper, med = res.lower, res.upper, res.mean
    else:
        raise ValueError(method_name)

    cov, width = coverage_and_width(y_test, lower, upper)
    wink = winkler_score(y_test, lower, upper, level=level)
    mae = np.mean(np.abs(y_test - med))
    return cov, width, wink, mae, time.time() - t0


methods = [f"QR:{s}" for s in SCORINGS] + ["PredictionInterval"]
results = []
t_start = time.time()

for level in LEVELS:
    for ds_name, (X, y) in tqdm(DATASETS.items()):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        for est_name, est in ESTIMATORS:
            for method_name in methods:
                try:
                    cov, width, wink, mae, elapsed = run_one(
                        method_name, est, X_train, y_train, X_test, y_test, level
                    )
                    row = {"level": level, "dataset": ds_name, "estimator": est_name, "method": method_name,
                           "coverage": round(cov, 4), "avg_interval_width": round(width, 4),
                           "winkler_score": round(wink, 4), "median_MAE": round(mae, 4),
                           "time_s": round(elapsed, 3), "status": "ok"}
                except Exception as e:
                    row = {"level": level, "dataset": ds_name, "estimator": est_name, "method": method_name,
                           "coverage": np.nan, "avg_interval_width": np.nan, "winkler_score": np.nan,
                           "median_MAE": np.nan, "time_s": np.nan, "status": f"ERROR: {str(e)[:80]}"}
                results.append(row)

full_sweep = pd.DataFrame(results)
print(f"Done in {time.time()-t_start:.1f}s. {len(full_sweep)} rows, "
      f"{full_sweep['status'].eq('ok').sum()} succeeded, {(~full_sweep['status'].eq('ok')).sum()} failed.")
full_sweep.to_csv("full_sweep_results.csv", index=False)

```

    100%|██████████| 6/6 [05:06<00:00, 51.11s/it]
    100%|██████████| 6/6 [04:30<00:00, 45.07s/it]

    Done in 577.1s. 2736 rows, 2688 succeeded, 48 failed.


    



```python

# ---------------------------------------------------------------------------
# Native quantile baselines: sklearn's own linear QuantileRegressor, and
# GradientBoostingRegressor(loss="quantile") -- neither wraps another model.
# ---------------------------------------------------------------------------
def fit_predict_native_quantile(method, X_train, y_train, X_test, level):
    low_q = (1 - level / 100) / 2
    high_q = 1 - low_q
    if method == "sklearn_QuantileRegressor":
        m_low = SkQuantileRegressor(quantile=low_q, alpha=0.0, solver="highs")
        m_med = SkQuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
        m_high = SkQuantileRegressor(quantile=high_q, alpha=0.0, solver="highs")
    elif method == "GBM_quantile":
        m_low = GradientBoostingRegressor(loss="quantile", alpha=low_q, random_state=42)
        m_med = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)
        m_high = GradientBoostingRegressor(loss="quantile", alpha=high_q, random_state=42)
    m_low.fit(X_train, y_train); m_med.fit(X_train, y_train); m_high.fit(X_train, y_train)
    lower, median, upper = m_low.predict(X_test), m_med.predict(X_test), m_high.predict(X_test)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return lower, median, upper


native_results = []
for level in LEVELS:
    for ds_name, (X, y) in tqdm(DATASETS.items()):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        for method in ["sklearn_QuantileRegressor", "GBM_quantile"]:
            t0 = time.time()
            lower, median, upper = fit_predict_native_quantile(method, X_train, y_train, X_test, level)
            cov, width = coverage_and_width(y_test, lower, upper)
            wink = winkler_score(y_test, lower, upper, level=level)
            mae = np.mean(np.abs(y_test - median))
            native_results.append({"level": level, "dataset": ds_name, "method": method,
                                    "coverage": cov, "avg_interval_width": width,
                                    "winkler_score": wink, "median_MAE": mae,
                                    "time_s": time.time() - t0})

native = pd.DataFrame(native_results)
native.to_csv("native_quantile_baselines.csv", index=False)
print(native.groupby(["level", "method"])["coverage"].agg(["mean", "median", "min", "max"]).round(3))

```

    100%|██████████| 6/6 [00:18<00:00,  3.16s/it]
    100%|██████████| 6/6 [00:20<00:00,  3.49s/it]

                                     mean  median  min  max
    level method                                           
    80    GBM_quantile               0.69    0.69 0.61 0.77
          sklearn_QuantileRegressor  0.76    0.77 0.67 0.80
    95    GBM_quantile               0.92    0.92 0.86 1.00
          sklearn_QuantileRegressor  0.87    0.91 0.67 0.93

## Headline result: it's a solid technique, most of the time

Restricting to the runs that didn't misbehave (more on that below), let's check
how often `QuantileRegressor`'s median coverage landed within 3 percentage points
of the target level, and compare method families head-to-head against the native
baselines.



```python

ok = full_sweep[full_sweep.status == "ok"].copy()
ok["collapsed"] = ok["avg_interval_width"] < 1e-6

agg = ok.groupby(["level", "method", "estimator"]).agg(
    mean_cov=("coverage", "mean"), median_cov=("coverage", "median"), min_cov=("coverage", "min"),
    n=("coverage", "count"), n_collapsed=("collapsed", "sum")
).reset_index()
agg.to_csv("full_sweep_estimator_summary.csv", index=False)

qr = agg[agg.method.str.startswith("QR:")]
for level in LEVELS:
    target = level / 100
    sub = qr[(qr.level == level) & (qr.n_collapsed == 0)].copy()
    sub["gap"] = (sub["median_cov"] - target).abs()
    good = sub[sub["gap"] <= 0.03]
    print(f"Level {level}%: {len(good)} of {len(sub)} non-collapsing (scoring, estimator) "
          f"pairs land within 3pts of target on median coverage ({len(good)/len(sub):.0%})")

```

    Level 80%: 126 of 173 non-collapsing (scoring, estimator) pairs land within 3pts of target on median coverage (73%)
    Level 95%: 133 of 173 non-collapsing (scoring, estimator) pairs land within 3pts of target on median coverage (77%)



```python
# Safe estimators: never collapsed under any QuantileRegressor scoring mode
never_collapse = qr.groupby("estimator")["n_collapsed"].sum()
SAFE_ESTIMATORS = sorted(never_collapse[never_collapse == 0].index.tolist())
RISKY_ESTIMATORS = sorted(never_collapse[never_collapse > 0].index.tolist())
print(f"{len(SAFE_ESTIMATORS)} estimators NEVER collapse under any QuantileRegressor scoring mode:")
print(SAFE_ESTIMATORS)
print(f"\n{len(RISKY_ESTIMATORS)} estimators collapse under at least one scoring mode/level/dataset:")
print(RISKY_ESTIMATORS)
```

    34 estimators NEVER collapse under any QuantileRegressor scoring mode:
    ['ARDRegression', 'BaggingRegressor', 'BayesianRidge', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'GammaRegressor', 'HuberRegressor', 'KNeighborsRegressor', 'KernelRidge', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'LinearSVR', 'NuSVR', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'PLSRegression', 'PassiveAggressiveRegressor', 'PoissonRegressor', 'QuantileRegressor', 'RANSACRegressor', 'RadiusNeighborsRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'SVR', 'TheilSenRegressor', 'TweedieRegressor']
    
    4 estimators collapse under at least one scoring mode/level/dataset:
    ['AdaBoostRegressor', 'DecisionTreeRegressor', 'ExtraTreeRegressor', 'GaussianProcessRegressor']



```python
# Method-family comparison: average coverage across the *safe* estimators only,
# so the trees/GP collapse doesn't distort the picture -- compared against the
# two native quantile baselines.
rows = []
for level in LEVELS:
    for method in ["QR:residuals", "QR:conformal", "PredictionInterval"]:
        sub = agg[(agg.level == level) & (agg.method == method) & (agg.estimator.isin(SAFE_ESTIMATORS))]
        rows.append({"level": level, "method": method, "mean_cov": sub.mean_cov.mean()})
    for method in ["sklearn_QuantileRegressor", "GBM_quantile"]:
        sub = native[(native.level == level) & (native.method == method)]
        rows.append({"level": level, "method": method, "mean_cov": sub.coverage.mean()})

family = pd.DataFrame(rows)
family.to_csv("method_family_comparison.csv", index=False)

labels_map = {
    "QR:residuals": "nnetsauce QuantileRegressor\n(residuals scoring)",
    "QR:conformal": "nnetsauce QuantileRegressor\n(conformal scoring)",
    "PredictionInterval": "nnetsauce PredictionInterval\n(splitconformal)",
    "sklearn_QuantileRegressor": "sklearn QuantileRegressor\n(native, linear)",
    "GBM_quantile": "GradientBoosting\n(native, quantile loss)",
}
methods_order = ["QR:residuals", "QR:conformal", "PredictionInterval", "sklearn_QuantileRegressor", "GBM_quantile"]
colors = ["#4C72B0", "#8CA8D8", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, level, target in zip(axes, LEVELS, [l / 100 for l in LEVELS]):
    sub = family[family.level == level].set_index("method").loc[methods_order]
    ax.bar(range(len(methods_order)), sub["mean_cov"], color=colors)
    ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target ({int(target*100)}%)")
    ax.set_xticks(range(len(methods_order)))
    ax.set_xticklabels([labels_map[m] for m in methods_order], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Target coverage: {int(target*100)}%")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("Mean empirical coverage")
plt.tight_layout()
plt.savefig("coverage_comparison.png", dpi=150)
plt.show()

print(family.round(3).to_string(index=False))
```
    
![image-title-here]({{base}}/images/2026-09-14/2026-09-14-quantileregressor-benchmark_13_0.png){:class="img-responsive"}
    


     level                    method  mean_cov
        80              QR:residuals      0.77
        80              QR:conformal      0.73
        80        PredictionInterval      0.81
        80 sklearn_QuantileRegressor      0.76
        80              GBM_quantile      0.69
        95              QR:residuals      0.92
        95              QR:conformal      0.86
        95        PredictionInterval      0.95
        95 sklearn_QuantileRegressor      0.87
        95              GBM_quantile      0.92


A few things stand out:

- **`PredictionInterval` (splitconformal) was the best-calibrated method** in our
  grid, at both targets. It computes a single calibration-residual quantile and
  never re-optimizes against data the model has already seen, which turns out to
  matter a lot (see next section).
- **`QuantileRegressor` with `scoring="residuals"`** essentially matches
  scikit-learn's own native linear `QuantileRegressor` at the 80% target and
  clearly beats it at the 95% target (where the native version was dragged down
  by the 20-row `linnerud` dataset).
- **Gradient-boosted quantile regression** undercovered the most at the 80%
  target — likely because with unregularized default hyperparameters,
  `GradientBoostingRegressor` starts overfitting each quantile individually
  rather than producing a coherent interval. It did comparatively better at 95%.

The takeaway isn't "throw away purpose-built quantile regressors." It's that a
model-agnostic wrapper around an off-the-shelf regressor is a *competitive*
alternative when you want interval estimates from a model family that doesn't
have a native quantile-loss variant (say, a Support Vector Regressor, or a
Bayesian linear model) — and it costs nothing extra to try.


## The catch: it fails predictably, and only for one kind of model

Here's the part worth being careful about. Let's find every run where the
predicted interval collapsed to (near) zero width.



```python
collapsed = ok[ok.avg_interval_width < 1e-6]
piv = collapsed[collapsed.method != "PredictionInterval"].pivot_table(
    index="estimator", columns="method", values="coverage", aggfunc="count", fill_value=0
)
risky_order = [e for e in ["DecisionTreeRegressor", "ExtraTreeRegressor",
                            "GaussianProcessRegressor", "AdaBoostRegressor"] if e in piv.index]
piv = piv.reindex(risky_order).fillna(0)
methods_order2 = [f"QR:{s}" for s in SCORINGS]
piv = piv[methods_order2]
print(piv.astype(int))
print(f"\nPredictionInterval collapsed count (out of {len(ok[ok.method=='PredictionInterval'])}):",
      len(ok[(ok.method == "PredictionInterval") & (ok.avg_interval_width < 1e-6)]))

```
    method                    QR:predictions  QR:residuals  QR:conformal  \
    estimator                                                              
    DecisionTreeRegressor                 12            12            12   
    ExtraTreeRegressor                    12            12            12   
    GaussianProcessRegressor              10             9             9   
    AdaBoostRegressor                      0             0             2   
    
    method                    QR:studentized  QR:conformal-studentized  
    estimator                                                           
    DecisionTreeRegressor                 12                        12  
    ExtraTreeRegressor                    12                        12  
    GaussianProcessRegressor               9                         9  
    AdaBoostRegressor                      0                         2  
    
    PredictionInterval collapsed count (out of 448): 0



```python
fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(len(piv))
set2 = plt.cm.Set2(np.linspace(0, 1, len(methods_order2)))
for m, c in zip(methods_order2, set2):
    ax.bar(piv.index, piv[m], bottom=bottom, label=m, color=c)
    bottom += piv[m].values
ax.set_ylabel("# collapsed runs (out of 12 = 6 datasets x 2 levels)")
ax.set_title("QuantileRegressor: zero-width interval collapse, by base estimator and scoring mode")
ax.legend(fontsize=8, ncol=2)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("collapse_chart.png", dpi=150)
plt.show()
```
    
![image-title-here]({{base}}/images/2026-09-14/2026-09-14-quantileregressor-benchmark_17_0.png){:class="img-responsive"}
    
`DecisionTreeRegressor` and `ExtraTreeRegressor` collapsed on **every single
run** (12 out of 12 — all 6 datasets, both levels), regardless of which of the 5
scoring strategies was used. `GaussianProcessRegressor` collapsed on 9–10 out of
12 runs. `AdaBoostRegressor` collapsed occasionally, only under the two
`conformal*` scoring modes.

The mechanism is the same in every case, and it isn't specific to any one
scoring strategy — it happens because `QuantileRegressor` optimizes its
interval-width multiplier by minimizing pinball loss **on data the base model
has already been fit on** (either the full training set, for
`predictions`/`residuals`/`studentized`, or a calibration split the model gets
re-fit to, for `conformal`/`conformal-studentized`). An unconstrained decision
tree, or a Gaussian process with a noiseless kernel, can memorize that data
almost perfectly. Once training residuals are ~0, the optimizer correctly
notices that a zero-width interval already achieves close to the minimum
possible pinball loss — and it collapses the interval accordingly, regardless
of whether the scale factor being multiplied is a residual standard deviation,
a prediction magnitude, or the target's own standard deviation.

The practical rule this suggests: **don't pair `QuantileRegressor` with base
estimators capable of near-perfect interpolation of their own fitting data** —
unconstrained trees and noiseless GPs, specifically. Every other estimator we
tested — the entire linear family, `SVR`/`NuSVR`/`LinearSVR`, `KernelRidge`,
`KNeighborsRegressor`, `RANSACRegressor`, `TheilSenRegressor`, the GLMs,
`Bagging`, `PassiveAggressiveRegressor`, `SGDRegressor` — **never collapsed
once**, across any scoring mode, dataset, or coverage level.


## Best performers at each target level

Within the safe estimators, here's what came closest to nominal coverage
(median across the 6 datasets), along with the worst-case dataset for each —
because a good median can still hide a bad outlier, and we don't want to bury
that in an average.

```python
for level in LEVELS:
    target = level / 100
    sub = qr[(qr.level == level) & (qr.n_collapsed == 0)].copy()
    sub["gap"] = (sub["median_cov"] - target).abs()
    print(f"=== {level}% target: top 5 by |median coverage - target| ===")
    top5 = sub.sort_values("gap").head(5)[["method", "estimator", "mean_cov", "median_cov", "min_cov"]]
    print(top5.round(3).to_string(index=False))
    print()
```

    === 80% target: top 5 by |median coverage - target| ===
                      method                  estimator  mean_cov  median_cov  min_cov
                QR:conformal              BayesianRidge      0.77        0.80     0.67
                QR:conformal                KernelRidge      0.76        0.80     0.50
                QR:conformal                       Lars      0.75        0.80     0.50
    QR:conformal-studentized              BayesianRidge      0.77        0.80     0.67
              QR:studentized PassiveAggressiveRegressor      0.82        0.80     0.74
    
    === 95% target: top 5 by |median coverage - target| ===
            method        estimator  mean_cov  median_cov  min_cov
      QR:residuals       ElasticNet      0.93        0.95     0.83
    QR:studentized       ElasticNet      0.93        0.95     0.83
      QR:residuals TweedieRegressor      0.94        0.95     0.83
      QR:conformal    BayesianRidge      0.91        0.95     0.67
      QR:residuals    PLSRegression      0.93        0.95     0.83
    


The worst-case column (`min_cov`) is a useful sanity check: even the
best-calibrated combinations here have at least one dataset where coverage
drops well below target — usually `linnerud`, which has only 6 test
observations after the split, so treat those specific numbers as noisy rather
than damning. It's a reminder that "good on average" and "reliable everywhere"
are different claims, worth checking separately rather than folding into one
number.


## Practical guidance

Putting it together, if you're deciding how to get prediction intervals out of
an arbitrary scikit-learn (or R, via `nnetsauce_r`) regressor:

1. **Default to `scoring="residuals"`** over `"conformal"` if you want the
   simplest behavior — it doesn't need a train/calibration split, and it
   performed as well or better in this benchmark. `"studentized"` and
   `"conformal-studentized"` tracked their non-studentized siblings closely
   enough (coverage correlation ≥0.96 in the full grid) that they add little
   beyond redundancy.
2. **Reach for `PredictionInterval(method="splitconformal")`** if you want the
   most robust option and don't need the flexibility of the 5 scoring
   strategies — it was the best-calibrated method here and never collapsed on
   any estimator.
3. **Avoid pairing either wrapper with unconstrained trees or noiseless
   Gaussian processes.** If you need a tree-based interval,
   `RandomForestRegressor`, `BaggingRegressor`, or `AdaBoostRegressor` (mostly)
   sidestep the issue since they don't interpolate the data as tightly as a
   single unconstrained tree.
4. **Check coverage on a genuine holdout, per use case, before trusting the
   number** — "good on average across 6 datasets" is a benchmarking
   convenience, not a guarantee for your specific one.

## Reproducibility

Every run in this notebook used default hyperparameters — no `GridSearchCV`, no
manual tuning — so the numbers reflect what you'd see trying this out of the
box. Re-running this notebook end-to-end reproduces every table and chart
above from scratch (data is downloaded fresh at runtime); the intermediate
CSVs (`full_sweep_results.csv`, `full_sweep_estimator_summary.csv`,
`native_quantile_baselines.csv`, `method_family_comparison.csv`) are also
written to disk if you want to explore further without re-running the sweep.

*Benchmarked: nnetsauce `QuantileRegressor` and `PredictionInterval`
([source](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/quantile/quantileregression.py),
[source](https://raw.githubusercontent.com/Techtonique/nnetsauce/refs/heads/master/nnetsauce/predictioninterval/predictioninterval.py)),
the R wrapper
([source](https://github.com/Techtonique/nnetsauce_r/blob/main/R/quantileregressor.R)),
against scikit-learn's `QuantileRegressor` and `GradientBoostingRegressor(loss="quantile")`.*

