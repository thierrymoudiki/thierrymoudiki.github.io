---
layout: post
title: "ahead (Time Series Forecasting with uncertainty quantification) gets a lot faster to install: most dependencies are now optional"
description: "ahead (R) 0.38.1 and its Python wrapper now install in a fraction of the time, by moving almost every heavy modeling dependency from Imports to Suggests and installing them at runtime, only when a function actually needs them."
date: 2026-09-08
categories: [R, Python]
comments: true
---

If you've used `ahead` before, you may remember `install.packages("ahead")`
or `pip install ahead` taking a while, since it dragged in two dozen
modeling packages up front regardless of which one you actually planned to
use. That's fixed now.

## The problem: one function's dependency, everyone's install time

`ahead` supports a lot of forecasting methods, and each one can lean on a
different modeling package: `forecast`, `randomForest`, `e1071`, `glmnet`,
`gam`, `quantreg`, `vars`, `fGarch`, `VineCopula`, `mboost`, `ranger`,
`ForecastComb`. Historically, most of these were plain `Imports`
in `DESCRIPTION`, which means R installs *all* of them before you can even
load the package — regardless of whether you plan to use `ahead::dynrmf` with a
random forest or `ahead::ridge2f` for a multivariate model with none of the above.

## The fix: (almost) everything becomes a Suggests

The new `DESCRIPTION` keeps only what the package's core code actually
needs at load time:

```
Imports:
    Rcpp (>= 1.0.6),
    foreach,
    tseries
Depends: R (>= 3.5.0)
Suggests:
    caret, cclust, dfoptim, doSNOW, doParallel, knitr, rmarkdown,
    testthat, fpp2, glmnet, e1071, gam, quantreg, randomForest,
    spatial, vars, roxygen2, ForecastComb, ranger, mboost, misc,
    Mcomp, fGarch, VineCopula, forecast (>= 8.0),
    ggplot2 (>= 3.0.0), randtoolbox (>= 1.17), simulatetimeseries
```

Everything else — `randomForest`, `e1071`, `forecast`, `glmnet`, `vars`,
`fGarch`, `VineCopula`, and the rest — moved to `Suggests`. That means R
installs in seconds, because `Rcpp`, `foreach`, and `tseries` are the only
hard requirements.

## Installing the missing piece, exactly when you need it

Moving packages to `Suggests` only helps if the package still works
smoothly when you actually call a function that needs one of them. That's
what a small internal helper, `check_suggested()`, takes care of:

```r
check_suggested <- function(pkg, ask = interactive()) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  do_install <- TRUE
  if (ask) {
    do_install <- utils::askYesNo(
      sprintf("Package '%s' is required but not installed. Install it now?", pkg)
    )
    do_install <- isTRUE(do_install)
  }
  if (do_install) {
    utils::install.packages(
      pkg,
      repos = c("https://techtonique.r-universe.dev", "https://cloud.r-project.org")
    )
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf(
        "Package '%s' is required. Install it with install.packages('%s', repos = c('https://techtonique.r-universe.dev', 'https://cloud.r-project.org')).",
        pkg, pkg
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}
```

Every function (hopefully all of them!) that relies on a suggested package now calls
`check_suggested("that_package")` first. In practice, this means:

- If the package is already installed, nothing changes — no prompt, no
  delay.
- If it's missing and you're in an interactive session, you get asked
  before anything is installed on your behalf.
- If it's missing and the session isn't interactive (a script, a CI job),
  it installs automatically.
- If the install still fails, you get a clear error with the exact command
  to run, pointing at both CRAN and Techtonique's r-universe repository —
  useful since a couple of these dependencies (like `ForecastComb` and
  `misc`) aren't on CRAN at all.

So `install.packages("ahead")` is fast, and the first time you call
`dynrmf(..., fit_func = randomForest::randomForest)`, `randomForest` gets
installed for you, once, and never again after that.

## Same idea, carried into the Python wrapper

`ahead`'s Python package is a wrapper around the R package (via `rpy2`), so
the first call to any forecaster has always installed R-side dependencies
on demand — that's the "might take some time, but ONLY the 1st time it's
called" comment you'll see next to `DynamicRegressor` or `Ridge2Regressor`
in the examples. With the R package's own install now lean, that first-call
overhead on the Python side shrinks too: there's less to pull in before
`check_suggested()` even gets to the package your chosen method needs.

```notebook-python
import os
import numpy as np
import pandas as pd
from ahead import DynamicRegressor, EAT
from time import time

# Forecasting horizon
h = 25

# Data frame containing the time series
df = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/AirPassengers.csv").set_index('date')
df.index = pd.DatetimeIndex(df.index)
print(df)

# univariate ts forecasting
print("Example 1 -----")
d1 = DynamicRegressor(h=h, date_formatting="ms")
print(d1.__module__)

start = time()
d1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d1.averages_)
print("\n")
print("ranges: \n")
print(d1.ranges_)
print("\n")

print("Example 2 -----")
d2 = DynamicRegressor(h=h, type_pi="T", date_formatting="original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d2.averages_)
print("\n")
print("ranges: \n")
print(d2.ranges_)
print("\n")

d2.plot()
```

The first call to `DynamicRegressor.forecast()` in a fresh environment is
the one that pays the (now much smaller) one-time cost of installing the R
`ahead` package; every call after that — including `d2` above, in the same
session — runs at full speed.

## Why this matters in practice

- **Faster CI and Docker builds** — you're not compiling `gam`, `fGarch`,
  `VineCopula`, and a dozen others just to run a couple of unit tests that
  use two of them.
- **Less fragile installs** — a slow-to-compile or awkward-to-build modeling
  package no longer stands between you and a working `ahead` install if you
  don't even need that method.
- **Smaller footprint for the methods you actually use** — if you only ever
  call `ridge2f`, you never install the GARCH or copula-based dependencies
  at all.

If you want everything available up front (for example, to prepare an
offline environment), you can still install every suggested package
yourself:

```r
install.packages(
  c("caret", "cclust", "dfoptim", "doSNOW", "doParallel", "fpp2",
    "glmnet", "e1071", "gam", "quantreg", "randomForest", "spatial",
    "vars", "ranger", "mboost", "Mcomp", "fGarch", "VineCopula",
    "forecast", "ggplot2", "randtoolbox", "simulatetimeseries"),
  repos = c("https://techtonique.r-universe.dev", "https://cloud.r-project.org")
)
```

## Get it

- R: see the [README](https://github.com/Techtonique/ahead) for the Techtonique repo setup, version `0.38.1`.
- Python: `pip install ahead --verbose`, from [ahead_python](https://github.com/Techtonique/ahead_python).

As always, issues and contributions are welcome on both repos.

![image-title-here]({{base}}/images/2025-09-01/2025-09-01-external-regressors-in-dynrmf_1_0.png){:class="img-responsive"}
