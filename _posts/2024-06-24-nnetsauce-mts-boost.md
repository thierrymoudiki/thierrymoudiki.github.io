---
layout: post
title: "Forecasting with XGBoost embedded in Quasi-Randomized Neural Networks"
description: "Forecasting with XGBoost embedded in Quasi-Randomized Neural Networks
at the International Symposium on Forecasting."
date: 2024-06-24
categories: [Python, QuasiRandomizedNN, Forecasting]
comments: true
---

Next week, I'll present [`nnetsauce`](https://github.com/Techtonique/nnetsauce)'s univariate and multivariate (probabilistic) time series forecasting capabilities at the 44th [International Symposium on Forecasting (ISF)](https://isf.forecasters.org/) (ISF) 2024. ISF is the **premier forecasting conference, attracting the world’s leading forecasting (I don't only do forecasting though) researchers, practitioners, and students**. I hope to see you there.

In this post, I illustrate how to obtain predictive simulations with [`nnetsauce`](https://github.com/Techtonique/nnetsauce)'s `MTS` class using XGBoost as base learner, and I give some intuition behind the method `"kde"` employed for uncertainty quantification in this case.


**(Command line)**

```bash
!pip install nnetsauce --upgrade --no-cache-dir
```

**Import Python packages**

```python
import nnetsauce as ns
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/USAccDeaths.csv"
df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date)
df.drop(columns=['date'], inplace=True)
```

**Number of estimators for the base learner**

```python
n_estimators_list = [5, 20, 50, 100]
estimators = []
residuals = []

for n_estimators in n_estimators_list:
  # XGBoost regressor as base learner
  regr_xgb = ns.MTS(obj=xgb.XGBRegressor(n_estimators=n_estimators,
                                         learning_rate=0.1),
                    n_hidden_features=50,
                    replications=1000,
                    kernel='gaussian',
                    lags=25)
  regr_xgb.fit(df)
  # in sample residuals 
  residuals.append(regr_xgb.residuals_.ravel())
  regr_xgb.predict(h=30)
  estimators.append(regr_xgb)

residuals_df = pd.DataFrame(np.asarray(residuals).T,
                            columns=["n5", "n20", "n50", "n100"])
```

    100%|██████████| 1/1 [00:00<00:00,  4.55it/s]
    100%|██████████| 1000/1000 [00:00<00:00, 1797.56it/s]
    100%|██████████| 1000/1000 [00:00<00:00, 1583.30it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.30it/s]
    100%|██████████| 1000/1000 [00:01<00:00, 582.86it/s]
    100%|██████████| 1000/1000 [00:01<00:00, 882.61it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.83s/it]
    100%|██████████| 1000/1000 [00:01<00:00, 808.35it/s]
    100%|██████████| 1000/1000 [00:00<00:00, 1555.30it/s]
    100%|██████████| 1/1 [00:02<00:00,  2.37s/it]
    100%|██████████| 1000/1000 [00:01<00:00, 578.83it/s]
    100%|██████████| 1000/1000 [00:00<00:00, 1436.02it/s]



```python
sns.set_theme(style="darkgrid")

for est in estimators:
  est.plot(type_plot="spaghetti")
```

![xxx]({{base}}/images/2024-06-24/2024-06-24-image1.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image2.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image3.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image4.png){:class="img-responsive"}          

```python
for i in range(4):
  sns.kdeplot(residuals_df.iloc[:,i], fill=True, color="red")
  plt.show()
```

![xxx]({{base}}/images/2024-06-24/2024-06-24-image5.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image6.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image7.png){:class="img-responsive"}  
![xxx]({{base}}/images/2024-06-24/2024-06-24-image8.png){:class="img-responsive"}              

In order to obtain predictive simulations with `"kde"` method (and as seen last week in [#143](https://thierrymoudiki.github.io/blog/#list-posts)), a [Kernel Density Estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) is adjusted to in-sample residuals. The most intuitive piece I found on KDEs is the following presentation: [https://scholar.harvard.edu/files/montamat/files/nonparametric_estimation.pdf](https://scholar.harvard.edu/files/montamat/files/nonparametric_estimation.pdf).

When using a high number of `estimators` (with the other parameters kept constant), the `XGBRegressor` base learner will overfit the training set, so that the in-sample residuals will be very small, and the uncertainty can't be captured/estimated adequately: the predictions will consist of point forecasts. A compromise needs to be found by using cross-validation on the base learner's hyperparameters, with an uncertainty quantification metric. Other types of predictions intervals/predictive simulation methods will be available in future versions of `nnetsauce`. 