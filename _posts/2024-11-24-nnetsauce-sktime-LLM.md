---
layout: post
title: "You can beat Forecasting LLMs (Large Language Models a.k.a foundation models) with nnetsauce.MTS"
description: "Benchmarking nnetsauce.MTS against foundation models and statistical models"
date: 2024-11-24
categories: [R, Python, Forecasting, nnetsauce]
comments: true
---

In this post:
- I show how to use python package [`nnetsauce`](https://github.com/Techtonique/nnetsauce) alongside `sktime` for univariate and multivariate time series (probabilistic) forecasting. `sktime` useful for benchmarking a plethora of (probabilistic) forecasts with a unified interface
- I benchmark [`nnetsauce.MTS`](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)'s _armada_ (**not with `sktime`**) against foundation models ("LLMs", Amazon's _Chronos_, IBM's TinyTimeMixer) and _statistical_ models (ARIMA, ETS, Theta, VAR, VECM). Regarding the LLMs: If I'm not doing it well (I just _plugged and played_), do not hesitate to reach out.  

A link to the notebook is availble at the end of this post.

**Contents**

- [0 - Install `nnetsauce` and `mlsauce`](#0---install-nnetsauce-and-mlsauce)
  - [1 - Example 1: using `nnetsauce` with sktime](#1---example-1-using-nnetsauce-with-sktime)
    - [1 - 1 Point forecast with `nnetsauce`'s `sktime` interface](#1---1-point-forecast-with-nnetsauces-sktime-interface)
    - [1 - 2 Probabilistic forecasting with `nnetsauce`'s `sktime` interface](#1---2-probabilistic-forecasting-with-nnetsauces-sktime-interface)
- [2 - sktime foundation models and nnetsauce](#2---sktime-foundation-models-and-nnetsauce)
    - [2 - 1 - Example1 on macroeconomic data](#2---1---example1-on-macroeconomic-data)
    - [2 - 2 - Example2 on antidiabetic drug sales](#2---2---example2-on-antidiabetic-drug-sales)

# 0 - Install `nnetsauce` and `mlsauce`


```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```


```python
!pip install nnetsauce
```

## 1 - Example 1: using [`nnetsauce`]() with sktime


```python
!pip install git+https://github.com/thierrymoudiki/sktime.git --upgrade --no-cache-dir
```


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn import linear_model
from statsmodels.tsa.base.datetools import dates_from_str
from sktime.forecasting.nnetsaucemts import NnetsauceMTS
```

Macroeconomic data


```python
# some example data
mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[["year", "quarter"]].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
mdata = mdata[["realgovt", "tbilrate", "cpi"]]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()
data2 = mdata

n = data.shape[0]
max_idx_train = np.floor(n * 0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index, :]
print(df_train.tail())
df_test = data.iloc[testing_index, :]
print(df_test.head())
```

                realgovt  tbilrate       cpi
    2003-06-30  0.047086 -0.171850  0.002726
    2003-09-30  0.000981 -0.021053  0.006511
    2003-12-31  0.007267 -0.043485  0.007543
    2004-03-31  0.012745  0.043485  0.005887
    2004-06-30  0.005669  0.252496  0.009031
                realgovt  tbilrate       cpi
    2004-09-30  0.017200  0.297960  0.008950
    2004-12-31 -0.012387  0.299877  0.005227
    2005-03-31  0.004160  0.201084  0.010374
    2005-06-30  0.000966  0.112399  0.004633
    2005-09-30  0.023120  0.156521  0.022849



```python
n2 = data.shape[0]
max_idx_train2 = np.floor(n2 * 0.9)
training_index2 = np.arange(0, max_idx_train2)
testing_index2 = np.arange(max_idx_train2, n2)
df_train2 = data2.iloc[training_index2, :]
print(df_train2.tail())
df_test2 = data2.iloc[testing_index, :]
print(df_test.head())
```

                realgovt  tbilrate    cpi
    2003-03-31   800.196      1.14  183.2
    2003-06-30   838.775      0.96  183.7
    2003-09-30   839.598      0.94  184.9
    2003-12-31   845.722      0.90  186.3
    2004-03-31   856.570      0.94  187.4
                realgovt  tbilrate       cpi
    2004-09-30  0.017200  0.297960  0.008950
    2004-12-31 -0.012387  0.299877  0.005227
    2005-03-31  0.004160  0.201084  0.010374
    2005-06-30  0.000966  0.112399  0.004633
    2005-09-30  0.023120  0.156521  0.022849


### 1 - 1 Point forecast with `nnetsauce`'s `sktime` interface


```python
regr = linear_model.RidgeCV()

obj_MTS = NnetsauceMTS(regr, lags = 20, n_hidden_features=7, n_clusters=2,
                       type_pi="scp2-block-bootstrap",
                       kernel="gaussian",
                       replications=250)
obj_MTS.fit(df_train)

res = obj_MTS.predict(fh=[i for i in range(1, 20)])

print(res)
```

    100%|██████████| 3/3 [00:00<00:00, 121.23it/s]

                realgovt  tbilrate  cpi
    date                               
    2004-09-30      0.01      0.02 0.02
    2004-12-31      0.01      0.04 0.02
    2005-03-31      0.00      0.03 0.02
    2005-06-30      0.01      0.02 0.02
    2005-09-30      0.01      0.01 0.02
    2005-12-31      0.01      0.04 0.03
    2006-03-31      0.01      0.03 0.02
    2006-06-30      0.01      0.03 0.03
    2006-09-30      0.01      0.02 0.02
    2006-12-31      0.01      0.00 0.02
    2007-03-31      0.01     -0.00 0.02
    2007-06-30      0.01      0.02 0.02
    2007-09-30      0.01     -0.01 0.02
    2007-12-31      0.00     -0.01 0.02
    2008-03-31      0.01      0.02 0.02
    2008-06-30      0.00      0.03 0.02
    2008-09-30      0.00      0.03 0.02
    2008-12-31      0.00      0.01 0.02
    2009-03-31      0.00     -0.00 0.02


    


### 1 - 2 Probabilistic forecasting with `nnetsauce`'s `sktime` interface


```python
res = obj_MTS.predict_quantiles(fh=[i for i in range(1, 20)],
                                alpha=0.05)

print(res)
```

               realgovt tbilrate   cpi realgovt tbilrate  cpi
                   0.05     0.05  0.05     0.95     0.95 0.95
    date                                                     
    2004-09-30    -0.03    -0.47  0.01     0.07     0.44 0.04
    2004-12-31    -0.03    -0.31  0.01     0.06     0.66 0.04
    2005-03-31    -0.04    -0.41  0.00     0.06     0.65 0.04
    2005-06-30    -0.03    -0.43  0.01     0.06     0.46 0.04
    2005-09-30    -0.03    -0.44  0.01     0.06     0.37 0.04
    2005-12-31    -0.03    -0.30  0.01     0.06     0.46 0.04
    2006-03-31    -0.03    -0.40  0.01     0.06     0.46 0.04
    2006-06-30    -0.03    -0.40  0.01     0.06     0.46 0.04
    2006-09-30    -0.03    -0.31  0.01     0.06     0.66 0.04
    2006-12-31    -0.03    -0.44  0.00     0.05     0.61 0.04
    2007-03-31    -0.03    -0.44  0.01     0.05     0.37 0.04
    2007-06-30    -0.04    -0.31  0.01     0.05     0.37 0.04
    2007-09-30    -0.03    -0.31  0.01     0.06     0.36 0.04
    2007-12-31    -0.04    -0.44  0.01     0.05     0.59 0.04
    2008-03-31    -0.04    -0.47  0.00     0.07     0.43 0.04
    2008-06-30    -0.03    -0.31  0.01     0.06     0.66 0.04
    2008-09-30    -0.04    -0.41  0.00     0.06     0.65 0.04
    2008-12-31    -0.04    -0.44 -0.00     0.05     0.45 0.03
    2009-03-31    -0.04    -0.44 -0.00     0.05     0.36 0.03



```python
obj_MTS.fitter.plot(series="realgovt")
```

![xxx]({{base}}/images/2024-11-24/2024-11-24-image1.png){:class="img-responsive"}          


```python
obj_MTS.fitter.plot(series="cpi")
```


![xxx]({{base}}/images/2024-11-24/2024-11-24-image2.png){:class="img-responsive"}          
    


# 2 - sktime foundation models and nnetsauce

### 2 - 1 - Example1 on macroeconomic data


```python
# Do imports
import nnetsauce as ns
import mlsauce as ms
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.chronos import ChronosForecaster

# Initialise models
chronos = ChronosForecaster("amazon/chronos-t5-tiny")
ttm = TinyTimeMixerForecaster()
regr = linear_model.RidgeCV()
obj_MTS = NnetsauceMTS(regr, lags = 20, n_hidden_features=7, n_clusters=2,
                       type_pi="scp2-block-bootstrap",
                       kernel="gaussian",
                       replications=250)
regr2 = ms.GenericBoostingRegressor(regr)
obj_MTS2 = ns.MTS(obj=regr2)

# Fit
h = df_test.shape[0] + 1
chronos.fit(y=df_train, fh=range(1, h))
ttm.fit(y=df_train, fh=range(1, h))
obj_MTS.fit(y=df_train, fh=range(1, h))
obj_MTS2.fit(df_train)

# Predict
pred_chronos = chronos.predict(fh=[i for i in range(1, h)])
pred_ttm = ttm.predict(fh=[i for i in range(1, h)])
pred_MTS = obj_MTS.predict(fh=[i for i in range(1, h)])
pred_MTS2 = obj_MTS2.predict(h=h-1)
```


    config.json:   0%|          | 0.00/1.14k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/33.6M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/142 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/1.19k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/3.24M [00:00<?, ?B/s]


    100%|██████████| 3/3 [00:00<00:00, 17.87it/s]
      0%|          | 0/3 [00:00<?, ?it/s]
      0%|          | 0/100 [00:00<?, ?it/s][A
     22%|██▏       | 22/100 [00:00<00:00, 124.79it/s]
     33%|███▎      | 1/3 [00:00<00:00,  4.78it/s]
      0%|          | 0/22 [00:00<?, ?it/s][A
    100%|██████████| 22/22 [00:00<00:00, 120.57it/s]
     67%|██████▋   | 2/3 [00:00<00:00,  4.57it/s]
      0%|          | 0/22 [00:00<?, ?it/s][A
    100%|██████████| 22/22 [00:00<00:00, 121.13it/s]
    100%|██████████| 3/3 [00:00<00:00,  4.51it/s]



```python
pred_MTS2 = obj_MTS2.predict(h=h-1)
```


```python
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions.values - targets.values) ** 2).mean())

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def me(predictions, targets):
    return np.mean(predictions - targets)
```


```python
print("Chronos RMSE:", rmse(df_test, pred_chronos))
print("Chronos MAE:", mae(df_test, pred_chronos))
print("Chronos ME:", me(df_test, pred_chronos))

print("TinyTimeMixer RMSE:", rmse(df_test, pred_ttm))
print("TinyTimeMixer MAE:", mae(df_test, pred_ttm))
print("TinyTimeMixer ME:", me(df_test, pred_ttm))

print("NnetsauceMTS RMSE:", rmse(df_test, pred_MTS))
print("NnetsauceMTS MAE:", mae(df_test, pred_MTS))
print("NnetsauceMTS ME:", me(df_test, pred_MTS))
```

    Chronos RMSE: 0.3270528840422444
    Chronos MAE: 0.10750380038506846
    Chronos ME: -0.04182334654432299
    TinyTimeMixer RMSE: 0.3248244141056522
    TinyTimeMixer MAE: 0.11031492439459516
    TinyTimeMixer ME: -0.03476608913007449
    NnetsauceMTS RMSE: 0.320951903060047
    NnetsauceMTS MAE: 0.10903099364744497
    NnetsauceMTS ME: -0.0298461588803314


**Loop and leaderboard**


```python
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from tqdm import tqdm

results = []

results.append(["Chronos", rmse(df_test, pred_chronos), mae(df_test, pred_chronos), me(df_test, pred_chronos)])
results.append(["TinyTimeMixer", rmse(df_test, pred_ttm), mae(df_test, pred_ttm), me(df_test, pred_ttm)])
results.append(["NnetsauceMTS", rmse(df_test, pred_MTS), mae(df_test, pred_MTS), me(df_test, pred_MTS)])

# statistical models
for i, name in enumerate(["ARIMA", "ETS", "Theta", "VAR", "VECM"]):
  try:
    regr = ns.ClassicalMTS(model=name)
    regr.fit(df_train)
    X_pred = regr.predict(h=df_test.shape[0])
    results.append([name, rmse(df_test, X_pred.mean), mae(df_test, X_pred.mean), me(df_test, X_pred.mean)])
  except Exception:
    pass

for est in tqdm(all_estimators()):
  if (issubclass(est[1], RegressorMixin)):
    try:
      preds = ns.MTS(est[1](), verbose=0, show_progress=False).\
      fit(df_train).\
      predict(h=df_test.shape[0])
      results.append([est[0], rmse(df_test, preds), mae(df_test, preds), me(df_test, preds)])
    except Exception:
      pass
```

    100%|██████████| 206/206 [00:06<00:00, 29.81it/s]



```python
results_df = pd.DataFrame(results, columns=["model", "rmse", "mae", "me"])
```


```python
import pandas as pd

# Assuming 'results_df' is the DataFrame from the provided code
pd.options.display.float_format = '{:.5f}'.format
display(results_df.sort_values(by="rmse"))
```



  <div id="df-7b5127fb-9114-4766-9f9f-549f1cba2ca1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>BaggingRegressor</td>
      <td>0.30560</td>
      <td>0.10120</td>
      <td>-0.03167</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>0.31035</td>
      <td>0.11071</td>
      <td>-0.03677</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>0.32077</td>
      <td>0.11472</td>
      <td>-0.02632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>0.32095</td>
      <td>0.10903</td>
      <td>-0.02985</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TransformedTargetRegressor</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Ridge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>46</th>
      <td>RidgeCV</td>
      <td>0.32262</td>
      <td>0.10881</td>
      <td>-0.03281</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>0.32315</td>
      <td>0.10933</td>
      <td>-0.03600</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>0.32324</td>
      <td>0.10931</td>
      <td>-0.03319</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>0.32325</td>
      <td>0.10932</td>
      <td>-0.03322</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PLSRegression</td>
      <td>0.32429</td>
      <td>0.11177</td>
      <td>-0.03433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>0.32482</td>
      <td>0.11031</td>
      <td>-0.03477</td>
    </tr>
    <tr>
      <th>44</th>
      <td>RandomForestRegressor</td>
      <td>0.32501</td>
      <td>0.10802</td>
      <td>-0.03557</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CCA</td>
      <td>0.32521</td>
      <td>0.11041</td>
      <td>-0.03330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VAR</td>
      <td>0.32597</td>
      <td>0.10998</td>
      <td>-0.03593</td>
    </tr>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>0.32607</td>
      <td>0.10997</td>
      <td>-0.03581</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>0.32629</td>
      <td>0.11244</td>
      <td>-0.02872</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>0.32643</td>
      <td>0.11203</td>
      <td>-0.03014</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>0.32657</td>
      <td>0.11299</td>
      <td>-0.02783</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TweedieRegressor</td>
      <td>0.32662</td>
      <td>0.11159</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SGDRegressor</td>
      <td>0.32665</td>
      <td>0.11173</td>
      <td>-0.03125</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BayesianRidge</td>
      <td>0.32666</td>
      <td>0.11163</td>
      <td>-0.03166</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>0.32666</td>
      <td>0.11333</td>
      <td>-0.02833</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoostRegressor</td>
      <td>0.32669</td>
      <td>0.11199</td>
      <td>-0.03032</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>0.32672</td>
      <td>0.11153</td>
      <td>-0.03218</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>0.32688</td>
      <td>0.11144</td>
      <td>-0.03286</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>0.32690</td>
      <td>0.11143</td>
      <td>-0.03293</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>0.32696</td>
      <td>0.11143</td>
      <td>-0.03317</td>
    </tr>
    <tr>
      <th>42</th>
      <td>QuantileRegressor</td>
      <td>0.32700</td>
      <td>0.11163</td>
      <td>-0.03203</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>0.32705</td>
      <td>0.10750</td>
      <td>-0.04182</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARDRegression</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PassiveAggressiveRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>0.32748</td>
      <td>0.11165</td>
      <td>-0.03496</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SVR</td>
      <td>0.32749</td>
      <td>0.11168</td>
      <td>-0.03670</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TheilSenRegressor</td>
      <td>0.32770</td>
      <td>0.11191</td>
      <td>-0.03850</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VECM</td>
      <td>0.33830</td>
      <td>0.11309</td>
      <td>-0.05521</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>0.34038</td>
      <td>0.12121</td>
      <td>-0.03416</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>0.34571</td>
      <td>0.15000</td>
      <td>0.01448</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>0.37064</td>
      <td>0.13648</td>
      <td>-0.09831</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSCanonical</td>
      <td>831.49831</td>
      <td>269.94059</td>
      <td>-269.93944</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RANSACRegressor</td>
      <td>27213.86285</td>
      <td>6727.75305</td>
      <td>-6727.75305</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>2640594432241.43555</td>
      <td>711026457838.45886</td>
      <td>540623410390.19354</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7b5127fb-9114-4766-9f9f-549f1cba2ca1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7b5127fb-9114-4766-9f9f-549f1cba2ca1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7b5127fb-9114-4766-9f9f-549f1cba2ca1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b66034eb-d692-4924-bcc5-46ccd8b76081">
  <button class="colab-df-quickchart" onclick="quickchart('df-b66034eb-d692-4924-bcc5-46ccd8b76081')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b66034eb-d692-4924-bcc5-46ccd8b76081 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




```python
display(results_df.sort_values(by="mae"))
```



  <div id="df-98e24ee6-374e-429d-8794-ec44429ae258" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>BaggingRegressor</td>
      <td>0.30560</td>
      <td>0.10120</td>
      <td>-0.03167</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>0.32705</td>
      <td>0.10750</td>
      <td>-0.04182</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TransformedTargetRegressor</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>44</th>
      <td>RandomForestRegressor</td>
      <td>0.32501</td>
      <td>0.10802</td>
      <td>-0.03557</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Ridge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>46</th>
      <td>RidgeCV</td>
      <td>0.32262</td>
      <td>0.10881</td>
      <td>-0.03281</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>0.32095</td>
      <td>0.10903</td>
      <td>-0.02985</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>0.32324</td>
      <td>0.10931</td>
      <td>-0.03319</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>0.32325</td>
      <td>0.10932</td>
      <td>-0.03322</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>0.32315</td>
      <td>0.10933</td>
      <td>-0.03600</td>
    </tr>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>0.32607</td>
      <td>0.10997</td>
      <td>-0.03581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VAR</td>
      <td>0.32597</td>
      <td>0.10998</td>
      <td>-0.03593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>0.32482</td>
      <td>0.11031</td>
      <td>-0.03477</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CCA</td>
      <td>0.32521</td>
      <td>0.11041</td>
      <td>-0.03330</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>0.31035</td>
      <td>0.11071</td>
      <td>-0.03677</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>0.32696</td>
      <td>0.11143</td>
      <td>-0.03317</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>0.32690</td>
      <td>0.11143</td>
      <td>-0.03293</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>0.32688</td>
      <td>0.11144</td>
      <td>-0.03286</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PassiveAggressiveRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARDRegression</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>0.32672</td>
      <td>0.11153</td>
      <td>-0.03218</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TweedieRegressor</td>
      <td>0.32662</td>
      <td>0.11159</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BayesianRidge</td>
      <td>0.32666</td>
      <td>0.11163</td>
      <td>-0.03166</td>
    </tr>
    <tr>
      <th>42</th>
      <td>QuantileRegressor</td>
      <td>0.32700</td>
      <td>0.11163</td>
      <td>-0.03203</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>0.32748</td>
      <td>0.11165</td>
      <td>-0.03496</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SVR</td>
      <td>0.32749</td>
      <td>0.11168</td>
      <td>-0.03670</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SGDRegressor</td>
      <td>0.32665</td>
      <td>0.11173</td>
      <td>-0.03125</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PLSRegression</td>
      <td>0.32429</td>
      <td>0.11177</td>
      <td>-0.03433</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TheilSenRegressor</td>
      <td>0.32770</td>
      <td>0.11191</td>
      <td>-0.03850</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoostRegressor</td>
      <td>0.32669</td>
      <td>0.11199</td>
      <td>-0.03032</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>0.32643</td>
      <td>0.11203</td>
      <td>-0.03014</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>0.32629</td>
      <td>0.11244</td>
      <td>-0.02872</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>0.32657</td>
      <td>0.11299</td>
      <td>-0.02783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VECM</td>
      <td>0.33830</td>
      <td>0.11309</td>
      <td>-0.05521</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>0.32666</td>
      <td>0.11333</td>
      <td>-0.02833</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>0.32077</td>
      <td>0.11472</td>
      <td>-0.02632</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>0.34038</td>
      <td>0.12121</td>
      <td>-0.03416</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>0.37064</td>
      <td>0.13648</td>
      <td>-0.09831</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>0.34571</td>
      <td>0.15000</td>
      <td>0.01448</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSCanonical</td>
      <td>831.49831</td>
      <td>269.94059</td>
      <td>-269.93944</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RANSACRegressor</td>
      <td>27213.86285</td>
      <td>6727.75305</td>
      <td>-6727.75305</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>2640594432241.43555</td>
      <td>711026457838.45886</td>
      <td>540623410390.19354</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-98e24ee6-374e-429d-8794-ec44429ae258')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-98e24ee6-374e-429d-8794-ec44429ae258 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-98e24ee6-374e-429d-8794-ec44429ae258');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5c070a67-aacc-4bfe-b5c2-d0504a7cc5e8">
  <button class="colab-df-quickchart" onclick="quickchart('df-5c070a67-aacc-4bfe-b5c2-d0504a7cc5e8')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5c070a67-aacc-4bfe-b5c2-d0504a7cc5e8 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




```python
display(results_df.sort_values(by="me", ascending=False))
```



  <div id="df-12366b00-c072-4e0b-b36c-293c3be4d6b0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>2640594432241.43555</td>
      <td>711026457838.45886</td>
      <td>540623410390.19354</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>0.34571</td>
      <td>0.15000</td>
      <td>0.01448</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>0.32077</td>
      <td>0.11472</td>
      <td>-0.02632</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>0.32657</td>
      <td>0.11299</td>
      <td>-0.02783</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>0.32666</td>
      <td>0.11333</td>
      <td>-0.02833</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>0.32629</td>
      <td>0.11244</td>
      <td>-0.02872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>0.32095</td>
      <td>0.10903</td>
      <td>-0.02985</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>0.32643</td>
      <td>0.11203</td>
      <td>-0.03014</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoostRegressor</td>
      <td>0.32669</td>
      <td>0.11199</td>
      <td>-0.03032</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SGDRegressor</td>
      <td>0.32665</td>
      <td>0.11173</td>
      <td>-0.03125</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BayesianRidge</td>
      <td>0.32666</td>
      <td>0.11163</td>
      <td>-0.03166</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BaggingRegressor</td>
      <td>0.30560</td>
      <td>0.10120</td>
      <td>-0.03167</td>
    </tr>
    <tr>
      <th>42</th>
      <td>QuantileRegressor</td>
      <td>0.32700</td>
      <td>0.11163</td>
      <td>-0.03203</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>0.32672</td>
      <td>0.11153</td>
      <td>-0.03218</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>0.32676</td>
      <td>0.11152</td>
      <td>-0.03225</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TweedieRegressor</td>
      <td>0.32662</td>
      <td>0.11159</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Ridge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>0.32168</td>
      <td>0.10821</td>
      <td>-0.03250</td>
    </tr>
    <tr>
      <th>46</th>
      <td>RidgeCV</td>
      <td>0.32262</td>
      <td>0.10881</td>
      <td>-0.03281</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>0.32688</td>
      <td>0.11144</td>
      <td>-0.03286</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>0.32690</td>
      <td>0.11143</td>
      <td>-0.03293</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>0.32696</td>
      <td>0.11143</td>
      <td>-0.03317</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>0.32324</td>
      <td>0.10931</td>
      <td>-0.03319</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>0.32325</td>
      <td>0.10932</td>
      <td>-0.03322</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CCA</td>
      <td>0.32521</td>
      <td>0.11041</td>
      <td>-0.03330</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TransformedTargetRegressor</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>0.32154</td>
      <td>0.10754</td>
      <td>-0.03336</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>0.34038</td>
      <td>0.12121</td>
      <td>-0.03416</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARDRegression</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PassiveAggressiveRegressor</td>
      <td>0.32723</td>
      <td>0.11147</td>
      <td>-0.03430</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PLSRegression</td>
      <td>0.32429</td>
      <td>0.11177</td>
      <td>-0.03433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>0.32482</td>
      <td>0.11031</td>
      <td>-0.03477</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>0.32748</td>
      <td>0.11165</td>
      <td>-0.03496</td>
    </tr>
    <tr>
      <th>44</th>
      <td>RandomForestRegressor</td>
      <td>0.32501</td>
      <td>0.10802</td>
      <td>-0.03557</td>
    </tr>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>0.32607</td>
      <td>0.10997</td>
      <td>-0.03581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VAR</td>
      <td>0.32597</td>
      <td>0.10998</td>
      <td>-0.03593</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>0.32315</td>
      <td>0.10933</td>
      <td>-0.03600</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SVR</td>
      <td>0.32749</td>
      <td>0.11168</td>
      <td>-0.03670</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>0.31035</td>
      <td>0.11071</td>
      <td>-0.03677</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TheilSenRegressor</td>
      <td>0.32770</td>
      <td>0.11191</td>
      <td>-0.03850</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>0.32705</td>
      <td>0.10750</td>
      <td>-0.04182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VECM</td>
      <td>0.33830</td>
      <td>0.11309</td>
      <td>-0.05521</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>0.37064</td>
      <td>0.13648</td>
      <td>-0.09831</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSCanonical</td>
      <td>831.49831</td>
      <td>269.94059</td>
      <td>-269.93944</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RANSACRegressor</td>
      <td>27213.86285</td>
      <td>6727.75305</td>
      <td>-6727.75305</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-12366b00-c072-4e0b-b36c-293c3be4d6b0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-12366b00-c072-4e0b-b36c-293c3be4d6b0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-12366b00-c072-4e0b-b36c-293c3be4d6b0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b5247b26-224b-4ac4-8e33-fea2d92963f6">
  <button class="colab-df-quickchart" onclick="quickchart('df-b5247b26-224b-4ac4-8e33-fea2d92963f6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b5247b26-224b-4ac4-8e33-fea2d92963f6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



### 2 - 2 - Example2 on antidiabetic drug sales


```python
url = "https://raw.githubusercontent.com/Techtonique/"
url += "datasets/main/time_series/univariate/"
url += "a10.csv"
data = pd.read_csv(url)
data.index = pd.DatetimeIndex(data.date) # must have
data.drop(columns=['date'], inplace=True)
```


```python
n = data.shape[0]
max_idx_train = np.floor(n * 0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index, :]
print(df_train.tail())
df_test = data.iloc[testing_index, :]
print(df_test.head())
```

                  value
    date               
    2006-05-01 17.78306
    2006-06-01 16.29160
    2006-07-01 16.98028
    2006-08-01 18.61219
    2006-09-01 16.62334
                  value
    date               
    2006-10-01 21.43024
    2006-11-01 23.57552
    2006-12-01 23.33421
    2007-01-01 28.03838
    2007-02-01 16.76387



```python
df_train.plot()
```




    <Axes: xlabel='date'>




    
![xxx]({{base}}/images/2024-11-24/2024-11-24-image3.png){:class="img-responsive"}          

    



```python
# Do imports
import nnetsauce as ns
import mlsauce as ms
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.chronos import ChronosForecaster

# Initialise models
chronos = ChronosForecaster("amazon/chronos-t5-tiny")
ttm = TinyTimeMixerForecaster()
regr = linear_model.RidgeCV()
regr2 = ms.GenericBoostingRegressor(regr)
obj_MTS2 = ns.MTS(obj=regr2)

# Fit
h = df_test.shape[0] + 1
chronos.fit(y=df_train, fh=range(1, h))
ttm.fit(y=df_train, fh=range(1, h))
obj_MTS.fit(y=df_train, fh=range(1, h))
obj_MTS2.fit(df_train)

# Predict
pred_chronos = chronos.predict(fh=[i for i in range(1, h)])
pred_ttm = ttm.predict(fh=[i for i in range(1, h)])
pred_MTS2 = obj_MTS2.predict(h=h-1)
```

    100%|██████████| 100/100 [00:00<00:00, 327.48it/s]



```python
print("Chronos RMSE:", rmse(df_test, pred_chronos))
print("Chronos MAE:", mae(df_test, pred_chronos))
print("Chronos ME:", me(df_test, pred_chronos))

print("TinyTimeMixer RMSE:", rmse(df_test, pred_ttm))
print("TinyTimeMixer MAE:", mae(df_test, pred_ttm))
print("TinyTimeMixer ME:", me(df_test, pred_ttm))

print("NnetsauceMTS RMSE:", rmse(df_test, pred_MTS2))
print("NnetsauceMTS MAE:", mae(df_test, pred_MTS2))
print("NnetsauceMTS ME:", me(df_test, pred_MTS2))
```

    Chronos RMSE: 4.668968548036785
    Chronos MAE: 4.351116104707961
    Chronos ME: 3.249815388190104
    TinyTimeMixer RMSE: 6.6643723494125355
    TinyTimeMixer MAE: 5.881109575050688
    TinyTimeMixer ME: 5.876180504854445
    NnetsauceMTS RMSE: 7.449155397775451
    NnetsauceMTS MAE: 6.717429133757641
    NnetsauceMTS ME: 6.717429133757641



```python
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from tqdm import tqdm

results = []

# LLMs and sktime
results.append(["Chronos", rmse(df_test, pred_chronos), mae(df_test, pred_chronos), me(df_test, pred_chronos)])
results.append(["TinyTimeMixer", rmse(df_test, pred_ttm), mae(df_test, pred_ttm), me(df_test, pred_ttm)])
results.append(["NnetsauceMTS", rmse(df_test, pred_MTS), mae(df_test, pred_MTS), me(df_test, pred_MTS)])

# statistical models
for i, name in enumerate(["ARIMA", "ETS", "Theta", "VAR", "VECM"]):
  try:
    regr = ns.ClassicalMTS(model=name)
    regr.fit(df_train)
    X_pred = regr.predict(h=df_test.shape[0])
    results.append([name, rmse(df_test, X_pred.mean), mae(df_test, X_pred.mean), me(df_test, X_pred.mean)])
  except Exception:
    pass

for est in tqdm(all_estimators()):
  if (issubclass(est[1], RegressorMixin)):
    try:
      preds = ns.MTS(est[1](), lags=20, verbose=0, show_progress=False).\
      fit(df_train).\
      predict(h=df_test.shape[0])
      results.append([est[0], rmse(df_test, preds), mae(df_test, preds), me(df_test, preds)])
    except Exception:
      pass
```

    100%|██████████| 206/206 [00:05<00:00, 35.10it/s]



```python
results_df = pd.DataFrame(results, columns=["model", "rmse", "mae", "me"])
```


```python
import pandas as pd

# Assuming 'results_df' is the DataFrame from the provided code
pd.options.display.float_format = '{:.5f}'.format
display(results_df.sort_values(by="rmse"))
```



  <div id="df-e3268446-e766-46ff-b59f-d4f3a49beb83" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>3.04709</td>
      <td>2.69552</td>
      <td>2.59578</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RandomForestRegressor</td>
      <td>3.10382</td>
      <td>2.55888</td>
      <td>2.05839</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>3.12691</td>
      <td>2.68994</td>
      <td>2.67753</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SGDRegressor</td>
      <td>3.22877</td>
      <td>2.83957</td>
      <td>2.83957</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>3.26890</td>
      <td>2.73450</td>
      <td>2.15218</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BaggingRegressor</td>
      <td>3.27914</td>
      <td>2.83882</td>
      <td>2.50350</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>3.30520</td>
      <td>2.89079</td>
      <td>2.71086</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ARDRegression</td>
      <td>3.45668</td>
      <td>3.02580</td>
      <td>3.02408</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>3.49769</td>
      <td>2.77511</td>
      <td>2.41475</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TheilSenRegressor</td>
      <td>3.62027</td>
      <td>3.25754</td>
      <td>3.25754</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>3.65081</td>
      <td>2.91033</td>
      <td>1.08975</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AdaBoostRegressor</td>
      <td>3.69998</td>
      <td>3.40106</td>
      <td>2.85192</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>3.71916</td>
      <td>3.29246</td>
      <td>3.29246</td>
    </tr>
    <tr>
      <th>45</th>
      <td>RidgeCV</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Ridge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>42</th>
      <td>RANSACRegressor</td>
      <td>3.82726</td>
      <td>3.45743</td>
      <td>3.45743</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BayesianRidge</td>
      <td>3.92099</td>
      <td>3.51565</td>
      <td>3.51565</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TransformedTargetRegressor</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>4.15261</td>
      <td>3.64182</td>
      <td>3.64182</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSRegression</td>
      <td>4.53423</td>
      <td>3.99218</td>
      <td>3.95978</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Theta</td>
      <td>4.57439</td>
      <td>4.24487</td>
      <td>4.24487</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>4.58340</td>
      <td>4.14143</td>
      <td>4.14143</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>4.61862</td>
      <td>4.26089</td>
      <td>4.26089</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TweedieRegressor</td>
      <td>4.65986</td>
      <td>3.93009</td>
      <td>3.53582</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>4.66897</td>
      <td>4.35112</td>
      <td>3.24982</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>4.96950</td>
      <td>4.55567</td>
      <td>4.55567</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>5.41091</td>
      <td>4.67558</td>
      <td>4.60438</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>5.82768</td>
      <td>5.06658</td>
      <td>5.02970</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PassiveAggressiveRegressor</td>
      <td>5.89234</td>
      <td>4.31086</td>
      <td>-4.31086</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ETS</td>
      <td>6.18780</td>
      <td>5.37221</td>
      <td>5.28513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>6.66437</td>
      <td>5.88111</td>
      <td>5.87618</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SVR</td>
      <td>6.75271</td>
      <td>6.06309</td>
      <td>6.06309</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>6.76407</td>
      <td>6.06742</td>
      <td>6.06742</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>12.55786</td>
      <td>12.11319</td>
      <td>12.11319</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>12.71627</td>
      <td>12.30232</td>
      <td>12.30232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARIMA</td>
      <td>13.37258</td>
      <td>12.97959</td>
      <td>12.97959</td>
    </tr>
    <tr>
      <th>41</th>
      <td>QuantileRegressor</td>
      <td>13.47588</td>
      <td>13.08599</td>
      <td>13.08599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>22.57186</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e3268446-e766-46ff-b59f-d4f3a49beb83')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e3268446-e766-46ff-b59f-d4f3a49beb83 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e3268446-e766-46ff-b59f-d4f3a49beb83');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-96bb5d77-7875-405c-8cc2-586659af5338">
  <button class="colab-df-quickchart" onclick="quickchart('df-96bb5d77-7875-405c-8cc2-586659af5338')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-96bb5d77-7875-405c-8cc2-586659af5338 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




```python
display(results_df.sort_values(by="mae"))
```



  <div id="df-4f16b036-c695-46bf-bc59-e2e154ded8e5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>RandomForestRegressor</td>
      <td>3.10382</td>
      <td>2.55888</td>
      <td>2.05839</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>3.12691</td>
      <td>2.68994</td>
      <td>2.67753</td>
    </tr>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>3.04709</td>
      <td>2.69552</td>
      <td>2.59578</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>3.26890</td>
      <td>2.73450</td>
      <td>2.15218</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>3.49769</td>
      <td>2.77511</td>
      <td>2.41475</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BaggingRegressor</td>
      <td>3.27914</td>
      <td>2.83882</td>
      <td>2.50350</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SGDRegressor</td>
      <td>3.22877</td>
      <td>2.83957</td>
      <td>2.83957</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>3.30520</td>
      <td>2.89079</td>
      <td>2.71086</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>3.65081</td>
      <td>2.91033</td>
      <td>1.08975</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ARDRegression</td>
      <td>3.45668</td>
      <td>3.02580</td>
      <td>3.02408</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TheilSenRegressor</td>
      <td>3.62027</td>
      <td>3.25754</td>
      <td>3.25754</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>3.71916</td>
      <td>3.29246</td>
      <td>3.29246</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AdaBoostRegressor</td>
      <td>3.69998</td>
      <td>3.40106</td>
      <td>2.85192</td>
    </tr>
    <tr>
      <th>45</th>
      <td>RidgeCV</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Ridge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>42</th>
      <td>RANSACRegressor</td>
      <td>3.82726</td>
      <td>3.45743</td>
      <td>3.45743</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BayesianRidge</td>
      <td>3.92099</td>
      <td>3.51565</td>
      <td>3.51565</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>4.15261</td>
      <td>3.64182</td>
      <td>3.64182</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TransformedTargetRegressor</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TweedieRegressor</td>
      <td>4.65986</td>
      <td>3.93009</td>
      <td>3.53582</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSRegression</td>
      <td>4.53423</td>
      <td>3.99218</td>
      <td>3.95978</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>4.58340</td>
      <td>4.14143</td>
      <td>4.14143</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Theta</td>
      <td>4.57439</td>
      <td>4.24487</td>
      <td>4.24487</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>4.61862</td>
      <td>4.26089</td>
      <td>4.26089</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PassiveAggressiveRegressor</td>
      <td>5.89234</td>
      <td>4.31086</td>
      <td>-4.31086</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>4.66897</td>
      <td>4.35112</td>
      <td>3.24982</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>4.96950</td>
      <td>4.55567</td>
      <td>4.55567</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>5.41091</td>
      <td>4.67558</td>
      <td>4.60438</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>5.82768</td>
      <td>5.06658</td>
      <td>5.02970</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ETS</td>
      <td>6.18780</td>
      <td>5.37221</td>
      <td>5.28513</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>6.66437</td>
      <td>5.88111</td>
      <td>5.87618</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SVR</td>
      <td>6.75271</td>
      <td>6.06309</td>
      <td>6.06309</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>6.76407</td>
      <td>6.06742</td>
      <td>6.06742</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>12.55786</td>
      <td>12.11319</td>
      <td>12.11319</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>12.71627</td>
      <td>12.30232</td>
      <td>12.30232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARIMA</td>
      <td>13.37258</td>
      <td>12.97959</td>
      <td>12.97959</td>
    </tr>
    <tr>
      <th>41</th>
      <td>QuantileRegressor</td>
      <td>13.47588</td>
      <td>13.08599</td>
      <td>13.08599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>22.57186</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4f16b036-c695-46bf-bc59-e2e154ded8e5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4f16b036-c695-46bf-bc59-e2e154ded8e5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4f16b036-c695-46bf-bc59-e2e154ded8e5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c0048e37-1094-486c-8838-ed0ef0eead91">
  <button class="colab-df-quickchart" onclick="quickchart('df-c0048e37-1094-486c-8838-ed0ef0eead91')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c0048e37-1094-486c-8838-ed0ef0eead91 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




```python
display(results_df.sort_values(by="me"))
```



  <div id="df-b4b7f844-1c6f-4cd2-94eb-fc7e6b23853d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mae</th>
      <th>me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>PassiveAggressiveRegressor</td>
      <td>5.89234</td>
      <td>4.31086</td>
      <td>-4.31086</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MLPRegressor</td>
      <td>3.65081</td>
      <td>2.91033</td>
      <td>1.08975</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RandomForestRegressor</td>
      <td>3.10382</td>
      <td>2.55888</td>
      <td>2.05839</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DecisionTreeRegressor</td>
      <td>3.26890</td>
      <td>2.73450</td>
      <td>2.15218</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ExtraTreeRegressor</td>
      <td>3.49769</td>
      <td>2.77511</td>
      <td>2.41475</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BaggingRegressor</td>
      <td>3.27914</td>
      <td>2.83882</td>
      <td>2.50350</td>
    </tr>
    <tr>
      <th>37</th>
      <td>OrthogonalMatchingPursuit</td>
      <td>3.04709</td>
      <td>2.69552</td>
      <td>2.59578</td>
    </tr>
    <tr>
      <th>38</th>
      <td>OrthogonalMatchingPursuitCV</td>
      <td>3.12691</td>
      <td>2.68994</td>
      <td>2.67753</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ExtraTreesRegressor</td>
      <td>3.30520</td>
      <td>2.89079</td>
      <td>2.71086</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MultiTaskLassoCV</td>
      <td>3.14258</td>
      <td>2.78649</td>
      <td>2.71682</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LassoLarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LarsCV</td>
      <td>3.14308</td>
      <td>2.78698</td>
      <td>2.71755</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MultiTaskElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ElasticNetCV</td>
      <td>3.17701</td>
      <td>2.84284</td>
      <td>2.77308</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SGDRegressor</td>
      <td>3.22877</td>
      <td>2.83957</td>
      <td>2.83957</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AdaBoostRegressor</td>
      <td>3.69998</td>
      <td>3.40106</td>
      <td>2.85192</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ARDRegression</td>
      <td>3.45668</td>
      <td>3.02580</td>
      <td>3.02408</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>4.66897</td>
      <td>4.35112</td>
      <td>3.24982</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TheilSenRegressor</td>
      <td>3.62027</td>
      <td>3.25754</td>
      <td>3.25754</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LassoLarsIC</td>
      <td>3.71916</td>
      <td>3.29246</td>
      <td>3.29246</td>
    </tr>
    <tr>
      <th>45</th>
      <td>RidgeCV</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KernelRidge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Ridge</td>
      <td>3.77950</td>
      <td>3.40210</td>
      <td>3.40210</td>
    </tr>
    <tr>
      <th>42</th>
      <td>RANSACRegressor</td>
      <td>3.82726</td>
      <td>3.45743</td>
      <td>3.45743</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BayesianRidge</td>
      <td>3.92099</td>
      <td>3.51565</td>
      <td>3.51565</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TweedieRegressor</td>
      <td>4.65986</td>
      <td>3.93009</td>
      <td>3.53582</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lars</td>
      <td>4.15261</td>
      <td>3.64182</td>
      <td>3.64182</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LinearRegression</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TransformedTargetRegressor</td>
      <td>4.09708</td>
      <td>3.65104</td>
      <td>3.65104</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PLSRegression</td>
      <td>4.53423</td>
      <td>3.99218</td>
      <td>3.95978</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GradientBoostingRegressor</td>
      <td>4.58340</td>
      <td>4.14143</td>
      <td>4.14143</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Theta</td>
      <td>4.57439</td>
      <td>4.24487</td>
      <td>4.24487</td>
    </tr>
    <tr>
      <th>30</th>
      <td>LinearSVR</td>
      <td>4.61862</td>
      <td>4.26089</td>
      <td>4.26089</td>
    </tr>
    <tr>
      <th>19</th>
      <td>HuberRegressor</td>
      <td>4.96950</td>
      <td>4.55567</td>
      <td>4.55567</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HistGradientBoostingRegressor</td>
      <td>5.41091</td>
      <td>4.67558</td>
      <td>4.60438</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KNeighborsRegressor</td>
      <td>5.82768</td>
      <td>5.06658</td>
      <td>5.02970</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MultiTaskElasticNet</td>
      <td>5.85659</td>
      <td>5.10931</td>
      <td>5.10931</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ETS</td>
      <td>6.18780</td>
      <td>5.37221</td>
      <td>5.28513</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Lasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MultiTaskLasso</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LassoLars</td>
      <td>5.94227</td>
      <td>5.63437</td>
      <td>5.63437</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>6.66437</td>
      <td>5.88111</td>
      <td>5.87618</td>
    </tr>
    <tr>
      <th>47</th>
      <td>SVR</td>
      <td>6.75271</td>
      <td>6.06309</td>
      <td>6.06309</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NuSVR</td>
      <td>6.76407</td>
      <td>6.06742</td>
      <td>6.06742</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GaussianProcessRegressor</td>
      <td>12.55786</td>
      <td>12.11319</td>
      <td>12.11319</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DummyRegressor</td>
      <td>12.71627</td>
      <td>12.30232</td>
      <td>12.30232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARIMA</td>
      <td>13.37258</td>
      <td>12.97959</td>
      <td>12.97959</td>
    </tr>
    <tr>
      <th>41</th>
      <td>QuantileRegressor</td>
      <td>13.47588</td>
      <td>13.08599</td>
      <td>13.08599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NnetsauceMTS</td>
      <td>22.57186</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b4b7f844-1c6f-4cd2-94eb-fc7e6b23853d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b4b7f844-1c6f-4cd2-94eb-fc7e6b23853d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b4b7f844-1c6f-4cd2-94eb-fc7e6b23853d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f62bb7e9-75ab-4ab6-b87b-affe826ad1ac">
  <button class="colab-df-quickchart" onclick="quickchart('df-f62bb7e9-75ab-4ab6-b87b-affe826ad1ac')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f62bb7e9-75ab-4ab6-b87b-affe826ad1ac button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>

<p style="color: darkgreen;">PS: <b>remember to give a try to <a href="https://www.techtonique.net/">Techtonique web app</a></b>, a tool designed to help you make informed, data-driven decisions using Mathematics, Statistics, Machine Learning, and Data Visualization</p>


<a target="_blank" href="https://colab.research.google.com/github/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_2024-11-24_nnetsauce-sktime-LLM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
