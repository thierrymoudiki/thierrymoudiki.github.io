---
layout: post
title: "You can beat Forecasting LLMs (Large Language Models a.k.a foundation models) with nnetsauce.MTS Pt.2: Generic Gradient Boosting"
description: "Benchmarking nnetsauce.MTS against foundation models and statistical models Pt.2: Generic Gradient Boosting"
date: 2024-12-01
categories: Python
comments: true
---

In this post I benchmark [`nnetsauce.MTS`](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)'s _armada_ of base models against foundation models ("LLMs", Amazon's [_Chronos_](https://openreview.net/pdf?id=gerNCVqqtR), IBM's [_TinyTimeMixer_](https://arxiv.org/pdf/2401.03955)) and _statistical_ models. Regarding the LLMs: If I'm not doing it well (I just _plugged and played_), do not hesitate to reach out.  

The _armada_ is [now](https://thierrymoudiki.github.io/blog/2024/11/24/r/python/forecasting/nnetsauce/nnetsauce-sktime-LLM) made of Generic Gradient Boosters (see [https://www.researchgate.net/publication/386212136_Scalable_Gradient_Boosting_using_Randomized_Neural_Networks](https://www.researchgate.net/publication/386212136_Scalable_Gradient_Boosting_using_Randomized_Neural_Networks)).


# 0 - Install `nnetsauce` and `mlsauce`


```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```


```python
!pip install nnetsauce
```


```python
!pip install git+https://github.com/thierrymoudiki/sktime.git --upgrade --no-cache-dir
```


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nnetsauce as ns
import mlsauce as ms
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.chronos import ChronosForecaster

from sklearn import linear_model
from statsmodels.tsa.base.datetools import dates_from_str
from sktime.forecasting.nnetsaucemts import NnetsauceMTS
```

# 1 - sktime foundation models and nnetsauce


```python
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions.values - targets.values) ** 2).mean())

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def me(predictions, targets):
    return np.mean(predictions - targets)
```

### 1 - 2 - Example2 on antidiabetic drug sales with generic booster


```python
filenames = ["a10.csv", "austa.csv", "nile.csv"]
```


```python
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.base import RegressorMixin
from sklearn.utils import all_estimators
from tqdm import tqdm


# Function to process each estimator
def process_estimator(est, df_train, df_test):
    try:
      if issubclass(est[1], RegressorMixin):
          preds = ns.MTS(ms.GenericBoostingRegressor(est[1](), verbose=0), lags=20, verbose=0, show_progress=False).\
              fit(df_train).\
              predict(h=df_test.shape[0])
          return ["MTS(GenBoost(" + est[0] + "))", rmse(df_test, preds), mae(df_test, preds)]
    except Exception:
      try:
        if issubclass(est[1], RegressorMixin):
            preds = ns.MTS(ms.GenericBoostingRegressor(est[1](), verbose=0), lags=5, verbose=0, show_progress=False).\
                fit(df_train).\
                predict(h=df_test.shape[0])
            return ["MTS(GenBoost(" + est[0] + "))", rmse(df_test, preds), mae(df_test, preds)]
      except Exception:
        pass

for filename in filenames:

  print("filename: ", filename)

  url = "https://raw.githubusercontent.com/Techtonique/"
  url += "datasets/main/time_series/univariate/"
  url += filename
  data = pd.read_csv(url)
  data.index = pd.DatetimeIndex(data.date) # must have
  data.drop(columns=['date'], inplace=True)

  data.plot()

  n = data.shape[0]
  max_idx_train = np.floor(n * 0.9)
  training_index = np.arange(0, max_idx_train)
  testing_index = np.arange(max_idx_train, n)
  df_train = data.iloc[training_index, :]
  print(df_train.tail())
  df_test = data.iloc[testing_index, :]
  print(df_test.head())

  results1 = []
  results2 = []
  results = []

  # Initialise models
  chronos = ChronosForecaster("amazon/chronos-t5-tiny")
  ttm = TinyTimeMixerForecaster()
  regr = linear_model.RidgeCV()

  # Fit
  h = df_test.shape[0] + 1
  chronos.fit(y=df_train, fh=range(1, h))
  ttm.fit(y=df_train, fh=range(1, h))

  # Predict
  pred_chronos = chronos.predict(fh=[i for i in range(1, h)])
  pred_ttm = ttm.predict(fh=[i for i in range(1, h)])

  # LLMs and sktime
  results1.append(["Chronos", rmse(df_test, pred_chronos), mae(df_test, pred_chronos)])
  results1.append(["TinyTimeMixer", rmse(df_test, pred_ttm), mae(df_test, pred_ttm)])

  # statistical models
  for i, name in enumerate(["ARIMA", "ETS", "Theta", "VAR", "VECM"]):
    try:
      regr = ns.ClassicalMTS(model=name)
      regr.fit(df_train)
      X_pred = regr.predict(h=df_test.shape[0])
      results1.append([name, rmse(df_test, X_pred.mean), mae(df_test, X_pred.mean)])
    except Exception:
      pass

  # Parallel processing
  results2 = Parallel(n_jobs=-1)(delayed(process_estimator)(est, df_train, df_test) for est in tqdm(all_estimators()))

  for elt in results1:
    if elt is not None:
      results.append(elt)

  for elt in results2:
    if elt is not None:
      results.append(elt)

  results_df = pd.DataFrame(results, columns=["model", "rmse", "mae"])

  display(results_df.sort_values(by="rmse"))
```

    filename:  a10.csv
                value
    date             
    2006-05-01  17.78
    2006-06-01  16.29
    2006-07-01  16.98
    2006-08-01  18.61
    2006-09-01  16.62
                value
    date             
    2006-10-01  21.43
    2006-11-01  23.58
    2006-12-01  23.33
    2007-01-01  28.04
    2007-02-01  16.76

  <div id="df-7426d153-f465-4d37-bcab-ec2886c41d48" class="colab-df-container">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>MTS(GenBoost(ElasticNet))</td>
      <td>2.74</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuitCV))</td>
      <td>2.86</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MTS(GenBoost(LassoLars))</td>
      <td>3.27</td>
      <td>2.69</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MTS(GenBoost(Lasso))</td>
      <td>3.27</td>
      <td>2.69</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MTS(GenBoost(BaggingRegressor))</td>
      <td>3.33</td>
      <td>2.90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MTS(GenBoost(ARDRegression))</td>
      <td>3.35</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MTS(GenBoost(ElasticNetCV))</td>
      <td>3.38</td>
      <td>3.03</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MTS(GenBoost(LassoCV))</td>
      <td>3.38</td>
      <td>3.04</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MTS(GenBoost(RANSACRegressor))</td>
      <td>3.40</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MTS(GenBoost(DecisionTreeRegressor))</td>
      <td>3.41</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MTS(GenBoost(LassoLarsCV))</td>
      <td>3.41</td>
      <td>3.07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MTS(GenBoost(ExtraTreeRegressor))</td>
      <td>3.41</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>44</th>
      <td>MTS(GenBoost(TweedieRegressor))</td>
      <td>3.45</td>
      <td>3.04</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MTS(GenBoost(RandomForestRegressor))</td>
      <td>3.45</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MTS(GenBoost(LassoLarsIC))</td>
      <td>3.49</td>
      <td>3.11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MTS(GenBoost(BayesianRidge))</td>
      <td>3.52</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MTS(GenBoost(TheilSenRegressor))</td>
      <td>3.55</td>
      <td>3.17</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MTS(GenBoost(ExtraTreesRegressor))</td>
      <td>3.57</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuit))</td>
      <td>3.72</td>
      <td>3.27</td>
    </tr>
    <tr>
      <th>39</th>
      <td>MTS(GenBoost(RidgeCV))</td>
      <td>4.00</td>
      <td>3.54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MTS(GenBoost(AdaBoostRegressor))</td>
      <td>4.04</td>
      <td>3.58</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MTS(GenBoost(PLSRegression))</td>
      <td>4.06</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MTS(GenBoost(KernelRidge))</td>
      <td>4.08</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>43</th>
      <td>MTS(GenBoost(TransformedTargetRegressor))</td>
      <td>4.09</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MTS(GenBoost(LinearRegression))</td>
      <td>4.09</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>38</th>
      <td>MTS(GenBoost(Ridge))</td>
      <td>4.10</td>
      <td>3.64</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MTS(GenBoost(GradientBoostingRegressor))</td>
      <td>4.48</td>
      <td>4.04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Theta</td>
      <td>4.57</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MTS(GenBoost(MLPRegressor))</td>
      <td>4.70</td>
      <td>4.29</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MTS(GenBoost(HuberRegressor))</td>
      <td>4.77</td>
      <td>4.35</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MTS(GenBoost(LinearSVR))</td>
      <td>4.83</td>
      <td>4.42</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MTS(GenBoost(QuantileRegressor))</td>
      <td>5.02</td>
      <td>4.22</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>5.10</td>
      <td>4.81</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MTS(GenBoost(PassiveAggressiveRegressor))</td>
      <td>5.19</td>
      <td>4.66</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MTS(GenBoost(HistGradientBoostingRegressor))</td>
      <td>5.40</td>
      <td>4.65</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MTS(GenBoost(LarsCV))</td>
      <td>5.92</td>
      <td>5.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ETS</td>
      <td>6.19</td>
      <td>5.37</td>
    </tr>
    <tr>
      <th>30</th>
      <td>MTS(GenBoost(NuSVR))</td>
      <td>6.43</td>
      <td>5.69</td>
    </tr>
    <tr>
      <th>41</th>
      <td>MTS(GenBoost(SVR))</td>
      <td>6.53</td>
      <td>5.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>6.66</td>
      <td>5.88</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MTS(GenBoost(KNeighborsRegressor))</td>
      <td>9.12</td>
      <td>6.74</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MTS(GenBoost(GaussianProcessRegressor))</td>
      <td>12.71</td>
      <td>12.29</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MTS(GenBoost(DummyRegressor))</td>
      <td>12.72</td>
      <td>12.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARIMA</td>
      <td>13.37</td>
      <td>12.98</td>
    </tr>
    <tr>
      <th>40</th>
      <td>MTS(GenBoost(SGDRegressor))</td>
      <td>inf</td>
      <td>88755370094251662260627878082695870432152870817...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7426d153-f465-4d37-bcab-ec2886c41d48')"
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
        document.querySelector('#df-7426d153-f465-4d37-bcab-ec2886c41d48 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7426d153-f465-4d37-bcab-ec2886c41d48');
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


<div id="df-0ebe13b7-ba26-49b0-ac54-074dcc03758b">
  <button class="colab-df-quickchart" onclick="quickchart('df-0ebe13b7-ba26-49b0-ac54-074dcc03758b')"
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
        document.querySelector('#df-0ebe13b7-ba26-49b0-ac54-074dcc03758b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



    filename:  austa.csv
                value
    date             
    2002-01-01   4.46
    2003-01-01   4.38
    2004-01-01   4.80
    2005-01-01   5.05
    2006-01-01   5.10
                value
    date             
    2007-01-01   5.20
    2008-01-01   5.17
    2009-01-01   5.17
    2010-01-01   5.44

  <div id="df-c4a97183-0ab0-41a8-b295-6f45c33df71f" class="colab-df-container">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>MTS(GenBoost(BayesianRidge))</td>
      <td>0.09</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>45</th>
      <td>MTS(GenBoost(TweedieRegressor))</td>
      <td>0.10</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>41</th>
      <td>MTS(GenBoost(SGDRegressor))</td>
      <td>0.11</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Theta</td>
      <td>0.12</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuitCV))</td>
      <td>0.14</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>40</th>
      <td>MTS(GenBoost(RidgeCV))</td>
      <td>0.14</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MTS(GenBoost(PassiveAggressiveRegressor))</td>
      <td>0.15</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MTS(GenBoost(ARDRegression))</td>
      <td>0.15</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MTS(GenBoost(LassoCV))</td>
      <td>0.16</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MTS(GenBoost(PLSRegression))</td>
      <td>0.16</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MTS(GenBoost(KernelRidge))</td>
      <td>0.17</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>0.17</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MTS(GenBoost(ElasticNetCV))</td>
      <td>0.17</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MTS(GenBoost(QuantileRegressor))</td>
      <td>0.17</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ETS</td>
      <td>0.19</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>39</th>
      <td>MTS(GenBoost(Ridge))</td>
      <td>0.19</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MTS(GenBoost(LassoLarsCV))</td>
      <td>0.19</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MTS(GenBoost(LassoLarsIC))</td>
      <td>0.21</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MTS(GenBoost(SVR))</td>
      <td>0.21</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MTS(GenBoost(LassoLars))</td>
      <td>0.22</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MTS(GenBoost(Lasso))</td>
      <td>0.22</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MTS(GenBoost(DecisionTreeRegressor))</td>
      <td>0.23</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MTS(GenBoost(NuSVR))</td>
      <td>0.23</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MTS(GenBoost(ExtraTreeRegressor))</td>
      <td>0.23</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MTS(GenBoost(ExtraTreesRegressor))</td>
      <td>0.23</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MTS(GenBoost(RANSACRegressor))</td>
      <td>0.24</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MTS(GenBoost(BaggingRegressor))</td>
      <td>0.24</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>43</th>
      <td>MTS(GenBoost(TheilSenRegressor))</td>
      <td>0.24</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MTS(GenBoost(LinearSVR))</td>
      <td>0.25</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MTS(GenBoost(AdaBoostRegressor))</td>
      <td>0.25</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>0.25</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MTS(GenBoost(GradientBoostingRegressor))</td>
      <td>0.26</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>38</th>
      <td>MTS(GenBoost(RandomForestRegressor))</td>
      <td>0.26</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>30</th>
      <td>MTS(GenBoost(MLPRegressor))</td>
      <td>0.27</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MTS(GenBoost(LinearRegression))</td>
      <td>0.29</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>44</th>
      <td>MTS(GenBoost(TransformedTargetRegressor))</td>
      <td>0.29</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MTS(GenBoost(KNeighborsRegressor))</td>
      <td>0.30</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MTS(GenBoost(HuberRegressor))</td>
      <td>0.39</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MTS(GenBoost(ElasticNet))</td>
      <td>0.40</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuit))</td>
      <td>0.90</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MTS(GenBoost(GaussianProcessRegressor))</td>
      <td>1.21</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MTS(GenBoost(LarsCV))</td>
      <td>1.84</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MTS(GenBoost(DummyRegressor))</td>
      <td>1.95</td>
      <td>1.95</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MTS(GenBoost(HistGradientBoostingRegressor))</td>
      <td>1.95</td>
      <td>1.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARIMA</td>
      <td>2.40</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MTS(GenBoost(Lars))</td>
      <td>inf</td>
      <td>69642504841544336879259080339820128177381130744...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c4a97183-0ab0-41a8-b295-6f45c33df71f')"
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
        document.querySelector('#df-c4a97183-0ab0-41a8-b295-6f45c33df71f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c4a97183-0ab0-41a8-b295-6f45c33df71f');
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


<div id="df-b908eab8-bbd1-4a85-adf1-11c83f77a1df">
  <button class="colab-df-quickchart" onclick="quickchart('df-b908eab8-bbd1-4a85-adf1-11c83f77a1df')"
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
        document.querySelector('#df-b908eab8-bbd1-4a85-adf1-11c83f77a1df button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



    filename:  nile.csv
                 value
    date              
    1960-01-01  815.00
    1961-01-01 1020.00
    1962-01-01  906.00
    1963-01-01  901.00
    1964-01-01 1170.00
                value
    date             
    1965-01-01 912.00
    1966-01-01 746.00
    1967-01-01 919.00
    1968-01-01 718.00
    1969-01-01 714.00

  <div id="df-903566d7-b896-4755-92a7-9b14e8468510" class="colab-df-container">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>MTS(GenBoost(QuantileRegressor))</td>
      <td>104.93</td>
      <td>101.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARIMA</td>
      <td>111.78</td>
      <td>105.46</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MTS(GenBoost(NuSVR))</td>
      <td>114.60</td>
      <td>107.00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MTS(GenBoost(LassoLarsIC))</td>
      <td>115.42</td>
      <td>107.39</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MTS(GenBoost(SVR))</td>
      <td>117.68</td>
      <td>108.61</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MTS(GenBoost(HistGradientBoostingRegressor))</td>
      <td>117.75</td>
      <td>108.59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MTS(GenBoost(DummyRegressor))</td>
      <td>117.75</td>
      <td>108.59</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MTS(GenBoost(GaussianProcessRegressor))</td>
      <td>117.75</td>
      <td>108.59</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MTS(GenBoost(LassoLarsCV))</td>
      <td>130.02</td>
      <td>114.40</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chronos</td>
      <td>151.16</td>
      <td>136.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ETS</td>
      <td>170.52</td>
      <td>145.80</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MTS(GenBoost(GradientBoostingRegressor))</td>
      <td>177.24</td>
      <td>154.55</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MTS(GenBoost(LarsCV))</td>
      <td>178.74</td>
      <td>152.11</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MTS(GenBoost(AdaBoostRegressor))</td>
      <td>179.29</td>
      <td>156.22</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MTS(GenBoost(DecisionTreeRegressor))</td>
      <td>184.44</td>
      <td>164.52</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Theta</td>
      <td>195.86</td>
      <td>173.90</td>
    </tr>
    <tr>
      <th>38</th>
      <td>MTS(GenBoost(RandomForestRegressor))</td>
      <td>207.64</td>
      <td>177.91</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MTS(GenBoost(BaggingRegressor))</td>
      <td>209.69</td>
      <td>182.79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MTS(GenBoost(ElasticNetCV))</td>
      <td>216.37</td>
      <td>193.03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MTS(GenBoost(LassoCV))</td>
      <td>225.24</td>
      <td>201.83</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MTS(GenBoost(RANSACRegressor))</td>
      <td>296.22</td>
      <td>260.08</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MTS(GenBoost(BayesianRidge))</td>
      <td>310.59</td>
      <td>287.58</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MTS(GenBoost(ExtraTreesRegressor))</td>
      <td>318.41</td>
      <td>304.40</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MTS(GenBoost(ExtraTreeRegressor))</td>
      <td>349.23</td>
      <td>339.16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyTimeMixer</td>
      <td>396.16</td>
      <td>388.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MTS(GenBoost(ARDRegression))</td>
      <td>407.08</td>
      <td>377.50</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MTS(GenBoost(KNeighborsRegressor))</td>
      <td>409.69</td>
      <td>398.57</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuitCV))</td>
      <td>560.00</td>
      <td>518.03</td>
    </tr>
    <tr>
      <th>30</th>
      <td>MTS(GenBoost(MLPRegressor))</td>
      <td>622.93</td>
      <td>573.30</td>
    </tr>
    <tr>
      <th>32</th>
      <td>MTS(GenBoost(OrthogonalMatchingPursuit))</td>
      <td>658.25</td>
      <td>591.60</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MTS(GenBoost(LinearSVR))</td>
      <td>744.48</td>
      <td>693.50</td>
    </tr>
    <tr>
      <th>45</th>
      <td>MTS(GenBoost(TweedieRegressor))</td>
      <td>1170.09</td>
      <td>1020.41</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MTS(GenBoost(KernelRidge))</td>
      <td>1313.31</td>
      <td>1147.54</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MTS(GenBoost(ElasticNet))</td>
      <td>1339.16</td>
      <td>1152.93</td>
    </tr>
    <tr>
      <th>41</th>
      <td>MTS(GenBoost(SGDRegressor))</td>
      <td>1358.22</td>
      <td>1166.58</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MTS(GenBoost(PassiveAggressiveRegressor))</td>
      <td>1554.27</td>
      <td>1319.48</td>
    </tr>
    <tr>
      <th>40</th>
      <td>MTS(GenBoost(RidgeCV))</td>
      <td>1708.53</td>
      <td>1443.61</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MTS(GenBoost(Lasso))</td>
      <td>1815.57</td>
      <td>1534.60</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MTS(GenBoost(LassoLars))</td>
      <td>1910.40</td>
      <td>1612.20</td>
    </tr>
    <tr>
      <th>43</th>
      <td>MTS(GenBoost(TheilSenRegressor))</td>
      <td>2050.37</td>
      <td>1726.27</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MTS(GenBoost(PLSRegression))</td>
      <td>2119.73</td>
      <td>1770.69</td>
    </tr>
    <tr>
      <th>44</th>
      <td>MTS(GenBoost(TransformedTargetRegressor))</td>
      <td>2178.49</td>
      <td>1819.45</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MTS(GenBoost(LinearRegression))</td>
      <td>2178.49</td>
      <td>1819.45</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MTS(GenBoost(HuberRegressor))</td>
      <td>2267.51</td>
      <td>1882.66</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MTS(GenBoost(Lars))</td>
      <td>2696.45</td>
      <td>1941.89</td>
    </tr>
    <tr>
      <th>39</th>
      <td>MTS(GenBoost(Ridge))</td>
      <td>2769.55</td>
      <td>2245.39</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-903566d7-b896-4755-92a7-9b14e8468510')"
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
        document.querySelector('#df-903566d7-b896-4755-92a7-9b14e8468510 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-903566d7-b896-4755-92a7-9b14e8468510');
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


<div id="df-07e5b01e-afe4-4f71-a76e-b2bc19f4605b">
  <button class="colab-df-quickchart" onclick="quickchart('df-07e5b01e-afe4-4f71-a76e-b2bc19f4605b')"
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
        document.querySelector('#df-07e5b01e-afe4-4f71-a76e-b2bc19f4605b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>

![xxx]({{base}}/images/2024-12-01/2024-12-01-image1.png){:class="img-responsive"}          

![xxx]({{base}}/images/2024-12-01/2024-12-01-image2.png){:class="img-responsive"}          

![xxx]({{base}}/images/2024-12-01/2024-12-01-image3.png){:class="img-responsive"}          