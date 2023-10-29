---
layout: post
title: "AutoML in nnetsauce (randomized and quasi-randomized nnetworks) Pt.2: multivariate time series forecasting"
description: "AutoML with nnetsauce, that implements randomized and quasi-randomized 'neural' networks for supervised learning and time series forecasting"
date: 2023-10-29
categories: [Python, QuasiRandomizedNN]
comments: true
---

Last week, I talked about an AutoML method for regression and classification implemented in Python package `nnetsauce`. This week, my post is about the same AutoML method, applied this time to multivariate time series (MTS) forecasting. 

In the examples below, keep in mind that VAR (Vector Autoregression) and VECM (Vector Error Correction Model) forecasting models aren't thoroughly trained here. `nnetsauce.MTS` isn't really tuned either; this is just a demo. To finish, a probabilistic error metric (instead of the Root Mean Squared Error, RMSE) is better suited  for models capturing forecasting uncertainty.

**Contents**

- 1 - Install
- 2 - MTS
- 2 - 1 nnetsauce.MTS
- 2 - 2 statsmodels VAR
- 2 - 3 statsmodels VECM

# **1 - Install**


```python
!pip install git+https://github.com/Techtonique/nnetsauce.git@lazy-predict
```


```python
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
from sklearn.linear_model import LassoCV
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.base.datetools import dates_from_str
```

# **2 - MTS**

Macro data


```python
# some example data
mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgovt', 'tbilrate']]

mdata.index = pd.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

display(data)
```


```python
df = data

df.index.rename('date')

idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]
df_test = df.iloc[idx_train:idx_end,]

regr_mts = ns.LazyMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                      lags = 1, n_hidden_features=3, n_clusters=0, random_state=1)
models, predictions = regr_mts.fit(df_train, df_test)
model_dictionary = regr_mts.provide_models(df_train, df_test)
```


```python
display(models)
```



  <div id="df-eef99f30-beac-423e-8885-5adcb9ec742d" class="colab-df-container">
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
      <th>RMSE</th>
      <th>MAE</th>
      <th>MPL</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LassoCV</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>ElasticNetCV</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>LassoLarsCV</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>LarsCV</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>DummyRegressor</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>LassoLars</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>ExtraTreeRegressor</th>
      <td>0.22</td>
      <td>0.14</td>
      <td>0.07</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>KNeighborsRegressor</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>0.22</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>HistGradientBoostingRegressor</th>
      <td>0.23</td>
      <td>0.13</td>
      <td>0.06</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>NuSVR</th>
      <td>0.23</td>
      <td>0.13</td>
      <td>0.06</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>ExtraTreesRegressor</th>
      <td>0.24</td>
      <td>0.13</td>
      <td>0.07</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>0.24</td>
      <td>0.13</td>
      <td>0.07</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>0.26</td>
      <td>0.16</td>
      <td>0.08</td>
      <td>2.06</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>0.28</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>DecisionTreeRegressor</th>
      <td>0.28</td>
      <td>0.18</td>
      <td>0.09</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>BaggingRegressor</th>
      <td>0.28</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>8.26</td>
      <td>5.90</td>
      <td>2.95</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>11774168792.68</td>
      <td>3129885640.50</td>
      <td>1564942820.25</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>TweedieRegressor</th>
      <td>1066305878860.67</td>
      <td>263521546472.00</td>
      <td>131760773236.00</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>LassoLarsIC</th>
      <td>10841414830181.57</td>
      <td>2665022282527.50</td>
      <td>1332511141263.75</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>PassiveAggressiveRegressor</th>
      <td>200205325611502239744.00</td>
      <td>40689888595970097152.00</td>
      <td>20344944297985048576.00</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>SGDRegressor</th>
      <td>1383750703550277812748288.00</td>
      <td>269310062772019343130624.00</td>
      <td>134655031386009671565312.00</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>LinearSVR</th>
      <td>6205416599219790202011648.00</td>
      <td>1189414936788171753521152.00</td>
      <td>594707468394085876760576.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>OrthogonalMatchingPursuitCV</th>
      <td>18588484112627753604349952.00</td>
      <td>3542235944300533382119424.00</td>
      <td>1771117972150266691059712.00</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>OrthogonalMatchingPursuit</th>
      <td>18588484112627753604349952.00</td>
      <td>3542235944300533382119424.00</td>
      <td>1771117972150266691059712.00</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>HuberRegressor</th>
      <td>50554040814422644093913571262464.00</td>
      <td>9061839427591544042390898606080.00</td>
      <td>4530919713795772021195449303040.00</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>RidgeCV</th>
      <td>1788858960353426286932811384356864.00</td>
      <td>317940467527547291488891451736064.00</td>
      <td>158970233763773645744445725868032.00</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>RANSACRegressor</th>
      <td>352805899757804849079011831705501696.00</td>
      <td>61914238966205227684888230708117504.00</td>
      <td>30957119483102613842444115354058752.00</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>LinearRegression</th>
      <td>13408548756595947978849418193194188800.00</td>
      <td>2316276205868561893698967459810246656.00</td>
      <td>1158138102934280946849483729905123328.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>TransformedTargetRegressor</th>
      <td>13408548756595947978849418193194188800.00</td>
      <td>2316276205868561893698967459810246656.00</td>
      <td>1158138102934280946849483729905123328.00</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>Lars</th>
      <td>13408548756596845228481163425784791040.00</td>
      <td>2316276205868715960905471081985343488.00</td>
      <td>1158138102934357980452735540992671744.00</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>27935786184657480745080678989281886208.00</td>
      <td>4824713257018197525713060327109689344.00</td>
      <td>2412356628509098762856530163554844672.00</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>27935786184685139645570846501298503680.00</td>
      <td>4824713257022931107816326787730767872.00</td>
      <td>2412356628511465553908163393865383936.00</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>64247413650209509837810706524366567768365621314...</td>
      <td>10088348458681313437051396009759695398571807517...</td>
      <td>50441742293406567185256980048798476992859037587...</td>
      <td>0.42</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-eef99f30-beac-423e-8885-5adcb9ec742d')"
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
        document.querySelector('#df-eef99f30-beac-423e-8885-5adcb9ec742d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-eef99f30-beac-423e-8885-5adcb9ec742d');
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


<div id="df-42d2b1e2-c521-4586-8aae-051d4a08abbc">
  <button class="colab-df-quickchart" onclick="quickchart('df-42d2b1e2-c521-4586-8aae-051d4a08abbc')"
            title="Suggest charts."
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
        document.querySelector('#df-42d2b1e2-c521-4586-8aae-051d4a08abbc button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary['LassoCV']
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MTS(n_clusters=0, n_hidden_features=3, obj=LassoCV(random_state=1), seed=&#x27;mean&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">MTS</label><div class="sk-toggleable__content"><pre>MTS(n_clusters=0, n_hidden_features=3, obj=LassoCV(random_state=1), seed=&#x27;mean&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">obj: LassoCV</label><div class="sk-toggleable__content"><pre>LassoCV(random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">LassoCV</label><div class="sk-toggleable__content"><pre>LassoCV(random_state=1)</pre></div></div></div></div></div></div></div></div></div></div>



## **2 - 1 - `nnetsauce.MTS`**


```python
regr = ns.MTS(obj = LassoCV(random_state=1),
              lags = 1, n_hidden_features=3,
              n_clusters=0, replications = 250,
              kernel = "gaussian", verbose = 1)
```


```python
regr.fit(df_train)
```

    
     Adjusting LassoCV to multivariate time series... 
     


    100%|██████████| 2/2 [00:00<00:00,  6.22it/s]


    
     Simulate residuals using gaussian kernel... 
    
    
     Best parameters for gaussian kernel: {'bandwidth': 0.04037017258596558} 
    





<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MTS(kernel=&#x27;gaussian&#x27;, n_clusters=0, n_hidden_features=3,
    obj=LassoCV(random_state=1), replications=250, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">MTS</label><div class="sk-toggleable__content"><pre>MTS(kernel=&#x27;gaussian&#x27;, n_clusters=0, n_hidden_features=3,
    obj=LassoCV(random_state=1), replications=250, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">obj: LassoCV</label><div class="sk-toggleable__content"><pre>LassoCV(random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">LassoCV</label><div class="sk-toggleable__content"><pre>LassoCV(random_state=1)</pre></div></div></div></div></div></div></div></div></div></div>




```python
res = regr.predict(h=df_test.shape[0], level=95)
```

    100%|██████████| 250/250 [00:00<00:00, 3686.16it/s]
    100%|██████████| 250/250 [00:00<00:00, 6971.82it/s]



```python
regr.plot("realgovt")
regr.plot("tbilrate")
```

![image-title-here]({{base}}/images/2023-10-29/2023-10-29-image1.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2023-10-29/2023-10-29-image2.png){:class="img-responsive"}


## **2 - 2 - VAR**


```python
model = VAR(df_train)
results = model.fit(maxlags=5, ic='aic')
lag_order = results.k_ar
VAR_preds = results.forecast(df_train.values[-lag_order:], df_test.shape[0])
```


```python
results.plot_forecast(steps = df_test.shape[0]);
```

![image-title-here]({{base}}/images/2023-10-29/2023-10-29-image3.png){:class="img-responsive"}

## **2 - 3 - VECM**


```python
model = VECM(df_train, k_ar_diff=2, coint_rank=2)
vecm_res = model.fit()
vecm_res.gamma.round(4)
vecm_res.summary()
vecm_res.predict(steps=df_test.shape[0])
forecast, lower, upper = vecm_res.predict(df_test.shape[0], 0.05)
```


```python
vecm_res.plot_forecast(steps = df_test.shape[0])
```

![image-title-here]({{base}}/images/2023-10-29/2023-10-29-image5.png){:class="img-responsive"}


out-of-sample errors


```python
display([("nnetsauce.MTS+"+models.index[i], models["RMSE"].iloc[i]) for i in range(3)])
display(('VAR', mean_squared_error(df_test.values, VAR_preds, squared=False)))
display(('VECM', mean_squared_error(df_test.values, forecast, squared=False)))
```


    [('nnetsauce.MTS+LassoCV', 0.22102547609924011),
     ('nnetsauce.MTS+ElasticNetCV', 0.22103106562991648),
     ('nnetsauce.MTS+LassoLarsCV', 0.22103468506703655)]
    ('VAR', 0.22128770514262763)
    ('VECM', 0.22170093788693065)

