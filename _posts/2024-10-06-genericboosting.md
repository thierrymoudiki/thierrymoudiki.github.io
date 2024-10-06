---
layout: post
title: "Gradient-Boosting anything (alert: high performance)"
description: "Gradient boosting with any regression algorithm in Python package mlsauce"
date: 2024-10-06
categories: [Python, R]
comments: true
---

We've always been told that decision trees are _best_ for [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) Machine Learning. I've always wanted to see for myself. [AdaBoostClassifier](https://techtonique.github.io/nnetsauce/nnetsauce.html#AdaBoostClassifier) is working well, but is relatively _slow_ (by my own standards). A few days ago, I noticed that my Cython implementation of [LSBoost](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) in Python package mlsauce was already quite _generic_ (never noticed before), and I decided to adapt it to any machine learning model with `fit` and `predict` methods. It's worth mentioning that only regression algorithms are accepted as base learners, and classification is [regression-based](https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers). The results are promising indeed; I'll let you see for yourself below, for regression and classification. All the algorithms, including `xgboost` and `RandomForest`, are used with their default hyperparameters. Which means, there's still a room for improvement.


Install mlsauce (version 0.20.3) from GitHub:

```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose --upgrade --no-cache-dir
```

# 1 - Classification

```python
import os
import pandas as pd
import mlsauce as ms
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from time import time

load_models = [load_breast_cancer, load_wine, load_iris]

for model in load_models:

    data = model()
    X = data.data
    y= data.target
    X = pd.DataFrame(X, columns=data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    clf = ms.LazyBoostingClassifier(verbose=0, ignore_warnings=True,
                                    custom_metric=None, preprocess=False)

    start = time()
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\nElapsed: {time() - start} seconds\n")

    display(models)

```

    2it [00:01,  1.52it/s]
    100%|██████████| 30/30 [00:21<00:00,  1.38it/s]

    
    Elapsed: 23.019137859344482 seconds
    


    




  <div id="df-a9bd5504-b0ef-4739-86c8-abc3bb72eb34" class="colab-df-container">
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>GenericBooster(KernelRidge)</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>2.05</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.92</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.92</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>GenericBooster(RANSACRegressor)</th>
      <td>0.90</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.90</td>
      <td>15.22</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>0.88</td>
      <td>0.87</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.87</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.87</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.85</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.84</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.79</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.79</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.68</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.56</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a9bd5504-b0ef-4739-86c8-abc3bb72eb34')"
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
        document.querySelector('#df-a9bd5504-b0ef-4739-86c8-abc3bb72eb34 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a9bd5504-b0ef-4739-86c8-abc3bb72eb34');
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


<div id="df-4868e91e-0cae-473d-8635-7d7bb02ffe10">
  <button class="colab-df-quickchart" onclick="quickchart('df-4868e91e-0cae-473d-8635-7d7bb02ffe10')"
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
        document.querySelector('#df-4868e91e-0cae-473d-8635-7d7bb02ffe10 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_5b31c18b-eb5c-431c-b491-ccd75a68800d">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('models')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_5b31c18b-eb5c-431c-b491-ccd75a68800d button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('models');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    2it [00:00,  8.29it/s]
    100%|██████████| 30/30 [00:15<00:00,  1.92it/s]

    
    Elapsed: 15.911818265914917 seconds
    


    




  <div id="df-3433d120-1127-4c65-82fb-1cabc2bcb888" class="colab-df-container">
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RandomForestClassifier</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>GenericBooster(KernelRidge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.95</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.92</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>GenericBooster(RANSACRegressor)</th>
      <td>0.81</td>
      <td>0.81</td>
      <td>None</td>
      <td>0.80</td>
      <td>12.63</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.61</td>
      <td>0.53</td>
      <td>None</td>
      <td>0.53</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.42</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.25</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.42</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.25</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.42</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.25</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3433d120-1127-4c65-82fb-1cabc2bcb888')"
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
        document.querySelector('#df-3433d120-1127-4c65-82fb-1cabc2bcb888 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3433d120-1127-4c65-82fb-1cabc2bcb888');
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


<div id="df-3ee4587d-9197-4f03-b812-3a2dbfe006e1">
  <button class="colab-df-quickchart" onclick="quickchart('df-3ee4587d-9197-4f03-b812-3a2dbfe006e1')"
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
        document.querySelector('#df-3ee4587d-9197-4f03-b812-3a2dbfe006e1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_0ac5e38a-a46c-41d5-af0d-71cae17631b5">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('models')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0ac5e38a-a46c-41d5-af0d-71cae17631b5 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('models');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    2it [00:00,  5.14it/s]
    100%|██████████| 30/30 [00:15<00:00,  1.92it/s]

    
    Elapsed: 16.0275661945343 seconds
    


    




  <div id="df-94fd34fc-de7d-4ad7-a736-7e487ede6e99" class="colab-df-container">
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.93</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.93</td>
      <td>1.52</td>
    </tr>
    <tr>
      <th>GenericBooster(KernelRidge)</th>
      <td>0.87</td>
      <td>0.83</td>
      <td>None</td>
      <td>0.85</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>GenericBooster(RANSACRegressor)</th>
      <td>0.63</td>
      <td>0.59</td>
      <td>None</td>
      <td>0.61</td>
      <td>10.86</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.50</td>
      <td>0.46</td>
      <td>None</td>
      <td>0.48</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-94fd34fc-de7d-4ad7-a736-7e487ede6e99')"
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
        document.querySelector('#df-94fd34fc-de7d-4ad7-a736-7e487ede6e99 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-94fd34fc-de7d-4ad7-a736-7e487ede6e99');
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


<div id="df-4dfa8cc3-3b64-4e1a-8393-1be96ee84436">
  <button class="colab-df-quickchart" onclick="quickchart('df-4dfa8cc3-3b64-4e1a-8393-1be96ee84436')"
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
        document.querySelector('#df-4dfa8cc3-3b64-4e1a-8393-1be96ee84436 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_bb5336dd-7f1b-4d34-8d6c-e4bd7c297d18">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('models')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_bb5336dd-7f1b-4d34-8d6c-e4bd7c297d18 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('models');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
!pip install shap
```


```python
import shap

best_model = clf.get_best_model()

# load JS visualization code to notebook
shap.initjs()

# explain all the predictions in the test set
explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
# this is multiclass so we only visualize the contributions to first class (hence index 0)
shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)
```

![xxx]({{base}}/images/2024-10-06/2024-10-06-image1.png){:class="img-responsive"}  


# 2 - Classification

```python
import os
import mlsauce as ms
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True,
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
display(models)
```

    3it [00:00,  4.75it/s]
    100%|██████████| 30/30 [00:58<00:00,  1.95s/it]




  <div id="df-60b2034f-2490-4c58-9acc-8a6c74895bdb" class="colab-df-container">
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
      <th>Adjusted R-Squared</th>
      <th>R-Squared</th>
      <th>RMSE</th>
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
      <th>GenericBooster(HuberRegressor)</th>
      <td>0.55</td>
      <td>0.60</td>
      <td>50.13</td>
      <td>3.73</td>
    </tr>
    <tr>
      <th>GenericBooster(SGDRegressor)</th>
      <td>0.55</td>
      <td>0.60</td>
      <td>50.40</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.53</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearSVR)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.54</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>GenericBooster(PassiveAggressiveRegressor)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.63</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.70</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.75</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.54</td>
      <td>0.59</td>
      <td>50.75</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>GenericBooster(KernelRidge)</th>
      <td>0.53</td>
      <td>0.59</td>
      <td>50.99</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>GenericBooster(TweedieRegressor)</th>
      <td>0.53</td>
      <td>0.59</td>
      <td>51.10</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.17</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.17</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.24</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>GenericBooster(SVR)</th>
      <td>0.52</td>
      <td>0.57</td>
      <td>51.97</td>
      <td>3.54</td>
    </tr>
    <tr>
      <th>GenericBooster(BayesianRidge)</th>
      <td>0.50</td>
      <td>0.56</td>
      <td>52.93</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLarsIC)</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.20</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.23</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNetCV)</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.43</td>
      <td>3.73</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLarsCV)</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.44</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoCV)</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.45</td>
      <td>4.01</td>
    </tr>
    <tr>
      <th>GenericBooster(LarsCV)</th>
      <td>0.49</td>
      <td>0.54</td>
      <td>53.54</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>GenericBooster(NuSVR)</th>
      <td>0.46</td>
      <td>0.53</td>
      <td>54.67</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>0.46</td>
      <td>0.52</td>
      <td>55.16</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>GenericBooster(RANSACRegressor)</th>
      <td>0.44</td>
      <td>0.50</td>
      <td>56.14</td>
      <td>23.45</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.41</td>
      <td>0.47</td>
      <td>57.52</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>XGBRegressor</th>
      <td>0.31</td>
      <td>0.39</td>
      <td>61.96</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.28</td>
      <td>0.36</td>
      <td>63.57</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.19</td>
      <td>0.28</td>
      <td>67.43</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>-0.13</td>
      <td>-0.00</td>
      <td>79.39</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(QuantileRegressor)</th>
      <td>-0.15</td>
      <td>-0.02</td>
      <td>80.00</td>
      <td>3.37</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>-7.86</td>
      <td>-6.85</td>
      <td>222.42</td>
      <td>1.14</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-60b2034f-2490-4c58-9acc-8a6c74895bdb')"
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
        document.querySelector('#df-60b2034f-2490-4c58-9acc-8a6c74895bdb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-60b2034f-2490-4c58-9acc-8a6c74895bdb');
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


<div id="df-25d8694d-3c24-4a9a-9251-61d52c9eded1">
  <button class="colab-df-quickchart" onclick="quickchart('df-25d8694d-3c24-4a9a-9251-61d52c9eded1')"
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
        document.querySelector('#df-25d8694d-3c24-4a9a-9251-61d52c9eded1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_1c262ba7-3c20-461d-a166-1901992e4ed5">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('models')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_1c262ba7-3c20-461d-a166-1901992e4ed5 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('models');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
data = fetch_california_housing()
n_points = 1000
idx_inputs = range(n_points)
X = data.data[idx_inputs,:]
y= data.target[idx_inputs]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True,
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
display(models)
```

    3it [00:03,  1.01s/it]
    100%|██████████| 30/30 [02:32<00:00,  5.10s/it]




  <div id="df-5e557d80-500c-4561-be9b-b4fc7c599f63" class="colab-df-container">
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
      <th>Adjusted R-Squared</th>
      <th>R-Squared</th>
      <th>RMSE</th>
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
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.82</td>
      <td>0.83</td>
      <td>0.34</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>0.82</td>
      <td>0.83</td>
      <td>0.34</td>
      <td>1.27</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.37</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>GenericBooster(NuSVR)</th>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.37</td>
      <td>18.97</td>
    </tr>
    <tr>
      <th>GenericBooster(SVR)</th>
      <td>0.78</td>
      <td>0.79</td>
      <td>0.38</td>
      <td>15.78</td>
    </tr>
    <tr>
      <th>XGBRegressor</th>
      <td>0.78</td>
      <td>0.79</td>
      <td>0.38</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>GenericBooster(HuberRegressor)</th>
      <td>0.77</td>
      <td>0.78</td>
      <td>0.39</td>
      <td>5.49</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearSVR)</th>
      <td>0.77</td>
      <td>0.77</td>
      <td>0.39</td>
      <td>7.15</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.40</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.40</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.40</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>GenericBooster(RANSACRegressor)</th>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.41</td>
      <td>32.76</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.41</td>
      <td>2.54</td>
    </tr>
    <tr>
      <th>GenericBooster(PassiveAggressiveRegressor)</th>
      <td>0.74</td>
      <td>0.75</td>
      <td>0.41</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>GenericBooster(SGDRegressor)</th>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.42</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.42</td>
      <td>2.48</td>
    </tr>
    <tr>
      <th>GenericBooster(KernelRidge)</th>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.43</td>
      <td>13.27</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLarsIC)</th>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.44</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>GenericBooster(BayesianRidge)</th>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.44</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLarsCV)</th>
      <td>0.70</td>
      <td>0.71</td>
      <td>0.44</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoCV)</th>
      <td>0.70</td>
      <td>0.71</td>
      <td>0.44</td>
      <td>9.69</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNetCV)</th>
      <td>0.70</td>
      <td>0.71</td>
      <td>0.44</td>
      <td>10.05</td>
    </tr>
    <tr>
      <th>GenericBooster(TweedieRegressor)</th>
      <td>0.69</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>GenericBooster(LarsCV)</th>
      <td>0.69</td>
      <td>0.70</td>
      <td>0.45</td>
      <td>1.65</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.42</td>
      <td>0.44</td>
      <td>0.62</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.25</td>
      <td>0.28</td>
      <td>0.70</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>GenericBooster(QuantileRegressor)</th>
      <td>-0.04</td>
      <td>-0.00</td>
      <td>0.83</td>
      <td>10.72</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>-0.08</td>
      <td>-0.04</td>
      <td>0.84</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>-0.08</td>
      <td>-0.04</td>
      <td>0.84</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>-0.08</td>
      <td>-0.04</td>
      <td>0.84</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>-0.46</td>
      <td>-0.40</td>
      <td>0.98</td>
      <td>4.75</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5e557d80-500c-4561-be9b-b4fc7c599f63')"
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
        document.querySelector('#df-5e557d80-500c-4561-be9b-b4fc7c599f63 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5e557d80-500c-4561-be9b-b4fc7c599f63');
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


<div id="df-0ddaa0ee-af50-473b-8953-c90aeb73ae8f">
  <button class="colab-df-quickchart" onclick="quickchart('df-0ddaa0ee-af50-473b-8953-c90aeb73ae8f')"
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
        document.querySelector('#df-0ddaa0ee-af50-473b-8953-c90aeb73ae8f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_76882da1-01f1-4275-9835-91b1209655aa">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('models')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_76882da1-01f1-4275-9835-91b1209655aa button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('models');
      }
      })();
    </script>
  </div>

    </div>
  </div>


