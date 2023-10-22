---
layout: post
title: "AutoML in nnetsauce (randomized and quasi-randomized nnetworks)"
description: "AutoML with nnetsauce, that implements randomized and quasi-randomized 'neural' networks for supervised learning and time series forecasting"
date: 2023-10-22
categories: [Python, QuasiRandomizedNN]
comments: true
---

**Content:**

<ol>
  <li> Installing nnetsauce for Python </li>
  <li> Classification </li>
  <li> Regression </li>
</ol>

**Disclaimer**: I have no affiliation with the `lazypredict` project. 

A few days ago, I _stumbled accross_ a cool Python package called `lazypredict`. Pretty well-designed, _working_, and relying on `scikit-learn`'s design. 

With `lazypredict`, you can rapidly have an idea of which scikit-learn model (can also work with `xgboost`'s and `lightgbm`'s `scikit-learn`-like interfaces) performs best on a given data set, with a little **preprocessing**, and **without hyperparameters' tuning (this is important to note)**. 

I thought something similar could be beneficial to [nnetsauce](https://github.com/Techtonique/nnetsauce)'s classes `CustomClassifier`, `CustomRegressor` (see detailed examples below, and interact with the graphs) and `MTS`. For now.  

So far, in `nnetsauce` (Python version), I adapted the lazy prediction feature to regression (`CustomRegressor`) and classification (`CustomClassifier`). Not for univariate and multivariate time series forecasting (`MTS`) yet. You can try it from a GitHub branch. 



## **2 - Installation**


```python
!pip install git+https://github.com/Techtonique/nnetsauce.git@lazy-predict
```

## **2 - Classification**

## **2 - 1 Loading the Dataset**

```python
import nnetsauce as ns
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y= data.target
```

## **2 - 2 Building the classification model using LazyPredict**


```python
from sklearn.model_selection import train_test_split

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123)

# build the lazyclassifier
clf = ns.LazyClassifier(verbose=0, ignore_warnings=True,
                        custom_metric=None,
                        n_hidden_features=10,
                        col_sample=0.9)

# fit it
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

    100%|██████████| 27/27 [00:09<00:00,  2.71it/s]



```python
# print the best models
display(models)
```



  <div id="df-bd160ee7-3a8e-4b49-8d37-ce83dd26eace" class="colab-df-container">
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
      <th>LogisticRegression</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.96</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.96</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.96</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.96</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.95</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.95</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.94</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.94</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.94</td>
      <td>0.91</td>
      <td>0.91</td>
      <td>0.94</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.93</td>
      <td>0.91</td>
      <td>0.91</td>
      <td>0.93</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.92</td>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.92</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.92</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.92</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.64</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.27</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bd160ee7-3a8e-4b49-8d37-ce83dd26eace')"
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
        document.querySelector('#df-bd160ee7-3a8e-4b49-8d37-ce83dd26eace button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bd160ee7-3a8e-4b49-8d37-ce83dd26eace');
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


<div id="df-f8063f97-70a0-4842-bb84-e9b01250dd2e">
  <button class="colab-df-quickchart" onclick="quickchart('df-f8063f97-70a0-4842-bb84-e9b01250dd2e')"
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
        document.querySelector('#df-f8063f97-70a0-4842-bb84-e9b01250dd2e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
```


```python
model_dictionary['LogisticRegression']
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_low&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;c...
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_high&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OrdinalEncoder())]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;))])),
                (&#x27;classifier&#x27;,
                 CustomClassifier(col_sample=0.9, n_hidden_features=10,
                                  obj=LogisticRegression(random_state=42)))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_low&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;c...
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_high&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OrdinalEncoder())]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;))])),
                (&#x27;classifier&#x27;,
                 CustomClassifier(col_sample=0.9, n_hidden_features=10,
                                  obj=LogisticRegression(random_state=42)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           dtype=&#x27;int64&#x27;)),
                                (&#x27;categorical_low&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                strategy=&#x27;constant&#x27;)),
                                                 (&#x27;encoding&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                sparse=False))]),
                                 Int64Index([], dtype=&#x27;int64&#x27;)),
                                (&#x27;categorical_high&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                strategy=&#x27;constant&#x27;)),
                                                 (&#x27;encoding&#x27;,
                                                  OrdinalEncoder())]),
                                 Int64Index([], dtype=&#x27;int64&#x27;))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">numeric</label><div class="sk-toggleable__content"><pre>Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">categorical_low</label><div class="sk-toggleable__content"><pre>Int64Index([], dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value=&#x27;missing&#x27;, strategy=&#x27;constant&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;, sparse=False)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">categorical_high</label><div class="sk-toggleable__content"><pre>Int64Index([], dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value=&#x27;missing&#x27;, strategy=&#x27;constant&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">classifier: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(col_sample=0.9, n_hidden_features=10,
                 obj=LogisticRegression(random_state=42))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">obj: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
model_dictionary['LogisticRegression'].get_params()
```




    {'memory': None,
     'steps': [('preprocessor',
       ColumnTransformer(transformers=[('numeric',
                                        Pipeline(steps=[('imputer', SimpleImputer()),
                                                        ('scaler', StandardScaler())]),
                                        Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                  dtype='int64')),
                                       ('categorical_low',
                                        Pipeline(steps=[('imputer',
                                                         SimpleImputer(fill_value='missing',
                                                                       strategy='constant')),
                                                        ('encoding',
                                                         OneHotEncoder(handle_unknown='ignore',
                                                                       sparse=False))]),
                                        Int64Index([], dtype='int64')),
                                       ('categorical_high',
                                        Pipeline(steps=[('imputer',
                                                         SimpleImputer(fill_value='missing',
                                                                       strategy='constant')),
                                                        ('encoding',
                                                         OrdinalEncoder())]),
                                        Int64Index([], dtype='int64'))])),
      ('classifier',
       CustomClassifier(col_sample=0.9, n_hidden_features=10,
                        obj=LogisticRegression(random_state=42)))],
     'verbose': False,
     'preprocessor': ColumnTransformer(transformers=[('numeric',
                                      Pipeline(steps=[('imputer', SimpleImputer()),
                                                      ('scaler', StandardScaler())]),
                                      Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                dtype='int64')),
                                     ('categorical_low',
                                      Pipeline(steps=[('imputer',
                                                       SimpleImputer(fill_value='missing',
                                                                     strategy='constant')),
                                                      ('encoding',
                                                       OneHotEncoder(handle_unknown='ignore',
                                                                     sparse=False))]),
                                      Int64Index([], dtype='int64')),
                                     ('categorical_high',
                                      Pipeline(steps=[('imputer',
                                                       SimpleImputer(fill_value='missing',
                                                                     strategy='constant')),
                                                      ('encoding',
                                                       OrdinalEncoder())]),
                                      Int64Index([], dtype='int64'))]),
     'classifier': CustomClassifier(col_sample=0.9, n_hidden_features=10,
                      obj=LogisticRegression(random_state=42)),
     'preprocessor__n_jobs': None,
     'preprocessor__remainder': 'drop',
     'preprocessor__sparse_threshold': 0.3,
     'preprocessor__transformer_weights': None,
     'preprocessor__transformers': [('numeric',
       Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]),
       Int64Index([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                  dtype='int64')),
      ('categorical_low',
       Pipeline(steps=[('imputer',
                        SimpleImputer(fill_value='missing', strategy='constant')),
                       ('encoding',
                        OneHotEncoder(handle_unknown='ignore', sparse=False))]),
       Int64Index([], dtype='int64')),
      ('categorical_high',
       Pipeline(steps=[('imputer',
                        SimpleImputer(fill_value='missing', strategy='constant')),
                       ('encoding', OrdinalEncoder())]),
       Int64Index([], dtype='int64'))],
     'preprocessor__verbose': False,
     'preprocessor__verbose_feature_names_out': True,
     'preprocessor__numeric': Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]),
     'preprocessor__categorical_low': Pipeline(steps=[('imputer',
                      SimpleImputer(fill_value='missing', strategy='constant')),
                     ('encoding',
                      OneHotEncoder(handle_unknown='ignore', sparse=False))]),
     'preprocessor__categorical_high': Pipeline(steps=[('imputer',
                      SimpleImputer(fill_value='missing', strategy='constant')),
                     ('encoding', OrdinalEncoder())]),
     'preprocessor__numeric__memory': None,
     'preprocessor__numeric__steps': [('imputer', SimpleImputer()),
      ('scaler', StandardScaler())],
     'preprocessor__numeric__verbose': False,
     'preprocessor__numeric__imputer': SimpleImputer(),
     'preprocessor__numeric__scaler': StandardScaler(),
     'preprocessor__numeric__imputer__add_indicator': False,
     'preprocessor__numeric__imputer__copy': True,
     'preprocessor__numeric__imputer__fill_value': None,
     'preprocessor__numeric__imputer__keep_empty_features': False,
     'preprocessor__numeric__imputer__missing_values': nan,
     'preprocessor__numeric__imputer__strategy': 'mean',
     'preprocessor__numeric__imputer__verbose': 'deprecated',
     'preprocessor__numeric__scaler__copy': True,
     'preprocessor__numeric__scaler__with_mean': True,
     'preprocessor__numeric__scaler__with_std': True,
     'preprocessor__categorical_low__memory': None,
     'preprocessor__categorical_low__steps': [('imputer',
       SimpleImputer(fill_value='missing', strategy='constant')),
      ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False))],
     'preprocessor__categorical_low__verbose': False,
     'preprocessor__categorical_low__imputer': SimpleImputer(fill_value='missing', strategy='constant'),
     'preprocessor__categorical_low__encoding': OneHotEncoder(handle_unknown='ignore', sparse=False),
     'preprocessor__categorical_low__imputer__add_indicator': False,
     'preprocessor__categorical_low__imputer__copy': True,
     'preprocessor__categorical_low__imputer__fill_value': 'missing',
     'preprocessor__categorical_low__imputer__keep_empty_features': False,
     'preprocessor__categorical_low__imputer__missing_values': nan,
     'preprocessor__categorical_low__imputer__strategy': 'constant',
     'preprocessor__categorical_low__imputer__verbose': 'deprecated',
     'preprocessor__categorical_low__encoding__categories': 'auto',
     'preprocessor__categorical_low__encoding__drop': None,
     'preprocessor__categorical_low__encoding__dtype': numpy.float64,
     'preprocessor__categorical_low__encoding__handle_unknown': 'ignore',
     'preprocessor__categorical_low__encoding__max_categories': None,
     'preprocessor__categorical_low__encoding__min_frequency': None,
     'preprocessor__categorical_low__encoding__sparse': False,
     'preprocessor__categorical_low__encoding__sparse_output': True,
     'preprocessor__categorical_high__memory': None,
     'preprocessor__categorical_high__steps': [('imputer',
       SimpleImputer(fill_value='missing', strategy='constant')),
      ('encoding', OrdinalEncoder())],
     'preprocessor__categorical_high__verbose': False,
     'preprocessor__categorical_high__imputer': SimpleImputer(fill_value='missing', strategy='constant'),
     'preprocessor__categorical_high__encoding': OrdinalEncoder(),
     'preprocessor__categorical_high__imputer__add_indicator': False,
     'preprocessor__categorical_high__imputer__copy': True,
     'preprocessor__categorical_high__imputer__fill_value': 'missing',
     'preprocessor__categorical_high__imputer__keep_empty_features': False,
     'preprocessor__categorical_high__imputer__missing_values': nan,
     'preprocessor__categorical_high__imputer__strategy': 'constant',
     'preprocessor__categorical_high__imputer__verbose': 'deprecated',
     'preprocessor__categorical_high__encoding__categories': 'auto',
     'preprocessor__categorical_high__encoding__dtype': numpy.float64,
     'preprocessor__categorical_high__encoding__encoded_missing_value': nan,
     'preprocessor__categorical_high__encoding__handle_unknown': 'error',
     'preprocessor__categorical_high__encoding__unknown_value': None,
     'classifier__a': 0.01,
     'classifier__activation_name': 'relu',
     'classifier__backend': 'cpu',
     'classifier__bias': True,
     'classifier__cluster_encode': True,
     'classifier__col_sample': 0.9,
     'classifier__direct_link': True,
     'classifier__dropout': 0,
     'classifier__n_clusters': 2,
     'classifier__n_hidden_features': 10,
     'classifier__nodes_sim': 'sobol',
     'classifier__obj__C': 1.0,
     'classifier__obj__class_weight': None,
     'classifier__obj__dual': False,
     'classifier__obj__fit_intercept': True,
     'classifier__obj__intercept_scaling': 1,
     'classifier__obj__l1_ratio': None,
     'classifier__obj__max_iter': 100,
     'classifier__obj__multi_class': 'auto',
     'classifier__obj__n_jobs': None,
     'classifier__obj__penalty': 'l2',
     'classifier__obj__random_state': 42,
     'classifier__obj__solver': 'lbfgs',
     'classifier__obj__tol': 0.0001,
     'classifier__obj__verbose': 0,
     'classifier__obj__warm_start': False,
     'classifier__obj': LogisticRegression(random_state=42),
     'classifier__row_sample': 1,
     'classifier__seed': 123,
     'classifier__type_clust': 'kmeans',
     'classifier__type_scaling': ('std', 'std', 'std')}



# **3 - Regression**


```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
```


```python
data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ns.LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
```

    100%|██████████| 40/40 [00:03<00:00, 12.38it/s]



```python
display(models)
```



  <div id="df-695e43ac-df05-4d19-804c-be1fd887e834" class="colab-df-container">
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
      <th>LassoLarsIC</th>
      <td>0.53</td>
      <td>0.59</td>
      <td>51.11</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>SGDRegressor</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.24</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>HuberRegressor</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.26</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.37</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>KernelRidge</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.37</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>RidgeCV</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>51.37</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.52</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LassoLars</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.52</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LassoCV</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.58</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>LassoLarsCV</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.58</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>TransformedTargetRegressor</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.62</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LinearRegression</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.62</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>OrthogonalMatchingPursuitCV</th>
      <td>0.52</td>
      <td>0.58</td>
      <td>51.69</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>BayesianRidge</th>
      <td>0.52</td>
      <td>0.57</td>
      <td>51.77</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LinearSVR</th>
      <td>0.51</td>
      <td>0.57</td>
      <td>52.04</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>ElasticNetCV</th>
      <td>0.51</td>
      <td>0.56</td>
      <td>52.49</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>LarsCV</th>
      <td>0.50</td>
      <td>0.56</td>
      <td>52.79</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>PassiveAggressiveRegressor</th>
      <td>0.49</td>
      <td>0.55</td>
      <td>53.39</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GradientBoostingRegressor</th>
      <td>0.48</td>
      <td>0.54</td>
      <td>54.00</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>ElasticNet</th>
      <td>0.46</td>
      <td>0.52</td>
      <td>54.92</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>BaggingRegressor</th>
      <td>0.46</td>
      <td>0.52</td>
      <td>54.92</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>0.46</td>
      <td>0.52</td>
      <td>55.07</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>HistGradientBoostingRegressor</th>
      <td>0.45</td>
      <td>0.51</td>
      <td>55.42</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>ExtraTreesRegressor</th>
      <td>0.44</td>
      <td>0.51</td>
      <td>55.71</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>AdaBoostRegressor</th>
      <td>0.44</td>
      <td>0.51</td>
      <td>55.75</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>0.43</td>
      <td>0.50</td>
      <td>56.38</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>TweedieRegressor</th>
      <td>0.42</td>
      <td>0.48</td>
      <td>57.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>RANSACRegressor</th>
      <td>0.42</td>
      <td>0.48</td>
      <td>57.14</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>KNeighborsRegressor</th>
      <td>0.31</td>
      <td>0.39</td>
      <td>62.10</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>OrthogonalMatchingPursuit</th>
      <td>0.31</td>
      <td>0.38</td>
      <td>62.27</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>GaussianProcessRegressor</th>
      <td>0.19</td>
      <td>0.28</td>
      <td>67.13</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>ExtraTreeRegressor</th>
      <td>0.15</td>
      <td>0.24</td>
      <td>69.09</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>0.12</td>
      <td>0.22</td>
      <td>69.98</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>NuSVR</th>
      <td>0.12</td>
      <td>0.22</td>
      <td>70.14</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>DummyRegressor</th>
      <td>-0.13</td>
      <td>-0.00</td>
      <td>79.39</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>DecisionTreeRegressor</th>
      <td>-0.26</td>
      <td>-0.11</td>
      <td>83.75</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Lars</th>
      <td>-1.95</td>
      <td>-1.61</td>
      <td>128.28</td>
      <td>0.14</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-695e43ac-df05-4d19-804c-be1fd887e834')"
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
        document.querySelector('#df-695e43ac-df05-4d19-804c-be1fd887e834 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-695e43ac-df05-4d19-804c-be1fd887e834');
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


<div id="df-3eda0ebe-10d9-4980-9919-91c29452828d">
  <button class="colab-df-quickchart" onclick="quickchart('df-3eda0ebe-10d9-4980-9919-91c29452828d')"
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
        document.querySelector('#df-3eda0ebe-10d9-4980-9919-91c29452828d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary["LassoLarsIC"]
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_low&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_high&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OrdinalEncoder())]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;))])),
                (&#x27;regressor&#x27;, CustomRegressor(obj=LassoLarsIC()))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_low&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;)),
                                                 (&#x27;categorical_high&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoding&#x27;,
                                                                   OrdinalEncoder())]),
                                                  Int64Index([], dtype=&#x27;int64&#x27;))])),
                (&#x27;regressor&#x27;, CustomRegressor(obj=LassoLarsIC()))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;)),
                                (&#x27;categorical_low&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                strategy=&#x27;constant&#x27;)),
                                                 (&#x27;encoding&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,
                                                                sparse=False))]),
                                 Int64Index([], dtype=&#x27;int64&#x27;)),
                                (&#x27;categorical_high&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                strategy=&#x27;constant&#x27;)),
                                                 (&#x27;encoding&#x27;,
                                                  OrdinalEncoder())]),
                                 Int64Index([], dtype=&#x27;int64&#x27;))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-31" type="checkbox" ><label for="sk-estimator-id-31" class="sk-toggleable__label sk-toggleable__label-arrow">numeric</label><div class="sk-toggleable__content"><pre>Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-32" type="checkbox" ><label for="sk-estimator-id-32" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-33" type="checkbox" ><label for="sk-estimator-id-33" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-34" type="checkbox" ><label for="sk-estimator-id-34" class="sk-toggleable__label sk-toggleable__label-arrow">categorical_low</label><div class="sk-toggleable__content"><pre>Int64Index([], dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-35" type="checkbox" ><label for="sk-estimator-id-35" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value=&#x27;missing&#x27;, strategy=&#x27;constant&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-36" type="checkbox" ><label for="sk-estimator-id-36" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;, sparse=False)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-37" type="checkbox" ><label for="sk-estimator-id-37" class="sk-toggleable__label sk-toggleable__label-arrow">categorical_high</label><div class="sk-toggleable__content"><pre>Int64Index([], dtype=&#x27;int64&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-38" type="checkbox" ><label for="sk-estimator-id-38" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value=&#x27;missing&#x27;, strategy=&#x27;constant&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-39" type="checkbox" ><label for="sk-estimator-id-39" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-40" type="checkbox" ><label for="sk-estimator-id-40" class="sk-toggleable__label sk-toggleable__label-arrow">regressor: CustomRegressor</label><div class="sk-toggleable__content"><pre>CustomRegressor(obj=LassoLarsIC())</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-41" type="checkbox" ><label for="sk-estimator-id-41" class="sk-toggleable__label sk-toggleable__label-arrow">obj: LassoLarsIC</label><div class="sk-toggleable__content"><pre>LassoLarsIC()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-42" type="checkbox" ><label for="sk-estimator-id-42" class="sk-toggleable__label sk-toggleable__label-arrow">LassoLarsIC</label><div class="sk-toggleable__content"><pre>LassoLarsIC()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
model_dictionary["LassoLarsIC"].get_params()
```




    {'memory': None,
     'steps': [('preprocessor',
       ColumnTransformer(transformers=[('numeric',
                                        Pipeline(steps=[('imputer', SimpleImputer()),
                                                        ('scaler', StandardScaler())]),
                                        Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')),
                                       ('categorical_low',
                                        Pipeline(steps=[('imputer',
                                                         SimpleImputer(fill_value='missing',
                                                                       strategy='constant')),
                                                        ('encoding',
                                                         OneHotEncoder(handle_unknown='ignore',
                                                                       sparse=False))]),
                                        Int64Index([], dtype='int64')),
                                       ('categorical_high',
                                        Pipeline(steps=[('imputer',
                                                         SimpleImputer(fill_value='missing',
                                                                       strategy='constant')),
                                                        ('encoding',
                                                         OrdinalEncoder())]),
                                        Int64Index([], dtype='int64'))])),
      ('regressor', CustomRegressor(obj=LassoLarsIC()))],
     'verbose': False,
     'preprocessor': ColumnTransformer(transformers=[('numeric',
                                      Pipeline(steps=[('imputer', SimpleImputer()),
                                                      ('scaler', StandardScaler())]),
                                      Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')),
                                     ('categorical_low',
                                      Pipeline(steps=[('imputer',
                                                       SimpleImputer(fill_value='missing',
                                                                     strategy='constant')),
                                                      ('encoding',
                                                       OneHotEncoder(handle_unknown='ignore',
                                                                     sparse=False))]),
                                      Int64Index([], dtype='int64')),
                                     ('categorical_high',
                                      Pipeline(steps=[('imputer',
                                                       SimpleImputer(fill_value='missing',
                                                                     strategy='constant')),
                                                      ('encoding',
                                                       OrdinalEncoder())]),
                                      Int64Index([], dtype='int64'))]),
     'regressor': CustomRegressor(obj=LassoLarsIC()),
     'preprocessor__n_jobs': None,
     'preprocessor__remainder': 'drop',
     'preprocessor__sparse_threshold': 0.3,
     'preprocessor__transformer_weights': None,
     'preprocessor__transformers': [('numeric',
       Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]),
       Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')),
      ('categorical_low',
       Pipeline(steps=[('imputer',
                        SimpleImputer(fill_value='missing', strategy='constant')),
                       ('encoding',
                        OneHotEncoder(handle_unknown='ignore', sparse=False))]),
       Int64Index([], dtype='int64')),
      ('categorical_high',
       Pipeline(steps=[('imputer',
                        SimpleImputer(fill_value='missing', strategy='constant')),
                       ('encoding', OrdinalEncoder())]),
       Int64Index([], dtype='int64'))],
     'preprocessor__verbose': False,
     'preprocessor__verbose_feature_names_out': True,
     'preprocessor__numeric': Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]),
     'preprocessor__categorical_low': Pipeline(steps=[('imputer',
                      SimpleImputer(fill_value='missing', strategy='constant')),
                     ('encoding',
                      OneHotEncoder(handle_unknown='ignore', sparse=False))]),
     'preprocessor__categorical_high': Pipeline(steps=[('imputer',
                      SimpleImputer(fill_value='missing', strategy='constant')),
                     ('encoding', OrdinalEncoder())]),
     'preprocessor__numeric__memory': None,
     'preprocessor__numeric__steps': [('imputer', SimpleImputer()),
      ('scaler', StandardScaler())],
     'preprocessor__numeric__verbose': False,
     'preprocessor__numeric__imputer': SimpleImputer(),
     'preprocessor__numeric__scaler': StandardScaler(),
     'preprocessor__numeric__imputer__add_indicator': False,
     'preprocessor__numeric__imputer__copy': True,
     'preprocessor__numeric__imputer__fill_value': None,
     'preprocessor__numeric__imputer__keep_empty_features': False,
     'preprocessor__numeric__imputer__missing_values': nan,
     'preprocessor__numeric__imputer__strategy': 'mean',
     'preprocessor__numeric__imputer__verbose': 'deprecated',
     'preprocessor__numeric__scaler__copy': True,
     'preprocessor__numeric__scaler__with_mean': True,
     'preprocessor__numeric__scaler__with_std': True,
     'preprocessor__categorical_low__memory': None,
     'preprocessor__categorical_low__steps': [('imputer',
       SimpleImputer(fill_value='missing', strategy='constant')),
      ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False))],
     'preprocessor__categorical_low__verbose': False,
     'preprocessor__categorical_low__imputer': SimpleImputer(fill_value='missing', strategy='constant'),
     'preprocessor__categorical_low__encoding': OneHotEncoder(handle_unknown='ignore', sparse=False),
     'preprocessor__categorical_low__imputer__add_indicator': False,
     'preprocessor__categorical_low__imputer__copy': True,
     'preprocessor__categorical_low__imputer__fill_value': 'missing',
     'preprocessor__categorical_low__imputer__keep_empty_features': False,
     'preprocessor__categorical_low__imputer__missing_values': nan,
     'preprocessor__categorical_low__imputer__strategy': 'constant',
     'preprocessor__categorical_low__imputer__verbose': 'deprecated',
     'preprocessor__categorical_low__encoding__categories': 'auto',
     'preprocessor__categorical_low__encoding__drop': None,
     'preprocessor__categorical_low__encoding__dtype': numpy.float64,
     'preprocessor__categorical_low__encoding__handle_unknown': 'ignore',
     'preprocessor__categorical_low__encoding__max_categories': None,
     'preprocessor__categorical_low__encoding__min_frequency': None,
     'preprocessor__categorical_low__encoding__sparse': False,
     'preprocessor__categorical_low__encoding__sparse_output': True,
     'preprocessor__categorical_high__memory': None,
     'preprocessor__categorical_high__steps': [('imputer',
       SimpleImputer(fill_value='missing', strategy='constant')),
      ('encoding', OrdinalEncoder())],
     'preprocessor__categorical_high__verbose': False,
     'preprocessor__categorical_high__imputer': SimpleImputer(fill_value='missing', strategy='constant'),
     'preprocessor__categorical_high__encoding': OrdinalEncoder(),
     'preprocessor__categorical_high__imputer__add_indicator': False,
     'preprocessor__categorical_high__imputer__copy': True,
     'preprocessor__categorical_high__imputer__fill_value': 'missing',
     'preprocessor__categorical_high__imputer__keep_empty_features': False,
     'preprocessor__categorical_high__imputer__missing_values': nan,
     'preprocessor__categorical_high__imputer__strategy': 'constant',
     'preprocessor__categorical_high__imputer__verbose': 'deprecated',
     'preprocessor__categorical_high__encoding__categories': 'auto',
     'preprocessor__categorical_high__encoding__dtype': numpy.float64,
     'preprocessor__categorical_high__encoding__encoded_missing_value': nan,
     'preprocessor__categorical_high__encoding__handle_unknown': 'error',
     'preprocessor__categorical_high__encoding__unknown_value': None,
     'regressor__a': 0.01,
     'regressor__activation_name': 'relu',
     'regressor__backend': 'cpu',
     'regressor__bias': True,
     'regressor__cluster_encode': True,
     'regressor__col_sample': 1,
     'regressor__direct_link': True,
     'regressor__dropout': 0,
     'regressor__n_clusters': 2,
     'regressor__n_hidden_features': 5,
     'regressor__nodes_sim': 'sobol',
     'regressor__obj__copy_X': True,
     'regressor__obj__criterion': 'aic',
     'regressor__obj__eps': 2.220446049250313e-16,
     'regressor__obj__fit_intercept': True,
     'regressor__obj__max_iter': 500,
     'regressor__obj__noise_variance': None,
     'regressor__obj__normalize': 'deprecated',
     'regressor__obj__positive': False,
     'regressor__obj__precompute': 'auto',
     'regressor__obj__verbose': False,
     'regressor__obj': LassoLarsIC(),
     'regressor__row_sample': 1,
     'regressor__seed': 123,
     'regressor__type_clust': 'kmeans',
     'regressor__type_scaling': ('std', 'std', 'std')}


