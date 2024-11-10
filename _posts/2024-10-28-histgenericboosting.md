---
layout: post
title: "Gradient-Boosting anything (alert: high performance): Part3, Histogram-based boosting"
description: "Gradient boosting with any regression algorithm in Python and R package mlsauce. Part3, Histogram-based boosting"
date: 2024-10-28
categories: [Python, R, QuasiRandomizedNN]
comments: true
---
**Update 2024-10-29**: Fixed an error, and the histogram-based is actually failing miserably. Still trying to wrap my head around it (why is it not only failing, but so badly). The original implementation of the **`GenericBooster` is still doing great** as shown below.

A few weeks ago, I introduced a model-agnostic gradient boosting (XGBoost, LightGBM, CatBoost-like) procedure for supervised regression and classification, that can use any base learner (available in R and Python package `mlsauce`): 

- [https://thierrymoudiki.github.io/blog/2024/10/06/python/r/genericboosting](https://thierrymoudiki.github.io/blog/2024/10/06/python/r/genericboosting)
- [https://thierrymoudiki.github.io/blog/2024/10/14/r/genericboosting-r](https://thierrymoudiki.github.io/blog/2024/10/14/r/genericboosting-r)

The rationale is different from other histogram-based gradient boosting algorithms, as histograms are _only_ used here for **feature engineering of continuous features**. So far, I don't see huge differences with the original implementation of the `GenericBooster`, but it's still a work in progress. I envisage to try it out on a data set that contains a 'higher' mix of continuous and categorical features (as categorical features are not _histogram-engineered_).

Here are a few results that can give you an idea of the performance of the algorithm. Keep in mind that the models are not tuned, and that the `GenericBooster` can be tuned (in addition to the boosting model's hyperparamters) with the base learner's hyperparameters. That makes, potentially, a lot of degrees of freedom and room for improvement/exploration. 

```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose --upgrade --no-cache-dir
```


```python
import os
import mlsauce as ms
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from time import time

load_models = [load_breast_cancer, load_iris, load_wine, load_digits]

for model in load_models:

    data = model()
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    clf = ms.LazyBoostingClassifier(verbose=0, ignore_warnings=True, #n_jobs=2,
                                    custom_metric=None, preprocess=False)

    start = time()
    models, predictioms = clf.fit(X_train, X_test, y_train, y_test, hist=True)
    models2, predictioms = clf.fit(X_train, X_test, y_train, y_test, hist=False)
    print(f"\nElapsed: {time() - start} seconds\n")

    display(models)
    display(models2)
```

    2it [00:00,  2.27it/s]
    100%|██████████| 38/38 [00:41<00:00,  1.09s/it]
    2it [00:00,  5.14it/s]
    100%|██████████| 38/38 [00:43<00:00,  1.14s/it]

    
    Elapsed: 85.95083284378052 seconds
    


    




  <div id="df-b9197984-0fec-4ecd-a9d2-7a70c1735f73" class="colab-df-container">
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
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.37</td>
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
      <th>GenericBooster(RidgeCV)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.28</td>
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
      <th>XGBClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.94</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.94</td>
      <td>4.97</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.87</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.87</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>0.88</td>
      <td>0.87</td>
      <td>2.24</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.87</td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.86</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.86</td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.85</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.85</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.84</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.85</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.84</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.82</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.80</td>
      <td>10.42</td>
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
      <td>0.08</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.81</td>
      <td>0.69</td>
      <td>0.69</td>
      <td>0.78</td>
      <td>14.75</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.68</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.56</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.46</td>
      <td>0.51</td>
      <td>1.67</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b9197984-0fec-4ecd-a9d2-7a70c1735f73')"
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
        document.querySelector('#df-b9197984-0fec-4ecd-a9d2-7a70c1735f73 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b9197984-0fec-4ecd-a9d2-7a70c1735f73');
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


<div id="df-9af71557-b4cc-44ba-8f62-fdd6e5393341">
  <button class="colab-df-quickchart" onclick="quickchart('df-9af71557-b4cc-44ba-8f62-fdd6e5393341')"
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
        document.querySelector('#df-9af71557-b4cc-44ba-8f62-fdd6e5393341 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_7fa3aef9-d639-4fa5-b88c-df337d8d1035">
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
        document.querySelector('#id_7fa3aef9-d639-4fa5-b88c-df337d8d1035 button.colab-df-generate');
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





  <div id="df-ad02b617-6c48-47f6-83ca-cce2b9ebfe2f" class="colab-df-container">
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
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.67</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.94</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.94</td>
      <td>7.81</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.92</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.92</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.87</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.87</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>0.88</td>
      <td>0.87</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.87</td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.86</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.86</td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.85</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.85</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.84</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.85</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.84</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.82</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.80</td>
      <td>10.57</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.79</td>
      <td>0.09</td>
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
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.81</td>
      <td>0.69</td>
      <td>0.69</td>
      <td>0.78</td>
      <td>14.20</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.68</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.56</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.46</td>
      <td>0.51</td>
      <td>1.33</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ad02b617-6c48-47f6-83ca-cce2b9ebfe2f')"
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
        document.querySelector('#df-ad02b617-6c48-47f6-83ca-cce2b9ebfe2f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ad02b617-6c48-47f6-83ca-cce2b9ebfe2f');
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


<div id="df-abec044e-44dc-4a0c-bb87-e40630552a7f">
  <button class="colab-df-quickchart" onclick="quickchart('df-abec044e-44dc-4a0c-bb87-e40630552a7f')"
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
        document.querySelector('#df-abec044e-44dc-4a0c-bb87-e40630552a7f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_53cb3359-d7a0-4b1e-8095-2af1d21a8b35">
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
    <button class="colab-df-generate" onclick="generateWithVariable('predictioms')"
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
        document.querySelector('#id_53cb3359-d7a0-4b1e-8095-2af1d21a8b35 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('predictioms');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    2it [00:00,  6.46it/s]
    100%|██████████| 38/38 [00:12<00:00,  3.11it/s]
    2it [00:00, 10.38it/s]
    100%|██████████| 38/38 [00:11<00:00,  3.18it/s]

    
    Elapsed: 24.71835470199585 seconds
    


    




  <div id="df-c4ff18f2-25d3-40d8-bfd9-13173b638bdd" class="colab-df-container">
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
      <th>GenericBooster(RidgeCV)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.93</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.93</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.90</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.90</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.90</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.90</td>
      <td>1.61</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.80</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.80</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.80</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.80</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.63</td>
      <td>0.72</td>
      <td>None</td>
      <td>0.57</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.57</td>
      <td>0.67</td>
      <td>None</td>
      <td>0.45</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.50</td>
      <td>0.46</td>
      <td>None</td>
      <td>0.48</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.43</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.26</td>
      <td>2.19</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
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
      <th>GenericBooster(ElasticNet)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-c4ff18f2-25d3-40d8-bfd9-13173b638bdd')"
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
        document.querySelector('#df-c4ff18f2-25d3-40d8-bfd9-13173b638bdd button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c4ff18f2-25d3-40d8-bfd9-13173b638bdd');
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


<div id="df-0f9a9a0f-3cb9-454d-8afc-7f2fd0c7c11b">
  <button class="colab-df-quickchart" onclick="quickchart('df-0f9a9a0f-3cb9-454d-8afc-7f2fd0c7c11b')"
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
        document.querySelector('#df-0f9a9a0f-3cb9-454d-8afc-7f2fd0c7c11b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_532a8704-8c84-413c-84f2-34f737983c28">
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
        document.querySelector('#id_532a8704-8c84-413c-84f2-34f737983c28 button.colab-df-generate');
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





  <div id="df-0a3bb6bd-7f81-4a71-bd39-97f8f8c405bb" class="colab-df-container">
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
      <th>GenericBooster(RidgeCV)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.93</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.90</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.90</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.90</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.90</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.80</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.80</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.80</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.80</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.63</td>
      <td>0.72</td>
      <td>None</td>
      <td>0.57</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.57</td>
      <td>0.67</td>
      <td>None</td>
      <td>0.45</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.50</td>
      <td>0.46</td>
      <td>None</td>
      <td>0.48</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.43</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.26</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
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
      <th>GenericBooster(ElasticNet)</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.11</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-0a3bb6bd-7f81-4a71-bd39-97f8f8c405bb')"
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
        document.querySelector('#df-0a3bb6bd-7f81-4a71-bd39-97f8f8c405bb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0a3bb6bd-7f81-4a71-bd39-97f8f8c405bb');
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


<div id="df-148b7175-8bc2-41e3-80d3-d01a875e7a0e">
  <button class="colab-df-quickchart" onclick="quickchart('df-148b7175-8bc2-41e3-80d3-d01a875e7a0e')"
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
        document.querySelector('#df-148b7175-8bc2-41e3-80d3-d01a875e7a0e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_bc8689ec-915d-487d-9d29-65d2a4df285f">
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
    <button class="colab-df-generate" onclick="generateWithVariable('predictioms')"
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
        document.querySelector('#id_bc8689ec-915d-487d-9d29-65d2a4df285f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('predictioms');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    2it [00:00,  5.45it/s]
    100%|██████████| 38/38 [00:14<00:00,  2.63it/s]
    2it [00:00,  9.26it/s]
    100%|██████████| 38/38 [00:14<00:00,  2.58it/s]

    
    Elapsed: 29.76035761833191 seconds
    


    




  <div id="df-c1b766f2-9f13-4e0f-885b-8dd563162984" class="colab-df-container">
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
      <td>0.30</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.17</td>
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
      <th>GenericBooster(RidgeCV)</th>
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
      <td>0.15</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.15</td>
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
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>3.71</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.86</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.95</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.92</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.69</td>
      <td>0.61</td>
      <td>None</td>
      <td>0.61</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.61</td>
      <td>0.53</td>
      <td>None</td>
      <td>0.53</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.42</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.25</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
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
      <td>0.01</td>
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
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.25</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.10</td>
      <td>2.73</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c1b766f2-9f13-4e0f-885b-8dd563162984')"
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
        document.querySelector('#df-c1b766f2-9f13-4e0f-885b-8dd563162984 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c1b766f2-9f13-4e0f-885b-8dd563162984');
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


<div id="df-9497af82-f7e0-4991-8bba-2c63cf5d6f93">
  <button class="colab-df-quickchart" onclick="quickchart('df-9497af82-f7e0-4991-8bba-2c63cf5d6f93')"
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
        document.querySelector('#df-9497af82-f7e0-4991-8bba-2c63cf5d6f93 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_249a5758-edd1-49bf-8cd7-4e0702d3b83d">
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
        document.querySelector('#id_249a5758-edd1-49bf-8cd7-4e0702d3b83d button.colab-df-generate');
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





  <div id="df-be416627-a3d4-4b3e-90d0-a5abb204257a" class="colab-df-container">
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
      <td>0.15</td>
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
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.15</td>
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
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>3.41</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.95</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.92</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.20</td>
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
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.69</td>
      <td>0.61</td>
      <td>None</td>
      <td>0.61</td>
      <td>0.03</td>
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
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.42</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.25</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
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
      <td>0.01</td>
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
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.25</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.10</td>
      <td>2.78</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-be416627-a3d4-4b3e-90d0-a5abb204257a')"
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
        document.querySelector('#df-be416627-a3d4-4b3e-90d0-a5abb204257a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-be416627-a3d4-4b3e-90d0-a5abb204257a');
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


<div id="df-a8769dc2-b88a-43fb-8bc4-4a478107b97d">
  <button class="colab-df-quickchart" onclick="quickchart('df-a8769dc2-b88a-43fb-8bc4-4a478107b97d')"
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
        document.querySelector('#df-a8769dc2-b88a-43fb-8bc4-4a478107b97d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_8c998e05-306b-4bb2-9771-7063c4a5acf0">
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
    <button class="colab-df-generate" onclick="generateWithVariable('predictioms')"
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
        document.querySelector('#id_8c998e05-306b-4bb2-9771-7063c4a5acf0 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('predictioms');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    2it [00:01,  1.90it/s]
    100%|██████████| 38/38 [09:30<00:00, 15.02s/it]
    2it [00:01,  1.03it/s]
    100%|██████████| 38/38 [09:27<00:00, 14.94s/it]

    
    Elapsed: 1141.7054164409637 seconds
    


    




  <div id="df-589ff3a6-9d19-4831-ab3f-a7a011f34bb0" class="colab-df-container">
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
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.95</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.95</td>
      <td>4.34</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>4.47</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>51.97</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>2.54</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>4.55</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.93</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.93</td>
      <td>13.86</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.88</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.88</td>
      <td>6.14</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.79</td>
      <td>0.79</td>
      <td>None</td>
      <td>0.80</td>
      <td>13.46</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.37</td>
      <td>0.39</td>
      <td>None</td>
      <td>0.26</td>
      <td>297.07</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.20</td>
      <td>0.20</td>
      <td>None</td>
      <td>0.21</td>
      <td>19.23</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.12</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.03</td>
      <td>140.91</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.10</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.06</td>
      <td>9.46</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-589ff3a6-9d19-4831-ab3f-a7a011f34bb0')"
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
        document.querySelector('#df-589ff3a6-9d19-4831-ab3f-a7a011f34bb0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-589ff3a6-9d19-4831-ab3f-a7a011f34bb0');
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


<div id="df-57046e92-ddc1-48bd-ae7e-4125d199e9b3">
  <button class="colab-df-quickchart" onclick="quickchart('df-57046e92-ddc1-48bd-ae7e-4125d199e9b3')"
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
        document.querySelector('#df-57046e92-ddc1-48bd-ae7e-4125d199e9b3 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_3bfd0acd-408b-4254-a168-cd55057dba6b">
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
        document.querySelector('#id_3bfd0acd-408b-4254-a168-cd55057dba6b button.colab-df-generate');
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





  <div id="df-b34028a0-fdcc-44a3-b90f-147c3f2ae608" class="colab-df-container">
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
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.27</td>
    </tr>
    <tr>
      <th>GenericBooster(ExtraTreeRegressor)</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>1.69</td>
    </tr>
    <tr>
      <th>GenericBooster(KNeighborsRegressor)</th>
      <td>0.95</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.95</td>
      <td>4.76</td>
    </tr>
    <tr>
      <th>GenericBooster(LinearRegression)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>2.01</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(BayesianRidge))</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>46.87</td>
    </tr>
    <tr>
      <th>GenericBooster(TransformedTargetRegressor)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>5.40</td>
    </tr>
    <tr>
      <th>GenericBooster(RidgeCV)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>3.93</td>
    </tr>
    <tr>
      <th>GenericBooster(Ridge)</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(TweedieRegressor))</th>
      <td>0.93</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.93</td>
      <td>14.96</td>
    </tr>
    <tr>
      <th>GenericBooster(DecisionTreeRegressor)</th>
      <td>0.88</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.88</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(PassiveAggressiveRegressor))</th>
      <td>0.79</td>
      <td>0.79</td>
      <td>None</td>
      <td>0.80</td>
      <td>12.68</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(LinearSVR))</th>
      <td>0.37</td>
      <td>0.39</td>
      <td>None</td>
      <td>0.26</td>
      <td>294.88</td>
    </tr>
    <tr>
      <th>GenericBooster(Lars)</th>
      <td>0.20</td>
      <td>0.20</td>
      <td>None</td>
      <td>0.21</td>
      <td>19.40</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(QuantileRegressor))</th>
      <td>0.12</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.03</td>
      <td>145.91</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTask(SGDRegressor))</th>
      <td>0.10</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.06</td>
      <td>10.30</td>
    </tr>
    <tr>
      <th>GenericBooster(LassoLars)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(Lasso)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskLasso)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(ElasticNet)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GenericBooster(DummyRegressor)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GenericBooster(MultiTaskElasticNet)</th>
      <td>0.07</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b34028a0-fdcc-44a3-b90f-147c3f2ae608')"
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
        document.querySelector('#df-b34028a0-fdcc-44a3-b90f-147c3f2ae608 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b34028a0-fdcc-44a3-b90f-147c3f2ae608');
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


<div id="df-6956fbcc-da62-4e29-84f4-0227afb51069">
  <button class="colab-df-quickchart" onclick="quickchart('df-6956fbcc-da62-4e29-84f4-0227afb51069')"
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
        document.querySelector('#df-6956fbcc-da62-4e29-84f4-0227afb51069 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_c943059a-ddb8-4997-a228-f0a702a64dbb">
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
    <button class="colab-df-generate" onclick="generateWithVariable('predictioms')"
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
        document.querySelector('#id_c943059a-ddb8-4997-a228-f0a702a64dbb button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('predictioms');
      }
      })();
    </script>
  </div>

    </div>
  </div>


![xxx]({{base}}/images/2024-10-14/2024-10-14-image1.png){:class="img-responsive"}  