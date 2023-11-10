---
layout: post
title: "A classifier that's very accurate (and deep)"
description: "nnetsauce version 0.15.0, an example of a deep-layered classifier calibrated with Automated Machine Learning (AutoML)"
date: 2023-11-10
categories: [Python, R, QuasiRandomizedNN]
comments: true
---

Version `v0.15.0` of `nnetsauce` is now available on your favorite platforms: [PyPI](https://pypi.org/project/nnetsauce/), [conda](https://anaconda.org/conda-forge/nnetsauce) and [GitHub](https://github.com/Techtonique/nnetsauce). The changes in this version are mostly related to  **Automated Machine learning (AutoML)**: 

- _lazy_ prediction for classification and regression: see [this post](https://thierrymoudiki.github.io/blog/2023/10/22/python/quasirandomizednn/nnetsauce-lazy-predict-preview) for more details (and remember to use `pip install nnetsauce` instead of installing from a GitHub branch named `lazy-predict`)
- _lazy_ prediction for multivariate time series (MTS): see [this post](https://thierrymoudiki.github.io/blog/2023/10/29/python/quasirandomizednn/MTS-LazyPredict) for more details (and remember to use `pip install nnetsauce` instead of installing from a GitHub branch named `lazy-predict`)
- _lazy_ prediction with **_deep_ quasi-randomized _nnetworks_** will be described in this post

Note that in the example below, for the offically released version (v0.15.0), Gradient boosting classifiers are available. This doesn't change the best model chosen by the algorithm (which is never Gradient boosting). To finish, for Windows users, if you run into issues when trying to install `nnetsauce`: remember that you can use the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). 

<h1 id="contents">Contents</h1>

<ul>
  <li> <a href="#0---install-and-import-packages">0 - Install and import packages</a> </li>
  <li> <a href="#1---breast-cancer-data">1 - breast cancer data</a> </li>
  <li> <a href="#2---iris-data">2 - iris data</a> </li>
  <li> <a href="#3---wine-data">3 - wine data</a> </li>
  <li> <a href="#4---digits-data">4 - digits data</a> </li>
</ul>

Here is a [jupyter notebook](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_091123_lazy_deep_classifier.ipynb) reproducing these results. Do not hesitate 
to modify these examples by choosing -- in `LazyDeepClassifier` -- a different number of layers `n_layers` or the number of 
engineered features _per_ layer, `n_hidden_features`. 


# 0 - Install and import packages 

<a href="#contents">top</a>

```python
!pip install nnetsauce --upgrade
```

```python
import os
import nnetsauce as ns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from time import time
```

# 1 - breast cancer data

<a href="#contents">top</a>

```python
data = load_breast_cancer()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
start = time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(f"\n\n Elapsed: {time()-start} seconds \n")
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)

display(models)
```

    100%|██████████| 27/27 [00:44<00:00,  1.65s/it]

    
    
     Elapsed: 44.49836850166321 seconds 
    


    




  <div id="df-872aae81-abd7-4a72-930a-2b6cb3bb0d18" class="colab-df-container">
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
      <th>Perceptron</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>2.76</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>1.89</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>3.13</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>1.69</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>3.42</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>1.65</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>0.97</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>1.82</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>0.96</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.96</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.96</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.96</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.96</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.96</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.96</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.96</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.96</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.93</td>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.93</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.93</td>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.93</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.93</td>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.93</td>
      <td>1.27</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.91</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.91</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.64</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-872aae81-abd7-4a72-930a-2b6cb3bb0d18')"
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
        document.querySelector('#df-872aae81-abd7-4a72-930a-2b6cb3bb0d18 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-872aae81-abd7-4a72-930a-2b6cb3bb0d18');
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


<div id="df-4ab2cd36-6861-443a-9038-4712c083df25">
  <button class="colab-df-quickchart" onclick="quickchart('df-4ab2cd36-6861-443a-9038-4712c083df25')"
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
        document.querySelector('#df-4ab2cd36-6861-443a-9038-4712c083df25 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary["Perceptron"]
```




<style>#sk-container-id-18 {color: black;background-color: white;}#sk-container-id-18 pre{padding: 0;}#sk-container-id-18 div.sk-toggleable {background-color: white;}#sk-container-id-18 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-18 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-18 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-18 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-18 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-18 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-18 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-18 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-18 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-18 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-18 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-18 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-18 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-18 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-18 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-18 div.sk-item {position: relative;z-index: 1;}#sk-container-id-18 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-18 div.sk-item::before, #sk-container-id-18 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-18 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-18 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-18 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-18 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-18 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-18 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-18 div.sk-label-container {text-align: center;}#sk-container-id-18 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-18 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-18" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=Perceptron(random_state=42))))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-86" type="checkbox" ><label for="sk-estimator-id-86" class="sk-toggleable__label sk-toggleable__label-arrow">CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=Perceptron(random_state=42))))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-87" type="checkbox" ><label for="sk-estimator-id-87" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=Perceptron(random_state=42)))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-88" type="checkbox" ><label for="sk-estimator-id-88" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=Perceptron(random_state=42))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-89" type="checkbox" ><label for="sk-estimator-id-89" class="sk-toggleable__label sk-toggleable__label-arrow">obj: Perceptron</label><div class="sk-toggleable__content"><pre>Perceptron(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-90" type="checkbox" ><label for="sk-estimator-id-90" class="sk-toggleable__label sk-toggleable__label-arrow">Perceptron</label><div class="sk-toggleable__content"><pre>Perceptron(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
print(classification_report(y_test, model_dictionary["Perceptron"].fit(X_train, y_train).predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.98      0.99        41
               1       0.99      1.00      0.99        73
    
        accuracy                           0.99       114
       macro avg       0.99      0.99      0.99       114
    weighted avg       0.99      0.99      0.99       114
    



```python
ConfusionMatrixDisplay.from_estimator(model_dictionary["Perceptron"], X_test, y_test)
plt.show()
```


![image-title-here]({{base}}/images/2023-11-12/2023-11-12-image1.png){:class="img-responsive"}


# 2 - iris data

<a href="#contents">top</a>

```python
data = load_iris()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
start = time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(f"\n\n Elapsed: {time()-start} seconds \n")
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
display(models)
```

    100%|██████████| 27/27 [00:05<00:00,  4.51it/s]

    
    
     Elapsed: 5.992894172668457 seconds 
    


    




  <div id="df-91f9a67c-e65b-41bf-9ca4-79f5826d1670" class="colab-df-container">
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
      <th>BaggingClassifier</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.93</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.93</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.93</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.93</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.93</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.93</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.90</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.90</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.90</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.90</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.87</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.87</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.87</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.87</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.63</td>
      <td>0.67</td>
      <td>None</td>
      <td>0.54</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.20</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.07</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.20</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.07</td>
      <td>0.10</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-91f9a67c-e65b-41bf-9ca4-79f5826d1670')"
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
        document.querySelector('#df-91f9a67c-e65b-41bf-9ca4-79f5826d1670 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-91f9a67c-e65b-41bf-9ca4-79f5826d1670');
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


<div id="df-39c80c73-686d-41f9-bf1d-2877ef4799cc">
  <button class="colab-df-quickchart" onclick="quickchart('df-39c80c73-686d-41f9-bf1d-2877ef4799cc')"
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
        document.querySelector('#df-39c80c73-686d-41f9-bf1d-2877ef4799cc button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary["BaggingClassifier"]
```




<style>#sk-container-id-19 {color: black;background-color: white;}#sk-container-id-19 pre{padding: 0;}#sk-container-id-19 div.sk-toggleable {background-color: white;}#sk-container-id-19 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-19 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-19 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-19 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-19 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-19 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-19 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-19 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-19 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-19 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-19 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-19 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-19 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-19 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-19 div.sk-item {position: relative;z-index: 1;}#sk-container-id-19 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-19 div.sk-item::before, #sk-container-id-19 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-19 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-19 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-19 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-19 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-19 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-19 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-19 div.sk-label-container {text-align: center;}#sk-container-id-19 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-19 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-19" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=BaggingClassifier(random_state=42))))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-91" type="checkbox" ><label for="sk-estimator-id-91" class="sk-toggleable__label sk-toggleable__label-arrow">CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=BaggingClassifier(random_state=42))))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-92" type="checkbox" ><label for="sk-estimator-id-92" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=BaggingClassifier(random_state=42)))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-93" type="checkbox" ><label for="sk-estimator-id-93" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=BaggingClassifier(random_state=42))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-94" type="checkbox" ><label for="sk-estimator-id-94" class="sk-toggleable__label sk-toggleable__label-arrow">obj: BaggingClassifier</label><div class="sk-toggleable__content"><pre>BaggingClassifier(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-95" type="checkbox" ><label for="sk-estimator-id-95" class="sk-toggleable__label sk-toggleable__label-arrow">BaggingClassifier</label><div class="sk-toggleable__content"><pre>BaggingClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
print(classification_report(y_test, model_dictionary["BaggingClassifier"].fit(X_train, y_train).predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        13
               1       1.00      1.00      1.00         6
               2       1.00      1.00      1.00        11
    
        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
    



```python
ConfusionMatrixDisplay.from_estimator(model_dictionary["BaggingClassifier"], X_test, y_test)
plt.show()
```


![image-title-here]({{base}}/images/2023-11-12/2023-11-12-image2.png){:class="img-responsive"}


# 3 - wine data

<a href="#contents">top</a>

```python
data = load_wine()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
start = time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(f"\n\n Elapsed: {time()-start} seconds \n")
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
display(models)
```

    100%|██████████| 27/27 [00:05<00:00,  5.09it/s]

    
    
     Elapsed: 5.312330007553101 seconds 
    


    




  <div id="df-87fc304d-4050-431a-b9f9-e653f933c18a" class="colab-df-container">
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
      <th>RidgeClassifierCV</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>None</td>
      <td>1.00</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.89</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.89</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.83</td>
      <td>0.87</td>
      <td>None</td>
      <td>0.83</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.86</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.85</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.86</td>
      <td>0.81</td>
      <td>None</td>
      <td>0.86</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.83</td>
      <td>0.79</td>
      <td>None</td>
      <td>0.84</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.81</td>
      <td>0.78</td>
      <td>None</td>
      <td>0.81</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.81</td>
      <td>0.75</td>
      <td>None</td>
      <td>0.80</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>None</td>
      <td>0.52</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.31</td>
      <td>0.33</td>
      <td>None</td>
      <td>0.14</td>
      <td>0.13</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-87fc304d-4050-431a-b9f9-e653f933c18a')"
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
        document.querySelector('#df-87fc304d-4050-431a-b9f9-e653f933c18a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-87fc304d-4050-431a-b9f9-e653f933c18a');
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


<div id="df-dfe4398b-a1e1-4a09-9fb8-c4c415410427">
  <button class="colab-df-quickchart" onclick="quickchart('df-dfe4398b-a1e1-4a09-9fb8-c4c415410427')"
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
        document.querySelector('#df-dfe4398b-a1e1-4a09-9fb8-c4c415410427 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary["RidgeClassifierCV"]
```




<style>#sk-container-id-20 {color: black;background-color: white;}#sk-container-id-20 pre{padding: 0;}#sk-container-id-20 div.sk-toggleable {background-color: white;}#sk-container-id-20 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-20 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-20 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-20 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-20 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-20 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-20 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-20 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-20 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-20 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-20 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-20 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-20 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-20 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-20 div.sk-item {position: relative;z-index: 1;}#sk-container-id-20 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-20 div.sk-item::before, #sk-container-id-20 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-20 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-20 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-20 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-20 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-20 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-20 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-20 div.sk-label-container {text-align: center;}#sk-container-id-20 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-20 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-20" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=RidgeClassifierCV())))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-96" type="checkbox" ><label for="sk-estimator-id-96" class="sk-toggleable__label sk-toggleable__label-arrow">CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=RidgeClassifierCV())))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-97" type="checkbox" ><label for="sk-estimator-id-97" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=RidgeClassifierCV()))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-98" type="checkbox" ><label for="sk-estimator-id-98" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=RidgeClassifierCV())</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-99" type="checkbox" ><label for="sk-estimator-id-99" class="sk-toggleable__label sk-toggleable__label-arrow">obj: RidgeClassifierCV</label><div class="sk-toggleable__content"><pre>RidgeClassifierCV()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-100" type="checkbox" ><label for="sk-estimator-id-100" class="sk-toggleable__label sk-toggleable__label-arrow">RidgeClassifierCV</label><div class="sk-toggleable__content"><pre>RidgeClassifierCV()</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
print(classification_report(y_test, model_dictionary["RidgeClassifierCV"].fit(X_train, y_train).predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         8
               1       1.00      1.00      1.00        11
               2       1.00      1.00      1.00        17
    
        accuracy                           1.00        36
       macro avg       1.00      1.00      1.00        36
    weighted avg       1.00      1.00      1.00        36
    



```python
ConfusionMatrixDisplay.from_estimator(model_dictionary["RidgeClassifierCV"], X_test, y_test)
plt.show()
```


![image-title-here]({{base}}/images/2023-11-12/2023-11-12-image3.png){:class="img-responsive"}


# 4 - digits data

<a href="#contents">top</a>

```python
data = load_digits()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
start = time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(f"\n\n Elapsed: {time()-start} seconds \n")
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
display(models)
```

    100%|██████████| 27/27 [01:41<00:00,  3.75s/it]

    
    
     Elapsed: 101.15918207168579 seconds 
    


    




  <div id="df-367bea27-c33f-4097-8edb-529f6ccc659b" class="colab-df-container">
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
      <th>SVC</th>
      <td>0.99</td>
      <td>0.99</td>
      <td>None</td>
      <td>0.99</td>
      <td>3.47</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>3.67</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>8.78</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.97</td>
      <td>0.97</td>
      <td>None</td>
      <td>0.97</td>
      <td>4.46</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>4.78</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>4.18</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.96</td>
      <td>0.96</td>
      <td>None</td>
      <td>0.96</td>
      <td>2.62</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>4.45</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.94</td>
      <td>4.38</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>None</td>
      <td>0.93</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.93</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.93</td>
      <td>4.55</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.92</td>
      <td>4.84</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.91</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.92</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.91</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.92</td>
      <td>3.64</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.88</td>
      <td>0.88</td>
      <td>None</td>
      <td>0.88</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.85</td>
      <td>4.27</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.83</td>
      <td>0.83</td>
      <td>None</td>
      <td>0.83</td>
      <td>3.36</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.82</td>
      <td>0.82</td>
      <td>None</td>
      <td>0.82</td>
      <td>2.85</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.81</td>
      <td>0.81</td>
      <td>None</td>
      <td>0.80</td>
      <td>4.63</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.81</td>
      <td>0.80</td>
      <td>None</td>
      <td>0.81</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.39</td>
      <td>0.37</td>
      <td>None</td>
      <td>0.32</td>
      <td>5.05</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.08</td>
      <td>0.10</td>
      <td>None</td>
      <td>0.01</td>
      <td>4.98</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-367bea27-c33f-4097-8edb-529f6ccc659b')"
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
        document.querySelector('#df-367bea27-c33f-4097-8edb-529f6ccc659b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-367bea27-c33f-4097-8edb-529f6ccc659b');
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


<div id="df-e1caf652-22ca-4a3f-bef7-3ca976401433">
  <button class="colab-df-quickchart" onclick="quickchart('df-e1caf652-22ca-4a3f-bef7-3ca976401433')"
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
        document.querySelector('#df-e1caf652-22ca-4a3f-bef7-3ca976401433 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
model_dictionary["SVC"]
```




<style>#sk-container-id-21 {color: black;background-color: white;}#sk-container-id-21 pre{padding: 0;}#sk-container-id-21 div.sk-toggleable {background-color: white;}#sk-container-id-21 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-21 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-21 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-21 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-21 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-21 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-21 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-21 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-21 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-21 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-21 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-21 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-21 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-21 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-21 div.sk-item {position: relative;z-index: 1;}#sk-container-id-21 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-21 div.sk-item::before, #sk-container-id-21 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-21 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-21 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-21 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-21 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-21 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-21 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-21 div.sk-label-container {text-align: center;}#sk-container-id-21 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-21 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-21" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=SVC(random_state=42))))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-101" type="checkbox" ><label for="sk-estimator-id-101" class="sk-toggleable__label sk-toggleable__label-arrow">CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=CustomClassifier(obj=SVC(random_state=42))))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-102" type="checkbox" ><label for="sk-estimator-id-102" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=CustomClassifier(obj=SVC(random_state=42)))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-103" type="checkbox" ><label for="sk-estimator-id-103" class="sk-toggleable__label sk-toggleable__label-arrow">obj: CustomClassifier</label><div class="sk-toggleable__content"><pre>CustomClassifier(obj=SVC(random_state=42))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-104" type="checkbox" ><label for="sk-estimator-id-104" class="sk-toggleable__label sk-toggleable__label-arrow">obj: SVC</label><div class="sk-toggleable__content"><pre>SVC(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-105" type="checkbox" ><label for="sk-estimator-id-105" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
print(classification_report(y_test, model_dictionary["SVC"].fit(X_train, y_train).predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        39
               1       0.94      1.00      0.97        34
               2       1.00      0.97      0.99        36
               3       1.00      1.00      1.00        33
               4       0.95      1.00      0.98        42
               5       1.00      0.95      0.97        37
               6       1.00      1.00      1.00        43
               7       1.00      1.00      1.00        31
               8       1.00      0.95      0.97        37
               9       0.97      1.00      0.98        28
    
        accuracy                           0.99       360
       macro avg       0.99      0.99      0.99       360
    weighted avg       0.99      0.99      0.99       360
    



```python
ConfusionMatrixDisplay.from_estimator(model_dictionary["SVC"], X_test, y_test)
plt.show()
```


![image-title-here]({{base}}/images/2023-11-12/2023-11-12-image4.png){:class="img-responsive"}



```python
ConfusionMatrixDisplay.from_estimator(model_dictionary["LogisticRegression"], X_test, y_test)
plt.show()
```


![image-title-here]({{base}}/images/2023-11-12/2023-11-12-image5.png){:class="img-responsive"}

