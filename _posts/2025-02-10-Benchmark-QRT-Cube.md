---
layout: post
title: "Model-agnostic global Survival Prediction of Patients with Myeloid Leukemia in QRT/Gustave Roussy Challenge (challengedata.ens.fr): Python's survivalist Quickstart"
description: "Model-agnostic global Survival Prediction of Patients with Myeloid Leukemia in QRT/Gustave Roussy Challenge (challengedata.ens.fr): Python's survivalist Quickstart"
date: 2025-02-10
categories: Python
comments: true
---

In this post I provide a quickstart Python code based on the one provided in [https://challengedata.ens.fr/challenges/162](https://challengedata.ens.fr/challenges/162). In particular, I use [Python package survivalist](https://github.com/Techtonique/survivalist), that does Survival analysis with Machine Learning and uncertainty quantification.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Logo-gustave-roussy.jpg/1200px-Logo-gustave-roussy.jpg" alt="Logo 1" width="250"/>
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/Qube_Research_%26_Technologies_Logo.svg/1200px-Qube_Research_%26_Technologies_Logo.svg.png" alt="Logo 2" width="200" style="margin-left: 20px;"/>
</p>

# Data Challenge : Leukemia Risk Prediction


*GOAL OF THE CHALLENGE and WHY IT IS IMPORTANT:*

The goal of the challenge is to **predict disease risk for patients with blood cancer**, in the context of specific subtypes of adult myeloid leukemias.

The risk is measured through the **overall survival** of patients, i.e. the duration of survival from the diagnosis of the blood cancer to the time of death or last follow-up.

Estimating the prognosis of patients is critical for an optimal clinical management.
For exemple, patients with low risk-disease will be offered supportive care to improve blood counts and quality of life, while patients with high-risk disease will be considered for hematopoietic stem cell transplantion.

The performance metric used in the challenge is the **IPCW-C-Index**.

*THE DATASETS*

The **training set is made of 3,323 patients**.

The **test set is made of 1,193 patients**.

For each patient, you have acces to CLINICAL data and MOLECULAR data.

The details of the data are as follows:

- OUTCOME:
  * OS_YEARS = Overall survival time in years
  * OS_STATUS = 1 (death) , 0 (alive at the last follow-up)

- CLINICAL DATA, with one line per patient:
  
  * ID = unique identifier per patient
  * CENTER = clinical center
  * BM_BLAST = Bone marrow blasts in % (blasts are abnormal blood cells)
  * WBC = White Blood Cell count in Giga/L
  * ANC = Absolute Neutrophil count in Giga/L
  * MONOCYTES = Monocyte count in Giga/L
  * HB = Hemoglobin in g/dL
  * PLT = Platelets coutn in Giga/L
  * CYTOGENETICS = A description of the karyotype observed in the blood cells of the patients, measured by a cytogeneticist. Cytogenetics is the science of chromosomes. A karyotype is performed from the blood tumoral cells. The convention for notation is ISCN (https://en.wikipedia.org/wiki/International_System_for_Human_Cytogenomic_Nomenclature). Cytogenetic notation are: https://en.wikipedia.org/wiki/Cytogenetic_notation. Note that a karyotype can be normal or abnornal. The notation 46,XX denotes a normal karyotype in females (23 pairs of chromosomes including 2 chromosomes X) and 46,XY in males (23 pairs of chromosomes inclusing 1 chromosme X and 1 chromsome Y). A common abnormality in the blood cancerous cells might be for exemple a loss of chromosome 7 (monosomy 7, or -7), which is typically asssociated with higher risk disease

- GENE MOLECULAR DATA, with one line per patient per somatic mutation. Mutations are detected from the sequencing of the blood tumoral cells.
We call somatic (= acquired) mutations the mutations that are found in the tumoral cells but not in other cells of the body.

  * ID = unique identifier per patient
  * CHR START END = position of the mutation on the human genome
  * REF ALT = reference and alternate (=mutant) nucleotide
  * GENE = the affected gene
  * PROTEIN_CHANGE = the consequence of the mutation on the protei that is expressed by a given gene
  * EFFECT = a broad categorization of the mutation consequences on a given gene.
  * VAF = Variant Allele Fraction = it represents the **proportion** of cells with the deleterious mutations.


```python
!pip install survivalist --upgrade --no-cache-dir --verbose
```


```python
!pip install scikit-survival
```


```python
!pip install nnetsauce
```


```python
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

import os
os.environ['PATH'] = f"/root/.cargo/bin:{os.environ['PATH']}"

!echo $PATH
!rustc --version
!cargo --version


```


```python
!pip install glmnetforpython --verbose --upgrade --no-cache-dir
```


```python
!pip install git+https://github.com/Techtonique/genbooster.git --upgrade --no-cache-dir
```


```python
import numpy as np
import pandas as pd
import glmnetforpython as glmnet
from matplotlib import pyplot as plt
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster.genboosterregressor import BoosterRegressor
from genbooster.randombagregressor import RandomBagRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.utils.discovery import all_estimators
from time import time
```


```python
from survivalist.nonparametric import kaplan_meier_estimator
from survivalist.datasets import load_whas500, load_gbsg2, load_veterans_lung_cancer
from survivalist.custom import SurvivalCustom
from survivalist.custom import PISurvivalCustom
from survivalist.ensemble import GradientBoostingSurvivalAnalysis
```


```python

```


```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from sklearn.impute import SimpleImputer
from sksurv.util import Surv

# Clinical Data
df = pd.read_csv("./clinical_train.csv")
df_eval = pd.read_csv("./clinical_test.csv")

# Molecular Data
maf_df = pd.read_csv("./molecular_train.csv")
maf_eval = pd.read_csv("./molecular_test.csv")

target_df = pd.read_csv("./target_train.csv")
#target_df_test = pd.read_csv("./target_test.csv")

# Preview the data
df.head()
```





  <div id="df-5df4c0c1-de39-4c7a-ad7d-49b45050a3f0" class="colab-df-container">
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
      <th>ID</th>
      <th>CENTER</th>
      <th>BM_BLAST</th>
      <th>WBC</th>
      <th>ANC</th>
      <th>MONOCYTES</th>
      <th>HB</th>
      <th>PLT</th>
      <th>CYTOGENETICS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P132697</td>
      <td>MSK</td>
      <td>14.00</td>
      <td>2.80</td>
      <td>0.20</td>
      <td>0.70</td>
      <td>7.60</td>
      <td>119.00</td>
      <td>46,xy,del(20)(q12)[2]/46,xy[18]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P132698</td>
      <td>MSK</td>
      <td>1.00</td>
      <td>7.40</td>
      <td>2.40</td>
      <td>0.10</td>
      <td>11.60</td>
      <td>42.00</td>
      <td>46,xx</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P116889</td>
      <td>MSK</td>
      <td>15.00</td>
      <td>3.70</td>
      <td>2.10</td>
      <td>0.10</td>
      <td>14.20</td>
      <td>81.00</td>
      <td>46,xy,t(3;3)(q25;q27)[8]/46,xy[12]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P132699</td>
      <td>MSK</td>
      <td>1.00</td>
      <td>3.90</td>
      <td>1.90</td>
      <td>0.10</td>
      <td>8.90</td>
      <td>77.00</td>
      <td>46,xy,del(3)(q26q27)[15]/46,xy[5]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P132700</td>
      <td>MSK</td>
      <td>6.00</td>
      <td>128.00</td>
      <td>9.70</td>
      <td>0.90</td>
      <td>11.10</td>
      <td>195.00</td>
      <td>46,xx,t(3;9)(p13;q22)[10]/46,xx[10]</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5df4c0c1-de39-4c7a-ad7d-49b45050a3f0')"
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
        document.querySelector('#df-5df4c0c1-de39-4c7a-ad7d-49b45050a3f0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5df4c0c1-de39-4c7a-ad7d-49b45050a3f0');
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


<div id="df-cfc22259-218c-4e52-9fdc-c3aae5f3fc0a">
  <button class="colab-df-quickchart" onclick="quickchart('df-cfc22259-218c-4e52-9fdc-c3aae5f3fc0a')"
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
        document.querySelector('#df-cfc22259-218c-4e52-9fdc-c3aae5f3fc0a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### Step 1: Data Preparation (clinical data only)

For survival analysis, we’ll format the dataset so that OS_YEARS represents the time variable and OS_STATUS represents the event indicator.


```python
# Drop rows where 'OS_YEARS' is NaN if conversion caused any issues
target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)

# Check the data types to ensure 'OS_STATUS' is boolean and 'OS_YEARS' is numeric
print(target_df[['OS_STATUS', 'OS_YEARS']].dtypes)

# Contarget_dfvert 'OS_YEARS' to numeric if it isn’t already
target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')

# Ensure 'OS_STATUS' is boolean
target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

# Select features
features = ['BM_BLAST', 'HB', 'PLT']
target = ['OS_YEARS', 'OS_STATUS']

# Create the survival data format
X = df.loc[df['ID'].isin(target_df['ID']), features]
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)
```

    OS_STATUS    float64
    OS_YEARS     float64
    dtype: object


### Step 2: Splitting the Dataset
We’ll split the data into training and testing sets to evaluate the model’s performance.


```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
# Survival-aware imputation for missing values
imputer = SimpleImputer(strategy="median")
X_train[['BM_BLAST', 'HB', 'PLT']] = imputer.fit_transform(X_train[['BM_BLAST', 'HB', 'PLT']])
X_test[['BM_BLAST', 'HB', 'PLT']] = imputer.transform(X_test[['BM_BLAST', 'HB', 'PLT']])
```


```python
display(X_train.head())
display(y_train)
```



  <div id="df-15c1d8b9-868f-4d6c-935a-eb22ffc99d5d" class="colab-df-container">
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
      <th>BM_BLAST</th>
      <th>HB</th>
      <th>PLT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1048</th>
      <td>3.00</td>
      <td>9.10</td>
      <td>150.00</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>15.00</td>
      <td>11.00</td>
      <td>45.00</td>
    </tr>
    <tr>
      <th>214</th>
      <td>6.00</td>
      <td>6.90</td>
      <td>132.00</td>
    </tr>
    <tr>
      <th>2135</th>
      <td>2.00</td>
      <td>10.00</td>
      <td>178.00</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>10.00</td>
      <td>10.00</td>
      <td>53.00</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-15c1d8b9-868f-4d6c-935a-eb22ffc99d5d')"
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
        document.querySelector('#df-15c1d8b9-868f-4d6c-935a-eb22ffc99d5d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-15c1d8b9-868f-4d6c-935a-eb22ffc99d5d');
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


<div id="df-ea864a78-c64f-43ea-90b3-d17e14a6e746">
  <button class="colab-df-quickchart" onclick="quickchart('df-ea864a78-c64f-43ea-90b3-d17e14a6e746')"
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
        document.querySelector('#df-ea864a78-c64f-43ea-90b3-d17e14a6e746 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




    array([(False, 1.91780822), ( True, 1.28219178), ( True, 1.49041096), ...,
           (False, 8.63561644), (False, 0.47671233), (False, 1.29041096)],
          dtype=[('OS_STATUS', '?'), ('OS_YEARS', '<f8')])


### Step 3: Cox Proportional Hazards Model

To account for censoring in survival analysis, we use a Cox Proportional Hazards (Cox PH) model, a widely used method that estimates the effect of covariates on survival times without assuming a specific baseline survival distribution. The Cox PH model is based on the hazard function, $h(t | X)$, which represents the instantaneous risk of an event (e.g., death) at time $t$ given covariates $X$. The model assumes that the hazard can be expressed as:

$$h(t | X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)$$


where $h_0(t)$ is the baseline hazard function, and $\beta$ values are coefficients for each covariate, representing the effect of $X$ on the hazard. Importantly, the proportional hazards assumption implies that the hazard ratios between individuals are constant over time. This approach effectively leverages both observed and censored survival times, making it a more suitable method for survival data compared to standard regression techniques that ignore censoring.



```python
# Initialize and train the Cox Proportional Hazards model
cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

# Evaluate the model using Concordance Index IPCW
cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.5f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
```

    Cox Proportional Hazard Model Concordance Index IPCW on train: 0.66
    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.66


### Step 4: other models

#### 4 - 1 demo


```python
import xgboost as xgb
import lightgbm as lgb

# Initialize and train the XGBoost model


event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=xgb.XGBRegressor(),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()

# Evaluate the model using Concordance Index IPCW
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).mean, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).lower, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).upper, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
```

![img1]({{base}}/images/2025-02-10/2025_02_09_Benchmark_QRT_Cube_26_0.png){:class="img-responsive"}
    


    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.60130
    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.60106
    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.59588



```python
import lightgbm as lgb

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")
estimator = PISurvivalCustom(regr=lgb.LGBMRegressor(verbose=0),
                             type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X_test.iloc[:1])

for fn in surv_funcs.mean:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.fill_between(fn.x, surv_funcs.lower[0].y, surv_funcs.upper[0].y, alpha=0.25, color="lightblue", step="post")
    plt.step(km[0], km[1], where="post", color="red", label="Kaplan-Meier")
    plt.fill_between(km[0], km[2][0], km[2][1], alpha=0.25, color="pink", step="post")
    plt.ylim(0, 1)
    plt.show()

# Evaluate the model using Concordance Index IPCW
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).mean, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).lower, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
cox_cindex_test = concordance_index_ipcw(y_train, y_test, estimator.predict(X_test.iloc[:]).upper, tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.5f}")
```


![img2]({{base}}/images/2025-02-10/2025_02_09_Benchmark_QRT_Cube_27_0.png){:class="img-responsive"}
    


    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.61745
    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.62040
    Cox Proportional Hazard Model Concordance Index IPCW on test: 0.61757


#### 4 - 2 models galore


```python
# prompt: loop on scikit-learn regressors
import nnetsauce as ns
import pandas as pd
import xgboost as xgb

from functools import partial
from sklearn.utils.discovery import all_estimators
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.impute import SimpleImputer


# Get all regressors from scikit-learn
regressors = [est for est in all_estimators()  if 'Regressor' in est[0]]

# Append xgb.XGBRegressor and lgb.LGBMRegressor as (name, class) tuples
regressors += [('XGBRegressor', xgb.XGBRegressor),
 ('LGBMRegressor', partial(lgb.LGBMRegressor, verbose=0))]

results = []

for name, Regressor in tqdm(regressors):

    print("\n\n ----- base learner", name)

    try:

      # Initialize and train the model
      estimator = PISurvivalCustom(regr=Regressor(), type_pi="kde")
      estimator.fit(X_train, y_train)
      # Make predictions and evaluate the model
      y_pred = estimator.predict(X_test.iloc[:])
      c_index = concordance_index_ipcw(y_train, y_test, y_pred.mean, tau=7)[0]
      c_index_upper = concordance_index_ipcw(y_train, y_test, y_pred.upper, tau=7)[0]
      c_index_lower = concordance_index_ipcw(y_train, y_test, y_pred.lower, tau=7)[0]
      print("\n c_index", c_index)
      results.append([name, c_index, c_index_lower, c_index_upper])

      # Initialize and train the model
      estimator = PISurvivalCustom(regr=ns.CustomRegressor(Regressor()), type_pi="kde")
      estimator.fit(X_train, y_train)
      # Make predictions and evaluate the model
      y_pred = estimator.predict(X_test.iloc[:])
      c_index = concordance_index_ipcw(y_train, y_test, y_pred.mean, tau=7)[0]
      c_index_upper = concordance_index_ipcw(y_train, y_test, y_pred.upper, tau=7)[0]
      c_index_lower = concordance_index_ipcw(y_train, y_test, y_pred.lower, tau=7)[0]
      print("\n c_index", c_index)
      results.append(["custom" + name, c_index, c_index_lower, c_index_upper])

      # Initialize and train the model
      estimator = PISurvivalCustom(regr=RandomBagRegressor(Regressor()), type_pi="kde")
      estimator.fit(X_train, y_train)
      # Make predictions and evaluate the model
      y_pred = estimator.predict(X_test.iloc[:])
      c_index = concordance_index_ipcw(y_train, y_test, y_pred.mean, tau=7)[0]
      c_index_upper = concordance_index_ipcw(y_train, y_test, y_pred.upper, tau=7)[0]
      c_index_lower = concordance_index_ipcw(y_train, y_test, y_pred.lower, tau=7)[0]
      print("\n c_index", c_index)
      results.append(["bagging" + name, c_index, c_index_lower, c_index_upper])

    except Exception as e:
      continue
```

```python
pd.options.display.float_format = '{:.5f}'.format
results_df = pd.DataFrame(results, columns=['Regressor', 'Concordance Index IPCW', 'lower bound', 'upper bound'])
results_df.drop(columns=['lower bound', 'upper bound'], inplace=True)
results_df.sort_values(by='Concordance Index IPCW', ascending=False, inplace=True)
results_df
```





  <div id="df-ed136350-0eb0-48d5-9c12-6e8ccdb5eddd" class="colab-df-container">
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
      <th>Regressor</th>
      <th>Concordance Index IPCW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>baggingPassiveAggressiveRegressor</td>
      <td>0.66627</td>
    </tr>
    <tr>
      <th>39</th>
      <td>baggingRANSACRegressor</td>
      <td>0.66416</td>
    </tr>
    <tr>
      <th>26</th>
      <td>baggingHuberRegressor</td>
      <td>0.66340</td>
    </tr>
    <tr>
      <th>48</th>
      <td>baggingTheilSenRegressor</td>
      <td>0.66338</td>
    </tr>
    <tr>
      <th>51</th>
      <td>baggingTransformedTargetRegressor</td>
      <td>0.66288</td>
    </tr>
    <tr>
      <th>47</th>
      <td>customTheilSenRegressor</td>
      <td>0.66249</td>
    </tr>
    <tr>
      <th>52</th>
      <td>TweedieRegressor</td>
      <td>0.66238</td>
    </tr>
    <tr>
      <th>24</th>
      <td>HuberRegressor</td>
      <td>0.66225</td>
    </tr>
    <tr>
      <th>46</th>
      <td>TheilSenRegressor</td>
      <td>0.66224</td>
    </tr>
    <tr>
      <th>45</th>
      <td>baggingSGDRegressor</td>
      <td>0.66125</td>
    </tr>
    <tr>
      <th>25</th>
      <td>customHuberRegressor</td>
      <td>0.66078</td>
    </tr>
    <tr>
      <th>32</th>
      <td>baggingMLPRegressor</td>
      <td>0.66022</td>
    </tr>
    <tr>
      <th>50</th>
      <td>customTransformedTargetRegressor</td>
      <td>0.66021</td>
    </tr>
    <tr>
      <th>54</th>
      <td>baggingTweedieRegressor</td>
      <td>0.65999</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TransformedTargetRegressor</td>
      <td>0.65943</td>
    </tr>
    <tr>
      <th>53</th>
      <td>customTweedieRegressor</td>
      <td>0.65934</td>
    </tr>
    <tr>
      <th>2</th>
      <td>baggingAdaBoostRegressor</td>
      <td>0.65665</td>
    </tr>
    <tr>
      <th>37</th>
      <td>RANSACRegressor</td>
      <td>0.65656</td>
    </tr>
    <tr>
      <th>31</th>
      <td>customMLPRegressor</td>
      <td>0.65547</td>
    </tr>
    <tr>
      <th>44</th>
      <td>customSGDRegressor</td>
      <td>0.65448</td>
    </tr>
    <tr>
      <th>20</th>
      <td>baggingGradientBoostingRegressor</td>
      <td>0.64801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>customAdaBoostRegressor</td>
      <td>0.64506</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AdaBoostRegressor</td>
      <td>0.64473</td>
    </tr>
    <tr>
      <th>42</th>
      <td>baggingRandomForestRegressor</td>
      <td>0.63564</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GradientBoostingRegressor</td>
      <td>0.63519</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baggingBaggingRegressor</td>
      <td>0.63496</td>
    </tr>
    <tr>
      <th>19</th>
      <td>customGradientBoostingRegressor</td>
      <td>0.63449</td>
    </tr>
    <tr>
      <th>59</th>
      <td>baggingLGBMRegressor</td>
      <td>0.63271</td>
    </tr>
    <tr>
      <th>23</th>
      <td>baggingHistGradientBoostingRegressor</td>
      <td>0.63223</td>
    </tr>
    <tr>
      <th>28</th>
      <td>customKNeighborsRegressor</td>
      <td>0.62687</td>
    </tr>
    <tr>
      <th>14</th>
      <td>baggingExtraTreesRegressor</td>
      <td>0.62214</td>
    </tr>
    <tr>
      <th>30</th>
      <td>MLPRegressor</td>
      <td>0.62160</td>
    </tr>
    <tr>
      <th>57</th>
      <td>LGBMRegressor</td>
      <td>0.62069</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HistGradientBoostingRegressor</td>
      <td>0.61921</td>
    </tr>
    <tr>
      <th>13</th>
      <td>customExtraTreesRegressor</td>
      <td>0.61834</td>
    </tr>
    <tr>
      <th>58</th>
      <td>customLGBMRegressor</td>
      <td>0.61702</td>
    </tr>
    <tr>
      <th>29</th>
      <td>baggingKNeighborsRegressor</td>
      <td>0.61671</td>
    </tr>
    <tr>
      <th>4</th>
      <td>customBaggingRegressor</td>
      <td>0.61650</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ExtraTreesRegressor</td>
      <td>0.61259</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BaggingRegressor</td>
      <td>0.61154</td>
    </tr>
    <tr>
      <th>36</th>
      <td>QuantileRegressor</td>
      <td>0.61013</td>
    </tr>
    <tr>
      <th>41</th>
      <td>customRandomForestRegressor</td>
      <td>0.60759</td>
    </tr>
    <tr>
      <th>40</th>
      <td>RandomForestRegressor</td>
      <td>0.60749</td>
    </tr>
    <tr>
      <th>38</th>
      <td>customRANSACRegressor</td>
      <td>0.60653</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PassiveAggressiveRegressor</td>
      <td>0.60526</td>
    </tr>
    <tr>
      <th>11</th>
      <td>baggingExtraTreeRegressor</td>
      <td>0.60368</td>
    </tr>
    <tr>
      <th>56</th>
      <td>customXGBRegressor</td>
      <td>0.60044</td>
    </tr>
    <tr>
      <th>22</th>
      <td>customHistGradientBoostingRegressor</td>
      <td>0.60003</td>
    </tr>
    <tr>
      <th>8</th>
      <td>baggingDecisionTreeRegressor</td>
      <td>0.59707</td>
    </tr>
    <tr>
      <th>27</th>
      <td>KNeighborsRegressor</td>
      <td>0.59139</td>
    </tr>
    <tr>
      <th>55</th>
      <td>XGBRegressor</td>
      <td>0.58723</td>
    </tr>
    <tr>
      <th>34</th>
      <td>customPassiveAggressiveRegressor</td>
      <td>0.58205</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ExtraTreeRegressor</td>
      <td>0.57820</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DecisionTreeRegressor</td>
      <td>0.56057</td>
    </tr>
    <tr>
      <th>10</th>
      <td>customExtraTreeRegressor</td>
      <td>0.55844</td>
    </tr>
    <tr>
      <th>7</th>
      <td>customDecisionTreeRegressor</td>
      <td>0.55117</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GaussianProcessRegressor</td>
      <td>0.53342</td>
    </tr>
    <tr>
      <th>16</th>
      <td>customGaussianProcessRegressor</td>
      <td>0.52007</td>
    </tr>
    <tr>
      <th>17</th>
      <td>baggingGaussianProcessRegressor</td>
      <td>0.49695</td>
    </tr>
    <tr>
      <th>43</th>
      <td>SGDRegressor</td>
      <td>0.40105</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ed136350-0eb0-48d5-9c12-6e8ccdb5eddd')"
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
        document.querySelector('#df-ed136350-0eb0-48d5-9c12-6e8ccdb5eddd button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ed136350-0eb0-48d5-9c12-6e8ccdb5eddd');
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


<div id="df-60fac9c9-a079-4edc-8cc2-43cf0cc6f933">
  <button class="colab-df-quickchart" onclick="quickchart('df-60fac9c9-a079-4edc-8cc2-43cf0cc6f933')"
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
        document.querySelector('#df-60fac9c9-a079-4edc-8cc2-43cf0cc6f933 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_a7624289-0674-40d9-b0a2-137c8f22c1c7">
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
    <button class="colab-df-generate" onclick="generateWithVariable('results_df')"
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
        document.querySelector('#id_a7624289-0674-40d9-b0a2-137c8f22c1c7 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('results_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
results_df.shape
```




    (60, 2)


