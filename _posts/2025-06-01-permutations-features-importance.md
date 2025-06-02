---
layout: post
title: "Permutations and SHAPley values for feature importance in techtonique dot net's API (with R + Python + the command line)"
description: "How to use techtonique.net's API to compute feature importance using permutations and SHAPley values"
date: 2025-06-01
categories: [R, Python, Techtonique]
comments: true
---

**Feature importance** is a crucial aspect of machine learning model interpretation, helping us understand which features contribute most to a model's predictions. In this blog post, we'll explore two popular methods for computing feature importance using techtonique.net's API:

1. **Permutation Importance**: This method measures how much a model's performance decreases when a feature is randomly shuffled. It's model-agnostic and provides a straightforward way to understand feature impact.

2. **SHAP (SHapley Additive exPlanations) Values**: Based on game theory, SHAP values provide a more detailed view of how each feature contributes to individual predictions, offering both global and local interpretability.

We'll demonstrate how to use these methods with both R and Python, showing how to:
- Send requests to techtonique.net's API
- Process and visualize the results


# 0 - Download data

```python
!wget https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/classification/breast_cancer_dataset2.csv
!wget https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv
```

# 1 - Load rpy2 extension


```python
%load_ext rpy2.ipython
```

# 2 - Send requests

Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.

```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@breast_cancer_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/mlclassification?base_model=RandomForestClassifier&n_hidden_features=5&interpretability=permutation" > res1.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  124k  100  4938  100  119k    927  23049  0:00:05  0:00:05 --:--:-- 26549


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.

```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@breast_cancer_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/gbdtclassification?model_type=xgboost&predict_proba=False&interpretability=permutation" > res2.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  122k  100  2751  100  119k   1130  50421  0:00:02  0:00:02 --:--:-- 51557


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.

```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@breast_cancer_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/mlclassification?base_model=RandomForestClassifier&n_hidden_features=5&interpretability=shap" > res3.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  236k  100  117k  100  119k   2780   2845  0:00:43  0:00:43 --:--:-- 26830

Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@breast_cancer_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/gbdtclassification?model_type=xgboost&predict_proba=False&interpretability=shap" > res4.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  173k  100 55122  100  119k   5159  11487  0:00:10  0:00:10 --:--:-- 13849


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@boston_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/mlregression?base_model=RidgeCV&n_hidden_features=5&return_pi=True&interpretability=permutation" > res5.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 52785  100 11903  100 40882   8541  29337  0:00:01  0:00:01 --:--:-- 37893


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@boston_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/mlregression?base_model=RidgeCV&n_hidden_features=5&return_pi=True&interpretability=shap" > res6.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 87409  100 46527  100 40882   4139   3637  0:00:11  0:00:11 --:--:-- 12810


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjIxMDF9.2tAjOBPLTC-YuGeu8Iep-QRrOP8DSo7l1B89lU1eI3Q" \
-F "file=@boston_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/gbdtregression?model_type=xgboost&interpretability=shap" > res7.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 40912  100    30  100 40882     34  46655 --:--:-- --:--:-- --:--:-- 46649


Note that you can use [https://curlconverter.com/](https://curlconverter.com/) to translate the following request in your favorite programming language.


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDg4MjU5MTh9.ARqoz9PWZGILxgx8tcUJEc1h19TY-1odEqDkpCUxYmE" \
-F "file=@boston_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/gbdtregression?model_type=xgboost&interpretability=permutation" > res8.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 47135  100  6253  100 40882   4012  26236  0:00:01  0:00:01 --:--:-- 30253


# 3 - Plot results


```python
import json
import matplotlib.pyplot as plt

with open('res1.json', 'r') as f:
    data = json.load(f)

for key, value in data.items():
    print(f"{key}: {value}")
```

    y_true: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    0: {'precision': 0.9508196721311475, 'recall': 0.90625, 'f1-score': 0.928, 'support': 64.0}
    1: {'precision': 0.9454545454545454, 'recall': 0.9719626168224299, 'f1-score': 0.9585253456221198, 'support': 107.0}
    accuracy: 0.9473684210526315
    macro avg: {'precision': 0.9481371087928465, 'recall': 0.939106308411215, 'f1-score': 0.9432626728110599, 'support': 171.0}
    weighted avg: {'precision': 0.9474625460820457, 'recall': 0.9473684210526315, 'f1-score': 0.9471006548629639, 'support': 171.0}
    proba: [[0.97, 0.03], [0.92, 0.08], [0.99, 0.01], [0.7, 0.3], [0.98, 0.02], [1.0, 0.0], [0.02, 0.98], [1.0, 0.0], [1.0, 0.0], [0.03, 0.97], [0.38, 0.62], [0.28, 0.72], [0.0, 1.0], [0.0, 1.0], [0.97, 0.03], [0.94, 0.06], [0.0, 1.0], [1.0, 0.0], [0.27, 0.73], [0.0, 1.0], [1.0, 0.0], [0.97, 0.03], [0.0, 1.0], [1.0, 0.0], [0.47, 0.53], [1.0, 0.0], [0.04, 0.96], [0.0, 1.0], [0.01, 0.99], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.97, 0.03], [0.03, 0.97], [0.97, 0.03], [0.5, 0.5], [0.92, 0.08], [0.0, 1.0], [0.0, 1.0], [0.98, 0.02], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.7, 0.3], [1.0, 0.0], [1.0, 0.0], [0.1, 0.9], [0.77, 0.23], [1.0, 0.0], [0.95, 0.05], [0.02, 0.98], [0.99, 0.01], [0.03, 0.97], [0.59, 0.41], [0.0, 1.0], [0.96, 0.04], [0.0, 1.0], [1.0, 0.0], [0.09, 0.91], [0.04, 0.96], [0.02, 0.98], [0.99, 0.01], [0.31, 0.69], [0.02, 0.98], [0.24, 0.76], [1.0, 0.0], [0.98, 0.02], [1.0, 0.0], [0.78, 0.22], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.22, 0.78], [0.01, 0.99], [0.08, 0.92], [0.19, 0.81], [0.01, 0.99], [0.01, 0.99], [0.0, 1.0], [0.12, 0.88], [0.01, 0.99], [0.0, 1.0], [0.02, 0.98], [0.01, 0.99], [0.0, 1.0], [0.0, 1.0], [0.94, 0.06], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.29, 0.71], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.98, 0.02], [0.0, 1.0], [0.01, 0.99], [0.03, 0.97], [0.01, 0.99], [0.0, 1.0], [0.69, 0.31], [1.0, 0.0], [0.98, 0.02], [0.44, 0.56], [0.2, 0.8], [0.0, 1.0], [0.02, 0.98], [0.01, 0.99], [0.0, 1.0], [0.08, 0.92], [0.42, 0.58], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.06, 0.94], [0.0, 1.0], [0.02, 0.98], [1.0, 0.0], [0.02, 0.98], [0.01, 0.99], [0.19, 0.81], [0.9, 0.1], [0.01, 0.99], [0.1, 0.9], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.03, 0.97], [0.08, 0.92], [0.0, 1.0], [0.98, 0.02], [0.04, 0.96], [0.51, 0.49], [0.0, 1.0], [1.0, 0.0], [0.01, 0.99], [0.1, 0.9], [0.11, 0.89], [0.99, 0.01], [0.86, 0.14], [0.18, 0.82], [0.01, 0.99], [0.03, 0.97], [0.35, 0.65], [0.98, 0.02], [0.09, 0.91], [1.0, 0.0], [0.12, 0.88], [0.01, 0.99], [0.02, 0.98], [0.15, 0.85], [0.02, 0.98], [0.14, 0.86], [0.01, 0.99], [1.0, 0.0], [0.42, 0.58], [0.14, 0.86], [0.0, 1.0], [0.44, 0.56], [0.17, 0.83], [0.01, 0.99], [0.01, 0.99], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.01, 0.99]]
    interpretability: {'method': 'permutation', 'importances_mean': [-0.008771929824561441, 0.007602339181286511, -0.002923976608187162, -0.004678362573099448, -0.0052631578947368585, -0.009356725146198841, -0.004093567251462038, -0.007017543859649145, -0.0005847953216374324, -0.0035087719298245836, -0.0005847953216374435, 0.002339181286549685, -0.0005847953216374324, -0.0035087719298245944, -0.006432748538011734, -0.0005847953216374324, -0.0005847953216374324, -0.0046783625730994595, -0.0011695906432748649, 0.0, -0.0017543859649122972, 0.006432748538011668, -0.009356725146198864, -0.0040935672514620155, -0.006432748538011734, -0.0035087719298245944, -0.005263157894736891, -0.006432748538011734, 0.0, 0.0], 'importances_std': [0.005391546466253153, 0.0026798688274595806, 0.002923976608187162, 0.0035087719298245723, 0.004857674773636302, 0.003879093322053089, 0.0037445170979139527, 0.004376207469911037, 0.0017543859649122972, 0.0038790933220531126, 0.003149219185458781, 0.0028649002839569063, 0.0017543859649122972, 0.0028649002839569605, 0.00485767477363631, 0.0017543859649122972, 0.0017543859649122972, 0.0023391812865497298, 0.0023391812865497298, 0.0, 0.006944059700022174, 0.004093567251461992, 0.006512005102725163, 0.007420220783888606, 0.004857674773636309, 0.0028649002839569605, 0.0017543859649122972, 0.006105442402871663, 0.0, 0.0], 'feature_names': ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']}


## 3 - 1 Permutation importance


```r
%%R

library(jsonlite)
library(ggplot2)

# Load JSON data
res1 <- fromJSON("res1.json")

# Extract feature names, importances_mean, and importances_std
feature_names <- res1$interpretability$feature_names
importances_mean <- res1$interpretability$importances_mean
importances_std <- res1$interpretability$importances_std

# Create a data frame
df <- data.frame(
  feature_names = factor(feature_names, levels = feature_names), # Maintain order
  importances_mean = importances_mean,
  importances_std = importances_std
)

# Create the bar plot
ggplot(df, aes(x = feature_names, y = importances_mean)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = importances_mean - importances_std, ymax = importances_mean + importances_std), width = 0.2) +
  labs(title = "Feature importances via permutation on test set", y = "Mean accuracy decrease") +
  theme_minimal()
```

![image-title-here]({{base}}/images/2025-06-01/2025-06-01-image1.png){:class="img-responsive"}    

```python
import matplotlib.pyplot as plt
feature_names = data['interpretability']['feature_names']
importances_mean = data['interpretability']['importances_mean']
importances_std = data['interpretability']['importances_std']

fig, ax = plt.subplots()
ax.bar(feature_names, importances_mean, yerr=importances_std)
ax.set_title("Feature importances via permutation on test set")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
```

![image-title-here]({{base}}/images/2025-06-01/2025-06-01-image2.png){:class="img-responsive"}        


## 3 - 2 SHAPley values


```python
import matplotlib.pyplot as plt
import numpy as np


with open('res3.json', 'r') as f:
    data = json.load(f)

# Assuming 'shap_values' and 'feature_names' are keys in the JSON response for SHAP
shap_values = data['interpretability']['values']
feature_names = data['interpretability']['feature_names']

# Install shap if not already installed
try:
    import shap
except ImportError:
    !pip install shap
    import shap

print(shap_values[0])

abs_shap_values = [abs(s[1]) for s in shap_values[0]]

# Plotting using matplotlib
plt.figure(figsize=(10, 6))
plt.barh(feature_names, abs_shap_values)
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Mean Absolute SHAP Values per Feature")
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.tight_layout()
plt.show()
```

    [[0.0467927752729715, -0.04679277527297147], [0.0, 0.0], [0.03065755727391224, -0.030657557273912207], [0.05967603575097714, -0.0596760357509771], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.039243510520076554, -0.039243510520076526], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.040102414661446345, -0.04010241466144643], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.12096136416860846, -0.12096136416860838], [0.0, 0.0], [0.10703338349800118, -0.10703338349800114], [0.11830883652988157, -0.11830883652988157], [0.0, 0.0], [0.0, 0.0], [0.004369056993017484, -0.004369056993017595], [0.04197566834618272, -0.04197566834618292], [0.0, 0.0], [0.0, 0.0]]


![image-title-here]({{base}}/images/2025-06-01/2025-06-01-image3.png){:class="img-responsive"}            

