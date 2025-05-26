---
layout: post
title: "Harnessing the Power of techtonique.net: A Comprehensive Guide to Machine Learning Classification via an API"
description: "How to make API calls to techtonique.net for machine learning classification tasks using curl, Python requests, and R httr"
date: 2025-05-26
categories: [R, Python]
comments: true
---

In today's data-driven world, machine learning classification tasks are ubiquitous across various domains. While building and deploying machine learning models can be complex, APIs provide a convenient way to leverage powerful classification capabilities without the need for extensive setup or infrastructure.

This blog post demonstrates how to use the machine learning classification API provided by [techtonique.net](https://www.techtonique.net) using three different methods:

1. **curl** - For command-line users and shell scripts
2. **Python requests** - For Python developers
3. **R httr** - For R users
4. **Excel** - For Excel users

We'll walk through examples using two classic datasets:
- The Iris dataset for multi-class classification
- The Breast Cancer dataset for binary classification

Each example will show how to:
- Make API calls with proper authentication
- Handle the response data

Let's get started!

Get a token from: [https://www.techtonique.net/token](https://www.techtonique.net/token).

Then download the classification dataset:

```python
!wget https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/classification/iris_dataset2.csv
!wget https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/classification/breast_cancer_dataset2.csv
```

    --2025-05-26 23:15:42--  https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/classification/iris_dataset2.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3092 (3.0K) [text/plain]
    Saving to: ‘iris_dataset2.csv’
    
    iris_dataset2.csv   100%[===================>]   3.02K  --.-KB/s    in 0s      
    
    2025-05-26 23:15:43 (39.9 MB/s) - ‘iris_dataset2.csv’ saved [3092/3092]
    
    --2025-05-26 23:15:43--  https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/classification/breast_cancer_dataset2.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 122538 (120K) [text/plain]
    Saving to: ‘breast_cancer_dataset2.csv’
    
    breast_cancer_datas 100%[===================>] 119.67K  --.-KB/s    in 0.01s   
    
    2025-05-26 23:15:43 (8.73 MB/s) - ‘breast_cancer_dataset2.csv’ saved [122538/122538]
    


# 1 - curl


```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDgzMDM0NzJ9.vmc6czfUZo2jJEsKCTcZBPA1yYd2vToB6VpXm2Ty04E" \
-F "file=@iris_dataset2.csv;type=text/csv" \
"https://www.techtonique.net/mlclassification?base_model=GradientBoostingClassifier&n_hidden_features=5&predict_proba=True"
```

    {"y_true":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"y_pred":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,2,2,1,2,2,2,2,2,2],"0":{"precision":1.0,"recall":1.0,"f1-score":1.0,"support":15.0},"1":{"precision":0.7894736842105263,"recall":1.0,"f1-score":0.8823529411764706,"support":15.0},"2":{"precision":1.0,"recall":0.7333333333333333,"f1-score":0.8461538461538461,"support":15.0},"accuracy":0.9111111111111111,"macro avg":{"precision":0.9298245614035089,"recall":0.9111111111111111,"f1-score":0.9095022624434389,"support":45.0},"weighted avg":{"precision":0.9298245614035088,"recall":0.9111111111111111,"f1-score":0.9095022624434389,"support":45.0},"proba":[[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.9999960244140262,3.334039683616685e-06,6.415462901466744e-07],[0.9999960244140262,3.334039683616685e-06,6.415462901466744e-07],[0.9999960244140262,3.334039683616685e-06,6.415462901466744e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[0.999996024414054,3.3340396558155455e-06,6.415462901466922e-07],[3.045033869468758e-06,0.9999937901439727,3.164822157743673e-06],[3.4085260162096436e-06,0.9999931695319606,3.4219420233008224e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[9.588330585448842e-06,0.9998587655346274,0.0001316461347872],[3.4085260162096436e-06,0.9999931695319606,3.4219420233008224e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[3.4085260162096436e-06,0.9999931695319606,3.4219420233008224e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[0.004932272272005095,0.9799043377817493,0.015163389946245788],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[9.588330585508172e-06,0.9998587655408142,0.00013164612860042232],[3.4085260162096436e-06,0.9999931695319606,3.4219420233008224e-06],[3.1524138718715695e-06,0.9999936827643102,3.1648218179040277e-06],[8.748137691624919e-07,2.8391363939081826e-06,0.9999962860498369],[3.4506054062196775e-05,0.007887348624295493,0.9920781453216423],[8.620706075486918e-07,2.5556823801935716e-06,0.9999965822470123],[0.007316994751226876,0.9353406793501166,0.05734232589865652],[0.006129685125947995,0.7712671846855687,0.22260313018848327],[0.003273968018685709,0.969594174153424,0.027131857827890307],[7.169724003427306e-05,0.004745498561275488,0.9951828041986902],[2.245977897875399e-06,8.537222217088907e-06,0.9999892167998851],[0.00016867136329469363,0.568295956512619,0.4315353721240864],[3.1992229758050182e-06,0.0008659343830747356,0.9991308663939493],[2.301715170556224e-05,0.005261230377511008,0.9947157524707835],[8.748139925370251e-07,2.5837975927175516e-06,0.9999965413884147],[8.748137418487531e-07,2.8703586517843243e-06,0.9999962548276062],[2.2462428700241693e-06,1.3402409372558758e-05,0.9999843513477573],[8.748139925370251e-07,2.5837975927175516e-06,0.9999965413884147]]}

Use [https://curlconverter.com/](https://curlconverter.com/) for translating this request.

# 2 - Python


```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_TOKEN_HERE',
}

params = {
    'base_model': 'RandomForestClassifier',
    'n_hidden_features': '5',
    'predict_proba': 'True',
}

files = {
    'file': ('breast_cancer_dataset2.csv', open('breast_cancer_dataset2.csv', 'rb'), 'text/csv'),
}

response = requests.post('https://www.techtonique.net/mlclassification',
                         params=params, headers=headers,
                         files=files)
```


```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
def plot_calibration_curve(y_true, y_prob, n_bins=10, title='Calibration Plot'):
    """Plot calibration curve for a single class."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

```


```python
response_json = response.json()
y_true = response_json['y_true']
y_prob = [response_json['proba'][i][1] for i in range(len(y_true))]
plot_calibration_curve(y_true, y_prob)
```

![image-title-here]({{base}}/images/2025-05-26/2025-05-26-image1.png)    


# 3 - R


```python
# prompt: load rpy2 extension

%load_ext rpy2.ipython
```


```r
%%R

library(httr)

headers = c(
  Authorization = "Bearer YOUR_TOKEN_HERE"
)

params = list(
  base_model = "GradientBoostingClassifier",
  n_hidden_features = "5",
  predict_proba = "True"
)

# file from:
files = list(
  file = upload_file("iris_dataset2.csv")
)

res <- httr::POST(url = "https://www.techtonique.net/mlclassification", httr::add_headers(.headers=headers), query = params, body = files, encode = "multipart")

print(res)
```

    Response [https://www.techtonique.net/mlclassification?base_model=GradientBoostingClassifier&n_hidden_features=5&predict_proba=True]
      Date: 2025-05-26 23:16
      Status: 200
      Content-Type: application/json
      Size: 3.67 kB
    


## 4 - Excel

For more ways to interact, e.g using Excel:

- [https://thierrymoudiki.github.io/blog/2025/03/28/python/xlwings-lite-techtonique](https://thierrymoudiki.github.io/blog/2025/03/28/python/xlwings-lite-techtonique)
- [https://thierrymoudiki.github.io/blog/2024/11/03/python/r/techtonique/scenario-simulations-techtonique](https://thierrymoudiki.github.io/blog/2024/11/03/python/r/techtonique/scenario-simulations-techtonique)
- [https://thierrymoudiki.github.io/blog/2024/09/30/python/vba-web-forecasting](https://thierrymoudiki.github.io/blog/2024/09/30/python/vba-web-forecasting)
