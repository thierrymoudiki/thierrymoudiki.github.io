---
layout: post
title: "mlsauce version 0.8.10: Statistical/Machine Learning with Python and R"
description: "Statistical/Machine Learning with Python and R, using mlsauce's AdaOpt and LSBoost"
date: 2023-11-05
categories: [Python, R, AdaOpt, LSBoost]
comments: true
---

This week, among other things, I've been working on updating [mlsauce](https://github.com/Techtonique/mlsauce) for both Python and R (that's version `0.8.10` of the package). 

`mlsauce` is a package for Statistical/Machine Learning that contains in particular: 


<ul>
  <li> <a href="https://www.researchgate.net/publication/341409169_AdaOpt_Multivariable_optimization_for_classification">AdaOpt</a>, a probabilistic classifier which uses <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">nearest neighbors</a> to obtain predictions. Interestingly, with AdaOpt, one neighbor can suffice to obtain a high accuracy. </li> 

  <li> <a href="https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares">LSBoost</a>
  , a gradient boosting algorithm based on randomized nnetworks (similar to XGBoost, LightGBM or Catboost, but not using Gradient Boosted Decision Trees a.k.a GBDT). </li>
</ul>


Not a lot of GitHub stars for `mlsauce`'s repository but someday, to my surprise, I noticed that `mlsauce.LSBoost`'s 2020 "paper" had more than  2000 reads [on ResearchGate](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares). Well, people, _starring_ the repository on GitHub is pretty cool too. 

Then, I had a ResearchGate recommendation on that same `mlsauce.LSBoost`'s "paper", and I told to myself: 'I've probably been missing something in this work for 3 years'. Yes I know I designed it from beginning to  end, but some people can be using it better than I did so far! 

Indeed, I've never obtained great results with `mlsauce.LSBoost` **IN THE PAST**. Eventually, as of today, my feelings are: `mlsauce` is _fast_, thanks to [Cython](https://cython.org/) (which is not easy to package though, IMHO), and quite competitive when well-tuned; as you'll see below. 

In this post, I revisit `mlsauce`, with examples of use of `AdaOpt` and `LSBoostclassifier`.`AdaOpt` is used for digits recognition (and seems to be doing well on this type of tasks, more on this in the future). `LSBoostclassifier` is used on toy examples from scikit-learn as done in the paper, but with better hyperparameters' tuning. For both models, `AdaOpt` and `LSBoostclassifier`, a **distribution of test set accuracy** is presented.  

**Contents**
<ol>
  <li>Install and import Python packages</li>
  <li><code>AdaOpt</code> Python -- with test set accuracy's distribution</li>
  <li><code>LSBoostclassifier</code> Python -- with test set accuracy's distribution</li>
  <li>R example</li>  
</ol>
 
 A notebook can also be found here: [https://github.com/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki_051123_GPopt_mlsauce_classification.ipynb](https://github.com/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki_051123_GPopt_mlsauce_classification.ipynb). 

# 1 - Install and import Python packages

```bash
!pip install mlsauce
```

```bash
!pip install GPopt # a package that implements Bayesian optimization, used here for hyperparameters' tuning
```

```python
import GPopt as gp
import mlsauce as ms
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from time import time
```

# 2 - `AdaOpt` Python -- with test set accuracy's distribution


```python
import numpy as np
from sklearn.datasets import load_digits # a dataset for digits recognition
from sklearn.model_selection import train_test_split, cross_val_score
from time import time


digits = load_digits()
Z = digits.data
t = digits.target
np.random.seed(13239)
X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                    test_size=0.2)

obj = ms.AdaOpt(n_iterations=50,
           learning_rate=0.3,
           reg_lambda=0.1,
           reg_alpha=0.5,
           eta=0.01,
           gamma=0.01,
           tolerance=1e-4,
           row_sample=1,
           k=1,
           n_jobs=3, type_dist="euclidean", verbose=1)

start = time()
obj.fit(X_train, y_train)
print(f"\n\n Elapsed train: {time()-start} \n")

start = time()
print(f"\n\n Accuracy: {obj.score(X_test, y_test)}")
print(f"\n Elapsed predict: {time()-start}")
```

    100%|██████████| 360/360 [00:00<00:00, 1979.13it/s]    
    
     Elapsed train: 0.01917862892150879 
    
     Accuracy: 0.9916666666666667
    
     Elapsed predict: 0.19308829307556152


**Obtaining test set accuracy distribution with the same hyperparameters**


```python
from collections import namedtuple
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy import stats
```


```python
def eval_adaopt(k=1, B=250):

  res_metric = []
  training_times = []
  testing_times = []

  DescribeResult = namedtuple('DescribeResult', ('accuracy',
                                                 'training_time',
                                                 'testing_time'))
  obj = ms.AdaOpt(n_iterations=50,
              learning_rate=0.3,
              reg_lambda=0.1,
              reg_alpha=0.5,
              eta=0.01,
              gamma=0.01,
              tolerance=1e-4,
              row_sample=1,
              k=k,
              n_jobs=-1, type_dist="euclidean", verbose=0)

  for i in tqdm(range(B)):

    np.random.seed(10*i+100)
    X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                        test_size=0.2)

    start = time()
    obj.fit(X_train, y_train)
    training_times.append(time()-start)
    start = time()
    res_metric.append(obj.score(X_test, y_test))
    testing_times.append(time()-start)

  return DescribeResult(res_metric, training_times, testing_times), stats.describe(res_metric), stats.describe(training_times), stats.describe(testing_times)
```

```python
res_k1_B250 = eval_adaopt(k=1, B=250)
res_k2_B250 = eval_adaopt(k=2, B=250)
res_k3_B250 = eval_adaopt(k=3, B=250)
res_k4_B250 = eval_adaopt(k=4, B=250)
res_k5_B250 = eval_adaopt(k=5, B=250)
```

    100%|██████████| 250/250 [00:50<00:00,  4.96it/s]
    100%|██████████| 250/250 [00:50<00:00,  4.94it/s]
    100%|██████████| 250/250 [00:50<00:00,  4.96it/s]
    100%|██████████| 250/250 [00:51<00:00,  4.90it/s]
    100%|██████████| 250/250 [00:51<00:00,  4.90it/s]

```python
display(res_k1_B250[1])
display(res_k2_B250[1])
display(res_k3_B250[1])
display(res_k4_B250[1])
display(res_k5_B250[1])
```

    DescribeResult(nobs=250, minmax=(0.9722222222222222, 1.0), mean=0.9872888888888888, variance=2.5628935495066882e-05, skewness=-0.13898324248427138, kurtosis=0.22445816198359791)

    DescribeResult(nobs=250, minmax=(0.9666666666666667, 0.9972222222222222), mean=0.9846888888888888, variance=3.354355694382497e-05, skewness=-0.2014633213050366, kurtosis=-0.16851847469456605)

    DescribeResult(nobs=250, minmax=(0.9611111111111111, 0.9972222222222222), mean=0.9836666666666666, variance=3.45951708066838e-05, skewness=-0.3714590259216959, kurtosis=0.264762318251484)

    DescribeResult(nobs=250, minmax=(0.9555555555555556, 1.0), mean=0.9793777777777778, variance=4.80023798899302e-05, skewness=-0.24910751075977636, kurtosis=0.4395617044106124)

    DescribeResult(nobs=250, minmax=(0.9555555555555556, 0.9972222222222222), mean=0.9770444444444444, variance=5.1334225792057076e-05, skewness=-0.12883539300214827, kurtosis=0.1411098033435696)


**Obtaining a distribution of training timings**

```python
display(res_k1_B250[2])
display(res_k2_B250[2])
display(res_k3_B250[2])
display(res_k4_B250[2])
display(res_k5_B250[2])
```

    DescribeResult(nobs=250, minmax=(0.00498199462890625, 0.021169185638427734), mean=0.007840995788574218, variance=4.368068123193988e-06, skewness=2.175594596266775, kurtosis=7.499194342725625)

    DescribeResult(nobs=250, minmax=(0.005329132080078125, 0.016299962997436523), mean=0.007670882225036621, variance=3.612048206608975e-06, skewness=1.7118375802873183, kurtosis=3.358366931595608)

    DescribeResult(nobs=250, minmax=(0.0053746700286865234, 0.015506505966186523), mean=0.007794314384460449, variance=2.920214088930605e-06, skewness=1.6360801483869196, kurtosis=3.2315493234819064)

    DescribeResult(nobs=250, minmax=(0.005369901657104492, 0.02190709114074707), mean=0.007874348640441894, variance=4.55353231021138e-06, skewness=2.3223174208412916, kurtosis=8.922678944294534)

    DescribeResult(nobs=250, minmax=(0.005362033843994141, 0.017331361770629883), mean=0.00786894702911377, variance=4.207144846754069e-06, skewness=1.8494401442014954, kurtosis=3.8446086533270085)

**Obtaining a distribution of testing timings**

```python
display(res_k1_B250[3])
display(res_k2_B250[3])
display(res_k3_B250[3])
display(res_k4_B250[3])
display(res_k5_B250[3])
```

    DescribeResult(nobs=250, minmax=(0.1675705909729004, 0.3001070022583008), mean=0.19125074195861816, variance=0.0003424395337048105, skewness=2.2500063799757677, kurtosis=5.9722526245151375)

    DescribeResult(nobs=250, minmax=(0.16643667221069336, 0.31163525581359863), mean=0.1923248109817505, variance=0.0003310783018211768, skewness=2.476834016032642, kurtosis=8.109087286878708)

    DescribeResult(nobs=250, minmax=(0.17519187927246094, 0.37604689598083496), mean=0.1916730365753174, variance=0.0003895799321858523, skewness=4.280046315900402, kurtosis=30.357835694940057)

    DescribeResult(nobs=250, minmax=(0.17512750625610352, 0.3540067672729492), mean=0.19378959369659424, variance=0.00035161275596300016, skewness=3.595469226517824, kurtosis=21.271489103625353)

    DescribeResult(nobs=250, minmax=(0.17573857307434082, 0.2584831714630127), mean=0.19390375328063963, variance=0.0002475867594812809, skewness=2.0323201018310013, kurtosis=3.343216700759352)

**Graph: distribution of test set accuracy for different numbers of neighbors (1 to 4)**

```python
# library & dataset
import pandas as pd
import seaborn as sns
df = pd.DataFrame(np.column_stack((res_k1_B250[0][0],
                               res_k2_B250[0][0],
                               res_k3_B250[0][0],
                               res_k4_B250[0][0])),
               columns=['k1', 'k2', 'k3', 'k4'])
```


```python
# Plot the histogram thanks to the distplot function
sns.distplot(a=df["k1"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k2"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k3"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k4"], hist=True, kde=True, rug=True)
```

![image-title-here]({{base}}/images/2023-11-05/2023-11-05-image1.png){:class="img-responsive"}

**Graph: distribution of training timings for different numbers of neighbors (1 to 4)**

```python
df = pd.DataFrame(np.column_stack((res_k1_B250[0][1],
                               res_k2_B250[0][1],
                               res_k3_B250[0][1],
                               res_k4_B250[0][1])),
               columns=['k1', 'k2', 'k3', 'k4'])

# Plot the histogram thanks to the distplot function
sns.distplot(a=df["k1"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k2"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k3"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k4"], hist=True, kde=True, rug=True)
```

![image-title-here]({{base}}/images/2023-11-05/2023-11-05-image2.png){:class="img-responsive"}

**Graph: distribution of testing timings for different numbers of neighbors (1 to 4)**

```python
df = pd.DataFrame(np.column_stack((res_k1_B250[0][2],
                               res_k2_B250[0][2],
                               res_k3_B250[0][2],
                               res_k4_B250[0][2])),
               columns=['k1', 'k2', 'k3', 'k4'])

# Plot the histogram thanks to the distplot function
sns.distplot(a=df["k1"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k2"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k3"], hist=True, kde=True, rug=True)
sns.distplot(a=df["k4"], hist=True, kde=True, rug=True)
```

![image-title-here]({{base}}/images/2023-11-05/2023-11-05-image3.png){:class="img-responsive"}


# 3 - `LSBoostClassifier` Python -- with test set accuracy's distribution

## 3 - 1 **Classification of Breast Cancer dataset**

```python
data = load_breast_cancer()
X = data.data
y = data.target
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=13)
```


```python
def lsboost_cv(X_train, y_train,
               n_estimators=100,
               learning_rate=0.1,
               n_hidden_features=5,
               reg_lambda=0.1,
               dropout=0,
               tolerance=1e-4,
               seed=123):

  estimator = ms.LSBoostClassifier(n_estimators=int(n_estimators),
                                   learning_rate=learning_rate,
                                   n_hidden_features=int(n_hidden_features),
                                   reg_lambda=reg_lambda,
                                   dropout=dropout,
                                   tolerance=tolerance,
                                   seed=seed, verbose=0)

  return -cross_val_score(estimator, X_train, y_train,
                          scoring='accuracy', cv=5, n_jobs=-1).mean()

```


```python
def optimize_lsboost(X_train, y_train):
  
  # objective function for hyperparams tuning
  def crossval_objective(x):

    return lsboost_cv(
      X_train=X_train,
      y_train=y_train,
      n_estimators=int(x[0]),
      learning_rate=x[1],
      n_hidden_features=int(x[2]),
      reg_lambda=x[3],
      dropout=x[4],
      tolerance=x[5])

  gp_opt = gp.GPOpt(objective_func=crossval_objective,
                      lower_bound = np.array([10, 0.001, 5, 1e-2, 0, 0]),
                      upper_bound = np.array([100, 0.4, 250, 1e4, 0.7, 1e-1]),
                      n_init=10, n_iter=190, seed=123)
  return {'parameters': gp_opt.optimize(verbose=2, abs_tol=1e-2), 'opt_object':  gp_opt}
```

```python
# hyperparams tuning
res1 = optimize_lsboost(X_train, y_train)
print(res1)
parameters = res1["parameters"]
start = time()
estimator = ms.LSBoostClassifier(n_estimators=int(parameters[0][0]),
                                   learning_rate=parameters[0][1],
                                   n_hidden_features=int(parameters[0][2]),
                                   reg_lambda=parameters[0][3],
                                   dropout=parameters[0][4],
                                   tolerance=parameters[0][5],
                                   seed=123, verbose=1).fit(X_train, y_train)
```

```python
print(f"\n\n Test set accuracy: {estimator.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
``` 
    
     Test set accuracy: 0.9912280701754386
    
     Elapsed: 0.11275959014892578

```python
from collections import namedtuple
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy import stats
```

**Distribution of test set accuracy of LSBoost on Breast Cancer dataset**

```python
def eval_lsboost(B=250):

  res_metric = []
  training_times = []
  testing_times = []

  DescribeResult = namedtuple('DescribeResult', ('accuracy',
                                                 'training_time',
                                                 'testing_time'))

  for i in tqdm(range(B)):

    np.random.seed(10*i+100)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)

    #try:
    start = time()
    obj = ms.LSBoostClassifier(n_estimators=int(parameters[0][0]),
                                   learning_rate=parameters[0][1],
                                   n_hidden_features=int(parameters[0][2]),
                                   reg_lambda=parameters[0][3],
                                   dropout=parameters[0][4],
                                   tolerance=parameters[0][5],
                                   seed=123, verbose=0).fit(X_train, y_train)
    training_times.append(time()-start)
    start = time()
    res_metric.append(obj.score(X_test, y_test))
    testing_times.append(time()-start)

  return DescribeResult(res_metric, training_times, testing_times), stats.describe(res_metric), stats.describe(training_times), stats.describe(testing_times)
```

```python
res_lsboost_B250 = eval_lsboost(B=250)
```

    100%|██████████| 250/250 [00:11<00:00, 21.07it/s]


```python
# library & dataset
import pandas as pd
import seaborn as sns
df = pd.DataFrame(res_lsboost_B250[0][0],
                  columns=["accuracy"])
```

```python
# Plot the histogram thanks to the distplot function
sns.distplot(a=df["accuracy"], hist=True, kde=True, rug=True)
```

![image-title-here]({{base}}/images/2023-11-05/2023-11-05-image4.png){:class="img-responsive"}

## 3 - 2 **Classification of Wine dataset**

```python
data = load_wine()
X = data.data
y = data.target
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=13)
```

```python
res2 = optimize_lsboost(X_train, y_train)
print(res2)
parameters = res2["parameters"]
start = time()
estimator = ms.LSBoostClassifier(n_estimators=int(parameters[0][0]),
                                   learning_rate=parameters[0][1],
                                   n_hidden_features=int(parameters[0][2]),
                                   reg_lambda=parameters[0][3],
                                   dropout=parameters[0][4],
                                   tolerance=parameters[0][5],
                                   seed=123, verbose=1).fit(X_train, y_train)
```

```python
print(f"\n\n Test set accuracy: {estimator.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
```
    
     Test set accuracy: 1.0
    
     Elapsed: 0.6752924919128418


**test set accuracy's distribution**


```python
def eval_lsboost2(B=250):

  res_metric = []
  training_times = []
  testing_times = []

  DescribeResult = namedtuple('DescribeResult', ('accuracy',
                                                 'training_time',
                                                 'testing_time'))

  for i in tqdm(range(B)):

    np.random.seed(10*i+100)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)

    start = time()
    obj = ms.LSBoostClassifier(n_estimators=int(parameters[0][0]),
                                   learning_rate=parameters[0][1],
                                   n_hidden_features=int(parameters[0][2]),
                                   reg_lambda=parameters[0][3],
                                   dropout=parameters[0][4],
                                   tolerance=parameters[0][5],
                                   seed=123, verbose=0).fit(X_train, y_train)
    training_times.append(time()-start)
    start = time()
    res_metric.append(obj.score(X_test, y_test))
    testing_times.append(time()-start)

  return DescribeResult(res_metric, training_times, testing_times), stats.describe(res_metric), stats.describe(training_times), stats.describe(testing_times)
```

```python
res_lsboost2_B250 = eval_lsboost2(B=250)
```

    100%|██████████| 250/250 [01:23<00:00,  3.01it/s]


```python
# library & dataset
import pandas as pd
import seaborn as sns
df = pd.DataFrame(res_lsboost2_B250[0][0],
                  columns=["accuracy"])
```

```python
# Plot the histogram thanks to the distplot function
sns.distplot(a=df["accuracy"], hist=True, kde=True, rug=True)
```

![image-title-here]({{base}}/images/2023-11-05/2023-11-05-image5.png){:class="img-responsive"}

# 4 - R example

```R
install.packages("remotes")

remotes::install_github("Techtonique/mlsauce/R-package")

library(datasets)

X <- as.matrix(iris[, 1:4])
y <- as.integer(iris[, 5]) - 1L

n <- dim(X)[1]
p <- dim(X)[2]
set.seed(21341)
train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
test_index <- -train_index
X_train <- as.matrix(iris[train_index, 1:4])
y_train <- as.integer(iris[train_index, 5]) - 1L
X_test <- as.matrix(iris[test_index, 1:4])
y_test <- as.integer(iris[test_index, 5]) - 1L

obj <- mlsauce::AdaOpt()

print(obj$get_params())

obj$fit(X_train, y_train)

# Accuracy (\~ 97\%)
print(obj$score(X_test, y_test))
```