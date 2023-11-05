```python

```


```python
!pip uninstall mlsauce --yes
```

    [33mWARNING: Skipping mlsauce as it is not installed.[0m[33m
    [0m


```python
!pip install GPopt
```

    Collecting GPopt
      Downloading GPopt-0.2.4-py2.py3-none-any.whl (69 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m69.0/69.0 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from GPopt) (1.3.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from GPopt) (1.23.5)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from GPopt) (1.5.3)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from GPopt) (1.11.3)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from GPopt) (1.2.2)
    Requirement already satisfied: threadpoolctl in /usr/local/lib/python3.10/dist-packages (from GPopt) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->GPopt) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->GPopt) (2023.3.post1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->GPopt) (1.16.0)
    Installing collected packages: GPopt
    Successfully installed GPopt-0.2.4



```python
!pip install mlsauce
```

    Collecting mlsauce
      Downloading mlsauce-0.8.10.tar.gz (33 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: jax in /usr/local/lib/python3.10/dist-packages (from mlsauce) (0.4.16)
    Requirement already satisfied: jaxlib in /usr/local/lib/python3.10/dist-packages (from mlsauce) (0.4.16+cuda11.cudnn86)
    Requirement already satisfied: numpy>=1.13.0 in /usr/local/lib/python3.10/dist-packages (from mlsauce) (1.23.5)
    Requirement already satisfied: scipy>=0.19.0 in /usr/local/lib/python3.10/dist-packages (from mlsauce) (1.11.3)
    Requirement already satisfied: joblib>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from mlsauce) (1.3.2)
    Requirement already satisfied: scikit-learn>=0.18.0 in /usr/local/lib/python3.10/dist-packages (from mlsauce) (1.2.2)
    Requirement already satisfied: pandas>=0.25.3 in /usr/local/lib/python3.10/dist-packages (from mlsauce) (1.5.3)
    Collecting tqdm==4.48.1 (from mlsauce)
      Downloading tqdm-4.48.1-py2.py3-none-any.whl (68 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m68.3/68.3 kB[0m [31m4.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.25.3->mlsauce) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.25.3->mlsauce) (2023.3.post1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.18.0->mlsauce) (3.2.0)
    Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax->mlsauce) (0.2.0)
    Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax->mlsauce) (3.3.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas>=0.25.3->mlsauce) (1.16.0)
    Building wheels for collected packages: mlsauce
      Building wheel for mlsauce (setup.py) ... [?25l[?25hdone
      Created wheel for mlsauce: filename=mlsauce-0.8.10-cp310-cp310-linux_x86_64.whl size=3136743 sha256=fded06f7390f3fdd6f96eab1dd2a694ea04ea632a78e7855f09927b41373d2f7
      Stored in directory: /root/.cache/pip/wheels/44/e3/e1/ac4c001ea496899483d6c6d2e6e6493ffab64a6210886a1125
    Successfully built mlsauce
    Installing collected packages: tqdm, mlsauce
      Attempting uninstall: tqdm
        Found existing installation: tqdm 4.66.1
        Uninstalling tqdm-4.66.1:
          Successfully uninstalled tqdm-4.66.1
    Successfully installed mlsauce-0.8.10 tqdm-4.48.1



```python

```

# 1 - LSBoostClassifier


```python

```


```python
import GPopt as gp
import mlsauce as ms
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from time import time
from functools import cache
```

## 1 - 1 breast cancer


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

    
     Creating initial design... 
    
    point: [5.500000e+01 2.005000e-01 1.275000e+02 5.000005e+03 3.500000e-01
     5.000000e-02]; score: -0.9076923076923077
    point: [7.7500000e+01 1.0075000e-01 1.8875000e+02 2.5000075e+03 5.2500000e-01
     2.5000000e-02]; score: -0.9252747252747253
    point: [3.2500000e+01 3.0025000e-01 6.6250000e+01 7.5000025e+03 1.7500000e-01
     7.5000000e-02]; score: -0.9010989010989011
    point: [4.37500000e+01 1.50625000e-01 1.58125000e+02 1.25000875e+03
     6.12500000e-01 8.75000000e-02]; score: -0.9252747252747253
    point: [8.87500000e+01 3.50125000e-01 3.56250000e+01 6.25000375e+03
     2.62500000e-01 3.75000000e-02]; score: -0.9208791208791209
    point: [6.62500000e+01 5.08750000e-02 9.68750000e+01 3.75000625e+03
     8.75000000e-02 6.25000000e-02]; score: -0.8791208791208792
    point: [2.12500000e+01 2.50375000e-01 2.19375000e+02 8.75000125e+03
     4.37500000e-01 1.25000000e-02]; score: -0.9032967032967033
    point: [2.68750000e+01 1.25687500e-01 8.15625000e+01 6.87500313e+03
     3.93750000e-01 1.87500000e-02]; score: -0.8923076923076924
    point: [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]; score: -0.9340659340659341
    point: [9.43750000e+01 2.59375000e-02 1.42812500e+02 9.37500063e+03
     2.18750000e-01 4.37500000e-02]; score: -0.843956043956044
    
     ...Done. 
    
    
     Optimization loop... 
    
    iteration 1 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9340659340659341
    next parameter: [1.47845459e+01 2.24195496e-01 9.23187256e+00 2.07825499e+03
     1.48297119e-01 4.30480957e-02]
    score for next parameter: -0.9340659340659341 
    
    iteration 2 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9340659340659341
    next parameter: [8.55282593e+01 2.30288696e-03 6.56265259e+00 1.69709082e+03
     1.23367310e-01 1.44744873e-02]
    score for next parameter: -0.6153846153846154 
    
    iteration 3 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9340659340659341
    next parameter: [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    score for next parameter: -0.9494505494505494 
    
    iteration 4 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [9.57592773e+01 1.33577881e-01 1.54835205e+02 2.52198013e+03
     3.98706055e-01 9.97802734e-02]
    score for next parameter: -0.8989010989010989 
    
    iteration 5 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [9.16943359e+01 1.40532227e-02 1.52980957e+02 2.48535908e+03
     3.62646484e-01 5.65917969e-02]
    score for next parameter: -0.8065934065934066 
    
    iteration 6 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [7.21578979e+01 1.98442169e-01 1.70543976e+02 2.53083022e+03
     2.22573853e-01 6.14593506e-02]
    score for next parameter: -0.9164835164835166 
    
    iteration 7 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [4.30661011e+01 7.72615051e-02 1.98537140e+02 1.87653400e+03
     4.21499634e-01 1.05804443e-02]
    score for next parameter: -0.9274725274725275 
    
    iteration 8 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [5.53378296e+01 3.85132477e-01 2.24406891e+02 1.85822347e+03
     2.03689575e-01 2.62420654e-02]
    score for next parameter: -0.9494505494505494 
    
    iteration 9 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [7.68353271e+01 2.17181824e-01 2.40354919e+02 1.86341146e+03
     2.66174316e-02 8.47778320e-03]
    score for next parameter: -0.9472527472527472 
    
    iteration 10 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [6.82357788e+01 1.33906647e-01 2.40138092e+02 1.88996172e+03
     2.74270630e-01 9.80438232e-02]
    score for next parameter: -0.9054945054945055 
    
    iteration 11 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [7.49511719e+01 3.87958984e-02 2.23920898e+02 1.82618005e+03
     3.86230469e-01 5.32226562e-02]
    score for next parameter: -0.8945054945054945 
    
    iteration 12 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [4.90124512e+01 3.62144775e-02 2.49491577e+02 1.85913900e+03
     5.25939941e-01 9.07104492e-02]
    score for next parameter: -0.8791208791208792 
    
    iteration 13 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [9.17437744e+01 5.75720825e-02 2.19569397e+02 1.85730795e+03
     5.17523193e-01 2.39929199e-02]
    score for next parameter: -0.9186813186813186 
    
    iteration 14 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [4.85263062e+01 3.20645660e-01 1.93423004e+02 1.84357505e+03
     3.93386841e-01 4.01580811e-02]
    score for next parameter: -0.9406593406593406 
    
    iteration 15 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [1.96075439e+01 3.28280334e-01 3.01370239e+01 2.10877254e+03
     3.81530762e-02 6.64978027e-02]
    score for next parameter: -0.9318681318681319 
    
    iteration 16 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [1.28756714e+01 3.10076447e-01 4.98384094e+01 2.06696350e+03
     6.92544556e-01 4.89105225e-02]
    score for next parameter: -0.9362637362637363 
    
    iteration 17 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [4.94738770e+01 3.38094604e-01 4.00811768e+01 2.08130675e+03
     4.95690918e-01 3.53637695e-02]
    score for next parameter: -0.9384615384615385 
    
    iteration 18 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [3.27938843e+01 3.65919495e-02 6.97267151e+01 2.08771543e+03
     4.67898560e-01 4.18182373e-02]
    score for next parameter: -0.879120879120879 
    
    iteration 19 -----
    current minimum:  [8.25811768e+01 3.78739807e-01 1.80749817e+02 2.50427996e+03
     4.30877686e-01 1.52770996e-02]
    current minimum score:  -0.9494505494505494
    next parameter: [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    score for next parameter: -0.9516483516483516 
    
    iteration 20 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.80490112e+01 2.14443665e-02 9.34402466e+00 2.05963929e+03
     5.02676392e-01 7.03216553e-02]
    score for next parameter: -0.8483516483516483 
    
    iteration 21 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [1.39550781e+01 4.73681641e-02 2.53369141e+01 2.04102358e+03
     4.88769531e-01 9.52148438e-02]
    score for next parameter: -0.8373626373626374 
    
    iteration 22 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [2.80917358e+01 3.81235992e-01 2.48658752e+01 2.07306701e+03
     3.22634888e-01 2.21405029e-02]
    score for next parameter: -0.9428571428571428 
    
    iteration 23 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [4.38763428e+01 3.28781128e-02 4.67953491e+01 2.06848938e+03
     2.86682129e-02 1.34887695e-03]
    score for next parameter: -0.8923076923076924 
    
    iteration 24 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [3.73504639e+01 1.98965759e-01 1.99386597e+01 2.10022763e+03
     5.05987549e-01 5.11291504e-02]
    score for next parameter: -0.9296703296703297 
    
    iteration 25 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.93536377e+01 3.32176819e-01 2.30635071e+02 1.85486654e+03
     1.37957764e-01 8.19274902e-02]
    score for next parameter: -0.9362637362637363 
    
    iteration 26 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [3.93499756e+01 2.78161804e-01 2.08174744e+02 1.83899742e+03
     5.58111572e-01 1.88781738e-02]
    score for next parameter: -0.9428571428571428 
    
    iteration 27 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [5.04818726e+01 6.88125610e-03 3.48548889e+01 2.10480526e+03
     6.85794067e-01 8.30047607e-02]
    score for next parameter: -0.6131868131868132 
    
    iteration 28 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [5.22534180e+01 1.13705811e-01 2.20631104e+02 1.84815268e+03
     6.87866211e-01 1.72607422e-02]
    score for next parameter: -0.9362637362637363 
    
    iteration 29 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [2.32687378e+01 2.71014191e-01 1.54450989e+01 2.10358456e+03
     2.13858032e-01 9.86907959e-02]
    score for next parameter: -0.9208791208791209 
    
    iteration 30 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [4.19976807e+01 5.36755981e-02 2.19424438e+01 2.07215148e+03
     3.57562256e-01 2.77038574e-02]
    score for next parameter: -0.9164835164835164 
    
    iteration 31 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [5.33712769e+01 2.31464874e-01 2.14387970e+02 1.87409260e+03
     4.92422485e-01 6.35833740e-02]
    score for next parameter: -0.9274725274725275 
    
    iteration 32 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [8.53854370e+01 1.39020782e-01 2.35263214e+02 1.86188558e+03
     6.95962524e-01 4.21722412e-02]
    score for next parameter: -0.9296703296703297 
    
    iteration 33 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.24926758e+01 1.80530518e-01 1.89647217e+02 1.87256672e+03
     1.70043945e-01 3.83300781e-03]
    score for next parameter: -0.9472527472527472 
    
    iteration 34 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [5.52169800e+01 4.87197571e-02 1.80174103e+02 1.88385821e+03
     3.91998291e-02 5.05950928e-02]
    score for next parameter: -0.8967032967032967 
    
    iteration 35 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.83676147e+01 5.14960022e-02 1.76136627e+02 2.50031268e+03
     2.91702271e-01 3.49090576e-02]
    score for next parameter: -0.9010989010989011 
    
    iteration 36 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [3.66308594e+01 1.46338867e-01 1.86596680e+02 1.86524251e+03
     4.65527344e-01 5.75195313e-02]
    score for next parameter: -0.9208791208791208 
    
    iteration 37 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [2.60922241e+01 9.63055725e-02 2.03172760e+02 1.86554769e+03
     2.54211426e-03 5.39520264e-02]
    score for next parameter: -0.9120879120879122 
    
    iteration 38 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [3.36975098e+01 1.97236694e-01 2.14859009e+02 1.82251794e+03
     3.14880371e-01 4.12719727e-02]
    score for next parameter: -0.9362637362637363 
    
    iteration 39 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [3.07614136e+01 1.56749786e-01 2.05236359e+02 1.83747154e+03
     1.61734009e-01 3.05023193e-02]
    score for next parameter: -0.9362637362637363 
    
    iteration 40 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [4.01025391e+01 5.57456055e-02 1.93894043e+02 1.81153163e+03
     1.99267578e-01 6.78222656e-02]
    score for next parameter: -0.8945054945054945 
    
    iteration 41 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [7.88293457e+01 3.46666870e-01 2.08997192e+02 1.86646321e+03
     3.79479980e-01 1.29028320e-02]
    score for next parameter: -0.9494505494505494 
    
    iteration 42 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [9.10681152e+01 1.98892700e-01 1.94342651e+02 1.88355304e+03
     4.51086426e-01 5.97045898e-02]
    score for next parameter: -0.9274725274725275 
    
    iteration 43 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [9.55148315e+01 3.93363800e-01 2.19771271e+02 1.88874102e+03
     5.70693970e-01 8.20892334e-02]
    score for next parameter: -0.9318681318681319 
    
    iteration 44 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [7.29379272e+01 3.55421783e-01 1.82835846e+02 1.83625084e+03
     7.56866455e-02 9.85809326e-02]
    score for next parameter: -0.9296703296703297 
    
    iteration 45 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [5.98944092e+01 2.49814880e-01 1.67201843e+02 1.84143882e+03
     2.41778564e-01 6.67053223e-02]
    score for next parameter: -0.9296703296703297 
    
    iteration 46 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.82907104e+01 2.96633575e-01 1.65758820e+02 1.86676839e+03
     8.85894775e-02 2.20062256e-02]
    score for next parameter: -0.945054945054945 
    
    iteration 47 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [8.23284912e+01 1.74904968e-01 1.60562439e+02 1.86829426e+03
     5.88873291e-01 2.73742676e-02]
    score for next parameter: -0.9384615384615385 
    
    iteration 48 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [6.85791016e+01 1.60171387e-01 1.54655762e+02 1.85059409e+03
     3.05908203e-01 7.13378906e-02]
    score for next parameter: -0.9186813186813186 
    
    iteration 49 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [7.69561768e+01 1.54302307e-01 1.65437317e+02 1.87928058e+03
     2.12127686e-01 8.40270996e-02]
    score for next parameter: -0.9142857142857143 
    
    iteration 50 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [9.07247925e+01 1.21973663e-01 1.69646759e+02 1.84479575e+03
     4.79434204e-01 8.30169678e-02]
    score for next parameter: -0.9142857142857143 
    


      0%|          | 0/36 [00:00<?, ?it/s]

    iteration 51 -----
    current minimum:  [3.63891602e+01 3.71847900e-01 3.43688965e+01 2.05811341e+03
     1.69360352e-01 1.21337891e-02]
    current minimum score:  -0.9516483516483516
    next parameter: [9.55477905e+01 3.11368713e-02 1.75478668e+02 1.87287190e+03
     4.06375122e-01 6.46667480e-03]
    score for next parameter: -0.9186813186813186 
    
    {'parameters': (array([3.63891602e+01, 3.71847900e-01, 3.43688965e+01, 2.05811341e+03,
           1.69360352e-01, 1.21337891e-02]), -0.9516483516483516), 'opt_object': <GPopt.GPOpt.GPOpt.GPOpt object at 0x7847834fece0>}


     81%|████████  | 29/36 [00:00<00:00, 355.27it/s]



```python
print(f"\n\n Test set accuracy: {estimator.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
```

    
    
     Test set accuracy: 0.9912280701754386
    
     Elapsed: 0.11275959014892578


**test set accuracy's distribution**


```python

```


```python
from collections import namedtuple
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy import stats
```


```python
@cache
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
    #except ValueError:
    #  continue

    res = tuple()

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
print(df.head())
```

       accuracy
    0  0.947368
    1  0.982456
    2  0.956140
    3  0.956140
    4  0.947368



```python
# Plot the histogram thanks to the distplot function
sns.distplot(a=df["accuracy"], hist=True, kde=True, rug=True)
```




    <Axes: xlabel='accuracy', ylabel='Density'>




![png](thierrymoudiki_051123_GPopt_mlsauce_classification_files/thierrymoudiki_051123_GPopt_mlsauce_classification_21_1.png)



```python

```

## 1 - 2 wine


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

    
     Creating initial design... 
    
    point: [5.500000e+01 2.005000e-01 1.275000e+02 5.000005e+03 3.500000e-01
     5.000000e-02]; score: -0.8950738916256158
    point: [7.7500000e+01 1.0075000e-01 1.8875000e+02 2.5000075e+03 5.2500000e-01
     2.5000000e-02]; score: -0.9302955665024631
    point: [3.2500000e+01 3.0025000e-01 6.6250000e+01 7.5000025e+03 1.7500000e-01
     7.5000000e-02]; score: -0.8448275862068965
    point: [4.37500000e+01 1.50625000e-01 1.58125000e+02 1.25000875e+03
     6.12500000e-01 8.75000000e-02]; score: -0.916256157635468
    point: [8.87500000e+01 3.50125000e-01 3.56250000e+01 6.25000375e+03
     2.62500000e-01 3.75000000e-02]; score: -0.9302955665024631
    point: [6.62500000e+01 5.08750000e-02 9.68750000e+01 3.75000625e+03
     8.75000000e-02 6.25000000e-02]; score: -0.6125615763546798
    point: [2.12500000e+01 2.50375000e-01 2.19375000e+02 8.75000125e+03
     4.37500000e-01 1.25000000e-02]; score: -0.8312807881773399
    point: [2.68750000e+01 1.25687500e-01 8.15625000e+01 6.87500313e+03
     3.93750000e-01 1.87500000e-02]; score: -0.8027093596059114
    point: [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]; score: -0.9371921182266011
    point: [9.43750000e+01 2.59375000e-02 1.42812500e+02 9.37500063e+03
     2.18750000e-01 4.37500000e-02]; score: -0.3943349753694581
    
     ...Done. 
    
    
     Optimization loop... 
    
    iteration 1 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9371921182266011
    next parameter: [8.93542480e+01 1.12195923e-01 1.56420288e+02 2.18140430e+03
     1.71154785e-01 4.68627930e-02]
    score for next parameter: -0.9233990147783251 
    
    iteration 2 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9371921182266011
    next parameter: [9.48776245e+01 1.53048126e-01 4.86421204e+01 5.72906921e+03
     6.27603149e-01 8.00628662e-02]
    score for next parameter: -0.6967980295566503 
    
    iteration 3 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9371921182266011
    next parameter: [1.49987793e+01 8.64791260e-02 8.55895996e+00 1.61499862e+03
     5.38928223e-01 4.70581055e-02]
    score for next parameter: -0.9017241379310346 
    
    iteration 4 -----
    current minimum:  [7.18750000e+01 3.25187500e-01 2.04062500e+02 1.87500813e+03
     4.37500000e-02 6.87500000e-02]
    current minimum score:  -0.9371921182266011
    next parameter: [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    score for next parameter: -0.9509852216748769 
    
    iteration 5 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [1.54547119e+01 1.46411926e-01 2.47323303e+02 8.22448908e+03
     2.79632568e-01 8.53210449e-02]
    score for next parameter: -0.647536945812808 
    
    iteration 6 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [9.22381592e+01 8.14867554e-02 2.47592468e+02 6.37268429e+03
     3.18341064e-01 6.51428223e-02]
    score for next parameter: -0.6268472906403941 
    
    iteration 7 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [2.48590088e+01 2.22393372e-01 2.45169983e+02 1.69129249e+03
     2.68011475e-01 2.46765137e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 8 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [1.14996338e+01 3.00323059e-01 5.91217041e+00 2.39441679e+03
     3.82855225e-01 5.35827637e-02]
    score for next parameter: -0.9445812807881774 
    
    iteration 9 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [2.51226807e+01 2.78113098e-01 6.62994385e+00 2.69715086e+03
     2.26312256e-01 7.14538574e-02]
    score for next parameter: -0.9305418719211822 
    
    iteration 10 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [1.32135010e+01 4.37395630e-02 6.27105713e+00 6.16882707e+03
     1.13861084e-01 7.99987793e-02]
    score for next parameter: -0.3943349753694581 
    
    iteration 11 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [3.37167358e+01 1.56798492e-01 9.55337524e+00 2.69806638e+03
     4.53884888e-01 7.83905029e-02]
    score for next parameter: -0.9305418719211822 
    
    iteration 12 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [5.87957764e+01 2.59750916e-01 2.37184753e+02 1.59241563e+03
     5.30596924e-01 2.73010254e-02]
    score for next parameter: -0.9440886699507389 
    
    iteration 13 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [9.11614990e+01 1.80165222e-01 5.20590210e+01 6.30676639e+03
     6.26129150e-01 9.92004395e-02]
    score for next parameter: -0.6406403940886699 
    
    iteration 14 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [7.46051025e+01 3.08895325e-01 2.36078186e+02 1.55457388e+03
     4.90863037e-01 9.47937012e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 15 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [7.00787354e+01 7.29631958e-02 2.39068909e+02 1.64734722e+03
     4.26263428e-01 4.66979980e-02]
    score for next parameter: -0.9019704433497537 
    
    iteration 16 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [1.36639404e+01 1.05707397e-02 2.47113953e+02 1.62659529e+03
     1.63848877e-01 5.18188477e-03]
    score for next parameter: -0.5564039408866994 
    
    iteration 17 -----
    current minimum:  [9.70391846e+01 8.47500610e-02 2.44751282e+02 1.53260124e+03
     1.65985107e-01 1.19079590e-02]
    current minimum score:  -0.9509852216748769
    next parameter: [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    score for next parameter: -0.958128078817734 
    
    iteration 18 -----
    current minimum:  [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    current minimum score:  -0.958128078817734
    next parameter: [4.72930908e+01 1.97748108e-01 2.35659485e+02 1.70838232e+03
     6.48602295e-01 8.28063965e-02]
    score for next parameter: -0.9022167487684729 
    
    iteration 19 -----
    current minimum:  [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    current minimum score:  -0.958128078817734
    next parameter: [8.66049194e+01 1.03124420e-01 2.02096100e+02 1.60675888e+03
     5.18313599e-01 4.93133545e-02]
    score for next parameter: -0.916256157635468 
    
    iteration 20 -----
    current minimum:  [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    current minimum score:  -0.958128078817734
    next parameter: [4.23574829e+01 1.35246063e-01 1.70152283e+01 2.42279810e+03
     4.31710815e-01 6.13189697e-02]
    score for next parameter: -0.923152709359606 
    
    iteration 21 -----
    current minimum:  [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    current minimum score:  -0.958128078817734
    next parameter: [3.34777832e+01 2.54709839e-01 3.00323486e+01 2.37427520e+03
     3.71276855e-01 9.68872070e-02]
    score for next parameter: -0.9302955665024631 
    
    iteration 22 -----
    current minimum:  [8.97140503e+01 1.82174347e-01 2.42261505e+02 1.59577256e+03
     1.45498657e-01 2.33001709e-02]
    current minimum score:  -0.958128078817734
    next parameter: [9.51303101e+01 1.95471100e-01 2.04249420e+02 1.51886834e+03
     6.77590942e-01 5.27038574e-03]
    score for next parameter: -0.9583743842364532 
    
    iteration 23 -----
    current minimum:  [9.51303101e+01 1.95471100e-01 2.04249420e+02 1.51886834e+03
     6.77590942e-01 5.27038574e-03]
    current minimum score:  -0.9583743842364532
    next parameter: [9.20321655e+01 3.02332184e-01 2.30986481e+02 1.48224729e+03
     1.05593872e-01 4.78729248e-02]
    score for next parameter: -0.958128078817734 
    
    iteration 24 -----
    current minimum:  [9.51303101e+01 1.95471100e-01 2.04249420e+02 1.51886834e+03
     6.77590942e-01 5.27038574e-03]
    current minimum score:  -0.9583743842364532
    next parameter: [5.94329834e+01 3.75379089e-01 1.99681091e+02 1.49231808e+03
     1.61968994e-01 7.15637207e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 25 -----
    current minimum:  [9.51303101e+01 1.95471100e-01 2.04249420e+02 1.51886834e+03
     6.77590942e-01 5.27038574e-03]
    current minimum score:  -0.9583743842364532
    next parameter: [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    score for next parameter: -0.9650246305418719 
    
    iteration 26 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [9.97308350e+01 3.38522339e-02 1.54341736e+02 1.49720089e+03
     6.28521729e-01 8.92883301e-02]
    score for next parameter: -0.41477832512315266 
    
    iteration 27 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [9.23425293e+01 4.28385010e-02 2.11449585e+02 1.45874878e+03
     2.87365723e-01 9.54956055e-02]
    score for next parameter: -0.6965517241379311 
    
    iteration 28 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [9.13867188e+01 3.89869141e-01 2.39951172e+02 1.50391475e+03
     3.59570312e-01 2.79296875e-02]
    score for next parameter: -0.958128078817734 
    
    iteration 29 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [7.22595215e+01 1.37328247e-01 2.19703979e+02 1.51245966e+03
     1.51672363e-01 4.34936523e-02]
    score for next parameter: -0.9302955665024631 
    
    iteration 30 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [9.04830933e+01 5.40774231e-02 2.02245636e+02 1.49079220e+03
     5.57876587e-01 7.86956787e-02]
    score for next parameter: -0.8238916256157636 
    
    iteration 31 -----
    current minimum:  [9.52593994e+01 3.84876770e-01 1.87030334e+02 1.46668334e+03
     4.17419434e-02 3.41491699e-02]
    current minimum score:  -0.9650246305418719
    next parameter: [7.33334351e+01 3.41978912e-01 2.21475983e+02 1.59699327e+03
     3.54934692e-01 7.61413574e-03]
    score for next parameter: -0.9652709359605911 
    
    iteration 32 -----
    current minimum:  [7.33334351e+01 3.41978912e-01 2.21475983e+02 1.59699327e+03
     3.54934692e-01 7.61413574e-03]
    current minimum score:  -0.9652709359605911
    next parameter: [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    score for next parameter: -0.9790640394088671 
    
    iteration 33 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [9.59432983e+01 6.79586487e-02 2.28713531e+02 1.56525502e+03
     4.09942627e-02 9.98748779e-02]
    score for next parameter: -0.8381773399014778 
    
    iteration 34 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.18066406e+01 3.07653320e-01 2.27270508e+02 1.61133651e+03
     5.51660156e-01 2.06054688e-02]
    score for next parameter: -0.9302955665024631 
    
    iteration 35 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [9.59020996e+01 1.57979614e-01 1.72570190e+02 1.44410035e+03
     3.82043457e-01 7.91381836e-02]
    score for next parameter: -0.9233990147783251 
    
    iteration 36 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [6.50305176e+01 9.45643311e-02 1.92966919e+02 1.54419791e+03
     3.61022949e-01 9.32739258e-02]
    score for next parameter: -0.8807881773399014 
    
    iteration 37 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.08370972e+01 1.15544464e-01 2.29102325e+02 1.52741326e+03
     3.48526001e-01 2.84759521e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 38 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.58441162e+01 2.96475281e-01 2.46695251e+02 1.47766966e+03
     2.77154541e-01 1.72180176e-02]
    score for next parameter: -0.958128078817734 
    
    iteration 39 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [1.65203857e+01 2.48451111e-01 3.45333862e+01 2.40296170e+03
     5.02569580e-01 3.01330566e-02]
    score for next parameter: -0.9233990147783251 
    
    iteration 40 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [4.80429077e+01 2.93834534e-02 1.02711487e+01 2.39533231e+03
     3.36990356e-01 9.66522217e-02]
    score for next parameter: -0.3943349753694581 
    
    iteration 41 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [2.47052002e+01 2.68274475e-01 4.01858521e+01 2.36267854e+03
     1.73504639e-01 3.41186523e-03]
    score for next parameter: -0.9302955665024631 
    
    iteration 42 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [3.59002686e+01 3.30472107e-01 4.44625854e+01 2.43714135e+03
     6.38946533e-01 8.35998535e-02]
    score for next parameter: -0.9233990147783251 
    
    iteration 43 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [2.84460449e+01 2.06880493e-01 1.45404053e+01 2.42554468e+03
     2.92150879e-01 8.85620117e-02]
    score for next parameter: -0.923152709359606 
    
    iteration 44 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [6.89965820e+01 1.60988770e-02 2.48504639e+02 1.50147334e+03
     3.46923828e-02 6.69189453e-02]
    score for next parameter: -0.3943349753694581 
    
    iteration 45 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [9.48254395e+01 2.61723511e-01 2.30530396e+02 1.49536983e+03
     5.03894043e-01 5.13793945e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 46 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [6.42559814e+01 2.73729553e-01 2.09101868e+02 1.52283563e+03
     2.60620117e-03 4.50134277e-02]
    score for next parameter: -0.951231527093596 
    
    iteration 47 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.68685913e+01 3.31848053e-01 1.74102936e+02 1.46027465e+03
     3.26223755e-01 7.73406982e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 48 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [1.71411133e+01 1.99953613e-02 2.02526855e+01 2.38037871e+03
     1.39965820e-01 8.15673828e-02]
    score for next parameter: -0.3943349753694581 
    
    iteration 49 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.74591064e+01 1.50029907e-02 2.26328430e+02 1.60096055e+03
     3.25262451e-01 4.26696777e-02]
    score for next parameter: -0.4862068965517241 
    
    iteration 50 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [7.74313354e+01 1.75063263e-01 2.44235382e+02 1.47614378e+03
     4.00308228e-01 3.21624756e-02]
    score for next parameter: -0.9371921182266011 
    
    iteration 51 -----
    current minimum:  [8.76403809e+01 3.19294067e-01 2.03673706e+02 1.54175651e+03
     3.06762695e-02 8.17871094e-04]
    current minimum score:  -0.9790640394088671
    next parameter: [8.10375977e+01 2.30015869e-01 2.05677490e+02 1.55030142e+03
     1.33813477e-01 7.73681641e-02]
    score for next parameter: -0.9302955665024631 
    


     10%|█         | 9/87 [00:00<00:00, 89.68it/s]

    {'parameters': (array([8.76403809e+01, 3.19294067e-01, 2.03673706e+02, 1.54175651e+03,
           3.06762695e-02, 8.17871094e-04]), -0.9790640394088671), 'opt_object': <GPopt.GPOpt.GPOpt.GPOpt object at 0x784772e73c70>}


    100%|██████████| 87/87 [00:00<00:00, 137.31it/s]



```python
print(f"\n\n Test set accuracy: {estimator.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
```

    
    
     Test set accuracy: 1.0
    
     Elapsed: 0.6752924919128418


**test set accuracy's distribution**


```python
@cache
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
    #except ValueError:
    #  continue

    res = tuple()

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
print(df.head())
```

       accuracy
    0  0.972222
    1  0.861111
    2  0.972222
    3  0.972222
    4  1.000000



```python
# Plot the histogram thanks to the distplot function
sns.distplot(a=df["accuracy"], hist=True, kde=True, rug=True)
```




    <Axes: xlabel='accuracy', ylabel='Density'>




![png](thierrymoudiki_051123_GPopt_mlsauce_classification_files/thierrymoudiki_051123_GPopt_mlsauce_classification_32_1.png)



```python

```

# 2 - AdaOpt


```python
import numpy as np
from sklearn.datasets import load_digits
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


    


**test set accuracy's distribution**


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

    #try:
    start = time()
    obj.fit(X_train, y_train)
    training_times.append(time()-start)
    start = time()
    res_metric.append(obj.score(X_test, y_test))
    testing_times.append(time()-start)
    #except ValueError:
    #  continue

    res = tuple()

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



## 2 - 2 accuracy


```python

```


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




    <Axes: xlabel='k4', ylabel='Density'>




![png](thierrymoudiki_051123_GPopt_mlsauce_classification_files/thierrymoudiki_051123_GPopt_mlsauce_classification_46_1.png)


## 2 - 3 training time


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




    <Axes: xlabel='k4', ylabel='Density'>




![png](thierrymoudiki_051123_GPopt_mlsauce_classification_files/thierrymoudiki_051123_GPopt_mlsauce_classification_48_1.png)


## 2 - 4 testing time


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




    <Axes: xlabel='k4', ylabel='Density'>




![png](thierrymoudiki_051123_GPopt_mlsauce_classification_files/thierrymoudiki_051123_GPopt_mlsauce_classification_50_1.png)

