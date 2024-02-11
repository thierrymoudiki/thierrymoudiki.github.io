---
layout: post
title: "`nnetsauce`'s introduction as of 2024-02-11 (new version `0.17.0`)"
description: "`nnetsauce`'s version `0.17.0` is available"
date: 2024-02-11
categories: [Python, QuasiRandomizedNN]
comments: true
---

## 0 - What is `nnetsauce`?

`nnetsauce` is a general-purpose computational tool for
statistical/Machine Learning (ML). As of 2024-02-11, `nnetsauce` is
available for R and Python. It contains various implementations of –
shallow and deep – regression, classification, univariate and
multivariate probabilistic time series forecasting models.

The specificity of `nnetsauce` resides on the fact that the great
majority of these ML models are powered by Quasi-Randomized Networks
(QRNNs, see Moudiki, Planchet, and Cousin (2018) and Moudiki
(2019-2024)). Indeed, whereas traditional *neural* networks (NNs, see
Goodfellow, Bengio, and Courville (2016)) are trained by using
gradient-based optimization algorithms, QRNNs do actually engineer
features by creating new, randomized or deterministic ones.

The package lives in and relies on a **well-supported and documented Python
ecosystem**; the one introduced by Pedregosa et al. (2011):

-   ML models are trained by using a method called `fit`
-   predictions on unseen data are made by a method called `predict`
-   thanks to object-oriented programming’s inheritance, the models are
    a 100% compatible with `scikit-learn`’s `Pipeline`s,
    `GridSearchCV`s, `RandomizedSearchCV`s, and `cross_val_score`s
    (tools for model calibration).

Changes in version `0.17.0` (2024-02-11) include: 

- Attribute `estimators` (a list of `Estimator`’s as strings) for `LazyClassifier`,
`LazyRegressor`, `LazyDeepClassifier`, `LazyDeepRegressor`, `LazyMTS`,
and `LazyDeepMTS` 
- New documentation for the package, using `pdoc`
(highly customizable, based on the excellent Jinja2, HTML and CSS):
<https://techtonique.github.io/nnetsauce/> 
- Remove external regressors `xreg` at inference time for time series forecasting classes `MTS` and
`DeepMTS` 
- New class `Downloader`: querying the R universe API for
datasets (see
https://thierrymoudiki.github.io/blog/2023/12/25/python/r/misc/mlsauce/runiverse-api2
for similar example in `mlsauce`) 
- Add custom metrics to `Lazy*` 
- Rename Deep regressors and classifiers to `Deep*` in `Lazy*` 
- Add attribute `sort_by` to `Lazy*` – sort the data frame output by a given
metric 
- Add attribute `classes_` to classifiers (ensure consistency
with `scikit-learn`)

The *best* way to illustrate the use of `nnetsauce` is to showcase its
integrated automated machine learning (AutoML) functionalities for
solving a real-world problem. This will be done by applying a
classification task on the Wisconsin breast cancer data set available in
`scikit-learn` (Pedregosa et al. (2011)). The same functionality is also
available for regression and time series forecasting tasks.

**Contents**

- [1 - Installing `nnetsauce` for Python](#1---installing-nnetsauce-for-python)
- [2 - Automated Machine Learning (AutoML) with `nnetsauce` classifiers on Wisconsin breast cancer data set](#2---automated-machine-learning-automl-with-nnetsauce-classifiers-on-wisconsin-breast-cancer-data-set)
- [dependencies](#dependencies)

# 1 - Installing `nnetsauce` for Python

There are three ways to install `nnetsauce` for Python:

-   **1st method**: from PyPI by using `pip` at the command line (stable
    version):

``` bash
pip install nnetsauce
```

-   **2nd method**: using `conda` (Linux and macOS only for now)

``` bash
conda install -c conda-forge nnetsauce 
```

-   **3rd method**: from Github (development version)

``` bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

or

``` bash
git clone https://github.com/Techtonique/nnetsauce.git
cd nnetsauce
make install
```

One way to **run all the examples** available in the repository after
cloning it (as done in this 3rd installation option) is to navigate into
it and run:

``` bash
cd nnetsauce
make install
make run-examples 
```

There are also: - Several Jupyter notebooks available in
<https://github.com/Techtonique/nnetsauce/tree/master/nnetsauce/demo>. -
Blog posts available in
<https://thierrymoudiki.github.io/blog#QuasiRandomizedNN>.

# 2 - Automated Machine Learning (AutoML) with `nnetsauce` classifiers on Wisconsin breast cancer data set

**Import the packages required for the examples:**


```
import nnetsauce as ns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import  load_breast_cancer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
```

**Run AutoML with `nnetsauce` classifiers on Wisconsin breast cancer
(identify and benign or malignant) *toy* data set:**


```
# load the whole data set 
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# splitting data in a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)
```

**Build the AutoML classifier**

There are more than 90 classifiers available in `nnetsauce` as of
2024-02-11. The `LazyClassifier` is a meta-estimator that fits a large
number of classifiers of different types, and provides the *best* model.
It is a good starting point for solving a classification problem.


```
# build the AutoML classifier
clf = ns.LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, 
n_hidden_features=10, col_sample=0.9)

# adjust on training and evaluate on test data
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

    
      0%|          | 0/92 [00:00<?, ?it/s]
      1%|1         | 1/92 [00:00<01:06,  1.38it/s]
      2%|2         | 2/92 [00:00<00:37,  2.41it/s]
      4%|4         | 4/92 [00:01<00:18,  4.72it/s]
      8%|7         | 7/92 [00:01<00:09,  8.75it/s]
     10%|9         | 9/92 [00:01<00:10,  7.74it/s]
     12%|#1        | 11/92 [00:01<00:08,  9.47it/s]
     15%|#5        | 14/92 [00:01<00:06, 11.41it/s]
     18%|#8        | 17/92 [00:02<00:07,  9.84it/s]
     21%|##        | 19/92 [00:02<00:12,  5.75it/s]
     24%|##3       | 22/92 [00:03<00:08,  7.90it/s]
     26%|##6       | 24/92 [00:03<00:10,  6.36it/s]
     28%|##8       | 26/92 [00:03<00:08,  7.62it/s]
     30%|###       | 28/92 [00:03<00:07,  9.11it/s]
     33%|###2      | 30/92 [00:04<00:14,  4.14it/s]
     35%|###4      | 32/92 [00:05<00:12,  4.91it/s]
     37%|###6      | 34/92 [00:05<00:09,  5.92it/s]
     39%|###9      | 36/92 [00:06<00:18,  3.06it/s]
     40%|####      | 37/92 [00:07<00:21,  2.55it/s]
     42%|####2     | 39/92 [00:07<00:16,  3.24it/s]
     45%|####4     | 41/92 [00:08<00:13,  3.65it/s]
     46%|####5     | 42/92 [00:08<00:14,  3.45it/s]
     47%|####6     | 43/92 [00:08<00:12,  3.93it/s]
     48%|####7     | 44/92 [00:10<00:24,  1.96it/s]
     49%|####8     | 45/92 [00:10<00:19,  2.39it/s]
     50%|#####     | 46/92 [00:10<00:18,  2.55it/s]
     51%|#####1    | 47/92 [00:10<00:14,  3.07it/s]
     52%|#####2    | 48/92 [00:10<00:11,  3.79it/s]
     53%|#####3    | 49/92 [00:11<00:12,  3.58it/s]
     54%|#####4    | 50/92 [00:12<00:22,  1.90it/s]
     57%|#####6    | 52/92 [00:12<00:12,  3.11it/s]
     58%|#####7    | 53/92 [00:12<00:11,  3.49it/s]
     59%|#####8    | 54/92 [00:12<00:11,  3.44it/s]
     60%|#####9    | 55/92 [00:14<00:21,  1.69it/s]
     62%|######1   | 57/92 [00:14<00:12,  2.77it/s]
     64%|######4   | 59/92 [00:14<00:08,  4.09it/s]
     66%|######6   | 61/92 [00:14<00:07,  4.06it/s]
     67%|######7   | 62/92 [00:15<00:07,  3.82it/s]
     70%|######9   | 64/92 [00:15<00:05,  5.32it/s]
     73%|#######2  | 67/92 [00:16<00:04,  5.08it/s]
     75%|#######5  | 69/92 [00:16<00:04,  4.62it/s]
     77%|#######7  | 71/92 [00:16<00:03,  5.78it/s]
     80%|########  | 74/92 [00:16<00:02,  7.62it/s]
     83%|########2 | 76/92 [00:17<00:03,  4.52it/s]
     85%|########4 | 78/92 [00:18<00:02,  5.09it/s]
     87%|########6 | 80/92 [00:18<00:01,  6.28it/s]
     89%|########9 | 82/92 [00:19<00:03,  3.12it/s]
     92%|#########2| 85/92 [00:19<00:01,  4.35it/s]
     93%|#########3| 86/92 [00:20<00:01,  4.13it/s]
     95%|#########4| 87/92 [00:21<00:02,  2.35it/s]
    100%|##########| 92/92 [00:21<00:00,  4.27it/s]

**Print the models’ leaderboard**


```
print(models)
```

                                                    Accuracy  ...  Time Taken
    Model                                                     ...            
    SimpleMultitaskClassifier(ExtraTreesRegressor)      0.99  ...        0.50
    CustomClassifier(RandomForestClassifier)            0.99  ...        0.43
    CustomClassifier(MLPClassifier)                     0.99  ...        0.73
    CustomClassifier(Perceptron)                        0.99  ...        0.04
    CustomClassifier(LogisticRegression)                0.99  ...        0.05
    ...                                                  ...  ...         ...
    MultitaskClassifier(DummyRegressor)                 0.64  ...        0.08
    SimpleMultitaskClassifier(LassoLars)                0.64  ...        0.05
    SimpleMultitaskClassifier(Lasso)                    0.64  ...        0.03
    SimpleMultitaskClassifier(Lars)                     0.41  ...        0.05
    MultitaskClassifier(Lars)                           0.39  ...        0.31
    
    [89 rows x 5 columns]

**Provide the *best* model**


```
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
fit_obj = model_dictionary['SimpleMultitaskClassifier(ExtraTreesRegressor)']
fit_obj
```

    CustomClassifier(col_sample=0.9, n_hidden_features=10,
                     obj=SimpleMultitaskClassifier(obj=ExtraTreesRegressor()))

**Classification report**


```
# Classification report
start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}")
```

    Elapsed 0.04414701461791992              precision    recall  f1-score   support
    
               0       0.98      1.00      0.99        40
               1       1.00      0.99      0.99        74
    
        accuracy                           0.99       114
       macro avg       0.99      0.99      0.99       114
    weighted avg       0.99      0.99      0.99       114

**Confusion matrix**


```
metrics.ConfusionMatrixDisplay.from_estimator(fit_obj, X_test, y_test)
```

    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x12ccf3050>


    
![xxx]({{base}}/images/2024-02-11/2024-02-11-image1.png){:class="img-responsive"}  
    


# dependencies


```
python3 -m pip freeze
```

    anyio==4.2.0
    appnope==0.1.3
    argon2-cffi==23.1.0
    argon2-cffi-bindings==21.2.0
    arrow==1.3.0
    asttokens==2.4.1
    async-lru==2.0.4
    attrs==23.2.0
    Babel==2.14.0
    beautifulsoup4==4.12.2
    binaryornot==0.4.4
    bleach==6.1.0
    certifi==2023.11.17
    cffi==1.16.0
    chardet==5.2.0
    charset-normalizer==3.3.2
    click==8.1.7
    comm==0.2.0
    contourpy==1.2.0
    cookiecutter==2.5.0
    cycler==0.12.1
    debugpy==1.8.0
    decorator==5.1.1
    defusedxml==0.7.1
    distlib==0.3.8
    docutils==0.20.1
    executing==2.0.1
    fastjsonschema==2.19.1
    filelock==3.13.1
    fonttools==4.47.2
    fqdn==1.5.1
    fsspec==2023.12.2
    idna==3.6
    importlib-metadata==7.0.1
    ipykernel==6.28.0
    ipython==8.19.0
    ipywidgets==8.1.1
    isoduration==20.11.0
    jaraco.classes==3.3.0
    jax==0.4.23
    jaxlib==0.4.23
    jedi==0.19.1
    Jinja2==3.1.2
    joblib==1.3.2
    json5==0.9.14
    jsonpointer==2.4
    jsonschema==4.20.0
    jsonschema-specifications==2023.12.1
    jupyter==1.0.0
    jupyter-console==6.6.3
    jupyter-events==0.9.0
    jupyter-lsp==2.2.1
    jupyter_client==8.6.0
    jupyter_core==5.5.1
    jupyter_server==2.12.1
    jupyter_server_terminals==0.5.1
    jupyterlab==4.0.10
    jupyterlab-widgets==3.0.9
    jupyterlab_pygments==0.3.0
    jupyterlab_server==2.25.2
    keyring==24.3.0
    kiwisolver==1.4.5
    liac-arff==2.5.0
    markdown-it-py==3.0.0
    MarkupSafe==2.1.3
    matplotlib==3.8.2
    matplotlib-inline==0.1.6
    mdurl==0.1.2
    minio==7.2.3
    mistune==3.0.2
    ml-dtypes==0.3.2
    more-itertools==10.1.0
    mpmath==1.3.0
    nbclient==0.9.0
    nbconvert==7.13.1
    nbformat==5.9.2
    nest-asyncio==1.5.8
    networkx==3.2.1
    nh3==0.2.15
    nnetsauce @ file:///Users/t/Documents/Python_Packages/nnetsauce
    notebook==7.0.6
    notebook_shim==0.2.3
    numpy==1.26.3
    openml==0.14.2
    opt-einsum==3.3.0
    overrides==7.4.0
    packaging==23.2
    pandas==2.2.0
    pandocfilters==1.5.0
    parso==0.8.3
    patsy==0.5.6
    pexpect==4.9.0
    pillow==10.2.0
    pkginfo==1.9.6
    platformdirs==4.1.0
    prometheus-client==0.19.0
    prompt-toolkit==3.0.43
    psutil==5.9.7
    ptyprocess==0.7.0
    pure-eval==0.2.2
    pyarrow==15.0.0
    pycparser==2.21
    pycryptodome==3.20.0
    Pygments==2.17.2
    pyparsing==3.1.1
    python-dateutil==2.8.2
    python-json-logger==2.0.7
    python-slugify==8.0.1
    pytz==2023.4
    PyYAML==6.0.1
    pyzmq==25.1.2
    qtconsole==5.5.1
    QtPy==2.4.1
    readme-renderer==42.0
    referencing==0.32.0
    requests==2.31.0
    requests-toolbelt==1.0.0
    rfc3339-validator==0.1.4
    rfc3986==2.0.0
    rfc3986-validator==0.1.1
    rich==13.7.0
    rpds-py==0.16.2
    scikit-learn==1.4.0
    scipy==1.12.0
    Send2Trash==1.8.2
    six==1.16.0
    sniffio==1.3.0
    soupsieve==2.5
    stack-data==0.6.3
    statsmodels==0.14.1
    sympy==1.12
    tabpfn==0.1.9
    terminado==0.18.0
    text-unidecode==1.3
    threadpoolctl==3.2.0
    tinycss2==1.2.1
    torch==2.2.0
    tornado==6.4
    tqdm==4.66.1
    traitlets==5.14.0
    twine==4.0.2
    types-python-dateutil==2.8.19.14
    typing_extensions==4.9.0
    tzdata==2023.4
    uri-template==1.3.0
    urllib3==2.1.0
    virtualenv==20.25.0
    wcwidth==0.2.12
    webcolors==1.13
    webencodings==0.5.1
    websocket-client==1.7.0
    widgetsnbextension==4.0.9
    xmltodict==0.13.0
    zipp==3.17.0


```
sessionInfo()
```

    R version 4.3.2 (2023-10-31)
    Platform: x86_64-apple-darwin20 (64-bit)
    Running under: macOS Sonoma 14.2
    
    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/lib/libRblas.0.dylib 
    LAPACK: /Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/lib/libRlapack.dylib;  LAPACK version 3.11.0
    
    locale:
    [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    
    time zone: Europe/Paris
    tzcode source: internal
    
    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base     
    
    loaded via a namespace (and not attached):
     [1] digest_0.6.34     fastmap_1.1.1     xfun_0.41         Matrix_1.6-1.1   
     [5] lattice_0.21-9    reticulate_1.34.0 knitr_1.45        htmltools_0.5.7  
     [9] png_0.1-8         rmarkdown_2.25    cli_3.6.2         grid_4.3.2       
    [13] compiler_4.3.2    rprojroot_2.0.4   here_1.0.1        rstudioapi_0.15.0
    [17] tools_4.3.2       evaluate_0.23     Rcpp_1.0.12       yaml_2.3.8       
    [21] rlang_1.1.3       jsonlite_1.8.8   

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. 2016. *Deep
Learning*. MIT press.

Moudiki, T. 2019-2024. “Nnetsauce, A Package for Statistical/Machine
Learning Using Randomized and Quasi-Randomized (Neural) Networks.”
<https://github.com/thierrymoudiki/nnetsauce>.

Moudiki, T., Frédéric Planchet, and Areski Cousin. 2018. “Multiple Time
Series Forecasting Using Quasi-Randomized Functional Link Neural
Networks.” *Risks* 6 (1): 22.

Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
Bertrand Thirion, Olivier Grisel, Mathieu Blondel, et al. 2011.
“Scikit-Learn: Machine Learning in Python.” *The Journal of Machine
Learning Research* 12: 2825–30.
