---
layout: post
title: "Conformalized adaptive (online/streaming) learning using learningmachine in Python and R"
description: "Adaptive (online/streaming) learning with uncertainty quantification and explanations using learningmachine in Python and R"
date: 2024-08-19
categories: [R, Python]
comments: true
---

The model presented here is a frequentist -- [conformalized](https://conformalpredictionintro.github.io/) -- version of the Bayesian one presented last week in [#152](https://thierrymoudiki.github.io/blog/2024/08/12/r/bqrvfl). The model is implemented in `learningmachine`, both in [Python](https://github.com/Techtonique/learningmachine_python) and [R](https://github.com/Techtonique/learningmachine). Model explanations are given as sensitivity analyses. 

# 0 - install packages

**For R**


```R
utils::install.packages(c("rmarkdown", "reticulate", "remotes"))
```

    Installing packages into '/cloud/lib/x86_64-pc-linux-gnu-library/4.4'
    (as 'lib' is unspecified)


```R
remotes::install_github("thierrymoudiki/bayesianrvfl")
```

    Skipping install of 'bayesianrvfl' from a github remote, the SHA1 (a8e9e78a) has not changed since last install.
      Use `force = TRUE` to force installationSkipping install of 'learningmachine' from a github remote, the SHA1 (6b930284) has not changed since last install.
      Use `force = TRUE` to force installation


```R
library("learningmachine")
```

    Loading required package: randtoolboxLoading required package: rngWELLThis is randtoolbox. For an overview, type 'help("randtoolbox")'.Loading required package: tseriesRegistered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo Loading required package: memoiseLoading required package: foreachLoading required package: skimrLoading required package: snowLoading required package: doSNOWLoading required package: iterators

**For Python**


```bash
pip install matplotlib
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: matplotlib in /cloud/python/lib/python3.8/site-packages (3.7.5)
    Requirement already satisfied: pyparsing>=2.3.1 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (3.1.2)
    Requirement already satisfied: packaging>=20.0 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (24.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (1.4.5)
    Requirement already satisfied: importlib-resources>=3.2.0 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (6.4.3)
    Requirement already satisfied: numpy<2,>=1.20 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (1.24.4)
    Requirement already satisfied: pillow>=6.2.0 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (10.4.0)
    Requirement already satisfied: cycler>=0.10 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: python-dateutil>=2.7 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: fonttools>=4.22.0 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (4.53.1)
    Requirement already satisfied: contourpy>=1.0.1 in /cloud/python/lib/python3.8/site-packages (from matplotlib) (1.1.1)
    Requirement already satisfied: zipp>=3.1.0 in /cloud/python/lib/python3.8/site-packages (from importlib-resources>=3.2.0->matplotlib) (3.20.0)
    Requirement already satisfied: six>=1.5 in /cloud/python/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    
    [notice] A new release of pip is available: 23.0.1 -> 24.2
    [notice] To update, run: /opt/python/3.8.17/bin/python3.8 -m pip install --upgrade pip


```bash
pip install git+https://github.com/Techtonique/learningmachine_python.git
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting git+https://github.com/Techtonique/learningmachine_python.git
      Cloning https://github.com/Techtonique/learningmachine_python.git to /tmp/pip-req-build-37h1oa6g
      Running command git clone --filter=blob:none --quiet https://github.com/Techtonique/learningmachine_python.git /tmp/pip-req-build-37h1oa6g
      Resolved https://github.com/Techtonique/learningmachine_python.git to commit 3ec7ca96df71add6218d55db9ef5d8eb40275877
      Preparing metadata (setup.py): started
      Preparing metadata (setup.py): finished with status 'done'
    Requirement already satisfied: numpy in /cloud/python/lib/python3.8/site-packages (from learningmachine==2.2.2) (1.24.4)
    Requirement already satisfied: pandas in /cloud/python/lib/python3.8/site-packages (from learningmachine==2.2.2) (2.0.3)
    Requirement already satisfied: rpy2>=3.4.5 in /cloud/python/lib/python3.8/site-packages (from learningmachine==2.2.2) (3.5.16)
    Requirement already satisfied: scikit-learn in /cloud/python/lib/python3.8/site-packages (from learningmachine==2.2.2) (1.3.2)
    Requirement already satisfied: scipy in /cloud/python/lib/python3.8/site-packages (from learningmachine==2.2.2) (1.10.1)
    Requirement already satisfied: jinja2 in /cloud/python/lib/python3.8/site-packages (from rpy2>=3.4.5->learningmachine==2.2.2) (3.1.4)
    Requirement already satisfied: tzlocal in /cloud/python/lib/python3.8/site-packages (from rpy2>=3.4.5->learningmachine==2.2.2) (5.2)
    Requirement already satisfied: backports.zoneinfo in /cloud/python/lib/python3.8/site-packages (from rpy2>=3.4.5->learningmachine==2.2.2) (0.2.1)
    Requirement already satisfied: cffi>=1.15.1 in /cloud/python/lib/python3.8/site-packages (from rpy2>=3.4.5->learningmachine==2.2.2) (1.17.0)
    Requirement already satisfied: tzdata>=2022.1 in /cloud/python/lib/python3.8/site-packages (from pandas->learningmachine==2.2.2) (2024.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /cloud/python/lib/python3.8/site-packages (from pandas->learningmachine==2.2.2) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /cloud/python/lib/python3.8/site-packages (from pandas->learningmachine==2.2.2) (2024.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /cloud/python/lib/python3.8/site-packages (from scikit-learn->learningmachine==2.2.2) (3.5.0)
    Requirement already satisfied: joblib>=1.1.1 in /cloud/python/lib/python3.8/site-packages (from scikit-learn->learningmachine==2.2.2) (1.4.2)
    Requirement already satisfied: pycparser in /cloud/python/lib/python3.8/site-packages (from cffi>=1.15.1->rpy2>=3.4.5->learningmachine==2.2.2) (2.22)
    Requirement already satisfied: six>=1.5 in /cloud/python/lib/python3.8/site-packages (from python-dateutil>=2.8.2->pandas->learningmachine==2.2.2) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /cloud/python/lib/python3.8/site-packages (from jinja2->rpy2>=3.4.5->learningmachine==2.2.2) (2.1.5)
    
    [notice] A new release of pip is available: 23.0.1 -> 24.2
    [notice] To update, run: /opt/python/3.8.17/bin/python3.8 -m pip install --upgrade pip

# 1 - Python code version

**import packages**


```python
import numpy as np 
import warnings 
import learningmachine as lm
```

    /cloud/python/lib/python3.8/site-packages/rpy2/rinterface_lib/embedded.py:276: UserWarning: R was initialized outside of rpy2 (R_NilValue != NULL). Trying to use it nevertheless.
      warnings.warn(msg)
    R was initialized outside of rpy2 (R_NilValue != NULL). Trying to use it nevertheless.

**plotting function**


```python
warnings.filterwarnings('ignore')

split_color = 'green'
split_color2 = 'orange'
local_color = 'gray'

def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color=split_color,
              method_name="",
              title=""):

    fig = plt.figure()

    plt.plot(x, y, 'k.', alpha=.3, markersize=10,
             fillstyle='full', label=u'Test set observations')

    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_u, y_l[::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' Prediction interval')

    if pred is not None:
        plt.plot(x, pred, 'k--', lw=2, alpha=0.9,
                 label=u'Predicted value')

    #plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.show()
```


```python
fit_obj = lm.Regressor(method="rvfl",
pi_method = "kdejackknifeplus",
nb_hidden = 3)

data = fetch_california_housing()

X, y = data.data[:600], data.target[:600]

X = pd.DataFrame(X, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=123)

start = time()

fit_obj.fit(X_train, y_train, reg_lambda=1)
```

    Regressor(method='rvfl', nb_hidden=3, pi_method='kdejackknifeplus')Elapsed time:  11.298886775970459preds: DescribeResult(preds=array([2.49357776, 1.43407668, 1.5975387 , 1.12443032, 2.82375138,
           2.4580493 , 3.54353746, 1.8909777 , 2.05285337, 3.77584837,
           1.83513261, 1.07393557, 1.94194497, 0.92784638, 1.05871262,
           2.22243527, 1.65294959, 1.91187611, 1.77770079, 1.25768064,
           1.87622468, 1.74061527, 2.92453382, 1.66174853, 1.6614039 ,
           1.77202212, 5.07081658, 2.38262843, 2.53746616, 1.363203  ,
           3.20583396, 1.86606062, 1.70192252, 1.07269937, 1.31627684,
           2.03799042, 3.26290966, 3.97461689, 1.13406842, 2.47295003,
           1.5368609 , 1.09332542, 1.4092314 , 2.60417043, 5.89174275,
           1.50619443, 2.4628304 , 2.19871466, 2.1841939 , 1.48221595,
           1.87870123, 0.77989849, 1.44449574, 2.38624907, 1.29244221,
           0.78829802, 1.41881675, 1.74509957, 1.11005633, 4.01926467,
           2.05504522, 1.71159601, 3.2461924 , 2.83470994, 2.19983398,
           1.59497106, 1.70886701, 1.90077656, 1.45410688, 1.66886891,
           1.75110881, 3.05357402, 3.6834588 , 1.08916238, 1.84916719,
           1.80851509, 2.15596533, 1.97719052, 1.92875913, 3.38678466,
           1.32564179, 1.81436977, 1.63403928, 1.18568231, 1.7315052 ,
           0.8090071 , 2.47389329, 1.19407051, 2.23409297, 2.30256486,
           5.37829418, 1.38890835, 2.00637569, 1.99204335, 1.85314134,
           2.86190699, 1.91334581, 1.24205971, 3.20558421, 4.26259962,
           2.01732404, 3.95679516, 1.39072672, 1.65839735, 2.30550471,
           2.3517598 , 2.13887703, 1.25020289, 1.88503591, 1.16169133,
           2.00430928, 3.26703677, 1.3242707 , 2.01948611, 2.43700375,
           1.69810225, 1.69708568, 2.18234363, 2.12271321, 0.94102412]), sims=array([[2.52197294, 2.65764086, 2.40986397, ..., 2.78250935, 2.00710776,
            2.45581687],
           [1.73158936, 1.64068277, 1.30405511, ..., 1.5192879 , 1.7531569 ,
            1.96604158],
           [1.9742429 , 1.39726812, 1.32734606, ..., 1.22639871, 1.46987265,
            1.40392566],
           ...,
           [2.0119616 , 2.40848185, 2.09426227, ..., 2.62052065, 2.51099616,
            2.36614074],
           [1.35463496, 2.3932347 , 2.05260157, ..., 2.40050845, 1.99334353,
            1.87698499],
           [0.48279083, 0.94856269, 0.97790733, ..., 1.0927788 , 1.23311912,
            0.40108719]]), lower=array([ 1.93689669e+00,  6.14177968e-01,  1.04195634e+00,  3.75503359e-01,
            2.02144872e+00,  1.82276306e+00,  2.64944272e+00,  1.34667642e+00,
            1.29086670e+00,  3.05297704e+00,  1.17765182e+00,  4.30461766e-01,
            1.37895403e+00,  2.98779763e-01,  4.40822194e-01,  1.55654853e+00,
            9.38628380e-01,  1.21829098e+00,  9.84558834e-01,  6.49532691e-01,
            9.01772850e-01,  1.19276603e+00,  2.17680551e+00,  1.09038146e+00,
            1.16893453e+00,  9.31502331e-01,  4.18264443e+00,  1.49607918e+00,
            1.57691294e+00,  3.28849227e-01,  2.39003022e+00,  1.07840109e+00,
            8.10677053e-01,  5.57945108e-01,  5.89483726e-01,  1.40646500e+00,
            2.38769699e+00,  3.33592642e+00,  4.92576112e-01,  1.64897773e+00,
            8.70602613e-01,  3.28203503e-01,  5.98006128e-01,  1.85519350e+00,
            5.28848929e+00,  8.98852307e-01,  1.62711360e+00,  1.55280346e+00,
            1.49136612e+00,  8.73686102e-01,  1.36286310e+00, -1.81572146e-01,
            7.77535746e-01,  1.55053396e+00,  7.13006914e-01,  4.64261897e-03,
            8.01311589e-01,  7.48462573e-01,  4.37894150e-01,  3.20953201e+00,
            1.01569966e+00,  1.04846962e+00,  2.60967933e+00,  1.88654294e+00,
            1.14477050e+00,  1.14136267e+00,  1.18280741e+00,  1.25494957e+00,
            8.98454204e-01,  2.10925704e-01,  1.05478650e+00,  2.35858707e+00,
            2.94164760e+00,  1.67571043e-01,  1.33878742e+00,  1.25839350e+00,
            1.53951656e+00,  9.05624320e-01,  1.17061173e+00,  2.54655499e+00,
            6.47466317e-01,  1.20289748e+00,  1.13571370e+00,  4.64266018e-01,
            1.14215867e+00,  1.49252246e-01,  1.82103747e+00, -2.49955629e-01,
            1.61755751e+00,  1.65686043e+00,  4.78967402e+00,  4.81742186e-01,
            1.44596520e+00,  1.32448131e+00,  8.89972080e-01,  2.34075771e+00,
            1.24303256e+00,  7.03972518e-01,  2.60285187e+00,  3.53153738e+00,
            1.35196678e+00,  3.50221519e+00,  4.47336205e-01,  9.67171390e-01,
            1.76976985e+00,  1.59672116e+00,  1.28269514e+00,  6.40395915e-01,
            1.29953005e+00,  5.27235568e-01,  1.46471898e+00,  1.66067091e+00,
            4.41349172e-01,  1.39726429e+00,  1.80795176e+00,  1.07103425e+00,
            5.76820176e-01,  1.21573235e+00,  1.34041046e+00,  1.73301081e-01]), upper=array([2.99397933, 1.96132754, 1.9741853 , 1.56915801, 3.40435554,
           2.91464536, 4.17157769, 2.35189282, 2.54248989, 4.34702931,
           2.28681813, 1.58798973, 2.46838793, 1.4842451 , 1.52209298,
           2.75057218, 2.25719594, 2.41607558, 2.30447366, 1.71324602,
           2.33541568, 2.24194145, 3.37712022, 2.16161502, 2.16743013,
           2.32044303, 5.7525094 , 2.95770978, 3.05001003, 1.96251894,
           3.65612204, 2.41472665, 2.15530985, 1.58579045, 1.84162227,
           2.53564354, 3.82321955, 4.44042528, 1.60388757, 2.96240096,
           2.12835682, 1.5469312 , 1.86880196, 3.16319094, 6.30956862,
           1.96174807, 2.99380439, 2.88775395, 2.60191973, 1.93653757,
           2.19505169, 1.30622414, 1.85682241, 2.95891228, 1.79928438,
           1.29625491, 2.09220303, 2.36290631, 1.53658861, 4.7506125 ,
           2.67337766, 2.17030365, 3.75023808, 3.35541687, 2.77520135,
           2.08450402, 2.16371497, 2.38837026, 1.87832477, 2.16419849,
           2.36599111, 3.50414737, 4.29523451, 1.53640485, 2.2309006 ,
           2.33145818, 2.66451633, 2.49759326, 2.43879385, 3.90232458,
           1.84661217, 2.27974655, 2.06887939, 1.71836616, 2.15279379,
           1.31422259, 2.90745441, 1.76813   , 2.71020272, 2.81272411,
           5.92280218, 1.86562672, 2.5298048 , 2.51531422, 2.39877546,
           3.36143692, 2.46371693, 1.76912589, 3.67435993, 4.72940884,
           2.48373427, 4.37798834, 2.15801521, 2.11703737, 3.27506479,
           2.88194859, 2.6982464 , 1.80432322, 2.45259713, 1.66864527,
           2.49172073, 3.85706896, 1.95383092, 2.49247639, 2.9572688 ,
           2.24464142, 2.27349167, 2.61578328, 2.64429111, 1.49723436]))coverage rate: 0.8


```python
plot_func(x = range(len(y_test)),
          y = y_test,
          y_u=preds.upper,
          y_l=preds.lower,
          pred=preds.preds,
          method_name="before update",
          title="")
```


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image1.png){:class="img-responsive"}      
    



```python
# update
fit_obj.update(X_test.iloc[0,:], y_test[0])
```

    Regressor(method='rvfl', nb_hidden=3, pi_method='kdejackknifeplus')coverage rate: 0.8067226890756303


```python
plot_func(x = range(len(y_test[:-1])),
          y = y_test[:-1],
          y_u=preds.upper,
          y_l=preds.lower,
          pred=preds.preds,
          method_name="after update",
          title="")
```


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image2.png){:class="img-responsive"}      
    


# 2 - R code version


```R
X <- as.matrix(mtcars[,-1])
y <- mtcars$mpg

set.seed(123)
(index_train <- base::sample.int(n = nrow(X),
                                 size = floor(0.7*nrow(X)),
                                 replace = FALSE))
```

     [1] 31 15 19 14  3 10 18 22 11  5 20 29 23 30  9 28  8 27  7 32 26 17[1] 22 10[1] 10 10



```R
obj <- learningmachine::Regressor$new(method = "rvfl",
nb_hidden = 50L, pi_method = "splitconformal")

obj$get_type()
```

    [1] "regression"[1] "Regressor"


```R
t0 <- proc.time()[3]
obj$fit(X_train, y_train, reg_lambda = 0.01)
cat("Elapsed: ", proc.time()[3] - t0, "s n")
```

    Elapsed:  0.01 s n


```R
print(obj$predict(X_test))
```

    $preds
              Mazda RX4       Mazda RX4 Wag      Hornet 4 Drive             Valiant 
              21.350888           19.789387           13.106761            9.695310 
             Merc 450SE          Merc 450SL Lincoln Continental       Toyota Corona 
              11.131161           12.568682            2.044672           19.289805 
             Camaro Z28    Pontiac Firebird 
              14.847878           12.282272 
    
    $lower
              Mazda RX4       Mazda RX4 Wag      Hornet 4 Drive             Valiant 
             12.3508879          10.7893873           4.1067608           0.6953102 
             Merc 450SE          Merc 450SL Lincoln Continental       Toyota Corona 
              2.1311611           3.5686817          -6.9553279          10.2898053 
             Camaro Z28    Pontiac Firebird 
              5.8478777           3.2822719 
    
    $upper
              Mazda RX4       Mazda RX4 Wag      Hornet 4 Drive             Valiant 
               30.35089            28.78939            22.10676            18.69531 
             Merc 450SE          Merc 450SL Lincoln Continental       Toyota Corona 
               20.13116            21.56868            11.04467            28.28981 
             Camaro Z28    Pontiac Firebird 
               23.84788            21.28227 


```R
obj$summary(X_test, y=y_test, show_progress=FALSE)
```

    $R_squared
    [1] -1.505856
    
    $R_squared_adj
    [1] 23.55271
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     -1.548   1.461   5.000   4.349   7.949   8.405 
    
    $Coverage_rate
    [1] 100
    
    $ttests
            estimate       lower        upper      p-value signif
    cyl   137.649985   39.777048  235.5229227 1.115728e-02      *
    disp   -2.406399   -4.650678   -0.1621204 3.825959e-02      *
    hp     -0.527573   -1.402043    0.3468975 2.054686e-01       
    drat  707.372951  246.095138 1168.6507638 7.059500e-03     **
    wt   -500.429007 -565.047979 -435.8100352 2.910469e-08    ***
    qsec  -89.930939 -124.899691  -54.9621860 2.537870e-04    ***
    vs    234.198406 -127.886990  596.2838006 1.774484e-01       
    am   -235.789718 -512.422513   40.8430776 8.592503e-02      .
    gear   52.646721   -6.640614  111.9340567 7.547657e-02      .
    carb  -17.100561  -87.819649   53.6185270 5.976705e-01       
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             10     
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean     sd      p0      p25      p50       p75     p100
     1 cyl            138.    137.     -8.40   75.8     91.1     98.6     394.   
     2 disp            -2.41    3.14   -8.46   -1.32    -1.08    -0.775    -0.300
     3 hp              -0.528   1.22   -3.40   -0.695   -0.188    0.0137    0.893
     4 drat           707.    645.     55.7   388.     482.     563.     1939.   
     5 wt            -500.     90.3  -698.   -538.    -500.    -458.     -377.   
     6 qsec           -89.9    48.9  -145.   -128.    -102.     -64.0       2.67 
     7 vs             234.    506.   -121.    -13.2     36.8     53.2    1269.   
     8 am            -236.    387.   -653.   -450.    -397.    -168.      519.   
     9 gear            52.6    82.9  -107.     -4.69    66.2    112.      170.   
    10 carb           -17.1    98.9  -117.    -64.6    -60.6    -17.5     171.   
       hist 
     1 ▂▇▁▁▂
     2 ▂▁▁▁▇
     3 ▁▁▁▇▂
     4 ▅▇▁▁▃
     5 ▂▁▆▇▃
     6 ▇▆▁▂▃
     7 ▇▁▁▁▂
     8 ▆▇▂▁▃
     9 ▂▅▅▅▇
    10 ▇▂▁▁▂


```R
t0 <- proc.time()[3]
obj$fit(X_train, y_train)
cat("Elapsed: ", proc.time()[3] - t0, "s n")
```

    Elapsed:  0.128 s n


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image3.png){:class="img-responsive"}      
    


    [1] 1

**update RVFL model**


```R
previous_coefs <- drop(obj$model$coef)
```


```R
newx <- X_test[1, ]

newy <- y_test[1]

new_X_test <- X_test[-1, ]

new_y_test <- y_test[-1]

t0 <- proc.time()[3]
obj$update(newx, newy)
cat("Elapsed: ", proc.time()[3] - t0, "s n")
```

    Elapsed:  0.242 s n


```R
summary(previous_coefs)
```

        Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    -0.68212 -0.26567 -0.05157  0.00700  0.21046  2.19222      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    -0.030666 -0.002610  0.004189  0.002917  0.011386  0.025243 


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image4.png){:class="img-responsive"}      
    



```R
obj$summary(new_X_test, y=new_y_test, show_progress=FALSE)
```

    $R_squared
    [1] -1.809339
    
    $R_squared_adj
    [1] 12.23735
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     -1.168   2.513   5.541   5.058   8.185   8.703 
    
    $Coverage_rate
    [1] 100
    
    $ttests
             estimate       lower        upper      p-value signif
    cyl   111.6701473   17.076928  206.2633669 2.615518e-02      *
    disp   -1.7983224   -3.876380    0.2797349 8.106884e-02      .
    hp     -0.4167545   -1.501658    0.6681495 4.015523e-01       
    drat  569.9102780  148.862037  990.9585186 1.420088e-02      *
    wt   -504.1496696 -583.757006 -424.5423330 4.741273e-07    ***
    qsec -107.9102921 -138.571336  -77.2492482 3.936777e-05    ***
    vs    145.0280002 -173.164419  463.2204193 3.239468e-01       
    am   -319.6910568 -566.618653  -72.7634604 1.745263e-02      *
    gear   57.7630332  -18.934712  134.4607782 1.206459e-01       
    carb  -42.9572292 -108.690903   22.7764447 1.702409e-01       
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             9      
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean     sd      p0      p25       p50       p75     p100
     1 cyl            112.    123.    -13.5    64.5     93.6      93.9     426.   
     2 disp            -1.80    2.70   -8.94   -1.41    -0.805    -0.689    -0.361
     3 hp              -0.417   1.41   -3.54   -0.679   -0.0942   -0.0556    1.19 
     4 drat           570.    548.     36.8   371.     439.      501.     1972.   
     5 wt            -504.    104.   -742.   -523.    -497.     -461.     -382.   
     6 qsec          -108.     39.9  -152.   -143.    -115.      -93.0     -35.9  
     7 vs             145.    414.   -116.    -23.9     51.1      81.2    1231.   
     8 am            -320.    321.   -575.   -479.    -395.     -368.      465.   
     9 gear            57.8    99.8  -113.      1.22    35.2     130.      196.   
    10 carb           -43.0    85.5  -129.    -79.6    -77.9     -22.5     165.   
       hist 
     1 ▅▇▁▁▂
     2 ▁▁▁▁▇
     3 ▂▁▂▇▃
     4 ▅▇▁▁▂
     5 ▂▁▂▇▃
     6 ▇▅▅▂▂
     7 ▇▁▁▁▁
     8 ▇▁▁▁▁
     9 ▃▇▇▇▇
    10 ▇▅▁▁▂


```R
res <- obj$predict(X = new_X_test)

new_y_train <- c(y_train, newy)

plot(c(new_y_train, res$preds), type='l',
main="",
ylab="",
ylim = c(min(c(res$upper, res$lower, y)),
max(c(res$upper, res$lower, y))))
lines(c(new_y_train, res$upper), col="gray60")
lines(c(new_y_train, res$lower), col="gray60")
lines(c(new_y_train, res$preds), col = "red")
lines(c(new_y_train, new_y_test), col = "blue")
abline(v = length(y_train), lty=2, col="black")
```


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image5.png){:class="img-responsive"}      
    


    [1] 1

**update RVFL model (Pt.2)**


```R
newx <- X_test[2, ]

newy <- y_test[2]

new_X_test <- X_test[-c(1, 2), ]

new_y_test <- y_test[-c(1, 2)]

t0 <- proc.time()[3]
obj$update(newx, newy)
cat("Elapsed: ", proc.time()[3] - t0, "s n")
```

    Elapsed:  0.077 s n


```R
obj$summary(new_X_test, y=new_y_test, show_progress=FALSE)
```

    $R_squared
    [1] -3.356623
    
    $R_squared_adj
    [1] 11.16545
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     -1.950   5.030   6.374   6.369   8.774  11.528 
    
    $Coverage_rate
    [1] 75
    
    $ttests
             estimate       lower        upper      p-value signif
    cyl    40.8981137    6.878148   74.9180798 2.494779e-02      *
    disp   -0.7335494   -1.206939   -0.2601595 8.026181e-03     **
    hp     -0.8233606   -2.198927    0.5522055 1.998737e-01       
    drat  549.7206897  416.053783  683.3875968 2.570765e-05    ***
    wt   -469.9351032 -535.877454 -403.9927527 6.344763e-07    ***
    qsec -116.6183871 -156.767393  -76.4693814 2.380078e-04    ***
    vs   -194.4213942 -288.046178 -100.7966103 1.732503e-03     **
    am   -395.7216847 -562.762331 -228.6810387 8.143911e-04    ***
    gear   53.0732573  -59.833653  165.9801679 3.030574e-01       
    carb  -25.9448064  -63.759959   11.8703467 1.487567e-01       
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             8      
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean      sd      p0     p25      p50      p75     p100
     1 cyl             40.9    40.7    -40.5    23.9    56.3     69.9     77.8  
     2 disp            -0.734   0.566   -1.64   -1.03   -0.571   -0.372   -0.139
     3 hp              -0.823   1.65    -3.99   -1.18   -0.974   -0.196    1.25 
     4 drat           550.    160.     170.    549.    606.     642.     643.   
     5 wt            -470.     78.9   -543.   -537.   -489.    -437.    -336.   
     6 qsec          -117.     48.0   -179.   -143.   -131.     -99.1    -29.9  
     7 vs            -194.    112.    -377.   -283.   -162.    -120.     -46.3  
     8 am            -396.    200.    -719.   -481.   -357.    -319.     -67.7  
     9 gear            53.1   135.    -143.    -23.9    16.5    172.     231.   
    10 carb           -25.9    45.2   -101.    -48.8   -23.8     -9.36    45.7  
       hist 
     1 ▂▂▂▁▇
     2 ▅▁▂▇▅
     3 ▂▁▇▂▃
     4 ▁▁▁▁▇
     5 ▇▅▂▁▅
     6 ▂▇▂▂▂
     7 ▂▅▂▇▂
     8 ▃▁▇▂▂
     9 ▂▅▅▁▇
    10 ▂▅▇▁▅


```R
res <- obj$predict(X = new_X_test)

new_y_train <- c(y_train, y_test[c(1, 2)])

plot(c(new_y_train, res$preds), type='l',
main="",
ylab="",
ylim = c(min(c(res$upper, res$lower, y)),
max(c(res$upper, res$lower, y))))
lines(c(new_y_train, res$upper), col="gray60")
lines(c(new_y_train, res$lower), col="gray60")
lines(c(new_y_train, res$preds), col = "red")
lines(c(new_y_train, new_y_test), col = "blue")
abline(v = length(y_train), lty=2, col="black")
```


    
![xxx]({{base}}/images/2024-08-19/2024-08-19-image6.png){:class="img-responsive"}      
    


    [1] 0.75
