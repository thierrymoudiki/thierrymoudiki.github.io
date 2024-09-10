---
layout: post
title: "Adaptive (online/streaming) learning with uncertainty quantification using Polyak averaging in learningmachine"
description: "Adaptive (online/streaming) learning with uncertainty quantification and explanations using learningmachine in Python and R"
date: 2024-09-10
categories: [Python, R]
comments: true
---

The model presented here is a frequentist -- [conformalized](https://conformalpredictionintro.github.io/) -- version of the Bayesian one presented in [#152](https://thierrymoudiki.github.io/blog/2024/08/12/r/bqrvfl). It is implemented in `learningmachine`, both in [Python](https://github.com/Techtonique/learningmachine_python) and [R](https://github.com/Techtonique/learningmachine), and is updated as new observations arrive, using [Polyak averaging](https://en.wikipedia.org/wiki/Stochastic_approximation). Model explanations are given as sensitivity analyses. 
  

## 1 - R version


```python
%load_ext rpy2.ipython
```


```r
%%R

utils::install.packages("bayesianrvfl", repos = c("https://techtonique.r-universe.dev", "https://cloud.r-project.org"))
utils::install.packages("learningmachine", repos = c("https://techtonique.r-universe.dev", "https://cloud.r-project.org"))
```


```r
%%R

library(learningmachine)

X <- as.matrix(mtcars[,-1])
y <- mtcars$mpg

set.seed(123)
(index_train <- base::sample.int(n = nrow(X),
                                 size = floor(0.6*nrow(X)),
                                 replace = FALSE))
##  [1] 31 15 19 14  3 10 18 22 11  5 20 29 23 30  9 28  8 27  7
X_train <- X[index_train, ]
y_train <- y[index_train]
X_test <- X[-index_train, ]
y_test <- y[-index_train]
dim(X_train)
## [1] 19 10
dim(X_test)
```

    [1] 13 10



```r
%%R

obj <- learningmachine::Regressor$new(method = "bayesianrvfl",
                                      nb_hidden = 5L)
obj$get_type()
```

    [1] "regression"



```r
%%R

obj_GCV <- bayesianrvfl::fit_rvfl(x = X_train, y = y_train)
(best_lambda <- obj_GCV$lambda[which.min(obj_GCV$GCV)])
```

    [1] 12.9155



```r
%%R

t0 <- proc.time()[3]
obj$fit(X_train, y_train, reg_lambda = best_lambda)
cat("Elapsed: ", proc.time()[3] - t0, "s \n")
```

    Elapsed:  0.01 s 



```r
%%R

previous_coefs <- drop(obj$model$coef)

newx <- X_test[1, ]
newy <- y_test[1]

new_X_test <- X_test[-1, ]
new_y_test <- y_test[-1]

t0 <- proc.time()[3]
obj$update(newx, newy, method = "polyak", alpha = 0.6)
cat("Elapsed: ", proc.time()[3] - t0, "s \n")

print(summary(previous_coefs))

print(summary(drop(obj$model$coef) - previous_coefs))


plot(drop(obj$model$coef) - previous_coefs, type='l')
abline(h = 0, lty=2, col="red")

```

    Elapsed:  0.003 s 
        Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    -0.96778 -0.51401 -0.16335 -0.05234  0.31900  0.98482 
         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    -0.065436 -0.002152  0.027994  0.015974  0.040033  0.058892 



    
![xxx]({{base}}/images/2024-09-10/2024-09-10-image1.png){:class="img-responsive"}    
    



```r
%%R

print(obj$summary(new_X_test, y=new_y_test, show_progress=FALSE))
```

    $R_squared
    [1] 0.6692541
    
    $R_squared_adj
    [1] -2.638205
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    -4.5014 -2.2111 -0.5532 -0.3928  1.3495  3.9206 
    
    $Coverage_rate
    [1] 100
    
    $citests
             estimate        lower        upper      p-value signif
    cyl   -41.4815528  -43.6039915  -39.3591140 1.306085e-13    ***
    disp   -0.5937584   -0.7014857   -0.4860311 1.040246e-07    ***
    hp     -1.0226867   -1.2175471   -0.8278263 1.719172e-07    ***
    drat   84.5859637   73.2987057   95.8732217 4.178658e-09    ***
    wt   -169.1047879 -189.5595154 -148.6500603 1.469605e-09    ***
    qsec   22.3026258   15.1341951   29.4710566 2.772362e-05    ***
    vs    113.3209911   88.3101728  138.3318093 7.599984e-07    ***
    am    175.1639102  139.5755741  210.7522464 3.304560e-07    ***
    gear   44.3270639   36.1456398   52.5084881 1.240722e-07    ***
    carb  -59.6511203  -69.8576126  -49.4446280 5.677270e-08    ***
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             12     
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean     sd       p0      p25      p50      p75     p100
     1 cyl            -41.5    3.34   -43.4    -43.4    -43.3    -41.7    -34.5  
     2 disp            -0.594  0.170   -0.916   -0.635   -0.505   -0.505   -0.356
     3 hp              -1.02   0.307   -1.44    -1.40    -0.877   -0.768   -0.768
     4 drat            84.6   17.8     59.5     76.7     89.5     89.5    128.   
     5 wt            -169.    32.2   -204.    -199.    -166.    -138.    -138.   
     6 qsec            22.3   11.3     13.3     13.3     17.4     29.2     40.1  
     7 vs             113.    39.4     59.6     94.4     94.4    117.     191.   
     8 am             175.    56.0    124.     124.     153.     226.     245.   
     9 gear            44.3   12.9     26.3     38.7     47.9     47.9     76.0  
    10 carb           -59.7   16.1    -77.3    -74.6    -58.2    -44.4    -44.4  
       hist 
     1 ▇▁▁▁▂
     2 ▂▁▃▇▁
     3 ▅▁▁▂▇
     4 ▂▃▇▁▁
     5 ▇▁▁▁▇
     6 ▇▂▁▁▃
     7 ▁▇▃▁▂
     8 ▇▁▁▂▃
     9 ▂▃▇▁▁
    10 ▇▁▁▁▇
    



```r
%%R

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


    
![xxx]({{base}}/images/2024-09-10/2024-09-10-image2.png){:class="img-responsive"}    
    



```r
%%R

newx <- X_test[2, ]
newy <- y_test[2]

new_X_test <- X_test[-c(1, 2), ]
new_y_test <- y_test[-c(1, 2)]

t0 <- proc.time()[3]
obj$update(newx, newy, method = "polyak", alpha = 0.9)
cat("Elapsed: ", proc.time()[3] - t0, "s \n")

print(obj$summary(new_X_test, y=new_y_test, show_progress=FALSE))


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

    Elapsed:  0.003 s 
    $R_squared
    [1] 0.6426871
    
    $R_squared_adj
    [1] -Inf
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    -4.5686 -2.4084 -1.0397 -0.3897  1.5507  4.0215 
    
    $Coverage_rate
    [1] 100
    
    $citests
             estimate        lower        upper      p-value signif
    cyl   -42.1261096  -44.5327541  -39.7194651 2.932516e-12    ***
    disp   -0.6256505   -0.7347381   -0.5165629 1.613495e-07    ***
    hp     -1.0139634   -1.2198651   -0.8080617 6.747693e-07    ***
    drat   82.8645391   74.8033348   90.9257434 5.680663e-10    ***
    wt   -170.7891742 -193.1932631 -148.3850853 1.053193e-08    ***
    qsec   22.2365552   13.9564091   30.5167012 1.350094e-04    ***
    vs    119.1784891   94.0163626  144.3406157 9.681321e-07    ***
    am    174.2138307  134.1390652  214.2885963 2.127371e-06    ***
    gear   42.7943293   36.9622907   48.6263678 1.523695e-08    ***
    carb  -59.4034661  -70.5135723  -48.2933599 3.127231e-07    ***
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             11     
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean     sd       p0      p25      p50      p75     p100
     1 cyl            -42.1    3.58   -44.1    -44.1    -44.1    -43.0    -34.9  
     2 disp            -0.626  0.162   -0.933   -0.643   -0.514   -0.514   -0.514
     3 hp              -1.01   0.306   -1.47    -1.24    -0.787   -0.787   -0.787
     4 drat            82.9   12.0     61.2     79.6     91.7     91.7     91.7  
     5 wt            -171.    33.3   -210.    -204.    -142.    -142.    -142.   
     6 qsec            22.2   12.3     13.2     13.2     13.2     30.7     41.2  
     7 vs             119.    37.5     96.0     96.0     96.0    117.     193.   
     8 am             174.    59.7    123.     123.     123.     233.     247.   
     9 gear            42.8    8.68    27.1     40.4     49.2     49.2     49.2  
    10 carb           -59.4   16.5    -78.8    -76.0    -45.1    -45.1    -45.1  
       hist 
     1 ▇▁▁▁▂
     2 ▂▁▁▃▇
     3 ▃▁▁▂▇
     4 ▂▁▁▃▇
     5 ▇▁▁▁▇
     6 ▇▂▁▁▃
     7 ▇▃▁▁▂
     8 ▇▁▁▂▃
     9 ▂▁▁▃▇
    10 ▇▁▁▁▇
    



    
![xxx]({{base}}/images/2024-09-10/2024-09-10-image3.png){:class="img-responsive"}    
    


## 2 - Python version


```python
!pip install git+https://github.com/Techtonique/learningmachine_python.git --verbose
```


```python

```


```python
import pandas as pd
import numpy as np
import warnings
import learningmachine as lm


# Load the mtcars dataset
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/mtcars.csv")
X = data.drop("mpg", axis=1).values
X = pd.DataFrame(X).iloc[:,1:]
X = X.astype(np.float16)
X.columns = ["cyl","disp","hp","drat","wt","qsec","vs","am","gear","carb"]
y = data["mpg"].values
```


```python
display(X.describe())
display(X.head())
display(X.dtypes)
```



  <div id="df-82d84f28-c904-4138-bb18-6321687bf588" class="colab-df-container">
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
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.0000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.187500</td>
      <td>230.750000</td>
      <td>146.7500</td>
      <td>3.595703</td>
      <td>3.216797</td>
      <td>17.843750</td>
      <td>0.437500</td>
      <td>0.406250</td>
      <td>3.687500</td>
      <td>2.812500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.786133</td>
      <td>123.937500</td>
      <td>68.5625</td>
      <td>0.534668</td>
      <td>0.978516</td>
      <td>1.788086</td>
      <td>0.503906</td>
      <td>0.499023</td>
      <td>0.737793</td>
      <td>1.615234</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>71.125000</td>
      <td>52.0000</td>
      <td>2.759766</td>
      <td>1.512695</td>
      <td>14.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>120.828125</td>
      <td>96.5000</td>
      <td>3.080078</td>
      <td>2.580566</td>
      <td>16.898438</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>196.312500</td>
      <td>123.0000</td>
      <td>3.694336</td>
      <td>3.325195</td>
      <td>17.703125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>326.000000</td>
      <td>180.0000</td>
      <td>3.919922</td>
      <td>3.610352</td>
      <td>18.906250</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>472.000000</td>
      <td>335.0000</td>
      <td>4.929688</td>
      <td>5.425781</td>
      <td>22.906250</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-82d84f28-c904-4138-bb18-6321687bf588')"
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
        document.querySelector('#df-82d84f28-c904-4138-bb18-6321687bf588 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-82d84f28-c904-4138-bb18-6321687bf588');
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


<div id="df-8f2853a0-5959-44e7-a495-054e1331dd3f">
  <button class="colab-df-quickchart" onclick="quickchart('df-8f2853a0-5959-44e7-a495-054e1331dd3f')"
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
        document.querySelector('#df-8f2853a0-5959-44e7-a495-054e1331dd3f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





  <div id="df-dca2125a-f091-4236-bd0e-260664c91541" class="colab-df-container">
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
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>160.0</td>
      <td>110.0</td>
      <td>3.900391</td>
      <td>2.619141</td>
      <td>16.453125</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>160.0</td>
      <td>110.0</td>
      <td>3.900391</td>
      <td>2.875000</td>
      <td>17.015625</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>108.0</td>
      <td>93.0</td>
      <td>3.849609</td>
      <td>2.320312</td>
      <td>18.609375</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>258.0</td>
      <td>110.0</td>
      <td>3.080078</td>
      <td>3.214844</td>
      <td>19.437500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>360.0</td>
      <td>175.0</td>
      <td>3.150391</td>
      <td>3.439453</td>
      <td>17.015625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-dca2125a-f091-4236-bd0e-260664c91541')"
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
        document.querySelector('#df-dca2125a-f091-4236-bd0e-260664c91541 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-dca2125a-f091-4236-bd0e-260664c91541');
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


<div id="df-b49a29f3-e4ce-43cd-b55c-e2e0038e845c">
  <button class="colab-df-quickchart" onclick="quickchart('df-b49a29f3-e4ce-43cd-b55c-e2e0038e845c')"
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
        document.querySelector('#df-b49a29f3-e4ce-43cd-b55c-e2e0038e845c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cyl</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>disp</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>hp</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>drat</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>wt</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>qsec</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>vs</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>am</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>gear</th>
      <td>float16</td>
    </tr>
    <tr>
      <th>carb</th>
      <td>float16</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



```python
y.dtype
```




    dtype('float64')




```python
X
```





  <div id="df-df5b51e6-22bd-4e6a-aa58-efde294d7a25" class="colab-df-container">
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
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>160.0000</td>
      <td>110.0</td>
      <td>3.900391</td>
      <td>2.619141</td>
      <td>16.453125</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>160.0000</td>
      <td>110.0</td>
      <td>3.900391</td>
      <td>2.875000</td>
      <td>17.015625</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>108.0000</td>
      <td>93.0</td>
      <td>3.849609</td>
      <td>2.320312</td>
      <td>18.609375</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>258.0000</td>
      <td>110.0</td>
      <td>3.080078</td>
      <td>3.214844</td>
      <td>19.437500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>360.0000</td>
      <td>175.0</td>
      <td>3.150391</td>
      <td>3.439453</td>
      <td>17.015625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>225.0000</td>
      <td>105.0</td>
      <td>2.759766</td>
      <td>3.460938</td>
      <td>20.218750</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>360.0000</td>
      <td>245.0</td>
      <td>3.210938</td>
      <td>3.570312</td>
      <td>15.843750</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>146.7500</td>
      <td>62.0</td>
      <td>3.689453</td>
      <td>3.189453</td>
      <td>20.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.0</td>
      <td>140.7500</td>
      <td>95.0</td>
      <td>3.919922</td>
      <td>3.150391</td>
      <td>22.906250</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.0</td>
      <td>167.6250</td>
      <td>123.0</td>
      <td>3.919922</td>
      <td>3.439453</td>
      <td>18.296875</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.0</td>
      <td>167.6250</td>
      <td>123.0</td>
      <td>3.919922</td>
      <td>3.439453</td>
      <td>18.906250</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.0</td>
      <td>275.7500</td>
      <td>180.0</td>
      <td>3.070312</td>
      <td>4.070312</td>
      <td>17.406250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8.0</td>
      <td>275.7500</td>
      <td>180.0</td>
      <td>3.070312</td>
      <td>3.730469</td>
      <td>17.593750</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8.0</td>
      <td>275.7500</td>
      <td>180.0</td>
      <td>3.070312</td>
      <td>3.779297</td>
      <td>18.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8.0</td>
      <td>472.0000</td>
      <td>205.0</td>
      <td>2.929688</td>
      <td>5.250000</td>
      <td>17.984375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.0</td>
      <td>460.0000</td>
      <td>215.0</td>
      <td>3.000000</td>
      <td>5.425781</td>
      <td>17.812500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.0</td>
      <td>440.0000</td>
      <td>230.0</td>
      <td>3.230469</td>
      <td>5.343750</td>
      <td>17.421875</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4.0</td>
      <td>78.6875</td>
      <td>66.0</td>
      <td>4.078125</td>
      <td>2.199219</td>
      <td>19.468750</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4.0</td>
      <td>75.6875</td>
      <td>52.0</td>
      <td>4.929688</td>
      <td>1.615234</td>
      <td>18.515625</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4.0</td>
      <td>71.1250</td>
      <td>65.0</td>
      <td>4.218750</td>
      <td>1.834961</td>
      <td>19.906250</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.0</td>
      <td>120.1250</td>
      <td>97.0</td>
      <td>3.699219</td>
      <td>2.464844</td>
      <td>20.015625</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8.0</td>
      <td>318.0000</td>
      <td>150.0</td>
      <td>2.759766</td>
      <td>3.519531</td>
      <td>16.875000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>8.0</td>
      <td>304.0000</td>
      <td>150.0</td>
      <td>3.150391</td>
      <td>3.435547</td>
      <td>17.296875</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8.0</td>
      <td>350.0000</td>
      <td>245.0</td>
      <td>3.730469</td>
      <td>3.839844</td>
      <td>15.406250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8.0</td>
      <td>400.0000</td>
      <td>175.0</td>
      <td>3.080078</td>
      <td>3.845703</td>
      <td>17.046875</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4.0</td>
      <td>79.0000</td>
      <td>66.0</td>
      <td>4.078125</td>
      <td>1.934570</td>
      <td>18.906250</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4.0</td>
      <td>120.3125</td>
      <td>91.0</td>
      <td>4.429688</td>
      <td>2.140625</td>
      <td>16.703125</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4.0</td>
      <td>95.1250</td>
      <td>113.0</td>
      <td>3.769531</td>
      <td>1.512695</td>
      <td>16.906250</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8.0</td>
      <td>351.0000</td>
      <td>264.0</td>
      <td>4.218750</td>
      <td>3.169922</td>
      <td>14.500000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6.0</td>
      <td>145.0000</td>
      <td>175.0</td>
      <td>3.619141</td>
      <td>2.769531</td>
      <td>15.500000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>8.0</td>
      <td>301.0000</td>
      <td>335.0</td>
      <td>3.539062</td>
      <td>3.570312</td>
      <td>14.601562</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4.0</td>
      <td>121.0000</td>
      <td>109.0</td>
      <td>4.109375</td>
      <td>2.779297</td>
      <td>18.593750</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df5b51e6-22bd-4e6a-aa58-efde294d7a25')"
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
        document.querySelector('#df-df5b51e6-22bd-4e6a-aa58-efde294d7a25 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df5b51e6-22bd-4e6a-aa58-efde294d7a25');
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


<div id="df-a8aa6779-0d83-4580-8d61-47c8e3c9aea5">
  <button class="colab-df-quickchart" onclick="quickchart('df-a8aa6779-0d83-4580-8d61-47c8e3c9aea5')"
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
        document.querySelector('#df-a8aa6779-0d83-4580-8d61-47c8e3c9aea5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_f479b25a-7be0-4535-a805-3fa5577c1283">
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
    <button class="colab-df-generate" onclick="generateWithVariable('X')"
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
        document.querySelector('#id_f479b25a-7be0-4535-a805-3fa5577c1283 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('X');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create a Bayesian RVFL regressor object
obj = lm.Regressor(method = "bayesianrvfl", nb_hidden = 5)

# Fit the model using the training data
obj.fit(X_train, y_train, reg_lambda=12.9155)

# Print the summary of the model
print(obj.summary(X_test, y=y_test, show_progress=False))
```

    $R_squared
    [1] 0.6416309
    
    $R_squared_adj
    [1] 1.537554
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    -4.0724 -2.0122 -0.1018 -0.1941  1.4361  3.9676 
    
    $Coverage_rate
    [1] 100
    
    $citests
             estimate       lower        upper      p-value signif
    cyl   -24.5943583  -40.407994   -8.7807230 8.909365e-03     **
    disp   -0.2419797   -0.370835   -0.1131245 3.711077e-03     **
    hp     -1.5734483   -1.722903   -1.4239939 2.255640e-07    ***
    drat  142.5646192  124.575179  160.5540599 1.217808e-06    ***
    wt   -144.8871352 -158.911143 -130.8631275 2.523441e-07    ***
    qsec   46.8290859   27.829411   65.8287611 9.388045e-04    ***
    vs     75.0555146   30.645127  119.4659017 6.110043e-03     **
    am    207.5935234  133.205572  281.9814744 4.843095e-04    ***
    gear   73.6892658   60.186232   87.1922995 1.091470e-05    ***
    carb  -71.2974988  -79.480400  -63.1145974 6.944475e-07    ***
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             7      
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean     sd       p0      p25      p50      p75       p100
     1 cyl            -24.6   17.1    -38.5    -38.5    -33.4    -12.4      1.66   
     2 disp            -0.242  0.139   -0.351   -0.351   -0.285   -0.181    0.00546
     3 hp              -1.57   0.162   -1.90    -1.61    -1.48    -1.47    -1.47   
     4 drat           143.    19.5    125.     125.     141.     154.     174.     
     5 wt            -145.    15.2   -167.    -152.    -142.    -142.    -117.     
     6 qsec            46.8   20.5     14.1     35.3     55.7     62.4     62.4    
     7 vs              75.1   48.0     37.2     37.2     58.7     93.9    167.     
     8 am             208.    80.4     64.3    168.     250.     267.     267.     
     9 gear            73.7   14.6     60.6     60.6     72.7     82.1     96.9    
    10 carb           -71.3    8.85   -84.4    -75.2    -69.7    -69.7    -55.2    
       hist 
     1 ▇▁▂▁▃
     2 ▇▂▁▂▂
     3 ▂▁▁▃▇
     4 ▇▂▂▂▂
     5 ▂▅▇▁▂
     6 ▃▁▁▂▇
     7 ▇▁▃▁▂
     8 ▂▂▁▂▇
     9 ▇▂▂▂▂
    10 ▂▅▇▁▂
    
    



```python
# Select the first test sample
newx = X_test.iloc[0,:]
newy = y_test[0]

# Update the model with the new sample
new_X_test = X_test[1:]
new_y_test = y_test[1:]
obj.update(newx, newy, method="polyak", alpha=0.9)

# Print the summary of the model
print(obj.summary(new_X_test, y=new_y_test, show_progress=False))
```

    $R_squared
    [1] 0.6051442
    
    $R_squared_adj
    [1] 1.394856
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    -4.6214 -2.5055 -1.5003 -0.8308  1.2738  3.2794 
    
    $Coverage_rate
    [1] 100
    
    $citests
             estimate        lower       upper      p-value signif
    cyl   -30.0502823  -48.4171958  -11.683369 8.442658e-03     **
    disp   -0.2958477   -0.4386085   -0.153087 3.121989e-03     **
    hp     -1.6053302   -1.6789750   -1.531685 3.424156e-08    ***
    drat  153.7968829  131.5239191  176.069847 1.041460e-05    ***
    wt   -155.1954135 -174.4144275 -135.976399 4.804729e-06    ***
    qsec   49.8967685   26.9993778   72.794159 2.504905e-03     **
    vs     87.4170764   32.4599776  142.374175 9.457226e-03     **
    am    214.5918910  119.8712855  309.312496 2.108825e-03     **
    gear   83.1355825   65.3159018  100.955263 7.110354e-05    ***
    carb  -77.1384645  -88.2087477  -66.068181 9.958425e-06    ***
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             6      
    Number of columns          10     
    _______________________           
    Column type frequency:            
      numeric                  10     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable     mean      sd       p0      p25      p50      p75      p100
     1 cyl            -30.1   17.5     -42.5    -42.5    -39.9    -17.7     -4.35  
     2 disp            -0.296  0.136    -0.377   -0.377   -0.343   -0.308   -0.0269
     3 hp              -1.61   0.0702   -1.70    -1.66    -1.57    -1.57    -1.53  
     4 drat           154.    21.2     137.     137.     144.     169.     185.    
     5 wt            -155.    18.3    -182.    -160.    -154.    -154.    -125.    
     6 qsec            49.9   21.8      18.8     33.3     60.6     66.6     66.6   
     7 vs              87.4   52.4      46.7     46.7     73.7    105.     178.    
     8 am             215.    90.3      65.6    169.     266.     275.     275.    
     9 gear            83.1   17.0      70.0     70.0     75.5     94.1    109.    
    10 carb           -77.1   10.5     -92.5    -79.9    -76.6    -76.6    -59.7   
       hist 
     1 ▇▁▁▁▃
     2 ▇▂▁▁▂
     3 ▅▁▁▇▂
     4 ▇▂▁▁▅
     5 ▂▂▇▁▂
     6 ▅▁▁▂▇
     7 ▇▁▅▁▂
     8 ▂▂▁▁▇
     9 ▇▂▁▂▂
    10 ▂▂▇▁▂
    
    

