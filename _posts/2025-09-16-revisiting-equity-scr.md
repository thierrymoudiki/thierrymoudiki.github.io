---
layout: post
title: "Reimagining Equity Solvency Capital Requirement Approximation (one of my Master's Thesis subjects): From Bilinear Interpolation to Probabilistic Machine Learning"
description: "Revisiting my 2007-2009 Master's Thesis work on SCR Equity approximation using  probabilistic machine learning techniques in R and Python."
date: 2025-09-16
categories: [R, Python]
comments: true
---


# Reimagining Equity Solvency Capital Requirement Approximation (one of my Master's Thesis subjects): From Bilinear Interpolation to Probabilistic Machine Learning

In the world of insurance and financial risk management, calculating the Solvency Capital Requirement (SCR) for equity risk could be a computationally intensive task that can make or break real-time decision making. Traditional approaches rely on expensive Monte Carlo simulations that can take hours to complete, forcing practitioners to **develop approximation schemes**. **Developing an approximation scheme was a project** I tackled back in 2007-2009 for my Master's Thesis in Actuarial Science (see references below).

## What I did back then

- **96 expensive ALIM simulations** were run across four key variables:
  - Minimum guaranteed rate (tmg): 1.75% to 6%
  - Percentage of investments in stocks: 2% to 6.25%  
  - Latent capital gains on equities: 2% to 6.25%
  - Profit sharing provisions (ppe): 3.5 to 10

- **Multi-stage interpolation strategy**: I decomposed the problem into multiple 2D approximation grids, then combined cross-sections to reconstruct the full 4D surface.

- **Validation through error analysis**: Rigorous comparison between simulation results and approximations to ensure the method's reliability.

## A Modern Probabilistic Approach

Today, I revisit this same challenge through the lens of **probabilistic machine learning**, and **obtain functional expressions/approximations** in R and Python. Fascinating how _easy_ it may look  now!  

This probabilistic approach offers several advantages:
- **Built-in uncertainty quantification**: Know not just the prediction, but how confident we should be
- **Automatic feature learning**: Let the model discover optimal representations
- **Fast**

Of course, having a functional probabilistic machine learning model, we can think of many ways to stress test (i.e obtain what-if analyses) these results, based on changes in one (or more) of the explanatory variables

---

**References:**
- Moudiki, T. (2012). *Modélisation du SCR Equity*. Institut des Actuaires. [PDF](https://www.institutdesactuaires.com/docs/mem/ed04ad3998627662cabf67ea465653be.pdf)
- ResearchGate version: [https://www.researchgate.net/publication/395528539_memoire_moudiki_2012](https://www.researchgate.net/publication/395528539_memoire_moudiki_2012)

# R version


```R
(scr_equity <- read.csv("ALIM4D.txt"))
```


<table class="dataframe">
<caption>A data.frame: 96 × 5</caption>
<thead>
	<tr><th scope=col>tmg</th><th scope=col>pct_actions</th><th scope=col>pvl_actions</th><th scope=col>ppe</th><th scope=col>SRC_Equity</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1.75</td><td>2.00</td><td>2.00</td><td>3.50</td><td>56471378</td></tr>
	<tr><td>1.75</td><td>2.00</td><td>2.00</td><td>9.00</td><td>48531931</td></tr>
	<tr><td>1.75</td><td>2.00</td><td>2.00</td><td>9.50</td><td>48558178</td></tr>
	<tr><td>1.75</td><td>2.00</td><td>2.00</td><td>9.75</td><td>48570523</td></tr>
	<tr><td>5.00</td><td>2.00</td><td>2.00</td><td>3.50</td><td>65111083</td></tr>
	<tr><td>5.00</td><td>2.00</td><td>2.00</td><td>9.00</td><td>54433115</td></tr>
	<tr><td>5.00</td><td>2.00</td><td>2.00</td><td>9.50</td><td>54436348</td></tr>
	<tr><td>5.00</td><td>2.00</td><td>2.00</td><td>9.75</td><td>54526734</td></tr>
	<tr><td>5.25</td><td>2.00</td><td>2.00</td><td>3.50</td><td>65244870</td></tr>
	<tr><td>5.25</td><td>2.00</td><td>2.00</td><td>9.00</td><td>54325632</td></tr>
	<tr><td>5.25</td><td>2.00</td><td>2.00</td><td>9.50</td><td>54387565</td></tr>
	<tr><td>5.25</td><td>2.00</td><td>2.00</td><td>9.75</td><td>54418533</td></tr>
	<tr><td>5.50</td><td>2.00</td><td>2.00</td><td>3.50</td><td>65396012</td></tr>
	<tr><td>5.50</td><td>2.00</td><td>2.00</td><td>9.00</td><td>54239282</td></tr>
	<tr><td>5.50</td><td>2.00</td><td>2.00</td><td>9.50</td><td>54302132</td></tr>
	<tr><td>5.50</td><td>2.00</td><td>2.00</td><td>9.75</td><td>54333018</td></tr>
	<tr><td>5.75</td><td>2.00</td><td>2.00</td><td>3.50</td><td>65581289</td></tr>
	<tr><td>5.75</td><td>2.00</td><td>2.00</td><td>9.00</td><td>54168174</td></tr>
	<tr><td>5.75</td><td>2.00</td><td>2.00</td><td>9.50</td><td>54209587</td></tr>
	<tr><td>5.75</td><td>2.00</td><td>2.00</td><td>9.75</td><td>54210481</td></tr>
	<tr><td>6.00</td><td>2.00</td><td>2.00</td><td>3.50</td><td>65785420</td></tr>
	<tr><td>6.00</td><td>2.00</td><td>2.00</td><td>9.00</td><td>54042241</td></tr>
	<tr><td>6.00</td><td>2.00</td><td>2.00</td><td>9.50</td><td>54103639</td></tr>
	<tr><td>6.00</td><td>2.00</td><td>2.00</td><td>9.75</td><td>54134241</td></tr>
	<tr><td>1.75</td><td>2.75</td><td>2.75</td><td>9.00</td><td>48435808</td></tr>
	<tr><td>1.75</td><td>2.75</td><td>2.75</td><td>9.25</td><td>48446558</td></tr>
	<tr><td>1.75</td><td>2.75</td><td>2.75</td><td>9.50</td><td>48459074</td></tr>
	<tr><td>1.75</td><td>2.75</td><td>2.75</td><td>9.75</td><td>48473874</td></tr>
	<tr><td>5.00</td><td>2.75</td><td>2.75</td><td>9.00</td><td>54501129</td></tr>
	<tr><td>5.00</td><td>2.75</td><td>2.75</td><td>9.25</td><td>54531852</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>5.75</td><td>6.00</td><td>6.00</td><td>9.50</td><td>53901737</td></tr>
	<tr><td>5.75</td><td>6.00</td><td>6.00</td><td>9.75</td><td>53968463</td></tr>
	<tr><td>6.00</td><td>6.00</td><td>6.00</td><td>3.50</td><td>62378886</td></tr>
	<tr><td>6.00</td><td>6.00</td><td>6.00</td><td>9.25</td><td>53780562</td></tr>
	<tr><td>6.00</td><td>6.00</td><td>6.00</td><td>9.50</td><td>53730182</td></tr>
	<tr><td>6.00</td><td>6.00</td><td>6.00</td><td>9.75</td><td>53950814</td></tr>
	<tr><td>1.75</td><td>6.25</td><td>6.25</td><td>3.50</td><td>51709654</td></tr>
	<tr><td>1.75</td><td>6.25</td><td>6.25</td><td>9.25</td><td>47537722</td></tr>
	<tr><td>1.75</td><td>6.25</td><td>6.25</td><td>9.50</td><td>47543381</td></tr>
	<tr><td>1.75</td><td>6.25</td><td>6.25</td><td>9.75</td><td>47555017</td></tr>
	<tr><td>5.00</td><td>6.25</td><td>6.25</td><td>3.50</td><td>61268505</td></tr>
	<tr><td>5.00</td><td>6.25</td><td>6.25</td><td>9.25</td><td>54207189</td></tr>
	<tr><td>5.00</td><td>6.25</td><td>6.25</td><td>9.50</td><td>54234608</td></tr>
	<tr><td>5.00</td><td>6.25</td><td>6.25</td><td>9.75</td><td>54268573</td></tr>
	<tr><td>5.25</td><td>6.25</td><td>6.25</td><td>3.50</td><td>61467297</td></tr>
	<tr><td>5.25</td><td>6.25</td><td>6.25</td><td>9.25</td><td>54070788</td></tr>
	<tr><td>5.25</td><td>6.25</td><td>6.25</td><td>9.50</td><td>54096598</td></tr>
	<tr><td>5.25</td><td>6.25</td><td>6.25</td><td>9.75</td><td>54154003</td></tr>
	<tr><td>5.50</td><td>6.25</td><td>6.25</td><td>3.50</td><td>61671008</td></tr>
	<tr><td>5.50</td><td>6.25</td><td>6.25</td><td>9.25</td><td>53964700</td></tr>
	<tr><td>5.50</td><td>6.25</td><td>6.25</td><td>9.50</td><td>53994107</td></tr>
	<tr><td>5.50</td><td>6.25</td><td>6.25</td><td>9.75</td><td>54052459</td></tr>
	<tr><td>5.75</td><td>6.25</td><td>6.25</td><td>3.50</td><td>61868864</td></tr>
	<tr><td>5.75</td><td>6.25</td><td>6.25</td><td>9.25</td><td>53862132</td></tr>
	<tr><td>5.75</td><td>6.25</td><td>6.25</td><td>9.50</td><td>53881640</td></tr>
	<tr><td>5.75</td><td>6.25</td><td>6.25</td><td>9.75</td><td>53941964</td></tr>
	<tr><td>6.00</td><td>6.25</td><td>6.25</td><td>3.50</td><td>62112237</td></tr>
	<tr><td>6.00</td><td>6.25</td><td>6.25</td><td>9.25</td><td>53734035</td></tr>
	<tr><td>6.00</td><td>6.25</td><td>6.25</td><td>9.50</td><td>53764618</td></tr>
	<tr><td>6.00</td><td>6.25</td><td>6.25</td><td>9.75</td><td>53825660</td></tr>
</tbody>
</table>




```R
scr_equity$SRC_Equity <- scr_equity$SRC_Equity/1e6
```


```R
options(repos = c(techtonique = "https://r-packages.techtonique.net",
                    CRAN = "https://cloud.r-project.org"))

install.packages(c("rvfl", "learningmachine"))
```


```R
set.seed(13)
train_idx <- sample(nrow(scr_equity), 0.8 * nrow(scr_equity))
X_train <- as.matrix(scr_equity[train_idx, -ncol(scr_equity)])
X_test <- as.matrix(scr_equity[-train_idx, -ncol(scr_equity)])
y_train <- scr_equity$SRC_Equity[train_idx]
y_test <- scr_equity$SRC_Equity[-train_idx]
```


```R
obj <- learningmachine::Regressor$new(method = "krr", pi_method = "none")
obj$get_type()
t0 <- proc.time()[3]
obj$fit(X_train, y_train, reg_lambda = 0.1)
cat("Elapsed: ", proc.time()[3] - t0, "s \n")
```


'regression'


    Elapsed:  0.005 s 



```R
print(sqrt(mean((obj$predict(X_test) - y_test)^2)))
```

    [1] 0.7250047



```R
obj$summary(X_test, y=y_test, show_progress=TRUE)
```

      |======================================================================| 100%



    $R_squared
    [1] 0.9306298
    
    $R_squared_adj
    [1] 0.9121311
    
    $Residuals
         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    -1.097222 -0.590318 -0.051308 -0.006375  0.447859  1.660139 
    
    $citests
                  estimate      lower       upper      p-value signif
    tmg          0.8311760 -0.8484270  2.51077903 3.133161e-01       
    pct_actions -0.4845265 -0.9327082 -0.03634475 3.555821e-02      *
    pvl_actions -0.4845265 -0.9327082 -0.03634475 3.555821e-02      *
    ppe         -2.2492137 -2.4397536 -2.05867385 6.622214e-16    ***
    
    $signif_codes
    [1] "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             20     
    Number of columns          4      
    _______________________           
    Column type frequency:            
      numeric                  4      
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
      skim_variable   mean    sd    p0    p25    p50    p75  p100 hist 


```R
obj <- learningmachine::Regressor$new(method = "rvfl",
                                      nb_hidden = 3L,
                                      pi_method = "kdesplitconformal")
```


```R
t0 <- proc.time()[3]
obj$fit(X_train, y_train, reg_lambda = 0.01)
cat("Elapsed: ", proc.time()[3] - t0, "s \n")
```

    Elapsed:  0.006 s 



```R
obj$summary(X_test, y=y_test, show_progress=FALSE)
```


    $R_squared
    [1] 0.8556358
    
    $R_squared_adj
    [1] 0.8171387
    
    $Residuals
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    -2.1720 -1.2977 -0.8132 -0.8003 -0.3254  0.5877 
    
    $Coverage_rate
    [1] 100
    
    $citests
                  estimate      lower      upper      p-value signif
    tmg          179.13631  162.48868  195.78394 3.639163e-15    ***
    pct_actions  -73.14222  -89.12337  -57.16108 1.046939e-08    ***
    pvl_actions   62.46782   46.48668   78.44896 1.199526e-07    ***
    ppe         -125.26721 -144.19952 -106.33490 2.223349e-11    ***
    
    $signif_codes
    [1] "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             20     
    Number of columns          4      
    _______________________           
    Column type frequency:            
      numeric                  4      
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
      skim_variable   mean   sd     p0    p25    p50   p75  p100 hist     



```R
obj$set_level(95)
res <- obj$predict(X = X_test)

plot(c(y_train, res$preds), type='l',
     main="(Probabilistic) Out-of-sample \n Equity Capital Requirement in m€",
     xlab="Observation Index",
     ylab="Equity Capital Requirement (m€)",
     ylim = c(min(c(res$upper, res$lower, y_test, y_train)),
              max(c(res$upper, res$lower, y_test, y_train))))
lines(c(y_train, res$upper), col="gray70")
lines(c(y_train, res$lower), col="gray70")
lines(c(y_train, res$preds), col = "red")
lines(c(y_train, y_test), col = "blue", lwd=2)
abline(v = length(y_train), lty=2, col="black", lwd=2)
```

![image-title-here]({{base}}/images/2025-09-16/2025-09-16-revisiting-equity-scr_13_0.png){:class="img-responsive"}    

```R
100*mean((y_test >= as.numeric(res$lower)) * (y_test <= as.numeric(res$upper)))
```


100


# Python version


```R
!pip install skimpy
```


```R
!pip install ydata-profiling
```


```R
!pip install nnetsauce
```


```R
import pandas as pd
from skimpy import skim
from ydata_profiling import ProfileReport
```


```R
scr_equity = pd.read_csv("ALIM4D.csv")
scr_equity['SRC_Equity'] = scr_equity['SRC_Equity']/1e6
```


```R
skim(scr_equity)
```


```R
ProfileReport(scr_equity)
```


```R
import nnetsauce as ns
import numpy as np

X, y = scr_equity.drop('SRC_Equity', axis=1), scr_equity['SRC_Equity'].values

```


```R
from sklearn.utils import all_estimators
from tqdm import tqdm
from sklearn.utils.multiclass import type_of_target
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time

# Get all scikit-learn regressors
estimators = all_estimators(type_filter='regressor')

results_regressors = []

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, (name, RegressorClass) in tqdm(enumerate(estimators)):

    if name in ['MultiOutputRegressor', 'MultiOutputClassifier', 'StackingRegressor', 'StackingClassifier',
                'VotingRegressor', 'VotingClassifier', 'TransformedTargetRegressor', 'RegressorChain',
                'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'RandomForestRegressor',
                'ExtraTreesRegressor', 'MLPRegressor']:

        continue

    for seed in seeds:

        try:

          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                              random_state=42+seed*1000)
          regr = ns.PredictionInterval(obj=ns.CustomRegressor(RegressorClass()),
                                       method="splitconformal",
                                       level=95,
                                       seed=312)
          start = time()
          regr.fit(X_train, y_train)
          print(f"Elapsed: {time() - start}s")
          preds = regr.predict(X_test, return_pi=True)
          coverage_rate = np.mean((preds.lower<=y_test)*(preds.upper>=y_test))
          rmse = np.sqrt(np.mean((preds-y_test)**2))
          results_regressors.append([name, seed, coverage_rate, rmse])

        except:

          continue
```


```R
results_df = pd.DataFrame(results_regressors, columns=['Regressor', 'Seed', 'Coverage Rate', 'RMSE'])
results_df.sort_values(by='Coverage Rate', ascending=False)
```


```R
results_df.dropna(inplace=True)
```


```R
results_df['logRMSE'] = np.log(results_df['RMSE'])
```
