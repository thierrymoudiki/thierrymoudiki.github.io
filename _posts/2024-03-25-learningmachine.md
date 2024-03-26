---
layout: post
title: "learningmachine v1.0.0: prediction intervals around the probability of the event 'a tumor being malignant'"
description: "learningmachine v1.0.0: prediction intervals around the probability of the event 'a tumor being malignant'; using conformal prediction and density estimation"
date: 2024-03-25
categories: R
comments: true
---

<span>
<a target="_blank" href="https://colab.research.google.com/github/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20240318_conformal_and_bayesian_regression.ipynb">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</span>

Considering the number of people who read [this post](https://thierrymoudiki.github.io/blog/2024/01/01/r/learningmachine/learningmachine), a lot of you are probably using `learningmachine` `v0.1.0`. Maybe because of the fancy name. Just so you know, `learningmachine` is only doing batch learning at the moment. Stay tuned. 

Well, today, there are good news and bad news. The good news is `learningmachine` is back with `v1.0.0` (Python port coming next week). The "bad" news is: jumping to v1.0.0 this early means there's a **change in the interface** (that won't change drastically anymore); with a lot of good reasons: 

- **Smaller codebase**: much easier to navigate and maintain, less error-prone 
- **Only 2 classes** in the interface: `Classifier`, `Regressor` with (currently) 7 machine learning `method`s; "bcn" ([Boosted Configuration Networks](https://thierrymoudiki.github.io/blog/2024/02/05/python/gpopt-new2)), "extratrees" (Extremely Randomized Trees), "glmnet" (Elastic Net), "krr" (Kernel Ridge Regression), "ranger" (Random Forest), "ridge" (Automatic Ridge Regression), "xgboost". 
- Every classifier is [regression-based](https://www.researchgate.net/publication/377227280_Regression-based_machine_learning_classifiers).

The new features are: 

- Summarizing supervised learning results: **interpretability _via_ sensitivity** of the response to small changes in the explanatory variables + coverage rates for probabilistic predictions
- Uncertainty quantification for both regressors and classifiers (as shown below for classifiers). Right now, only the 'Least Ambiguous set-valued' method (denoted as standard Spit Conformal Prediction [here](https://conformalpredictionintro.github.io/)) is implemented for classifiers, **with a twist** (won't necessarily 
  remain this way): for empty prediction sets, the class with the highest probability is chosen. This _may_ 
  lead to over-conservative prediction sets. 

`learningmachine` is still experimental, probably with some quirks (because achieving this level of abstraction required some effort), with no beautiful documentation, but you can already tinker it and do advanced analysis, as shown below. 


```R
utils::install.packages("caret")
```
```R
utils::install.packages("dfoptim")
```
```R
utils::install.packages("ggplot2")
```
```R
utils::install.packages("mlbench")
```
```R
utils::install.packages("ranger")
```
```R
utils::install.packages("remotes")
```
```R
remotes::install_github("Techtonique/learningmachine")
```
```R
library(learningmachine)
```

```R
library(ggplot2)
library(mlbench)
library(ranger)

data("BreastCancer")
BreastCancer$Id <- NULL
rownames(BreastCancer) <- NULL 
y <- as.factor(BreastCancer$Class)
X <- BreastCancer[,-10]
X$Bare.nuclei[is.na(X$Bare.nuclei)] <- median(as.numeric(BreastCancer$Bare.nuclei[!is.na(BreastCancer$Bare.nuclei)]))
apply(X, 2, function(x) sum(is.na(x)))
```

       Cl.thickness       Cell.size      Cell.shape   Marg.adhesion    Epith.c.size 
                  0               0               0               0               0 
        Bare.nuclei     Bl.cromatin Normal.nucleoli         Mitoses 
                  0               0               0               0 

```R
for (i in seq_len(ncol(X)))
{
  X[,i] <- as.numeric(X[,i])
}


index_train <- caret::createDataPartition(y, p = 0.8)$Resample1
X_train <- X[index_train, ]
y_train <- y[index_train]
X_test <- X[-index_train, ]
y_test <- y[-index_train]
dim(X_train)
```

    [1] 560   9

```R
dim(X_test)
```

    [1] 139   9

```R
obj <- learningmachine::Classifier$new(method = "ranger")
obj$get_type()
```

    [1] "classification"

```R
obj$get_name()
```

    [1] "Classifier"

```R
obj$set_B(10)
obj$set_level(95)

t0 <- proc.time()[3]
obj$fit(X_train, y_train, pi_method="kdesplitconformal") # this will be described in a paper
cat("Elapsed: ", proc.time()[3] - t0, "s \n")
```

    Elapsed:  0.123 s 

```R
probs <- obj$predict_proba(X_test)

obj$summary(X_test, y=y_test, 
            class_name = "malignant",
            show_progress=FALSE)
```

    $Coverage_rate
    [1] 95.68345

    $ttests
                         estimate         lower       upper      p-value signif
    Cl.thickness     0.0056807801  0.0024459156 0.008915645 0.0006893052    ***
    Cell.size        0.0039919446  0.0011625077 0.006821382 0.0060221736     **
    Cell.shape       0.0023459459  0.0005416303 0.004150262 0.0112039276      *
    Marg.adhesion    0.0042356479  0.0018622609 0.006609035 0.0005676013    ***
    Epith.c.size    -0.0001036245 -0.0013577745 0.001150525 0.8704619531       
    Bare.nuclei      0.0104212402  0.0031755384 0.017666942 0.0051349801     **
    Bl.cromatin      0.0051171380 -0.0002930096 0.010527286 0.0635723868      .
    Normal.nucleoli  0.0067594459  0.0024786650 0.011040227 0.0021872093     **
    Mitoses          0.0007052483 -0.0001171510 0.001527648 0.0922097961      .

    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             139    
    Number of columns          9      
    _______________________           
    Column type frequency:            
      numeric                  9      
    ________________________          
    Group variables            None   

    ── Variable type: numeric ──────────────────────────────────────────────────────
      skim_variable        mean      sd       p0 p25 p50 p75   p100 hist 
    1 Cl.thickness     0.00568  0.0193  -0.0178    0   0   0 0.158  ▇▁▁▁▁
    2 Cell.size        0.00399  0.0169  -0.0136    0   0   0 0.116  ▇▁▁▁▁
    3 Cell.shape       0.00235  0.0108  -0.0209    0   0   0 0.0827 ▁▇▁▁▁
    4 Marg.adhesion    0.00424  0.0142  -0.00497   0   0   0 0.116  ▇▁▁▁▁
    5 Epith.c.size    -0.000104 0.00748 -0.0371    0   0   0 0.0409 ▁▁▇▁▁
    6 Bare.nuclei      0.0104   0.0432   0         0   0   0 0.297  ▇▁▁▁▁
    7 Bl.cromatin      0.00512  0.0323  -0.0171    0   0   0 0.366  ▇▁▁▁▁
    8 Normal.nucleoli  0.00676  0.0255  -0.00125   0   0   0 0.126  ▇▁▁▁▁
    9 Mitoses          0.000705 0.00490  0         0   0   0 0.0507 ▇▁▁▁▁

```R
df <- reshape2::melt(probs$sims$malignant[c(1, 5), ])
df$Var2 <- NULL 
colnames(df) <- c("individual", "prob_malignant")
df$individual <- as.factor(df$individual)
ggplot2::ggplot(df, aes(x=prob_malignant, fill=individual)) + geom_histogram(alpha=.3) +
  theme(
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(fill='transparent')
  )
```

![xxx]({{base}}/images/2024-03-25/2024-03-25-image1.png){:class="img-responsive"}      

```R
t.test(subset(df, individual == 1)$prob_malignant)
```


        One Sample t-test

    data:  subset(df, individual == 1)$prob_malignant
    t = 323.02, df = 99, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     0.6990101 0.7076507
    sample estimates:
    mean of x 
    0.7033304 

```R
t.test(subset(df, individual == 2)$prob_malignant)
```


        One Sample t-test

    data:  subset(df, individual == 2)$prob_malignant
    t = 222.29, df = 99, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     0.5023095 0.5113577
    sample estimates:
    mean of x 
    0.5068336 

```R
t.test(prob_malignant ~ individual, data = df)
```


        Welch Two Sample t-test

    data:  prob_malignant by individual
    t = 62.327, df = 197.58, p-value < 2.2e-16
    alternative hypothesis: true difference in means between group 1 and group 2 is not equal to 0
    95 percent confidence interval:
     0.1902796 0.2027140
    sample estimates:
    mean in group 1 mean in group 2 
          0.7033304       0.5068336 
