---
layout: post
title: "Probability of receiving a loan; using learningmachine"
description: "Probability of receiving a loan; using learningmachine."
date: 2024-05-12
categories: [Python, R]
comments: true
---

In this post, I examine a data set available in the UCI Machine Learning
repository: `German Credit`. This data set contains characteristics of
1000 bank clients, explanatory variables, and a variable indicating
whether the client has *good* or *bad* chances of obtaining a credit.

We will use Machine Learning models available in R package
`learningmachine`. The package (also available [in
Python, click this link to see some examples](https://github.com/Techtonique/learningmachine_python/blob/main/learningmachine/demo/thierrymoudiki_20240508_calib.ipynb))
still has a lot of **ROUUUGH EDGES**, and no docs, but this situation
will be vastly improved by the end of june 2024. This, hopefully, won’t
prevent you from understanding its general philosophy and how to use it.
Feel free to submit [pull
requests](https://github.com/Techtonique/learningmachine/pulls).

# Contents

- [Contents](#contents)
- [0 - Install packages](#0---install-packages)
- [1 - Data preparation](#1---data-preparation)
- [2 Model training](#2-model-training)
- [3 - Model predictions](#3---model-predictions)

# 0 - Install packages


```
utils::install.packages("caret", repos = "https://cloud.r-project.org")
```

    
    The downloaded binary packages are in
        /var/folders/cp/q8d6040n3m38d22z3hkk1zc40000gn/T//RtmpEReNQT/downloaded_packages
    The downloaded binary packages are in
        /var/folders/cp/q8d6040n3m38d22z3hkk1zc40000gn/T//RtmpEReNQT/downloaded_packages
    The downloaded binary packages are in
        /var/folders/cp/q8d6040n3m38d22z3hkk1zc40000gn/T//RtmpEReNQT/downloaded_packages
    The downloaded binary packages are in
        /var/folders/cp/q8d6040n3m38d22z3hkk1zc40000gn/T//RtmpEReNQT/downloaded_packages


```
german_credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

german_credit$response <- german_credit$response - 1
german_credit$response[german_credit$response == 0] <- "bad"
german_credit$response[german_credit$response == 1] <- "good"
german_credit$response <- as.factor(german_credit$response)

print(table(german_credit$response)) # class imbalance, more bad credits, makes sense
```

    
     bad good 
     700  300 

# 1 - Data preparation


```
# obtain only numerical features 
X <- model.matrix.default(response ~ ., data = german_credit)[,-1] # explanatory variables
y <- german_credit$response # target variable
```


```
set.seed(123)
index_train <- caret::createDataPartition(y, p = 0.8)$Resample1
X_train <- X[index_train, ] # training set
y_train <- y[index_train] # training set
X_test <- X[-index_train, ] # test set
y_test <- y[-index_train] # test set
```

# 2 Model training

**With Random Forest**

The `nb_hidden` parameter is the number of nodes in the hidden layer of
the neural network. This is inspired from Quasi-Randomized _neural_ networks.


```
obj_rf <- learningmachine::Classifier$new(method = "ranger", nb_hidden = 25)
```


```
obj_rf$set_B(100L) # number of simulations for uncertainty estimation (buggy, fixed)
obj_rf$set_level(95) # confidence level
```


```
t0 <- proc.time()[3]
obj_rf$fit(X_train, y_train, pi_method="kdesplitconformal") # this will be described in a paper or something similar, obtains simulations for probabilities 
cat("Elapsed: ", proc.time()[3] - t0, "s \n")
```

    Elapsed:  0.848 s 

# 3 - Model predictions


```
probs <- obj_rf$predict_proba(X_test)
```

For individuals #158 and #171, we can plot the distribution of the
probability of having a good credit. This reads (among other possible
interpretations): with an error rate of 5% and an individual having the
same characteristics as #158, according to this specific model (a Random
Forest + quasi-random nodes in a layer), the probability of having a 
good credit is comprised between:


```
customer_index1 <- 158
customer_index2 <- 171
```


```
t.test(probs$sims$good[158, ])$conf.int
```

    [1] 0.4431368 0.5053498
    attr(,"conf.level")
    [1] 0.95

For an individual having the same characteristics as #171, the
probability of having a good credit is comprised between (credit can be
granted):


```
t.test(probs$sims$good[171, ])$conf.int
```

    [1] 0.6314353 0.7049742
    attr(,"conf.level")
    [1] 0.95


```
df <- data.frame(id=c(customer_index1, customer_index2), 
                 mean=c(mean(probs$sims$good[customer_index1,]), 
                        mean(probs$sims$good[customer_index2,])),
                 lower=c(quantile(probs$sims$good[customer_index1,], 0.025), 
                         quantile(probs$sims$good[customer_index2,], 0.025)),
                 upper=c(quantile(probs$sims$good[customer_index1,], 0.975), 
                         quantile(probs$sims$good[customer_index2,], 0.975)))
ggplot2::ggplot(df, aes(x=id, y=mean)) + 
  geom_point(size = 4) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.05) + 
  labs(title="Probability of having a good credit for individual #36 and #151", x="Individual", y="Probability") + 
  theme_minimal()
```

![pres-image]({{base}}/images/2024-05-12/2024-05-12-image1.png){:class="img-responsive"}        
        


This can be verified for these 2 individuals by looking at the actual
response:


```
print(y_test[c(customer_index1, customer_index2)])
```

    [1] bad  good
    Levels: bad good


```
print(X_test[c(customer_index1, customer_index2), ])
```

        chk_acctA12 chk_acctA13 chk_acctA14 duration credit_hisA31 credit_hisA32
    770           0           0           1       12             0             0
    815           0           0           0       48             0             1
        credit_hisA33 credit_hisA34 purposeA41 purposeA410 purposeA42 purposeA43
    770             0             1          0           0          0          1
    815             0             0          0           0          0          0
        purposeA44 purposeA45 purposeA46 purposeA48 purposeA49 amount
    770          0          0          0          0          0   1655
    815          0          0          0          0          0   3931
        saving_acctA62 saving_acctA63 saving_acctA64 saving_acctA65 present_empA72
    770              0              0              0              0              0
    815              0              0              0              0              0
        present_empA73 present_empA74 present_empA75 installment_rate sexA92 sexA93
    770              0              0              1                2      0      1
    815              0              1              0                4      0      1
        sexA94 other_debtorA102 other_debtorA103 present_resid propertyA122
    770      0                0                0             4            0
    815      0                0                0             4            0
        propertyA123 propertyA124 age other_installA142 other_installA143
    770            0            0  63                 0                 1
    815            0            1  46                 0                 1
        housingA152 housingA153 n_credits jobA172 jobA173 jobA174 n_people
    770           1           0         2       1       0       0        1
    815           0           1         1       0       1       0        2
        telephoneA192 foreignA202
    770             1           0
    815             0           0

Other insights can be obtained by looking at the sensitivity of these
probabilities to a small change in the input features:


```
obj_rf$summary(X_test, y_test, class_name = "good")
```

    
      |                                                                            
      |                                                                      |   0%
      |                                                                            
      |=                                                                     |   2%
      |                                                                            
      |===                                                                   |   4%
      |                                                                            
      |====                                                                  |   6%
      |                                                                            
      |======                                                                |   8%
      |                                                                            
      |=======                                                               |  10%
      |                                                                            
      |=========                                                             |  12%
      |                                                                            
      |==========                                                            |  15%
      |                                                                            
      |============                                                          |  17%
      |                                                                            
      |=============                                                         |  19%
      |                                                                            
      |===============                                                       |  21%
      |                                                                            
      |================                                                      |  23%
      |                                                                            
      |==================                                                    |  25%
      |                                                                            
      |===================                                                   |  27%
      |                                                                            
      |====================                                                  |  29%
      |                                                                            
      |======================                                                |  31%
      |                                                                            
      |=======================                                               |  33%
      |                                                                            
      |=========================                                             |  35%
      |                                                                            
      |==========================                                            |  38%
      |                                                                            
      |============================                                          |  40%
      |                                                                            
      |=============================                                         |  42%
      |                                                                            
      |===============================                                       |  44%
      |                                                                            
      |================================                                      |  46%
      |                                                                            
      |==================================                                    |  48%
      |                                                                            
      |===================================                                   |  50%
      |                                                                            
      |====================================                                  |  52%
      |                                                                            
      |======================================                                |  54%
      |                                                                            
      |=======================================                               |  56%
      |                                                                            
      |=========================================                             |  58%
      |                                                                            
      |==========================================                            |  60%
      |                                                                            
      |============================================                          |  62%
      |                                                                            
      |=============================================                         |  65%
      |                                                                            
      |===============================================                       |  67%
      |                                                                            
      |================================================                      |  69%
      |                                                                            
      |==================================================                    |  71%
      |                                                                            
      |===================================================                   |  73%
      |                                                                            
      |====================================================                  |  75%
      |                                                                            
      |======================================================                |  77%
      |                                                                            
      |=======================================================               |  79%
      |                                                                            
      |=========================================================             |  81%
      |                                                                            
      |==========================================================            |  83%
      |                                                                            
      |============================================================          |  85%
      |                                                                            
      |=============================================================         |  88%
      |                                                                            
      |===============================================================       |  90%
      |                                                                            
      |================================================================      |  92%
      |                                                                            
      |==================================================================    |  94%
      |                                                                            
      |===================================================================   |  96%
      |                                                                            
      |===================================================================== |  98%
      |                                                                            
      |======================================================================| 100%$ttests
                           estimate         lower         upper      p-value signif
    chk_acctA12        2.866628e-01 -0.1928077200  7.661333e-01 2.398135e-01       
    chk_acctA13        2.378946e-01 -0.4616138353  9.374030e-01 5.032280e-01       
    chk_acctA14        8.315866e-02 -0.4235356624  5.898530e-01 7.465524e-01       
    duration           9.503593e-03  0.0065260625  1.248112e-02 1.934730e-09    ***
    credit_hisA31     -1.463447e-01 -0.8962360480  6.035466e-01 7.007696e-01       
    credit_hisA32     -6.963395e-02 -0.2858254135  1.465575e-01 5.260569e-01       
    credit_hisA33      5.375104e-01  0.0196516923  1.055369e+00 4.199342e-02      *
    credit_hisA34      1.065668e-01 -0.4428390009  6.559726e-01 7.025022e-01       
    purposeA41         1.492564e-01 -0.5592482065  8.577610e-01 6.782819e-01       
    purposeA410        7.286522e-02 -1.1776013520  1.323332e+00 9.086349e-01       
    purposeA42         6.336391e-02 -0.5280075152  6.547353e-01 8.328772e-01       
    purposeA43        -5.145645e-01 -1.0662239242  3.709486e-02 6.735248e-02      .
    purposeA44         1.830964e-02 -1.1012994662  1.137919e+00 9.743061e-01       
    purposeA45        -2.082171e-01 -1.0634041966  6.469700e-01 6.316672e-01       
    purposeA46         1.607812e-01 -0.6056958203  9.272581e-01 6.795756e-01       
    purposeA48         1.934510e-01 -1.0192745472  1.406177e+00 7.534241e-01       
    purposeA49         1.141721e-01 -0.6032982799  8.316424e-01 7.540015e-01       
    amount            -9.233378e-06 -0.0000241999  5.733142e-06 2.252088e-01       
    saving_acctA62    -1.791712e-02 -0.4529104828  4.170762e-01 9.353457e-01       
    saving_acctA63    -2.877816e-01 -0.9103934331  3.348303e-01 3.631500e-01       
    saving_acctA64    -5.484981e-02 -0.6988110092  5.891114e-01 8.667832e-01       
    saving_acctA65    -2.320988e-01 -0.7750382076  3.108405e-01 4.002499e-01       
    present_empA72    -4.288099e-02 -0.6115880875  5.258261e-01 8.819511e-01       
    present_empA73    -2.535393e-02 -0.4317895531  3.810817e-01 9.022210e-01       
    present_empA74     1.065930e-01 -0.4069873095  6.201734e-01 6.827762e-01       
    present_empA75    -4.092760e-02 -0.5363583786  4.545032e-01 8.707597e-01       
    installment_rate   3.046334e-02  0.0127585556  4.816812e-02 8.340803e-04    ***
    sexA92             1.322121e-01 -0.0971965450  3.616208e-01 2.571257e-01       
    sexA93            -1.702534e-01 -0.5588612709  2.183544e-01 3.886630e-01       
    sexA94            -8.904505e-01 -1.5169988038 -2.639021e-01 5.571408e-03     **
    other_debtorA102  -1.703366e-01 -0.9532761016  6.126030e-01 6.683731e-01       
    other_debtorA103   2.025036e-01 -0.6345644743  1.039572e+00 6.338458e-01       
    present_resid      3.757648e-02  0.0186243133  5.652864e-02 1.266420e-04    ***
    propertyA122       3.373646e-01 -0.0364643491  7.111936e-01 7.666657e-02      .
    propertyA123       1.935712e-01 -0.3518870858  7.390296e-01 4.848668e-01       
    propertyA124       3.098673e-01 -0.2204829963  8.402176e-01 2.506403e-01       
    age               -7.948128e-03 -0.0102349661 -5.661289e-03 8.815565e-11    ***
    other_installA142  3.706325e-02 -0.6727217143  7.468482e-01 9.180898e-01       
    other_installA143  1.260310e-01 -0.0735411150  3.256031e-01 2.144849e-01       
    housingA152       -3.180369e-01 -0.6359422652 -1.315398e-04 4.990619e-02      *
    housingA153       -5.257518e-01 -1.1228101597  7.130662e-02 8.403259e-02      .
    n_credits          4.199850e-02  0.0090041088  7.499289e-02 1.286638e-02      *
    jobA172           -6.004859e-01 -1.1685872713 -3.238462e-02 3.840312e-02      *
    jobA173            1.502447e-01 -0.1516701848  4.521596e-01 3.276250e-01       
    jobA174           -7.947721e-02 -0.7201773080  5.612229e-01 8.070057e-01       
    n_people          -2.361480e-02 -0.0750490125  2.781941e-02 3.663605e-01       
    telephoneA192     -5.883641e-02 -0.3758199606  2.581471e-01 7.147377e-01       
    foreignA202       -1.297824e-01 -0.8829349965  6.233702e-01 7.343615e-01       
    
    $effects
    ── Data Summary ────────────────────────
                               Values 
    Name                       effects
    Number of rows             200    
    Number of columns          48     
    _______________________           
    Column type frequency:            
      numeric                  48     
    ________________________          
    Group variables            None   
    
    ── Variable type: numeric ──────────────────────────────────────────────────────
       skim_variable            mean       sd         p0        p25         p50
     1 chk_acctA12        0.287      3.44     -19.6       0          0         
     2 chk_acctA13        0.238      5.02     -19.9       0          0         
     3 chk_acctA14        0.0832     3.63     -19.6      -0.0115     0         
     4 duration           0.00950    0.0214    -0.0536   -0.00187    0.00614   
     5 credit_hisA31     -0.146      5.38     -35.3       0          0         
     6 credit_hisA32     -0.0696     1.55     -21.6       0          0         
     7 credit_hisA33      0.538      3.71     -13.6       0          0         
     8 credit_hisA34      0.107      3.94     -30.7       0          0         
     9 purposeA41         0.149      5.08     -22.4       0          0         
    10 purposeA410        0.0729     8.97     -37.6       0          0         
    11 purposeA42         0.0634     4.24     -35.3       0          0         
    12 purposeA43        -0.515      3.96     -30.7       0          0         
    13 purposeA44         0.0183     8.03     -29.9       0          0         
    14 purposeA45        -0.208      6.13     -35.9       0          0         
    15 purposeA46         0.161      5.50     -30.7       0          0         
    16 purposeA48         0.193      8.70     -32.2       0          0         
    17 purposeA49         0.114      5.15     -30.7       0          0         
    18 amount            -0.00000923 0.000107  -0.000527 -0.0000665  0.00000537
    19 saving_acctA62    -0.0179     3.12     -21.6       0          0         
    20 saving_acctA63    -0.288      4.47     -23.1       0          0         
    21 saving_acctA64    -0.0548     4.62     -22.5       0          0         
    22 saving_acctA65    -0.232      3.89     -30.7       0          0         
    23 present_empA72    -0.0429     4.08     -30.7       0          0         
    24 present_empA73    -0.0254     2.91     -19.9       0          0         
    25 present_empA74     0.107      3.68     -19.9       0          0         
    26 present_empA75    -0.0409     3.55     -19.6       0          0         
    27 installment_rate   0.0305     0.127     -0.312    -0.0500     0.0252    
    28 sexA92             0.132      1.65      -0.823     0          0         
    29 sexA93            -0.170      2.79     -19.9      -0.0552     0         
    30 sexA94            -0.890      4.49     -30.7       0          0         
    31 other_debtorA102  -0.170      5.61     -22.4       0          0         
    32 other_debtorA103   0.203      6.00     -23.1       0          0         
    33 present_resid      0.0376     0.136     -0.337    -0.0376     0.0328    
    34 propertyA122       0.337      2.68      -6.81      0          0         
    35 propertyA123       0.194      3.91     -30.7       0          0         
    36 propertyA124       0.310      3.80     -23.1       0          0         
    37 age               -0.00795    0.0164    -0.0691   -0.0152    -0.00689   
    38 other_installA142  0.0371     5.09     -35.3       0          0         
    39 other_installA143  0.126      1.43      -1.20     -0.0902     0         
    40 housingA152       -0.318      2.28     -19.6      -0.101      0         
    41 housingA153       -0.526      4.28     -30.7       0          0         
    42 n_credits          0.0420     0.237     -0.764    -0.0953     0.0332    
    43 jobA172           -0.600      4.07     -35.3       0          0         
    44 jobA173            0.150      2.17      -6.81     -0.159      0         
    45 jobA174           -0.0795     4.59     -23.1       0          0         
    46 n_people          -0.0236     0.369     -1.78     -0.203     -0.0173    
    47 telephoneA192     -0.0588     2.27     -19.6       0          0         
    48 foreignA202       -0.130      5.40     -35.3       0          0         
             p75      p100 hist 
     1 0         30.7      ▁▇▁▁▁
     2 0         35.3      ▁▇▁▁▁
     3 0         23.1      ▁▁▇▁▁
     4 0.0184     0.115    ▁▇▃▁▁
     5 0         22.4      ▁▁▁▇▁
     6 0.0753     1.39     ▁▁▁▁▇
     7 0         21.7      ▁▇▁▁▁
     8 0         21.7      ▁▁▇▁▁
     9 0         35.3      ▁▇▁▁▁
    10 0         35.3      ▁▁▇▁▁
    11 0         19.9      ▁▁▁▇▁
    12 0         15.5      ▁▁▁▇▁
    13 0         30.7      ▁▁▇▁▁
    14 0         30.7      ▁▁▇▁▁
    15 0         23.1      ▁▁▇▁▁
    16 0         41.0      ▁▁▇▁▁
    17 0         35.3      ▁▁▇▁▁
    18 0.0000579  0.000229 ▁▁▃▇▃
    19 0         30.7      ▁▁▇▁▁
    20 0         21.7      ▁▁▇▁▁
    21 0         22.4      ▁▁▇▁▁
    22 0         21.6      ▁▁▇▁▁
    23 0         35.3      ▁▁▇▁▁
    24 0         23.1      ▁▁▇▁▁
    25 0         23.1      ▁▁▇▁▁
    26 0         21.7      ▁▁▇▁▁
    27 0.0936     0.598    ▁▇▅▁▁
    28 0         23.1      ▇▁▁▁▁
    29 0.0324    19.2      ▁▁▇▁▁
    30 0         13.2      ▁▁▁▇▁
    31 0         23.1      ▁▁▇▁▁
    32 0         35.9      ▁▇▁▁▁
    33 0.110      0.680    ▁▇▅▁▁
    34 0         23.1      ▁▇▁▁▁
    35 0         23.1      ▁▁▇▁▁
    36 0         21.6      ▁▁▇▁▁
    37 0.00195    0.0540   ▁▂▇▂▁
    38 0         30.7      ▁▁▇▁▁
    39 0.174     19.9      ▇▁▁▁▁
    40 0.0725     0.851    ▁▁▁▁▇
    41 0         21.6      ▁▁▇▁▁
    42 0.161      0.879    ▁▃▇▂▁
    43 0         13.6      ▁▁▁▇▁
    44 0.00207   19.9      ▁▇▁▁▁
    45 0         35.3      ▁▇▁▁▁
    46 0.174      1.32     ▁▁▇▅▁
    47 0         19.9      ▁▁▇▁▁
    48 0         30.7      ▁▁▇▁▁
