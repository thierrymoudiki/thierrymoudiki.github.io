---
layout: post
title: "tisthemachinelearner: A Lightweight interface to scikit-learn with 2 classes, Classifier and Regressor (in Python and R)" 
description: "Demo usage of tisthemachinelearner, in Python and R"
date: 2025-02-17
categories: 
comments: true
---

# Contents 

- [Contents](#contents)
- [1 - Python version](#1---python-version)
- [2 - R version](#2---r-version)

# 1 - Python version


```python
!pip install tisthemachinelearner
```


```python
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import Classifier, Regressor

# Classification
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Classifier("LogisticRegression", random_state=42)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))

clf = Classifier("RandomForestClassifier", n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))

# Regression
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = Regressor("LinearRegression")
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))

reg = Regressor("RidgeCV", alphas=[0.01, 0.1, 1, 10])
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
```

    /usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
     1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
     1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
     1 0 0]
    0.956140350877193
    [1 0 0 1 1 0 0 0 0 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
     1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
     1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
     1 1 0]
    0.9649122807017544
    [139.5475584  179.51720835 134.03875572 291.41702925 123.78965872
      92.1723465  258.23238899 181.33732057  90.22411311 108.63375858
      94.13865744 168.43486358  53.5047888  206.63081659 100.12925869
     130.66657085 219.53071499 250.7803234  196.3688346  218.57511815
     207.35050182  88.48340941  70.43285917 188.95914235 154.8868162
     159.36170122 188.31263363 180.39094033  47.99046561 108.97453871
     174.77897633  86.36406656 132.95761215 184.53819483 173.83220911
     190.35858492 124.4156176  119.65110656 147.95168682  59.05405241
      71.62331856 107.68284704 165.45365458 155.00975931 171.04799096
      61.45761356  71.66672581 114.96732206  51.57975523 167.57599528
     152.52291955  62.95568515 103.49741722 109.20751489 175.64118426
     154.60296242  94.41704366 210.74209145 120.2566205   77.61585399
     187.93203995 206.49337474 140.63167076 105.59678023 130.70432536
     202.18534537 171.13039501 164.91423047 124.72472569 144.81030894
     181.99635452 199.41369642 234.21436188 145.95665512  79.86703276
     157.36941275 192.74412541 208.89814032 158.58722555 206.02195855
     107.47971675 140.93598906  54.82129332  55.92573195 115.01180018
      78.95584188  81.56087285  54.37997256 166.2543518 ]
    53.85344583676593
    [140.48932729 180.39358466 138.26095011 292.70472351 122.54953663
      93.61127853 256.94944065 185.46640503  86.4960167  110.59467587
      95.04571587 164.19550268  60.59798796 205.82695673  99.72760443
     131.91526636 220.91412088 247.87634694 195.84576355 215.78308828
     206.82609175  89.01546302  72.05374047 188.47495433 155.71143723
     161.25320029 189.08097216 178.04173865  49.65268248 110.50254797
     178.39994134  90.08024148 132.14592247 181.98946205 173.37370782
     190.81087767 123.38010922 118.90948131 146.69459204  60.67799313
      74.18510938 108.16651262 162.96843997 151.55290246 173.76202246
      64.5447612   76.57353392 109.83957197  56.57149752 163.18082268
     155.2330795   64.94611225 110.68142707 108.69309211 172.0029122
     157.94954707  94.8588743  208.43411608 118.81317959  72.11719648
     185.80485787 203.47916991 141.32147862 105.78698586 127.7320836
     202.81245148 168.55319265 162.78471685 120.58057123 142.15774259
     180.74853766 196.43247773 234.92016137 143.87413715  81.91095295
     153.24099082 193.15008313 206.58954277 158.12424491 201.30838954
     112.09889377 138.42466927  54.61388245  56.57971753 112.85843725
      83.27187052  81.11235009  59.60136702 164.50759424]
    53.68696471589718


# 2 - R version


```python
%load_ext rpy2.ipython
```


```r
%%R

install.packages("reticulate")
```


```r
%%R

library(reticulate)

# Importation des bibliothèques Python
np <- import("numpy")
sklearn <- import("sklearn")
datasets <- import("sklearn.datasets")
model_selection <- import("sklearn.model_selection")
tisthemachinelearner <- import("tisthemachinelearner")

# Classification
breast_cancer <- datasets$load_breast_cancer(return_X_y = TRUE)
X <- breast_cancer[[1]]
y <- breast_cancer[[2]]

split <- model_selection$train_test_split(X, y, test_size = 0.2, random_state = 42L)
X_train <- split[[1]]
X_test <- split[[2]]
y_train <- split[[3]]
y_test <- split[[4]]

# Logistic Regression
clf <- tisthemachinelearner$Classifier("LogisticRegression", random_state = 42L)
clf$fit(X_train, y_train)
print(clf$predict(X_test))
print(clf$score(X_test, y_test))

# Random Forest Classifier
clf <- tisthemachinelearner$Classifier("RandomForestClassifier", n_estimators = 100L, random_state = 42L)
clf$fit(X_train, y_train)
print(clf$predict(X_test))
print(clf$score(X_test, y_test))

# Regression
diabetes <- datasets$load_diabetes(return_X_y = TRUE)
X <- diabetes[[1]]
y <- diabetes[[2]]

split <- model_selection$train_test_split(X, y, test_size = 0.2, random_state = 42L)
X_train <- split[[1]]
X_test <- split[[2]]
y_train <- split[[3]]
y_test <- split[[4]]

# Linear Regression
reg <- tisthemachinelearner$Regressor("LinearRegression")
reg$fit(X_train, y_train)
y_pred <- reg$predict(X_test)
print(y_pred)
print(np$sqrt(np$mean((y_pred - y_test) ** 2)))

# Ridge Regression with Cross-Validation
reg <- tisthemachinelearner$Regressor("RidgeCV", alphas = c(0.01, 0.1, 1, 10))
reg$fit(X_train, y_train)
y_pred_ridge <- reg$predict(X_test)
print(y_pred_ridge)
print(np$sqrt(np$mean((y_pred_ridge - y_test) ** 2)))

```

      [1] 1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
     [38] 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
     [75] 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0

    /usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


     0 1 0 0 1 1 1 0 1 1 0
    [112] 1 0 0
    [1] 0.9561404
      [1] 1 0 0 1 1 0 0 0 0 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
     [38] 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
     [75] 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
    [112] 1 1 0
    [1] 0.9649123
     [1] 139.54756 179.51721 134.03876 291.41703 123.78966  92.17235 258.23239
     [8] 181.33732  90.22411 108.63376  94.13866 168.43486  53.50479 206.63082
    [15] 100.12926 130.66657 219.53071 250.78032 196.36883 218.57512 207.35050
    [22]  88.48341  70.43286 188.95914 154.88682 159.36170 188.31263 180.39094
    [29]  47.99047 108.97454 174.77898  86.36407 132.95761 184.53819 173.83221
    [36] 190.35858 124.41562 119.65111 147.95169  59.05405  71.62332 107.68285
    [43] 165.45365 155.00976 171.04799  61.45761  71.66673 114.96732  51.57976
    [50] 167.57600 152.52292  62.95569 103.49742 109.20751 175.64118 154.60296
    [57]  94.41704 210.74209 120.25662  77.61585 187.93204 206.49337 140.63167
    [64] 105.59678 130.70433 202.18535 171.13040 164.91423 124.72473 144.81031
    [71] 181.99635 199.41370 234.21436 145.95666  79.86703 157.36941 192.74413
    [78] 208.89814 158.58723 206.02196 107.47972 140.93599  54.82129  55.92573
    [85] 115.01180  78.95584  81.56087  54.37997 166.25435
    [1] 53.85345
     [1] 140.48933 180.39358 138.26095 292.70472 122.54954  93.61128 256.94944
     [8] 185.46641  86.49602 110.59468  95.04572 164.19550  60.59799 205.82696
    [15]  99.72760 131.91527 220.91412 247.87635 195.84576 215.78309 206.82609
    [22]  89.01546  72.05374 188.47495 155.71144 161.25320 189.08097 178.04174
    [29]  49.65268 110.50255 178.39994  90.08024 132.14592 181.98946 173.37371
    [36] 190.81088 123.38011 118.90948 146.69459  60.67799  74.18511 108.16651
    [43] 162.96844 151.55290 173.76202  64.54476  76.57353 109.83957  56.57150
    [50] 163.18082 155.23308  64.94611 110.68143 108.69309 172.00291 157.94955
    [57]  94.85887 208.43412 118.81318  72.11720 185.80486 203.47917 141.32148
    [64] 105.78699 127.73208 202.81245 168.55319 162.78472 120.58057 142.15774
    [71] 180.74854 196.43248 234.92016 143.87414  81.91095 153.24099 193.15008
    [78] 206.58954 158.12424 201.30839 112.09889 138.42467  54.61388  56.57972
    [85] 112.85844  83.27187  81.11235  59.60137 164.50759
    [1] 53.68696



```r
%%R

plot(y_pred_ridge, y_test, type="p", pch=19)
points(y_pred, y_test, col="blue", pch=19)
abline(a = 0, b = 1, col="red")
```

![image-title-here]({{base}}/images/2025-02-17/2025-02-17-image1.png)
    

