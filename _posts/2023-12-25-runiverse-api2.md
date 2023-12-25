---
layout: post
title: "A plethora of datasets at your fingertips Part2: how do couples cheat on each other?"
description: "Dowload data from R-Universe API"
date: 2023-12-25
categories: [Python, R, Misc, mlsauce]
comments: true
---

In `mlsauce`'s new release (`v0.9.0`, for [Python](https://github.com/Techtonique/mlsauce) and [R](https://github.com/Techtonique/mlsauce_r)), you're be able to download a plethora of datasets for your statistical/machine learning experiments. These datasets come from the [R-universe](https://r-universe.dev/search/), and you'll be able to use them **no matter whether you're working with Python or R**. 


**Contents**

<ul>
 <li> <a href="#dowload-a-dataset-in-python">Dowload a dataset in Python</a> </li>
 <li> <a href="#dowload-a-dataset-in-r">Dowload a dataset in R</a> </li>
</ul>


# Dowload a dataset in Python 

<a href="#">top</a>

**Install**

```bash
!pip install git+https://github.com/Techtonique/mlsauce.git@feature-branch
```

**Import data**

```Python
import mlsauce as ms 

# `ms.download` parameters 
# pkgname="MASS"
# dataset="Boston"
# source="https://cran.r-universe.dev/"

# the controversial Boston data set 
df1 = ms.download(dataset="Boston")

print(f"===== df1: \n {df1} \n")
print(f"===== df1.dtypes: \n {df1.dtypes}")

print("\n====================================================== \n")

# the controversial Boston data set 
df2 = ms.download(dataset="Insurance")
print(f"===== df2: \n {df2} \n")
print(f"===== df2.dtypes: \n {df2.dtypes}")
````
```Python
===== df1: 
        crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio   black  lstat  medv
0    0.0063  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
1    0.0273   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
2    0.0273   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7
3    0.0324   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7  394.63   2.94  33.4
4    0.0690   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7  396.90   5.33  36.2
..      ...   ...    ...   ...    ...    ...   ...     ...  ...  ...      ...     ...    ...   ...
501  0.0626   0.0  11.93     0  0.573  6.593  69.1  2.4786    1  273     21.0  391.99   9.67  22.4
502  0.0453   0.0  11.93     0  0.573  6.120  76.7  2.2875    1  273     21.0  396.90   9.08  20.6
503  0.0608   0.0  11.93     0  0.573  6.976  91.0  2.1675    1  273     21.0  396.90   5.64  23.9
504  0.1096   0.0  11.93     0  0.573  6.794  89.3  2.3889    1  273     21.0  393.45   6.48  22.0
505  0.0474   0.0  11.93     0  0.573  6.030  80.8  2.5050    1  273     21.0  396.90   7.88  11.9

[506 rows x 14 columns] 

===== df1.dtypes: 
 crim       float64
zn         float64
indus      float64
chas         int64
nox        float64
rm         float64
age        float64
dis        float64
rad          int64
tax          int64
ptratio    float64
black      float64
lstat      float64
medv       float64
dtype: object

====================================================== 

===== df2: 
    District   Group    Age  Holders  Claims
0         1     <1l    <25      197      38
1         1     <1l  25-29      264      35
2         1     <1l  30-35      246      20
3         1     <1l    >35     1680     156
4         1  1-1.5l    <25      284      63
..      ...     ...    ...      ...     ...
59        4  1.5-2l    >35      344      63
60        4     >2l    <25        3       0
61        4     >2l  25-29       16       6
62        4     >2l  30-35       25       8
63        4     >2l    >35      114      33

[64 rows x 5 columns] 

===== df2.dtypes: 
 District    object
Group       object
Age         object
Holders      int64
Claims       int64
dtype: object
```

# Dowload a dataset in R 

<a href="#">top</a>

**Install**

```R
remotes::install_github("Techtonique/mlsauce_r@dev-branch")
```

**Import data**

The controversial Boston dataset. 

```R
df <- mlsauce::download(pkgname = "MASS",
                        dataset = "Boston",
                        source = "https://cran.r-universe.dev/")
```

```R
print(head(df))
```

```R
    crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
1 0.0063 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
2 0.0273  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
3 0.0273  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
4 0.0324  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
5 0.0690  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
6 0.0298  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
```

```R
print(summary(lm(medv ~ ., data = df)))
```

```R
Call:
lm(formula = medv ~ ., data = df)

Residuals:
    Min      1Q  Median      3Q     Max 
-15.595  -2.730  -0.518   1.777  26.199 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
crim        -1.080e-01  3.286e-02  -3.287 0.001087 ** 
zn           4.642e-02  1.373e-02   3.382 0.000778 ***
indus        2.056e-02  6.150e-02   0.334 0.738288    
chas         2.687e+00  8.616e-01   3.118 0.001925 ** 
nox         -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
rm           3.810e+00  4.179e-01   9.116  < 2e-16 ***
age          6.922e-04  1.321e-02   0.052 0.958230    
dis         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
rad          3.060e-01  6.635e-02   4.613 5.07e-06 ***
tax         -1.233e-02  3.760e-03  -3.280 0.001112 ** 
ptratio     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
black        9.312e-03  2.686e-03   3.467 0.000573 ***
lstat       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.745 on 492 degrees of freedom
Multiple R-squared:  0.7406,	Adjusted R-squared:  0.7338 
F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16
```

![image-title-here]({{base}}/images/t-logo2.png){:class="img-responsive"}
