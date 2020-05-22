---
layout: post
title: "AdaOpt (a probabilistic classifier based on a mix of multivariable optimization and nearest neighbors) for R"
description: AdaOpt (a probabilistic classifier based on a mix of multivariable optimization and nearest neighbors) for R
date: 2020-05-22
categories: [R, Misc]
---


[Last week]({% post_url 2020-05-15-adaopt-classifier-1 %}) on this blog, I presented `AdaOpt` for Python on a __handwritten digits classification__ task. `AdaOpt` is a novel _probabilistic_ classifier, based on a mix of multivariable optimization and a _nearest neighbors_ algorithm. It's still very new and only time will allow to fully appreciate all of its features.

The tool is fast due to [Cython](https://cython.org/), and to the ubiquitous (and mighty) numpy, which both help in bringing C/C++ -like performances to Python. There are also a few _tricks_ available in `AdaOpt`, that allow to make it faster to train on _bigger_ datasets. More details about the algorithm can be found in [this (short) paper](https://www.researchgate.net/publication/341409169_AdaOpt_Multivariable_optimization_for_classification).


![image-title-here]({{base}}/images/2020-05-22/2020-05-22-image1.png){:class="img-responsive"}


`AdaOpt` is now __available to R users__, and I used [reticulate](https://rstudio.github.io/reticulate/) for porting it (as I did for [nnetsauce]({% post_url 2020-03-06-nnetsauce-R-notebooks %})). An R documentation for the package can be found [in this repo](https://github.com/thierrymoudiki/mlsauce/blob/master/R-package.Rcheck/mlsauce-manual.pdf).

Here is an __example of use__:

## 1 - Install packages 

`AdaOpt`'s development code is [available on GitHub](https://github.com/thierrymoudiki/mlsauce), and the package can be installed by using `devtools` in R console:

```{r}
library(devtools)
```
```{r}
devtools::install_github("thierrymoudiki/mlsauce/R-package/")
```
```{r}
library(mlsauce)
```

The package `datasets` is also used for training the model: 

```{r}
library(datasets)
```

## 2 - Dataset for classification

The `iris` dataset (from package `datasets`) is used for this simple demo: 

```{r}
# import iris dataset

X <- as.matrix(iris[, 1:4])
y <- as.integer(iris[, 5]) - 1L # the classifier will accept numbers starting at 0
n <- dim(X)[1]
p <- dim(X)[2]

# create training set and test set
set.seed(21341)
train_index <- sample(x = 1:n, size = floor(0.8*n), 
                      replace = TRUE)
test_index <- -train_index

X_train <- as.matrix(iris[train_index, 1:4])
y_train <- as.integer(iris[train_index, 5]) - 1L # the classifier will accept numbers starting at 0
X_test <- as.matrix(iris[test_index, 1:4])
y_test <- as.integer(iris[test_index, 5]) - 1L # the classifier will accept numbers starting at 0
```


## 3 - Model fitting and score on test set

Now, we create an `AdaOpt`object and print its attributes: 

```{r}
# create AdaOpt object with default parameters
obj <- mlsauce::AdaOpt()

# print object attributes
print(obj$get_params())
```
```
## $batch_size
## [1] 100
## 
## $cache
## [1] TRUE
## 
## $eta
## [1] 0.01
## 
## $gamma
## [1] 0.01
## 
## $k
## [1] 3
## 
## $learning_rate
## [1] 0.3
## 
## $n_clusters
## [1] 0
## 
## $n_iterations
## [1] 50
## 
## $reg_alpha
## [1] 0.5
## 
## $reg_lambda
## [1] 0.1
## 
## $row_sample
## [1] 1
## 
## $seed
## [1] 123
## 
## $tolerance
## [1] 0
## 
## $type_dist
## [1] "euclidean-f"
```

Model __fitting__: 

```{r}
# fit AdaOpt to iris dataset
obj$fit(X_train, y_train)
```
```
## AdaOpt(batch_size=100, cache=True, eta=0.01, gamma=0.01, k=3, learning_rate=0.3,
##        n_clusters=0.0, n_iterations=50, reg_alpha=0.5, reg_lambda=0.1,
##        row_sample=1.0, seed=123, tolerance=0.0, type_dist='euclidean-f')
```

Obtain __test set accuracy__:

```{r}
# accuracy on test set 
print(obj$score(X_test, y_test))
```
```
## [1] 0.9701493
``` 

Lastly, no this package is not going to end up on [CRAN](https://cran.r-project.org/). None of my packages will, starting from now. If you're planning to submit your package to this website, well, there's more to it than being proud of having it accepted. If I may: think about it longer. In particular: [read this document about licenses choices](https://choosealicense.com/), and know your rights regarding __your__ intellectual property...

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!