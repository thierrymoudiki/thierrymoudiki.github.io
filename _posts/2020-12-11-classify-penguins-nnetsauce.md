---
layout: post
title: "Classify penguins with nnetsauce's MultitaskClassifier"
description: Classify penguins with nnetsauce's MultitaskClassifier
date: 2020-12-11
categories: [R, QuasiRandomizedNN]
---


I've recently heard and read about  [`iris`](https://en.wikipedia.org/wiki/Iris_flower_data_set) dataset's [_retirement_](https://www.r-bloggers.com/2020/06/penguins-dataset-overview-iris-alternative-in-r/). `iris` had been, for years, a go-to dataset for testing classifiers. The _new_ `iris` is a dataset of palmer penguins, available in R through the package [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/index.html). 

In this blog post, after data preparation, I adjust a classifier -- nnetsauce's [`MultitaskClassifier`](https://thierrymoudiki.github.io/blog/2020/02/28/python/quasirandomizednn/r/nnetsauce-new-version) -- to the palmer penguins dataset. 

# 0 - Import data and packages

Install palmerpenguins R package:

```r
library(palmerpenguins) 
```

Install nnetsauce's R package:

```r
library(devtools)
devtools::install_github("Techtonique/nnetsauce/R-package")
library(nnetsauce)
```


# 1 - Data preparation

`penguins_` below, is a temporary dataset which will contain palmer penguins data after imputation of missing values (NAs).

```r
penguins_ <- as.data.frame(palmerpenguins::penguins)
```

In numerical 
variables, NAs are replaced by the median of the column excluding NAs. In categorical variables, NAs are replaced by the most frequent value. These choices have an impact on the result. For example, if NAs are replaced by the mean instead of the median, the results could be quite different. 


```r
# replacing NA's by the median

replacement <- median(palmerpenguins::penguins$bill_length_mm, na.rm = TRUE)
penguins_$bill_length_mm[is.na(palmerpenguins::penguins$bill_length_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$bill_depth_mm, na.rm = TRUE)
penguins_$bill_depth_mm[is.na(palmerpenguins::penguins$bill_depth_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$flipper_length_mm, na.rm = TRUE)
penguins_$flipper_length_mm[is.na(palmerpenguins::penguins$flipper_length_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$body_mass_g, na.rm = TRUE)
penguins_$body_mass_g[is.na(palmerpenguins::penguins$body_mass_g)] <- replacement
```

```r
# replacing NA's by the most frequent occurence
penguins_$sex[is.na(palmerpenguins::penguins$sex)] <- "male" # most frequent
```

__Check__: any NA remaining in `penguins_`?

```r
print(sum(is.na(penguins_)))
```

The data frame `penguins_mat` below will contain all the penguins data, with each categorical explanatory variable present in `penguins_` transformed into a numerical one (otherwise, no Statistical/Machine learning model can be trained):  

```{r}
# one-hot encoding
penguins_mat <- model.matrix(species ~., data=penguins_)[,-1]
penguins_mat <- cbind(penguins$species, penguins_mat)
penguins_mat <- as.data.frame(penguins_mat)
colnames(penguins_mat)[1] <- "species"
```


```{r}
print(head(penguins_mat))
print(tail(penguins_mat))
```

![pres-image]({{base}}/images/2020-12-11/2020-12-11-image1.png){:class="img-responsive"}
![pres-image]({{base}}/images/2020-12-11/2020-12-11-image2.png){:class="img-responsive"}

# 2 - Model training and testing

The model used here to identify penguins species is [nnetsauce](https://github.com/Techtonique/nnetsauce)'s `MultitaskClassifier` (the R version here, but there's a Python version too). 
Instead of solving the whole problem of _classifying these species_ directly, 
nnetsauce's [`MultitaskClassifier`](https://techtonique.github.io/nnetsauce/documentation/classifiers/#multitaskclassifier) 
considers __three different questions separately__: is this an 
Adelie or not? Is this a Chinstrap or not? Is this a Gentoo or not? 

Each one of these binary classification problems is solved by an embedded regression (regression 
meaning here, a learning model for continuous outputs) model, on augmented data. The relatively 
strong hypothesis made in this setup is that: each one of these binary classification problems 
is solved by the same embedded regression model.

# 2 - 1 __First attempt:__ with feature selection. 


At first, only a few features are selected to explain the response: the __most positively correlated feature__ `flipper_length_mm`

![pres-image]({{base}}/images/2020-12-11/2020-12-11-image3.png){:class="img-responsive"}


and another 
__an interesting feature: the penguin's location__:

```r
table(palmerpenguins::penguins$species, palmerpenguins::penguins$island)
```


![pres-image]({{base}}/images/2020-12-11/2020-12-11-image4.png){:class="img-responsive"}

__Splitting the data into a training set and a testing set__

```{r}
y <- as.integer(penguins_mat$species) - 1L
X <- as.matrix(penguins_mat[,2:ncol(penguins_mat)])

n <- nrow(X)
p <- ncol(X)

set.seed(123)
index_train <- sample(1:n, size=floor(0.8*n))

X_train2 <- X[index_train, c("islandDream", "islandTorgersen", "flipper_length_mm")]
y_train2 <- y[index_train]
X_test2 <- X[-index_train, c("islandDream", "islandTorgersen", "flipper_length_mm") ]
y_test2 <- y[-index_train]

obj3 <- nnetsauce::sklearn$linear_model$LinearRegression()
obj4 <- nnetsauce::MultitaskClassifier(obj3)

print(obj4$get_params())
```



__Fit and predict on test set:__

```{r}
obj4$fit(X_train2, y_train2)

# accuracy on test set
print(obj4$score(X_test2, y_test2))
```

![pres-image]({{base}}/images/2020-12-11/2020-12-11-image5.png){:class="img-responsive"}

Not bad, an accuracy of 9 penguins out of 10 recognized by the classifier, with manually selected features. Can we do better with 
the entire dataset (all the features).

# 2 - 2 __Second attempt:__ the entire dataset. 

```{r}
X_train <- X[index_train, ]
y_train <- y[index_train]
X_test <- X[-index_train, ]
y_test <- y[-index_train]

obj <- nnetsauce::sklearn$linear_model$LinearRegression()
obj2 <- nnetsauce::MultitaskClassifier(obj)

obj2$fit(X_train, y_train)

# accuracy on test set
print(obj2$score(X_test, y_test))
```

![pres-image]({{base}}/images/2020-12-11/2020-12-11-image6.png){:class="img-responsive"}

By using all the explanatory variables, 100% of the 69 test set penguins are now recognized, 
thanks to nnetsauce's `MultitaskClassifier`. 
