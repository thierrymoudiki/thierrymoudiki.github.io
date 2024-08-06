---
layout: post
title: "Auto XGBoost, Auto LightGBM, Auto CatBoost, Auto GradientBoosting"
description: "Auto XGBoost, Auto LightGBM, Auto CatBoost, Auto GradientBoosting"
date: 2024-08-05
categories: [Python, R]
comments: true
---

I've always wanted to have a minimal unified interface to `XGBoost`, `CatBoost`, `LightGBM` and `sklearn's` `GradientBoosting`, without worrying about the different parameters names aliases. So, I had a lot of fun  creating [`unifiedbooster`](https://github.com/thierrymoudiki/unifiedbooster) (which is not part of [Techtonique](https://github.com/Techtonique), but is a personal swiss knife tool, under the MIT License).

In `unifiedbooster`, there are 5 main __common__  parameters for each algorithm:

- `n_estimators`: maximum number of trees that can be built
- `learning_rate`: shrinkage rate; used for reducing the gradient step
- `max_depth`: maximum tree depth
- `rowsample`: subsample ratio of the training instances
- `colsample`: percentage of features to use at each node split

In many situations, these are enough for obtaining robust "baselines" (and the whole documentation can be found [here](https://techtonique.github.io/unifiedbooster/unifiedbooster.html)). Additional parameters can be provided thanks to the `**kwargs` (even though that's not the main philosophy of the tool).

I present a **Python version and an R version**.

# Python version

```python
!pip install unifiedbooster
```

There are many ways to calibrate the _boosters_, which all rely on [GPopt](https://github.com/Techtonique/GPopt). I'll present only one today (the other ones in a few weeks): [Bayesian optimization](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt).


```python
import unifiedbooster as ub
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from time import time

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target # data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # split data into training set and test set

# Find 'good' hyperparameters for LightGBM
# Obtain 'best' model's performance on test set
res = ub.cross_val_optim(X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          model_type="lightgbm", # or 'lightgbm', 'gradientboosting', 'catboost'
                          type_fit="classification",
                          scoring="accuracy",
                          n_estimators=250,
                          cv=5, # numbers of folds in cross-validation
                          verbose=1,
                          seed=123)
print(res)
```

    
     Creating initial design... 
    
    
     ...Done. 
    
    
     Optimization loop... 
    
    190/190 [██████████████████████████████] - 45s 237ms/step
    result(best_params={'learning_rate': 0.9611431739764045, 'max_depth': 1, 'rowsample': 0.597564697265625, 'colsample': 0.508392333984375, 'model_type': 'lightgbm', 'n_estimators': 250}, best_score=-0.9780219780219781, test_accuracy=0.9736842105263158)


How do we **verify what we just did?**


```python
# Initialize the unified clf 
clf = ub.GBDTClassifier(**res.best_params)

# Fit the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate model's accuracy on test set
print(accuracy_score(y_test, y_pred))
```

    0.9736842105263158

**Classification report**

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.95      0.96        43
               1       0.97      0.99      0.98        71
    
        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114
    

**Confusion matrix**

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix,
            annot=True,
            fmt='g',
            xticklabels=clf.classes_,
            yticklabels=clf.classes_,
    )
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()
```

![xxx]({{base}}/images/2024-08-05/2024-08-05-image1.png){:class="img-responsive"}      


# R version

In the same environment as the Python environment: 

```R
utils::install.packages("reticulate")
library("reticulate")
unifiedbooster <- import("unifiedbooster")
```

Get data: 

```R
utils::install.packages("palmerpenguins")
library("palmerpenguins")


penguins_ <- as.data.frame(palmerpenguins::penguins)
# replacing NA's by the median

replacement <- median(palmerpenguins::penguins$bill_length_mm, na.rm = TRUE)
penguins_$bill_length_mm[is.na(palmerpenguins::penguins$bill_length_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$bill_depth_mm, na.rm = TRUE)
penguins_$bill_depth_mm[is.na(palmerpenguins::penguins$bill_depth_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$flipper_length_mm, na.rm = TRUE)
penguins_$flipper_length_mm[is.na(palmerpenguins::penguins$flipper_length_mm)] <- replacement

replacement <- median(palmerpenguins::penguins$body_mass_g, na.rm = TRUE)
penguins_$body_mass_g[is.na(palmerpenguins::penguins$body_mass_g)] <- replacement

# replacing NA's by the most frequent occurence
penguins_$sex[is.na(palmerpenguins::penguins$sex)] <- "male" # most frequent

# one-hot encoding
penguins_mat <- model.matrix(species ~., data=penguins_)[,-1]
penguins_mat <- cbind(penguins$species, penguins_mat)
penguins_mat <- as.data.frame(penguins_mat)
colnames(penguins_mat)[1] <- "species"

y <- as.integer(penguins_mat$species) - 1L
X <- as.matrix(penguins_mat[,2:ncol(penguins_mat)])

n <- nrow(X)
p <- ncol(X)

set.seed(123)
index_train <- sample(1:n, size=floor(0.8*n))

X_train <- X[index_train, c("islandDream", "islandTorgersen", "flipper_length_mm")]
y_train <- y[index_train]
X_test <- X[-index_train, c("islandDream", "islandTorgersen", "flipper_length_mm") ]
y_test <- y[-index_train]
```

Find hyperparameters: 

```R
res <- unifiedbooster$cross_val_optim(X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          model_type="xgboost",
                          type_fit="classification",
                          scoring="accuracy",
                          n_estimators=100L,
                          cv=5L, # numbers of folds in cross-validation
                          verbose=1L,
                          seed=123L)
print(res)
```

**check**

```R
# Initialize the unified clf 
clf = do.call(unifiedbooster$GBDTClassifier, res$best_params)

# Fit the model
clf$fit(X_train, y_train)

# Predict on the test set
y_pred = clf$predict(X_test)

# Evaluate model's accuracy on test set
print(mean(y_test == y_pred))
```
