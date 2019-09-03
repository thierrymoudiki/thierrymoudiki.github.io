---
layout: post
title: "More nnetsauce (examples of use)"
description: Examples of use of Python package nnetsauce
date: 2019-05-09
---

As mentioned in a [previous]({% post_url 2019-03-13-nnetsauce %}) post, [`nnetsauce`](https://github.com/thierrymoudiki/nnetsauce) is a Python package for Statistical/Machine learning and deep learning, based on combinations of *neural* networks layers. It could be used for solving regression, classification and multivariate time series forecasting problems. This post makes a more detailed introduction of `nnetsauce`, with a few examples based on classification and deep learning.          

        
## Installing the package


 Currently, `nnetsauce` can be installed through [Github](https://github.com/thierrymoudiki/nnetsauce) (but it will be available on PyPi in a few weeks).


Here is how: 

```python        
git clone https://github.com/thierrymoudiki/nnetsauce.git
cd nnetsauce
python setup.py install
```
                
## Examples of use of `nnetsauce`


  Below, are two examples of use of [`nnetsauce`](https://github.com/thierrymoudiki/nnetsauce). A **classification** example based on breast cancer data, and an illustrative **deep learning** example. In the classification example, we show how a logistic regression model can be enhanced, for a higher accuracy (accuracy is used here for simplicity), by using `nnetsauce`. The deep learning example shows how custom building blocks of `nnetsauce` objects can be combined together, to form a - perfectible - deeper learning architecture. 


  `scikit-learn` models are heavily used in these examples, but `nnetsauce` **will work with any learning model possessing methods `fit()` and `predict()`** (plus, `predict_proba()` for a classifier). That is, it could be used in conjunction with [xgboost](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py), [LightGBM](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py), or [CatBoost](https://github.com/catboost) for example. For the purpose of **model validation**, `sklearn`'s  cross-validation functions such as `GridSearchCV` and `cross_val_score` can be employed (on `nnetsauce` models), as it will be shown in the classification example.


##	 Classification example

For this first example, we start by **fitting a logistic regression model to breast cancer data** on a training set, and measure its accuracy on a validation set: 

```python
    # 0 - Packages ----- 

    # Importing the packages that will be used in the demo
    import nnetsauce as ns
    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split
    

    # 1 - Datasets -----

    # Loading breast cancer data
    breast_cancer = datasets.load_breast_cancer()
    Z = breast_cancer.data
    t = breast_cancer.target

    
    # 2 - Data splitting -----            

    # Separating the data into training/testing set, and 
    # a validation set
    Z_train, Z_test, t_train, t_test = train_test_split(
        Z, t, test_size=0.2, random_state=42)


    # 3 - Logistic regression -----

    # Fitting the Logistic regression model on 
    # training set
    regr = linear_model.LogisticRegression()                        
    regr.fit(Z_train, t_train)

    # predictive accuracy of the model on test set
    regr.score(Z_test, t_test)  
```

The accuracy of this model is equal to `0.9561`. The **logistic regression is now augmented of `n_hidden_features` additional features** with `nnetsauce`. We use `GridSearchCV` to find a better combination of hyperparameters;  additional hyperparameters such as row subsampling (`row_sample`) and `dropout` are included and reseached: 

```python
    # Defining nnetsauce model
    # based on the logistic regression model
    # defined previously
    fit_obj = ns.CustomClassifier(
    obj=regr,
    n_hidden_features=10,
    direct_link=True,
    bias=True,
    nodes_sim="sobol",
    activation_name="relu", 
    seed = 123)
    
    # Grid search ---
    from sklearn.model_selection import GridSearchCV
    # grid search for finding better hyperparameters
    np.random.seed(123)
    clf = GridSearchCV(cv = 3, estimator = fit_obj,
                       param_grid={'n_hidden_features': range(5, 25), 
                                   'row_sample': [0.7,0.8, 0.9], 
                                   'dropout': [0.7, 0.8, 0.9], 
                                   'n_clusters': [0, 2, 3, 4]}, 
                                   verbose=2)
    
    # fitting the model
    clf.fit(Z_train, t_train)

    # 'best' hyperparameters found 
    print(clf.best_params_)
    print(clf.best_score_)

    # predictive accuracy on test set
    clf.best_estimator_.score(Z_test, t_test)
```


After using `nnetsauce`, the accuracy is now equal to `0.9692`.

## deep learning example


This second example, is an **illustrative** example of deep learning with [`nnetsauce`](https://github.com/thierrymoudiki/nnetsauce). Many, **more advanced things could be tried**. In this example, predictive accuracy of the model **increases as new layers are added** to the stack. 
</p>


The **first layer** is a Bayesian ridge regression. Model accuracy (Root Mean Squared Error, RMSE) is equal to `63.56`. The **second layer** notably uses 3 additional features, an hyperbolic tangent activation function and the first layer; accuracy is `61.76`. To finish, the **third layer** uses 5 additional features, a sigmoid activation function and the second layer. The final accuracy, after adding this third layer is equal to: `61.68`.
</p>


```python
    import nnetsauce as ns
    from sklearn import datasets, metrics

    diabetes = datasets.load_diabetes()
    X = diabetes.data 
    y = diabetes.target
    
    # layer 1 (base layer) ----
    layer1_regr = linear_model.BayesianRidge()
    layer1_regr.fit(X[0:100,:], y[0:100])
    # RMSE score
    np.sqrt(metrics.mean_squared_error(y[100:125], layer1_regr.predict(X[100:125,:])))


    # layer 2 using layer 1 ----
    layer2_regr = ns.CustomRegressor(obj = layer1_regr, n_hidden_features=3, 
                            direct_link=True, bias=True, 
                            nodes_sim='sobol', activation_name='tanh', 
                            n_clusters=2)
    layer2_regr.fit(X[0:100,:], y[0:100])


    # RMSE score
    np.sqrt(layer2_regr.score(X[100:125,:], y[100:125]))

    # layer 3 using layer 2 ----
    layer3_regr = ns.CustomRegressor(obj = layer2_regr, n_hidden_features=5, 
                direct_link=True, bias=True, 
                nodes_sim='hammersley', activation_name='sigmoid', 
                n_clusters=2)
    layer3_regr.fit(X[0:100,:], y[0:100])

    # RMSE score
    np.sqrt(layer3_regr.score(X[100:125,:], y[100:125]))
```
