---
layout: post
title: "Prediction intervals for nnetsauce models"
description: Prediction intervals for nnetsauce models 
date: 2019-10-18
categories: QuasiRandomizedNN
---



There is always some uncertainty attached to Statistical/Machine Learning (ML) model's predictions. 
In this post, we examine a few ways to obtain [prediction intervals](https://en.wikipedia.org/wiki/Prediction_interval) for [nnetsauce](https://github.com/Techtonique/nnetsauce)'s ML models. As a reminder, every model in the nnetsauce is based on a component __g(XW+b)__ where:

- __X__ is a matrix containing some explanatory variables and optional clustering information (taking into account input data's heterogeneity).
- __W__ creates additional explanatory variables from __X__ and is drawn from various random and quasirandom sequences.
- __b__ is an optional bias parameter.
- __g__ is an _activation function_ such as the hyperbolic tangent or the sigmoid function.  


If we consider again [our example with tomatoes and apples]({% post_url 2019-09-25-nnetsauce-randombag-1 %}), the __y__ below (figure) contains our model's decisions: either "observation is a tomato" or "observation is an apple"). The __X__'s are each fruit's characteristics (color, shape, etc.). 

![image-title-here]({{base}}/images/2019-10-18/2019-10-18-image1.png){:class="img-responsive"}

This figure presents the structure of a linear nnetsauce model but [it could be anything]({% post_url 2019-05-09-more-nnetsauce %}) including nonlinear models, as long as the model has methods `fit` and `predict`. There are many ways to obtain prediction intervals for nnetsauce models. Here, we present 3 ways for doing that: 

- Using a __bayesian linear model__ as a basis for our nnetsauce model 
- Using a uniform distribution for __W__
- Using the randomness of [__dropout__](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) regularization technique on __g(XW+b)__


Here are the Python packages needed for our demo: 

```python
import nnetsauce as ns
import numpy as np      
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

Loading dataset for model training: 

```python
# Boston Housing data
boston = datasets.load_boston()
X = boston.data 
y = np.log(boston.target)

# split data in training/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  
                                                    random_state=123)
```

##### Using a __bayesian model__ as a basis for our nnetsauce model


Fitting a Bayesian Ridge regression model: 

```python
# base model for nnetsauce
regr = linear_model.BayesianRidge()

# create nnetsauce model based on the bayesian linear model
fit_obj = ns.CustomRegressor(obj=regr, n_hidden_features=10, 
                             activation_name='relu', n_clusters=2)

# model fitting
fit_obj.fit(X_train, y_train)

# mean prediction and
# standard deviation of the predictions
mean_preds1, std_preds1 = fit_obj.predict(X_test, return_std=True)
```

The model's prediction interval (using the mean prediction and
the standard deviation of our predictions) is depicted below:

![image-title-here]({{base}}/images/2019-10-18/2019-10-18-image2.png){:class="img-responsive"}

Red line represents values of `y_test`; actual observed houses prices on unseen data (data not used for training the model). The black line represents model predictions on unseen data, and the shaded region is our prediction interval.

##### Using a uniform distribution for __W__

For doing that, we simulate the __W__ in a nnetsauce `CustomRegressor` 100 times. With `nodes_sim="uniform"`.

```python
# number of model simulations
n = 100 

# create nnetsauce models
fit_obj2 = [ns.CustomRegressor(obj=regr, n_hidden_features=10, 
                               activation_name='relu', n_clusters=2,
                               nodes_sim="uniform", 
                               seed=(i*1000 + 2)) for i in range(n)]

# model fitting
fit_obj2_fits = [fit_obj2[i].fit(X_train, y_train) for i in range(n)]

# models predictions 
preds2 = np.asarray([fit_obj2_fits[i].predict(X_test) for i in range(n)]).T

# mean prediction 
mean_preds2 = preds2.mean(axis=1)

# standard deviation of the predictions
std_preds2 = preds2.std(axis=1)

```

We obtain the following prediction interval: 

![image-title-here]({{base}}/images/2019-10-18/2019-10-18-image3.png){:class="img-responsive"}

##### Using the __dropout__ regularization technique on __g(XW+b)__

The dropout technique randomly drops some elements in the matrix __g(XW+b)__. We can use its randomness to incorporate some uncertainty into our model:

```python
# create nnetsauce models
fit_obj3 = [ns.CustomRegressor(obj=regr, n_hidden_features=10, 
                               activation_name='relu', n_clusters=2,
                               nodes_sim="sobol", dropout=0.1, 
                               seed=(i*1000 + 2)) for i in range(n)]

# models fitting
fit_obj3_fits = [fit_obj3[i].fit(X_train, y_train) for i in range(n)]

# model predictions 
preds3 = np.asarray([fit_obj3_fits[i].predict(X_test) for i in range(n)]).T

# mean prediction 
mean_preds3 = preds3.mean(axis=1)

# standard deviation of the predictions
std_preds3 = preds3.std(axis=1)
```

Using the dropout technique, we obtain: 

![image-title-here]({{base}}/images/2019-10-18/2019-10-18-image4.png){:class="img-responsive"}


Using a uniform distribution for __W__ seems to be the best way to obtain prediction intervals on this dataset and for this model. However, it's worth mentioning that __the base model (a bayesian linear model) is not [calibrated]({% post_url 2019-10-04-crossval-1 %})__. Calibrating the base model could lead to different results than those displayed here. 


__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!

