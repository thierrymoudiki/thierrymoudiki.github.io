---
layout: post
title: "GPopt for Machine Learning (hyperparameters' tuning)
description: "Finding the 'best' hyperparameters for machine learning models using Bayesian Optimization"
date: 2026-01-21
categories: Python
comments: true
---

In this post, we will explore how to use the GPopt package to find the 'best' hyperparameters for machine learning models using Bayesian Optimization. This is a different interface from the one we used in the previous post; this one being more Machine Learning-focused (sklearn-like).


```python
!pip install GPopt
```


```python
import os
import GPopt as gp
import numpy as np
from os import chdir
from scipy.optimize import minimize
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create simple dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter configuration
param_config = {
    'n_estimators': {
        'bounds': [10, 100],
        'dtype': 'int',
        'default': 50
    },
    'max_depth': {
        'bounds': [3, 10],
        'dtype': 'int',
        'default': 5
    }
}

# Create optimizer with smaller settings for testing
optimizer = gp.MLOptimizer(
    scoring="balanced_accuracy",
    cv=5,  # Smaller CV for testing
    n_init=10,  # Fewer initial points (initial design)
    n_iter=25,  # Fewer iterations
    seed=42
)

result = optimizer.optimize(X_train, y_train,
                            RandomForestClassifier(),
                            param_config, verbose=2)
print("Optimization successful!")
print(f"Best score: {optimizer.get_best_score():.4f}")
print(f"Best parameters: {optimizer.get_best_parameters()}")

# Test creating and fitting the optimized estimator
model = optimizer.fit_optimized_estimator()
```

    
     Creating initial design... 
    
    point: [55.   6.5]; score: -0.9522469274791255
    point: [77.5   4.75]; score: -0.9534639660186794
    point: [32.5   8.25]; score: -0.945289879687232
    point: [43.75   5.625]; score: -0.9427969240822875
    point: [88.75   9.125]; score: -0.9521578009194108
    point: [66.25   3.875]; score: -0.9446118061149555
    point: [21.25   7.375]; score: -0.933914516144686
    point: [26.875   5.1875]; score: -0.9392365525652597
    point: [71.875   8.6875]; score: -0.9487684035831789
    point: [94.375   3.4375]; score: -0.936944819174989
    
     ...Done. 
    
    
     Optimization loop... 
    
    iteration 1 -----
    current minimum:  [77.5   4.75]
    current minimum score:  -0.9534639660186794
    next parameter: [82.00164795  7.66702271]
    score for next parameter: -0.958129280420302 
    
    iteration 2 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [82.00027466  9.99946594]
    score for next parameter: -0.9535530925783942 
    
    iteration 3 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [82.90184021  5.47583771]
    score for next parameter: -0.9428271721161654 
    
    iteration 4 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [81.80664062  8.37988281]
    score for next parameter: -0.9522469274791255 
    
    iteration 5 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [82.45689392  7.63556671]
    score for next parameter: -0.9480298339431468 
    
    iteration 6 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [81.85539246  7.84555817]
    score for next parameter: -0.9487986516170567 
    
    iteration 7 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [81.93161011  7.63957214]
    score for next parameter: -0.9510601369734497 
    
    iteration 8 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [82.01469421  7.73490143]
    score for next parameter: -0.9565245170828602 
    
    iteration 9 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [82.13623047  7.68432617]
    score for next parameter: -0.9451777840322733 
    
    iteration 10 -----
    current minimum:  [82.00164795  7.66702271]
    current minimum score:  -0.958129280420302
    next parameter: [81.94946289  7.72106934]
    score for next parameter: -0.9594051974856928 
    
    iteration 11 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.96250916  7.6808548 ]
    score for next parameter: -0.9522771755130034 
    
    iteration 12 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.89796448  7.76235199]
    score for next parameter: -0.947551365043625 
    
    iteration 13 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [82.07443237  7.76689148]
    score for next parameter: -0.9433342715076461 
    
    iteration 14 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [82.03666687  9.93094635]
    score for next parameter: -0.9540013134440379 
    
    iteration 15 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [82.10533142  9.90509796]
    score for next parameter: -0.9505530375819689 
    
    iteration 16 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.77093506  8.32583618]
    score for next parameter: -0.9534639660186794 
    
    iteration 17 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.84165955  8.33966827]
    score for next parameter: -0.9530459931869133 
    
    iteration 18 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [82.05589294  7.57019806]
    score for next parameter: -0.9474924865177881 
    
    iteration 19 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [77.44094849  4.81867981]
    score for next parameter: -0.9547701311179478 
    
    iteration 20 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [77.39425659  4.87358093]
    score for next parameter: -0.9445513100471998 
    
    iteration 21 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.87667847  8.29838562]
    score for next parameter: -0.9411619127109679 
    
    iteration 22 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [82.06756592  9.86306763]
    score for next parameter: -0.9521880489532887 
    
    iteration 23 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [77.55905151  4.68132019]
    score for next parameter: -0.9463056960121122 
    
    iteration 24 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.92749023  9.91711426]
    score for next parameter: -0.9552183519835917 
    
    iteration 25 -----
    current minimum:  [81.94946289  7.72106934]
    current minimum score:  -0.9594051974856928
    next parameter: [81.89659119  9.87689972]
    score for next parameter: -0.9501048167163251 
    
    Optimization successful!
    Best score: 0.9594
    Best parameters: {'n_estimators': np.float64(81.949462890625), 'max_depth': np.float64(7.7210693359375)}



```python
best_params = optimizer.get_best_parameters()
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
obj = RandomForestClassifier(**best_params)
obj.fit(X_train, y_train)
obj.score(X_test, y_test)
```




    0.9649122807017544


