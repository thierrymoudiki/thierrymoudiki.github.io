---
layout: post
title: "scikit-learn, glmnet, xgboost, lightgbm, pytorch, keras, nnetsauce in probabilistic Machine Learning (for longitudinal data) Reserving (work in progress)"
description: "Examples of use of Probabilistic Machine Learning (for longitudinal data) Reserving with scikit-learn, glmnet, xgboost, lightgbm, pytorch, keras, nnetsauce"
date: 2025-06-06
categories: Python
comments: true
---

Claims reserving in insurance is a critical actuarial process that involves estimating the future payments that an insurance company will need to make for claims that have already occurred but haven't been fully settled yet. 

This post demonstrates how to use various machine learning models for probabilistic reserving in insurance. We'll explore implementations using popular Python libraries including scikit-learn, glmnet, xgboost, lightgbm, PyTorch, Keras, and nnetsauce.

The goal is to show how these different modeling approaches can be applied to insurance reserving problems, with a focus on generating probabilistic predictions and confidence intervals. We'll work with a real insurance dataset and compare the results across different model architectures.

Key aspects covered:
- Implementation of various ML models for reserving
- Generation of prediction intervals
- Calculation of IBNR (Incurred But Not Reported) estimates
- Visualization of results and uncertainty bounds

Note that this is a work in progress, and the models shown are not fully tuned. The focus is on demonstrating the methodology rather than achieving optimal performance. Link to notebook at the end. 

```bash
!pip install git+https://github.com/Techtonique/mlreserving.git --verbose
!pip install nnetsauce glmnetforpython torch tensorflow scikeras 
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlreserving import MLReserving

# Import models from scikit-learn
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge

# Import nnetsauce models
import nnetsauce as ns

# Import glmnet models
import glmnetforpython as glmnet

# Import xgboost
import xgboost as xgb

# Import lightgbm
import lightgbm as lgb

# Import pytorch
import torch
import torch.nn as nn

# Import keras through scikeras
from scikeras.wrappers import KerasRegressor
from tensorflow import keras

# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv"
df = pd.read_csv(url)


import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def get_reg(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    return model

reg = KerasRegressor(
    model=get_reg,
    loss="mse",
    metrics=[keras.metrics.R2Score],
    hidden_layer_sizes=(20, 20, 20),
    dropout=0.1,
    verbose=0,
    random_state=123
)

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=1, hidden_sizes=(64, 32), activation=nn.ReLU,
                 learning_rate=0.001, max_epochs=100, batch_size=32, random_state=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

    def _build_model(self):
        layers = []
        input_dim = self.input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation())
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def fit(self, X, y, sample_weight=None):

        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        X, y = self._prepare_data(X, y)
        self._build_model()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = self._prepare_data(X)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).squeeze()
        return predictions.numpy()

    def _prepare_data(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float32)
            return X, y
        return X

models_run = []

# Define a list of models to try
models = [
    # scikit-learn models
    RidgeCV(alphas=[10**i for i in range(-5, 5)]),
    ElasticNetCV(),
    RandomForestRegressor(n_estimators=100, random_state=42),
    ExtraTreeRegressor(random_state=42),
    KernelRidge(alpha=1.0),
    
    # nnetsauce models
    ns.Ridge2Regressor(),
    ns.CustomRegressor(RidgeCV(alphas=[10**i for i in range(-5, 5)])),
    ns.DeepRegressor(RidgeCV(alphas=[10**i for i in range(-5, 5)])),
    
    # glmnet models
    glmnet.GLMNet(),
    
    # xgboost models
    xgb.XGBRegressor(n_estimators=100, random_state=42),
    
    # lightgbm models
    lgb.LGBMRegressor(n_estimators=100, random_state=42),
    
    # Keras model
    KerasRegressor(
    model=get_reg,
    loss="mse",
    metrics=[keras.metrics.R2Score],
    hidden_layer_sizes=(20, 20, 20),
    dropout=0.1,
    verbose=0,
    random_state=123
),

    # pytorch
    MLPRegressorTorch(input_size=2+1,
                      hidden_sizes=(20, 20, 20),
                      max_epochs=200,
                      random_state=42),
    

      
]

# Function to evaluate model performance
def evaluate_model(model_name, result, ibnr):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    print("\nMean predictions:")
    print(result.mean)
    
    print("\nIBNR per origin year (mean):")
    print(ibnr.mean)
    
    print("\nSummary:")
    print(model.get_summary())
    
    # Display results
    print("\nMean predictions:")
    result.mean.plot()
    plt.title(f'Mean Predictions - {mdl.__class__.__name__}')
    plt.show()

    print("\nLower bound (95%):")
    result.lower.plot()
    plt.title(f'Lower Bound - {mdl.__class__.__name__}')
    plt.show()

    print("\nUpper bound (95%):")
    result.upper.plot()
    plt.title(f'Upper Bound - {mdl.__class__.__name__}')
    plt.show()

    # Plot IBNR
    plt.figure(figsize=(10, 6))
    plt.plot(ibnr.mean.index, ibnr.mean.values, 'b-', label='Mean IBNR')
    plt.fill_between(ibnr.mean.index, 
                      ibnr.lower.values, 
                      ibnr.upper.values, 
                      alpha=0.2, 
                      label='95% Confidence Interval')
    plt.title(f'IBNR per Origin Year - {mdl.__class__.__name__}')
    plt.xlabel('Origin Year')
    plt.ylabel('IBNR')
    plt.legend()
    plt.grid(True)
    plt.show()

# Try both factor and non-factor approaches
for mdl in models:
    try:
        # Initialize the model with prediction intervals
        model = MLReserving(
            model=mdl,
            level=95,
            random_state=42
        )
        
        # Fit the model
        model.fit(df, origin_col="origin", development_col="development", value_col="values")
        
        # Make predictions with intervals
        result = model.predict()
        ibnr = model.get_ibnr()
        
        # Evaluate the model
        evaluate_model(mdl.__class__.__name__, result, ibnr)
        
        models_run.append(mdl.__class__.__name__)

    except Exception as e:
        print(f"Error with model {mdl.__class__.__name__}: {str(e)}")
        continue 

print(models_run)
```

<a target="_blank" href="https://colab.research.google.com/github/Techtonique/mlreserving/blob/main/mlreserving/demo/2025_06_06_advanced_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>

![image-title-here]({{base}}/images/2025-06-06/2025-06-06-image1.png){:class="img-responsive"}    

