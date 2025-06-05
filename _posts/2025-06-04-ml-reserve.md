---
layout: post
title: "Machine Learning (for longitudinal data) Reserving (work in progress)"
description: "Examps of use of Machine Learning (for longitudinal data) Reserving"
date: 2025-06-04
categories: Python
comments: true
---

Work in progress. Comments welcome. Link to notebook at the end of this post. 

```bash
!pip install git+https://github.com/Techtonique/mlreserving.git --verbose
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlreserving import MLReserving
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv"
df = pd.read_csv(url)

print(df.head())
print(df.tail())
#df["values"] = df["values"]/1000

models = [RidgeCV(), ExtraTreesRegressor(), RandomForestRegressor()]

# Try both factor and non-factor approaches
for use_factors in [False, True]:
    print(f"\n{'='*50}")
    print(f"Using {'factors' if use_factors else 'log transformations'}")
    print(f"{'='*50}\n")
    
    for mdl in models: 
        # Initialize the model with prediction intervals
        model = MLReserving(model=mdl,
            level=95,  # 95% confidence level
            use_factors=use_factors,  # Use categorical encoding
            random_state=42
        )        

        # Fit the model
        model.fit(df, origin_col="origin", development_col="development", value_col="values")

        # Make predictions with intervals
        result = model.predict()

        print("\nMean predictions:")
        print(result.mean)
        print("\nIBNR per origin year (mean):")
        print(result.ibnr_mean)

        print("\nLower bound (95%):")
        print(result.lower)
        print("\nIBNR per origin year (lower):")
        print(result.ibnr_lower)

        print("\nUpper bound (95%):")
        print(result.upper)
        print("\nIBNR per origin year (upper):")
        print(result.ibnr_upper)

        # Display results
        print("\nMean predictions:")
        result.mean.plot()
        plt.title(f'Mean Predictions - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        print("\nLower bound (95%):")
        result.lower.plot()
        plt.title(f'Lower Bound - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        print("\nUpper bound (95%):")
        result.upper.plot()
        plt.title(f'Upper Bound - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        # Plot IBNR
        plt.figure(figsize=(10, 6))
        plt.plot(result.ibnr_mean.index, result.ibnr_mean.values, 'b-', label='Mean IBNR')
        plt.fill_between(result.ibnr_mean.index, 
                        result.ibnr_lower.values, 
                        result.ibnr_upper.values, 
                        alpha=0.2, 
                        label='95% Confidence Interval')
        plt.title(f'IBNR per Origin Year - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.xlabel('Origin Year')
        plt.ylabel('IBNR')
        plt.legend()
        plt.grid(True)
        plt.show()
```

![image-title-here]({{base}}/images/2025-06-04/2025-06-04-image1.png){:class="img-responsive"}    

<a target="_blank" href="https://colab.research.google.com/github/Techtonique/mlreserving/blob/main/mlreserving/demo/2025-06-05-ml-reserving.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>
