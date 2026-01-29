---
layout: post
title: "Overfitting and scaling (on GPU T4) tests on nnetsauce.CustomRegressor"
description: "Overfitting and scaling (on GPU T4) tests on nnetsauce.CustomRegressor"
date: 2026-01-29
categories: Python
comments: true
---

In this post, we will test the overfitting (if it can overfit and when it stops; if a it works well with a reasonable number of hidden features) and scaling properties of nnetsauce.CustomRegressor. Scaling tests were made on Colab with GPU T4. 

**Installing packages**

```python
!pip install nnetsauce
```

```python
!pip install mlsauce
```

**Overfitting** tests


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# NOTE: This script requires nnetsauce to be installed
# Install with: pip install nnetsauce
try:
    from nnetsauce import CustomRegressor
except ImportError:
    print("ERROR: nnetsauce is not installed. Please install it with:")
    print("pip install nnetsauce")
    exit(1)

# Set random seed for reproducibility
np.random.seed(42)

# Define a complex target function
def target_function(x):
    """Complex non-linear function to approximate"""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(8 * np.pi * x) + 0.3 * np.cos(5 * np.pi * x)

# Generate training and test data
n_train = 50
n_test = 200

X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
y_train = target_function(X_train.ravel()) + np.random.normal(0, 0.1, n_train)

X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
y_test = target_function(X_test.ravel())

# Test different numbers of hidden features (nodes)
# CustomRegressor adds hidden layers to boost the base model's capacity
n_hidden_features_list = [5, 10, 25, 50, 100, 200, 300, 400, 500]

# Create figure with subplots - FIXED: Changed from 2x3 to 3x3 to accommodate 9 plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

train_errors = []
test_errors = []

for idx, n_hidden in tqdm(enumerate(n_hidden_features_list)):
    # Create CustomRegressor with LinearRegression as base
    # n_hidden_features controls model capacity
    # activation_name='relu' uses ReLU activation for hidden features
    model = CustomRegressor(
        obj=LinearRegression(),
        n_hidden_features=n_hidden,
        activation_name='relu',  # or 'tanh', 'sigmoid'
        nodes_sim='sobol',  # quasi-random sampling
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)

    # Plot results
    ax = axes[idx]
    ax.scatter(X_train, y_train, c='red', s=30, alpha=0.6, label='Training data', zorder=3)
    ax.plot(X_test, y_test, 'b-', linewidth=2, label='True function', zorder=1)
    ax.plot(X_test, y_test_pred, 'g--', linewidth=2, label='Prediction', zorder=2)
    ax.set_title(f'Hidden Features: {n_hidden}\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nnetsauce_overfitting_demo.png', dpi=150, bbox_inches='tight')
print("Saved: nnetsauce_overfitting_demo.png")

# Create a second figure showing error vs model capacity
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot MSE vs number of hidden features
ax1.plot(n_hidden_features_list, train_errors, 'o-', linewidth=2, markersize=8, label='Training MSE')
ax1.plot(n_hidden_features_list, test_errors, 's-', linewidth=2, markersize=8, label='Test MSE')
ax1.set_xlabel('Number of Hidden Features (Model Capacity)', fontsize=12)
ax1.set_ylabel('Mean Squared Error', fontsize=12)
ax1.set_title('CustomRegressor: Error vs Model Capacity', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Demonstrate overfitting with very high capacity
n_overfit = 1000
model_overfit = CustomRegressor(
    obj=LinearRegression(),
    n_hidden_features=n_overfit,
    activation_name='relu',
    a=0.01,
    nodes_sim='sobol',
    bias=True,
    dropout=0.0,
    n_clusters=0,
)

model_overfit.fit(X_train, y_train)
y_train_overfit = model_overfit.predict(X_train)
y_test_overfit = model_overfit.predict(X_test)

ax2.scatter(X_train, y_train, c='red', s=40, alpha=0.7, label='Training data', zorder=3)
ax2.plot(X_test, y_test, 'b-', linewidth=2.5, label='True function', zorder=1)
ax2.plot(X_test, y_test_overfit, 'g--', linewidth=2, label=f'Prediction (n={n_overfit})', zorder=2)
ax2.set_title(f'High Capacity Model (Overfitting)\nTrain MSE: {mean_squared_error(y_train, y_train_overfit):.4f}, Test MSE: {mean_squared_error(y_test, y_test_overfit):.4f}',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nnetsauce_error_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: nnetsauce_error_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("OVERFITTING DEMONSTRATION WITH NNETSAUCE")
print("="*60)
print("\nModel: CustomRegressor(LinearRegression) with ReLU activation")
print(f"Training samples: {n_train}")
print(f"Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)")
print("\n" + "-"*60)
print(f"{'Hidden Features':<15} {'Train MSE':<15} {'Test MSE':<15} {'Ratio':<10}")
print("-"*60)

for n_hidden, train_err, test_err in zip(n_hidden_features_list, train_errors, test_errors):
    ratio = test_err / train_err if train_err > 0 else float('inf')
    print(f"{n_hidden:<15} {train_err:<15.6f} {test_err:<15.6f} {ratio:<10.2f}")

print("-"*60)
print(f"\n✓ As model capacity increases, training error decreases")
print(f"✓ Overfitting occurs when test error > training error significantly")
print(f"✓ Training MSE improved from {train_errors[0]:.4f} to {train_errors[-1]:.4f}")
print(f"✓ Test/Train ratio shows overfitting severity")

# Calculate overfitting indicator
best_idx = np.argmin([test_err / train_err for test_err, train_err in zip(test_errors, train_errors)])
print(f"\n✓ Best generalization at {n_hidden_features_list[best_idx]} hidden features")
print(f"  (Test/Train ratio = {test_errors[best_idx]/train_errors[best_idx]:.2f})")
print("="*60)

plt.show()
```

    9it [00:00, 15.71it/s]


    Saved: nnetsauce_overfitting_demo.png
    Saved: nnetsauce_error_analysis.png
    
    ============================================================
    OVERFITTING DEMONSTRATION WITH NNETSAUCE
    ============================================================
    
    Model: CustomRegressor(LinearRegression) with ReLU activation
    Training samples: 50
    Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)
    
    ------------------------------------------------------------
    Hidden Features Train MSE       Test MSE        Ratio     
    ------------------------------------------------------------
    5               0.202713        0.194247        0.96      
    10              0.085788        0.089940        1.05      
    25              0.021638        0.269249        12.44     
    50              0.012347        1.240659        100.48    
    100             0.004235        1.375602        324.85    
    200             0.003917        8.012315        2045.63   
    300             0.003917        0.999124        255.09    
    400             0.003917        1.917230        489.49    
    500             0.003388        1.793224        529.24    
    ------------------------------------------------------------
    
    ✓ As model capacity increases, training error decreases
    ✓ Overfitting occurs when test error > training error significantly
    ✓ Training MSE improved from 0.2027 to 0.0034
    ✓ Test/Train ratio shows overfitting severity
    
    ✓ Best generalization at 5 hidden features
      (Test/Train ratio = 0.96)
    ============================================================
    
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_6_2.png){:class="img-responsive"}
        
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_6_3.png){:class="img-responsive"}
    
**Scaling tests on nnetsauce.CustomRegressor+Housing dataset**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NOTE: This script requires nnetsauce and mlsauce to be installed
# Install with:
# pip install nnetsauce
# pip install git+https://github.com/Techtonique/mlsauce.git
try:
    from nnetsauce import CustomRegressor
except ImportError:
    print("ERROR: nnetsauce is not installed. Please install it with:")
    print("pip install nnetsauce")
    exit(1)

try:
    import mlsauce as ms
    MLSAUCE_AVAILABLE = True
except ImportError:
    print("WARNING: mlsauce is not installed. Will only compare with sklearn Ridge.")
    print("To install: pip install git+https://github.com/Techtonique/mlsauce.git")
    MLSAUCE_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

# Load California housing dataset
print("Loading California housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Use a subset for faster computation
subset_size = 2000
indices = np.random.choice(X.shape[0], subset_size, replace=False)
X = X[indices]
y = y[indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nDataset Info:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Target: Median house value (in $100,000s)")

# Test different numbers of hidden features
n_hidden_features_list = [5, 10, 25, 50, 100, 200, 300, 400, 500]

# Store results
results = {
    'sklearn_ridge': {'train_mse': [], 'test_mse': [], 'train_r2': [], 'test_r2': []},
}

if MLSAUCE_AVAILABLE:
    results['mlsauce_ridge'] = {'train_mse': [], 'test_mse': [], 'train_r2': [], 'test_r2': []}

print("\n" + "="*70)
print("COMPARING SKLEARN RIDGE VS MLSAUCE RIDGEREGRESSOR")
print("="*70)

# Train models with different capacities
for idx, n_hidden in tqdm(enumerate(n_hidden_features_list),
                          total=len(n_hidden_features_list),
                          desc="Training models"):

    # 1. CustomRegressor with sklearn Ridge
    model_sklearn = CustomRegressor(
        obj=Ridge(alpha=1.0),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
    )

    model_sklearn.fit(X_train, y_train)
    y_train_pred_sk = model_sklearn.predict(X_train)
    y_test_pred_sk = model_sklearn.predict(X_test)

    results['sklearn_ridge']['train_mse'].append(mean_squared_error(y_train, y_train_pred_sk))
    results['sklearn_ridge']['test_mse'].append(mean_squared_error(y_test, y_test_pred_sk))
    results['sklearn_ridge']['train_r2'].append(r2_score(y_train, y_train_pred_sk))
    results['sklearn_ridge']['test_r2'].append(r2_score(y_test, y_test_pred_sk))

    # 2. CustomRegressor with mlsauce RidgeRegressor (if available)
    if MLSAUCE_AVAILABLE:
        model_mlsauce = CustomRegressor(
            obj=ms.RidgeRegressor(reg_lambda=1.0, backend="cpu"),
            n_hidden_features=n_hidden,
            activation_name='relu',
            nodes_sim='sobol',
        )

        model_mlsauce.fit(X_train, y_train)
        y_train_pred_ml = model_mlsauce.predict(X_train)
        y_test_pred_ml = model_mlsauce.predict(X_test)

        results['mlsauce_ridge']['train_mse'].append(mean_squared_error(y_train, y_train_pred_ml))
        results['mlsauce_ridge']['test_mse'].append(mean_squared_error(y_test, y_test_pred_ml))
        results['mlsauce_ridge']['train_r2'].append(r2_score(y_train, y_train_pred_ml))
        results['mlsauce_ridge']['test_r2'].append(r2_score(y_test, y_test_pred_ml))

# Create visualization
n_plots = 2 if MLSAUCE_AVAILABLE else 1
fig, axes = plt.subplots(2, n_plots, figsize=(7*n_plots, 10))

if n_plots == 1:
    axes = axes.reshape(-1, 1)

# Plot 1: sklearn Ridge - MSE
ax = axes[0, 0]
ax.plot(n_hidden_features_list, results['sklearn_ridge']['train_mse'],
        'o-', linewidth=2, markersize=8, label='Training MSE', color='#2E86AB')
ax.plot(n_hidden_features_list, results['sklearn_ridge']['test_mse'],
        's-', linewidth=2, markersize=8, label='Test MSE', color='#A23B72')
ax.set_xlabel('Number of Hidden Features (Model Capacity)', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('CustomRegressor(sklearn Ridge): MSE vs Capacity', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Plot 2: sklearn Ridge - R²
ax = axes[1, 0]
ax.plot(n_hidden_features_list, results['sklearn_ridge']['train_r2'],
        'o-', linewidth=2, markersize=8, label='Training R²', color='#2E86AB')
ax.plot(n_hidden_features_list, results['sklearn_ridge']['test_r2'],
        's-', linewidth=2, markersize=8, label='Test R²', color='#A23B72')
ax.set_xlabel('Number of Hidden Features (Model Capacity)', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('CustomRegressor(sklearn Ridge): R² vs Capacity', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

if MLSAUCE_AVAILABLE:
    # Plot 3: mlsauce Ridge - MSE
    ax = axes[0, 1]
    ax.plot(n_hidden_features_list, results['mlsauce_ridge']['train_mse'],
            'o-', linewidth=2, markersize=8, label='Training MSE', color='#2E86AB')
    ax.plot(n_hidden_features_list, results['mlsauce_ridge']['test_mse'],
            's-', linewidth=2, markersize=8, label='Test MSE', color='#A23B72')
    ax.set_xlabel('Number of Hidden Features (Model Capacity)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('CustomRegressor(mlsauce Ridge): MSE vs Capacity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot 4: mlsauce Ridge - R²
    ax = axes[1, 1]
    ax.plot(n_hidden_features_list, results['mlsauce_ridge']['train_r2'],
            'o-', linewidth=2, markersize=8, label='Training R²', color='#2E86AB')
    ax.plot(n_hidden_features_list, results['mlsauce_ridge']['test_r2'],
            's-', linewidth=2, markersize=8, label='Test R²', color='#A23B72')
    ax.set_xlabel('Number of Hidden Features (Model Capacity)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('CustomRegressor(mlsauce Ridge): R² vs Capacity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('california_housing_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: california_housing_comparison.png")

# Print comparison table
print("\n" + "="*100)
print("RESULTS COMPARISON: CALIFORNIA HOUSING DATASET")
print("="*100)

print("\n" + "-"*100)
print(f"{'N_Hidden':<12} {'sklearn Ridge':<40} {'mlsauce Ridge':<40}")
print(f"{'Features':<12} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<12}")
print("-"*100)

for i, n_hidden in enumerate(n_hidden_features_list):
    sk_train_mse = results['sklearn_ridge']['train_mse'][i]
    sk_test_mse = results['sklearn_ridge']['test_mse'][i]
    sk_test_r2 = results['sklearn_ridge']['test_r2'][i]

    if MLSAUCE_AVAILABLE:
        ml_train_mse = results['mlsauce_ridge']['train_mse'][i]
        ml_test_mse = results['mlsauce_ridge']['test_mse'][i]
        ml_test_r2 = results['mlsauce_ridge']['test_r2'][i]

        print(f"{n_hidden:<12} {sk_train_mse:<12.4f} {sk_test_mse:<12.4f} {sk_test_r2:<12.4f} "
              f"{ml_train_mse:<12.4f} {ml_test_mse:<12.4f} {ml_test_r2:<12.4f}")
    else:
        print(f"{n_hidden:<12} {sk_train_mse:<12.4f} {sk_test_mse:<12.4f} {sk_test_r2:<12.4f} "
              f"{'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("-"*100)

# Summary statistics
print("\n" + "="*100)
print("SUMMARY")
print("="*100)

for model_name, model_results in results.items():
    print(f"\n{model_name.upper().replace('_', ' ')}:")
    best_test_idx = np.argmin(model_results['test_mse'])
    best_r2_idx = np.argmax(model_results['test_r2'])

    print(f"  ✓ Best Test MSE: {model_results['test_mse'][best_test_idx]:.4f} at {n_hidden_features_list[best_test_idx]} hidden features")
    print(f"  ✓ Best Test R²: {model_results['test_r2'][best_r2_idx]:.4f} at {n_hidden_features_list[best_r2_idx]} hidden features")

    # Calculate overfitting ratio
    ratios = [test/train if train > 0 else float('inf')
              for test, train in zip(model_results['test_mse'], model_results['train_mse'])]
    best_ratio_idx = np.argmin(ratios)
    print(f"  ✓ Best generalization (lowest Test/Train MSE ratio): {ratios[best_ratio_idx]:.2f} at {n_hidden_features_list[best_ratio_idx]} hidden features")

    # Detect overfitting
    overfit_indices = [i for i, r in enumerate(ratios) if r > 2.0]
    if overfit_indices:
        print(f"  ⚠ Overfitting detected (ratio > 2.0) at: {[n_hidden_features_list[i] for i in overfit_indices]} hidden features")

if MLSAUCE_AVAILABLE:
    print("\n" + "="*100)
    print("DIRECT COMPARISON")
    print("="*100)

    # Compare final performance
    sk_final_test_mse = results['sklearn_ridge']['test_mse'][-1]
    ml_final_test_mse = results['mlsauce_ridge']['test_mse'][-1]

    sk_best_test_mse = min(results['sklearn_ridge']['test_mse'])
    ml_best_test_mse = min(results['mlsauce_ridge']['test_mse'])

    print(f"\nAt highest capacity (500 hidden features):")
    print(f"  sklearn Ridge Test MSE: {sk_final_test_mse:.4f}")
    print(f"  mlsauce Ridge Test MSE: {ml_final_test_mse:.4f}")
    print(f"  Winner: {'mlsauce' if ml_final_test_mse < sk_final_test_mse else 'sklearn'}")

    print(f"\nBest overall performance:")
    print(f"  sklearn Ridge Best Test MSE: {sk_best_test_mse:.4f}")
    print(f"  mlsauce Ridge Best Test MSE: {ml_best_test_mse:.4f}")
    print(f"  Winner: {'mlsauce' if ml_best_test_mse < sk_best_test_mse else 'sklearn'}")

print("\n" + "="*100)

plt.show()
```

    Loading California housing dataset...
    
    Dataset Info:
    Training samples: 1400
    Test samples: 600
    Features: 8
    Target: Median house value (in $100,000s)
    
    ======================================================================
    COMPARING SKLEARN RIDGE VS MLSAUCE RIDGEREGRESSOR
    ======================================================================


    Training models: 100%|██████████| 9/9 [00:02<00:00,  3.38it/s]


    
    Saved: california_housing_comparison.png
    
    ====================================================================================================
    RESULTS COMPARISON: CALIFORNIA HOUSING DATASET
    ====================================================================================================
    
    ----------------------------------------------------------------------------------------------------
    N_Hidden     sklearn Ridge                            mlsauce Ridge                           
    Features     Train MSE    Test MSE     Test R²      Train MSE    Test MSE     Test R²     
    ----------------------------------------------------------------------------------------------------
    5            0.5428       0.5062       0.6235       0.5428       0.5062       0.6235      
    10           0.5255       0.4924       0.6337       0.5255       0.4924       0.6337      
    25           0.4919       0.4605       0.6575       0.4919       0.4605       0.6575      
    50           0.4366       0.4570       0.6601       0.4366       0.4570       0.6601      
    100          0.3841       0.4325       0.6783       0.3841       0.4325       0.6783      
    200          0.3228       0.4072       0.6972       0.3228       0.4072       0.6972      
    300          0.2902       0.3826       0.7154       0.2902       0.3826       0.7154      
    400          0.2679       0.3878       0.7115       0.2679       0.3878       0.7115      
    500          0.2498       0.3808       0.7167       0.2498       0.3808       0.7167      
    ----------------------------------------------------------------------------------------------------
    
    ====================================================================================================
    SUMMARY
    ====================================================================================================
    
    SKLEARN RIDGE:
      ✓ Best Test MSE: 0.3808 at 500 hidden features
      ✓ Best Test R²: 0.7167 at 500 hidden features
      ✓ Best generalization (lowest Test/Train MSE ratio): 0.93 at 5 hidden features
    
    MLSAUCE RIDGE:
      ✓ Best Test MSE: 0.3808 at 500 hidden features
      ✓ Best Test R²: 0.7167 at 500 hidden features
      ✓ Best generalization (lowest Test/Train MSE ratio): 0.93 at 5 hidden features
    
    ====================================================================================================
    DIRECT COMPARISON
    ====================================================================================================
    
    At highest capacity (500 hidden features):
      sklearn Ridge Test MSE: 0.3808
      mlsauce Ridge Test MSE: 0.3808
      Winner: mlsauce
    
    Best overall performance:
      sklearn Ridge Best Test MSE: 0.3808
      mlsauce Ridge Best Test MSE: 0.3808
      Winner: mlsauce
    
    ====================================================================================================

    
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_7_3.png){:class="img-responsive"}
    

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import warnings
warnings.filterwarnings('ignore')

# NOTE: This script requires nnetsauce and mlsauce
try:
    from nnetsauce import CustomRegressor
except ImportError:
    print("ERROR: nnetsauce is not installed.")
    exit(1)

try:
    import mlsauce as ms
    MLSAUCE_AVAILABLE = True
except ImportError:
    print("WARNING: mlsauce is not installed.")
    MLSAUCE_AVAILABLE = False

# Set random seed
np.random.seed(42)

# Load dataset
print("Loading California housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Use larger subset to see performance differences
subset_size = 5000
indices = np.random.choice(X.shape[0], subset_size, replace=False)
X = X[indices]
y = y[indices]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nDataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
print(f"Features: {X_train.shape[1]}")

# Configuration
n_hidden_features_list = [50, 100, 200, 500, 1000]
results = {
    'sklearn': {'times': [], 'test_mse': []},
}

if MLSAUCE_AVAILABLE:
    results['mlsauce_cpu'] = {'times': [], 'test_mse': []}
    results['mlsauce_gpu'] = {'times': [], 'test_mse': []}

print("\n" + "="*70)
print("PERFORMANCE COMPARISON: SKLEARN VS MLSAUCE (CPU/GPU)")
print("="*70)

for n_hidden in n_hidden_features_list:
    print(f"\nTesting with {n_hidden} hidden features...")

    # 1. sklearn Ridge (CPU)
    start = time()
    model_sklearn = CustomRegressor(
        obj=Ridge(alpha=1.0),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
    )
    model_sklearn.fit(X_train, y_train)
    y_test_pred = model_sklearn.predict(X_test)
    elapsed_sklearn = time() - start

    results['sklearn']['times'].append(elapsed_sklearn)
    results['sklearn']['test_mse'].append(mean_squared_error(y_test, y_test_pred))
    print(f"  sklearn Ridge (CPU): {elapsed_sklearn:.3f}s")

    if MLSAUCE_AVAILABLE:
        # 2. mlsauce Ridge (CPU)
        start = time()
        model_mlsauce_cpu = CustomRegressor(
            obj=ms.RidgeRegressor(reg_lambda=1.0, backend="cpu"),
            n_hidden_features=n_hidden,
            activation_name='relu',
            nodes_sim='sobol',
        )
        model_mlsauce_cpu.fit(X_train, y_train)
        y_test_pred = model_mlsauce_cpu.predict(X_test)
        elapsed_ml_cpu = time() - start

        results['mlsauce_cpu']['times'].append(elapsed_ml_cpu)
        results['mlsauce_cpu']['test_mse'].append(mean_squared_error(y_test, y_test_pred))
        print(f"  mlsauce Ridge (CPU): {elapsed_ml_cpu:.3f}s (speedup: {elapsed_sklearn/elapsed_ml_cpu:.2f}x)")

        # 3. mlsauce Ridge (GPU) - if available
        try:
            start = time()
            model_mlsauce_gpu = CustomRegressor(
                obj=ms.RidgeRegressor(reg_lambda=1.0, backend="gpu"),
                n_hidden_features=n_hidden,
                activation_name='relu',
                nodes_sim='sobol',
            )
            model_mlsauce_gpu.fit(X_train, y_train)
            y_test_pred = model_mlsauce_gpu.predict(X_test)
            elapsed_ml_gpu = time() - start

            results['mlsauce_gpu']['times'].append(elapsed_ml_gpu)
            results['mlsauce_gpu']['test_mse'].append(mean_squared_error(y_test, y_test_pred))
            print(f"  mlsauce Ridge (GPU): {elapsed_ml_gpu:.3f}s (speedup: {elapsed_sklearn/elapsed_ml_gpu:.2f}x)")
        except Exception as e:
            print(f"  mlsauce Ridge (GPU): FAILED ({str(e)[:50]}...)")
            results['mlsauce_gpu']['times'].append(None)
            results['mlsauce_gpu']['test_mse'].append(None)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Time
ax = axes[0]
ax.plot(n_hidden_features_list, results['sklearn']['times'],
        'o-', linewidth=2, markersize=8, label='sklearn Ridge (CPU)', color='#2E86AB')

if MLSAUCE_AVAILABLE:
    ax.plot(n_hidden_features_list, results['mlsauce_cpu']['times'],
            's-', linewidth=2, markersize=8, label='mlsauce Ridge (CPU)', color='#F18F01')

    if any(t is not None for t in results['mlsauce_gpu']['times']):
        valid_indices = [i for i, t in enumerate(results['mlsauce_gpu']['times']) if t is not None]
        valid_n_hidden = [n_hidden_features_list[i] for i in valid_indices]
        valid_times = [results['mlsauce_gpu']['times'][i] for i in valid_indices]
        ax.plot(valid_n_hidden, valid_times,
                '^-', linewidth=2, markersize=8, label='mlsauce Ridge (GPU)', color='#C73E1D')

ax.set_xlabel('Number of Hidden Features', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Training Time vs Model Capacity', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: Test MSE
ax = axes[1]
ax.plot(n_hidden_features_list, results['sklearn']['test_mse'],
        'o-', linewidth=2, markersize=8, label='sklearn Ridge (CPU)', color='#2E86AB')

if MLSAUCE_AVAILABLE:
    ax.plot(n_hidden_features_list, results['mlsauce_cpu']['test_mse'],
            's-', linewidth=2, markersize=8, label='mlsauce Ridge (CPU)', color='#F18F01')

    if any(t is not None for t in results['mlsauce_gpu']['test_mse']):
        valid_indices = [i for i, t in enumerate(results['mlsauce_gpu']['test_mse']) if t is not None]
        valid_n_hidden = [n_hidden_features_list[i] for i in valid_indices]
        valid_mse = [results['mlsauce_gpu']['test_mse'][i] for i in valid_indices]
        ax.plot(valid_n_hidden, valid_mse,
                '^-', linewidth=2, markersize=8, label='mlsauce Ridge (GPU)', color='#C73E1D')

ax.set_xlabel('Number of Hidden Features', fontsize=12)
ax.set_ylabel('Test MSE', fontsize=12)
ax.set_title('Test Error vs Model Capacity', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
print("\n\nSaved: performance_comparison.png")

# Summary table
print("\n" + "="*90)
print("PERFORMANCE SUMMARY")
print("="*90)
print(f"\n{'N_Hidden':<12} {'sklearn (CPU)':<20} {'mlsauce (CPU)':<20} {'mlsauce (GPU)':<20}")
print(f"{'Features':<12} {'Time (s)':<20} {'Time (s)':<20} {'Time (s)':<20}")
print("-"*90)

for i, n_hidden in enumerate(n_hidden_features_list):
    sk_time = results['sklearn']['times'][i]

    if MLSAUCE_AVAILABLE:
        ml_cpu_time = results['mlsauce_cpu']['times'][i]
        ml_gpu_time = results['mlsauce_gpu']['times'][i] if results['mlsauce_gpu']['times'][i] else 0

        if ml_gpu_time:
            print(f"{n_hidden:<12} {sk_time:<20.3f} {ml_cpu_time:<20.3f} {ml_gpu_time:<20.3f}")
        else:
            print(f"{n_hidden:<12} {sk_time:<20.3f} {ml_cpu_time:<20.3f} {'N/A':<20}")
    else:
        print(f"{n_hidden:<12} {sk_time:<20.3f} {'N/A':<20} {'N/A':<20}")

print("-"*90)

if MLSAUCE_AVAILABLE:
    # Calculate average speedups
    cpu_speedups = [sk_t / ml_t for sk_t, ml_t in
                    zip(results['sklearn']['times'], results['mlsauce_cpu']['times'])]
    print(f"\nAverage mlsauce CPU speedup: {np.mean(cpu_speedups):.2f}x")

    gpu_times_valid = [t for t in results['mlsauce_gpu']['times'] if t is not None]
    if gpu_times_valid:
        gpu_speedups = [sk_t / ml_t for sk_t, ml_t in
                       zip(results['sklearn']['times'][:len(gpu_times_valid)], gpu_times_valid)]
        print(f"Average mlsauce GPU speedup: {np.mean(gpu_speedups):.2f}x")
        print(f"GPU vs CPU speedup: {np.mean([c/g for c, g in zip(results['mlsauce_cpu']['times'][:len(gpu_times_valid)], gpu_times_valid)]):.2f}x")

print("\n" + "="*90)
print("\nNote: GPU acceleration is most beneficial with:")
print("  - Large datasets (10,000+ samples)")
print("  - High-dimensional features")
print("  - Large number of hidden features")
print("  - Multiple iterations/cross-validation")
print("="*90)

plt.show()
```

    Loading California housing dataset...
    
    Dataset: 3500 training samples, 1500 test samples
    Features: 8
    
    ======================================================================
    PERFORMANCE COMPARISON: SKLEARN VS MLSAUCE (CPU/GPU)
    ======================================================================
    
    Testing with 50 hidden features...
      sklearn Ridge (CPU): 0.069s
      mlsauce Ridge (CPU): 0.091s (speedup: 0.76x)
      mlsauce Ridge (GPU): 7.625s (speedup: 0.01x)
    
    Testing with 100 hidden features...
      sklearn Ridge (CPU): 0.094s
      mlsauce Ridge (CPU): 0.057s (speedup: 1.65x)
      mlsauce Ridge (GPU): 1.691s (speedup: 0.06x)
    
    Testing with 200 hidden features...
      sklearn Ridge (CPU): 0.206s
      mlsauce Ridge (CPU): 0.176s (speedup: 1.17x)
      mlsauce Ridge (GPU): 1.721s (speedup: 0.12x)
    
    Testing with 500 hidden features...
      sklearn Ridge (CPU): 0.350s
      mlsauce Ridge (CPU): 0.369s (speedup: 0.95x)
      mlsauce Ridge (GPU): 3.018s (speedup: 0.12x)
    
    Testing with 1000 hidden features...
      sklearn Ridge (CPU): 0.757s
      mlsauce Ridge (CPU): 0.745s (speedup: 1.02x)
      mlsauce Ridge (GPU): 2.856s (speedup: 0.27x)
    
    
    Saved: performance_comparison.png
    
    ==========================================================================================
    PERFORMANCE SUMMARY
    ==========================================================================================
    
    N_Hidden     sklearn (CPU)        mlsauce (CPU)        mlsauce (GPU)       
    Features     Time (s)             Time (s)             Time (s)            
    ------------------------------------------------------------------------------------------
    50           0.069                0.091                7.625               
    100          0.094                0.057                1.691               
    200          0.206                0.176                1.721               
    500          0.350                0.369                3.018               
    1000         0.757                0.745                2.856               
    ------------------------------------------------------------------------------------------
    
    Average mlsauce CPU speedup: 1.11x
    Average mlsauce GPU speedup: 0.11x
    GPU vs CPU speedup: 0.11x
    
    ==========================================================================================
    
    Note: GPU acceleration is most beneficial with:
      - Large datasets (10,000+ samples)
      - High-dimensional features
      - Large number of hidden features
      - Multiple iterations/cross-validation
    ==========================================================================================



    
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_8_1.png){:class="img-responsive"}
    

**GPU only for `RidgeRegressor`**


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from time import time
import warnings
warnings.filterwarnings('ignore')

try:
    from nnetsauce import CustomRegressor
except ImportError:
    print("ERROR: nnetsauce is not installed.")
    exit(1)

try:
    import mlsauce as ms
    MLSAUCE_AVAILABLE = True
except ImportError:
    print("WARNING: mlsauce is not installed.")
    MLSAUCE_AVAILABLE = False
    exit(1)

print("="*80)
print("LARGE-SCALE GPU BENCHMARK")
print("Simulating the PDF example: 10,000 samples × 100 features")
print("="*80)

# Configuration matching the PDF's large-scale example
np.random.seed(42)

# Test different dataset sizes
dataset_configs = [
    (1000, 50, "Small: 1K samples × 50 features"),
    (5000, 100, "Medium: 5K samples × 100 features"),
    (10000, 100, "Large: 10K samples × 100 features (PDF example)"),
    (20000, 150, "XLarge: 20K samples × 150 features"),
]

n_hidden = 100  # Fixed hidden features

results = {
    'config': [],
    'sklearn_cpu': [],
    'mlsauce_cpu': [],
    'mlsauce_gpu': [],
    'gpu_speedup': [],
}

print("\nRunning benchmarks...\n")

for n_samples, n_features, description in dataset_configs:
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    # Generate synthetic data
    print(f"Generating {n_samples:,} samples with {n_features} features...")
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results['config'].append(description)

    # 1. sklearn Ridge (CPU)
    print("\n1. Testing sklearn Ridge (CPU)...")
    start = time()
    model_sklearn = CustomRegressor(
        obj=Ridge(alpha=1.0),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
    )
    model_sklearn.fit(X_train, y_train)
    _ = model_sklearn.predict(X_test)
    elapsed_sklearn = time() - start
    results['sklearn_cpu'].append(elapsed_sklearn)
    print(f"   Time: {elapsed_sklearn:.3f}s")

    # 2. mlsauce Ridge (CPU)
    print("2. Testing mlsauce Ridge (CPU)...")
    start = time()
    model_mlsauce_cpu = CustomRegressor(
        obj=ms.RidgeRegressor(reg_lambda=1.0, backend="cpu"),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
        backend='cpu',
    )
    model_mlsauce_cpu.fit(X_train, y_train)
    _ = model_mlsauce_cpu.predict(X_test)
    elapsed_ml_cpu = time() - start
    results['mlsauce_cpu'].append(elapsed_ml_cpu)
    print(f"   Time: {elapsed_ml_cpu:.3f}s")
    print(f"   Speedup vs sklearn: {elapsed_sklearn/elapsed_ml_cpu:.2f}x")

    # 3. mlsauce Ridge (GPU)
    print("3. Testing mlsauce Ridge (GPU)...")
    try:
        start = time()
        model_mlsauce_gpu = CustomRegressor(
            obj=ms.RidgeRegressor(reg_lambda=1.0, backend="gpu"),
            n_hidden_features=n_hidden,
            activation_name='relu',
            nodes_sim='sobol',
            backend='cpu'
        )
        model_mlsauce_gpu.fit(X_train, y_train)
        _ = model_mlsauce_gpu.predict(X_test)
        elapsed_ml_gpu = time() - start
        results['mlsauce_gpu'].append(elapsed_ml_gpu)

        speedup = elapsed_sklearn / elapsed_ml_gpu
        results['gpu_speedup'].append(speedup)

        print(f"   Time: {elapsed_ml_gpu:.3f}s")
        print(f"   Speedup vs sklearn: {speedup:.2f}x")
        print(f"   Speedup vs mlsauce CPU: {elapsed_ml_cpu/elapsed_ml_gpu:.2f}x")

        if speedup > 1.0:
            print(f"   ✓ GPU IS FASTER!")
        else:
            print(f"   ✗ GPU overhead still dominates")

    except Exception as e:
        print(f"   FAILED: {str(e)[:60]}...")
        results['mlsauce_gpu'].append(None)
        results['gpu_speedup'].append(None)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Absolute times
x_pos = np.arange(len(results['config']))
width = 0.25

ax1.bar(x_pos - width, results['sklearn_cpu'], width,
        label='sklearn Ridge (CPU)', color='#2E86AB', alpha=0.8)
ax1.bar(x_pos, results['mlsauce_cpu'], width,
        label='mlsauce Ridge (CPU)', color='#F18F01', alpha=0.8)

gpu_times = [t if t is not None else 0 for t in results['mlsauce_gpu']]
ax1.bar(x_pos + width, gpu_times, width,
        label='mlsauce Ridge (GPU)', color='#C73E1D', alpha=0.8)

ax1.set_ylabel('Training Time (seconds)', fontsize=12)
ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([c.split(':')[0] for c in results['config']], rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(results['sklearn_cpu']):
    ax1.text(i - width, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(results['mlsauce_cpu']):
    ax1.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(gpu_times):
    if v > 0:
        ax1.text(i + width, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

# Plot 2: Speedup factors
valid_speedups = [s if s is not None else 0 for s in results['gpu_speedup']]
colors = ['green' if s > 1.0 else 'red' for s in valid_speedups]

bars = ax2.bar(x_pos, valid_speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even (1.0x)')
ax2.set_ylabel('GPU Speedup vs sklearn CPU', fontsize=12)
ax2.set_title('GPU Speedup Factor (>1.0 = GPU wins)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([c.split(':')[0] for c in results['config']], rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, valid_speedups)):
    if val > 0:
        label = f'{val:.2f}x'
        y_pos = val + 0.05 if val > 1.0 else val - 0.1
        ax2.text(i, y_pos, label, ha='center', va='bottom' if val > 1.0 else 'top',
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('large_scale_gpu_benchmark.png', dpi=150, bbox_inches='tight')
print("\n\nSaved: large_scale_gpu_benchmark.png")

# Summary table
print("\n" + "="*100)
print("BENCHMARK SUMMARY")
print("="*100)

print(f"\n{'Configuration':<40} {'sklearn CPU':<12} {'mlsauce CPU':<12} {'mlsauce GPU':<12} {'GPU Speedup':<12}")
print("-"*100)

for i, config in enumerate(results['config']):
    sk = results['sklearn_cpu'][i]
    ml_cpu = results['mlsauce_cpu'][i]
    ml_gpu = results['mlsauce_gpu'][i]
    speedup = results['gpu_speedup'][i]

    gpu_str = f"{ml_gpu:.3f}s" if ml_gpu else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

    print(f"{config:<40} {sk:<12.3f}s {ml_cpu:<12.3f}s {gpu_str:<12} {speedup_str:<12}")

print("-"*100)

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

gpu_wins = [i for i, s in enumerate(results['gpu_speedup']) if s and s > 1.0]
if gpu_wins:
    print(f"\n✓ GPU becomes advantageous at:")
    for i in gpu_wins:
        speedup = results['gpu_speedup'][i]
        print(f"  - {results['config'][i]}: {speedup:.2f}x speedup")
else:
    print("\n✗ GPU did not outperform CPU in any configuration tested")
    print("  Reasons:")
    print("  - GPU overhead (data transfer, compilation) > computation time")
    print("  - Dataset still too small to amortize GPU setup costs")

print("\n💡 For GPU to be beneficial, you typically need:")
print("  1. Dataset: 50,000+ samples (PDF showed 1M+ data points)")
print("  2. Multiple iterations (cross-validation, hyperparameter tuning)")
print("  3. Batch predictions (forecasting 100+ time series simultaneously)")
print("  4. High-dimensional features (200+)")
print("  5. Deep architectures (multiple hidden layers)")

print("\n" + "="*100)
```

    ================================================================================
    LARGE-SCALE GPU BENCHMARK
    Simulating the PDF example: 10,000 samples × 100 features
    ================================================================================
    
    Running benchmarks...
    
    
    ================================================================================
    Small: 1K samples × 50 features
    ================================================================================
    Generating 1,000 samples with 50 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 0.135s
    2. Testing mlsauce Ridge (CPU)...
       Time: 0.142s
       Speedup vs sklearn: 0.95x
    3. Testing mlsauce Ridge (GPU)...
       Time: 0.136s
       Speedup vs sklearn: 0.99x
       Speedup vs mlsauce CPU: 1.04x
       ✗ GPU overhead still dominates
    
    ================================================================================
    Medium: 5K samples × 100 features
    ================================================================================
    Generating 5,000 samples with 100 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 0.805s
    2. Testing mlsauce Ridge (CPU)...
       Time: 0.576s
       Speedup vs sklearn: 1.40x
    3. Testing mlsauce Ridge (GPU)...
       Time: 0.514s
       Speedup vs sklearn: 1.56x
       Speedup vs mlsauce CPU: 1.12x
       ✓ GPU IS FASTER!
    
    ================================================================================
    Large: 10K samples × 100 features (PDF example)
    ================================================================================
    Generating 10,000 samples with 100 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 1.025s
    2. Testing mlsauce Ridge (CPU)...
       Time: 0.963s
       Speedup vs sklearn: 1.06x
    3. Testing mlsauce Ridge (GPU)...
       Time: 0.966s
       Speedup vs sklearn: 1.06x
       Speedup vs mlsauce CPU: 1.00x
       ✓ GPU IS FASTER!
    
    ================================================================================
    XLarge: 20K samples × 150 features
    ================================================================================
    Generating 20,000 samples with 150 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 2.896s
    2. Testing mlsauce Ridge (CPU)...
       Time: 2.875s
       Speedup vs sklearn: 1.01x
    3. Testing mlsauce Ridge (GPU)...
       Time: 3.563s
       Speedup vs sklearn: 0.81x
       Speedup vs mlsauce CPU: 0.81x
       ✗ GPU overhead still dominates
    
    
    Saved: large_scale_gpu_benchmark.png
    
    ====================================================================================================
    BENCHMARK SUMMARY
    ====================================================================================================
    
    Configuration                            sklearn CPU  mlsauce CPU  mlsauce GPU  GPU Speedup 
    ----------------------------------------------------------------------------------------------------
    Small: 1K samples × 50 features          0.135       s 0.142       s 0.136s       0.99x       
    Medium: 5K samples × 100 features        0.805       s 0.576       s 0.514s       1.56x       
    Large: 10K samples × 100 features (PDF example) 1.025       s 0.963       s 0.966s       1.06x       
    XLarge: 20K samples × 150 features       2.896       s 2.875       s 3.563s       0.81x       
    ----------------------------------------------------------------------------------------------------
    
    ====================================================================================================
    KEY INSIGHTS
    ====================================================================================================
    
    ✓ GPU becomes advantageous at:
      - Medium: 5K samples × 100 features: 1.56x speedup
      - Large: 10K samples × 100 features (PDF example): 1.06x speedup
    
    💡 For GPU to be beneficial, you typically need:
      1. Dataset: 50,000+ samples (PDF showed 1M+ data points)
      2. Multiple iterations (cross-validation, hyperparameter tuning)
      3. Batch predictions (forecasting 100+ time series simultaneously)
      4. High-dimensional features (200+)
      5. Deep architectures (multiple hidden layers)
    
    ====================================================================================================



    
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_11_1.png){:class="img-responsive"}
    


**GPU also for `CustomRegressor`**


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from time import time
import warnings
warnings.filterwarnings('ignore')

try:
    from nnetsauce import CustomRegressor
except ImportError:
    print("ERROR: nnetsauce is not installed.")
    exit(1)

try:
    import mlsauce as ms
    MLSAUCE_AVAILABLE = True
except ImportError:
    print("WARNING: mlsauce is not installed.")
    MLSAUCE_AVAILABLE = False
    exit(1)

print("="*80)
print("LARGE-SCALE GPU BENCHMARK")
print("Simulating the PDF example: 10,000 samples × 100 features")
print("="*80)

# Configuration matching the PDF's large-scale example
np.random.seed(42)

# Test different dataset sizes
dataset_configs = [
    (1000, 50, "Small: 1K samples × 50 features"),
    (5000, 100, "Medium: 5K samples × 100 features"),
    (10000, 100, "Large: 10K samples × 100 features (PDF example)"),
    (20000, 150, "XLarge: 20K samples × 150 features"),
]

n_hidden = 100  # Fixed hidden features

results = {
    'config': [],
    'sklearn_cpu': [],
    'mlsauce_cpu': [],
    'mlsauce_gpu': [],
    'gpu_speedup': [],
}

print("\nRunning benchmarks...\n")

for n_samples, n_features, description in dataset_configs:
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    # Generate synthetic data
    print(f"Generating {n_samples:,} samples with {n_features} features...")
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results['config'].append(description)

    # 1. sklearn Ridge (CPU)
    print("\n1. Testing sklearn Ridge (CPU)...")
    start = time()
    model_sklearn = CustomRegressor(
        obj=Ridge(alpha=1.0),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
    )
    model_sklearn.fit(X_train, y_train)
    _ = model_sklearn.predict(X_test)
    elapsed_sklearn = time() - start
    results['sklearn_cpu'].append(elapsed_sklearn)
    print(f"   Time: {elapsed_sklearn:.3f}s")

    # 2. mlsauce Ridge (CPU)
    print("2. Testing mlsauce Ridge (CPU)...")
    start = time()
    model_mlsauce_cpu = CustomRegressor(
        obj=ms.RidgeRegressor(reg_lambda=1.0, backend="cpu"),
        n_hidden_features=n_hidden,
        activation_name='relu',
        nodes_sim='sobol',
        backend='cpu',
    )
    model_mlsauce_cpu.fit(X_train, y_train)
    _ = model_mlsauce_cpu.predict(X_test)
    elapsed_ml_cpu = time() - start
    results['mlsauce_cpu'].append(elapsed_ml_cpu)
    print(f"   Time: {elapsed_ml_cpu:.3f}s")
    print(f"   Speedup vs sklearn: {elapsed_sklearn/elapsed_ml_cpu:.2f}x")

    # 3. mlsauce Ridge (GPU)
    print("3. Testing mlsauce Ridge (GPU)...")
    try:
        start = time()
        model_mlsauce_gpu = CustomRegressor(
            obj=ms.RidgeRegressor(reg_lambda=1.0, backend="gpu"),
            n_hidden_features=n_hidden,
            activation_name='relu',
            nodes_sim='sobol',
            backend='gpu'
        )
        model_mlsauce_gpu.fit(X_train, y_train)
        _ = model_mlsauce_gpu.predict(X_test)
        elapsed_ml_gpu = time() - start
        results['mlsauce_gpu'].append(elapsed_ml_gpu)

        speedup = elapsed_sklearn / elapsed_ml_gpu
        results['gpu_speedup'].append(speedup)

        print(f"   Time: {elapsed_ml_gpu:.3f}s")
        print(f"   Speedup vs sklearn: {speedup:.2f}x")
        print(f"   Speedup vs mlsauce CPU: {elapsed_ml_cpu/elapsed_ml_gpu:.2f}x")

        if speedup > 1.0:
            print(f"   ✓ GPU IS FASTER!")
        else:
            print(f"   ✗ GPU overhead still dominates")

    except Exception as e:
        print(f"   FAILED: {str(e)[:60]}...")
        results['mlsauce_gpu'].append(None)
        results['gpu_speedup'].append(None)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Absolute times
x_pos = np.arange(len(results['config']))
width = 0.25

ax1.bar(x_pos - width, results['sklearn_cpu'], width,
        label='sklearn Ridge (CPU)', color='#2E86AB', alpha=0.8)
ax1.bar(x_pos, results['mlsauce_cpu'], width,
        label='mlsauce Ridge (CPU)', color='#F18F01', alpha=0.8)

gpu_times = [t if t is not None else 0 for t in results['mlsauce_gpu']]
ax1.bar(x_pos + width, gpu_times, width,
        label='mlsauce Ridge (GPU)', color='#C73E1D', alpha=0.8)

ax1.set_ylabel('Training Time (seconds)', fontsize=12)
ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([c.split(':')[0] for c in results['config']], rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(results['sklearn_cpu']):
    ax1.text(i - width, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(results['mlsauce_cpu']):
    ax1.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(gpu_times):
    if v > 0:
        ax1.text(i + width, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

# Plot 2: Speedup factors
valid_speedups = [s if s is not None else 0 for s in results['gpu_speedup']]
colors = ['green' if s > 1.0 else 'red' for s in valid_speedups]

bars = ax2.bar(x_pos, valid_speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even (1.0x)')
ax2.set_ylabel('GPU Speedup vs sklearn CPU', fontsize=12)
ax2.set_title('GPU Speedup Factor (>1.0 = GPU wins)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([c.split(':')[0] for c in results['config']], rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, valid_speedups)):
    if val > 0:
        label = f'{val:.2f}x'
        y_pos = val + 0.05 if val > 1.0 else val - 0.1
        ax2.text(i, y_pos, label, ha='center', va='bottom' if val > 1.0 else 'top',
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('large_scale_gpu_benchmark.png', dpi=150, bbox_inches='tight')
print("\n\nSaved: large_scale_gpu_benchmark.png")

# Summary table
print("\n" + "="*100)
print("BENCHMARK SUMMARY")
print("="*100)

print(f"\n{'Configuration':<40} {'sklearn CPU':<12} {'mlsauce CPU':<12} {'mlsauce GPU':<12} {'GPU Speedup':<12}")
print("-"*100)

for i, config in enumerate(results['config']):
    sk = results['sklearn_cpu'][i]
    ml_cpu = results['mlsauce_cpu'][i]
    ml_gpu = results['mlsauce_gpu'][i]
    speedup = results['gpu_speedup'][i]

    gpu_str = f"{ml_gpu:.3f}s" if ml_gpu else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

    print(f"{config:<40} {sk:<12.3f}s {ml_cpu:<12.3f}s {gpu_str:<12} {speedup_str:<12}")

print("-"*100)

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

gpu_wins = [i for i, s in enumerate(results['gpu_speedup']) if s and s > 1.0]
if gpu_wins:
    print(f"\n✓ GPU becomes advantageous at:")
    for i in gpu_wins:
        speedup = results['gpu_speedup'][i]
        print(f"  - {results['config'][i]}: {speedup:.2f}x speedup")
else:
    print("\n✗ GPU did not outperform CPU in any configuration tested")
    print("  Reasons:")
    print("  - GPU overhead (data transfer, compilation) > computation time")
    print("  - Dataset still too small to amortize GPU setup costs")

print("\n💡 For GPU to be beneficial, you typically need:")
print("  1. Dataset: 50,000+ samples (PDF showed 1M+ data points)")
print("  2. Multiple iterations (cross-validation, hyperparameter tuning)")
print("  3. Batch predictions (forecasting 100+ time series simultaneously)")
print("  4. High-dimensional features (200+)")
print("  5. Deep architectures (multiple hidden layers)")

print("\n" + "="*100)
```

    ================================================================================
    LARGE-SCALE GPU BENCHMARK
    Simulating the PDF example: 10,000 samples × 100 features
    ================================================================================
    
    Running benchmarks...
    
    
    ================================================================================
    Small: 1K samples × 50 features
    ================================================================================
    Generating 1,000 samples with 50 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 0.116s
    2. Testing mlsauce Ridge (CPU)...
       Time: 0.094s
       Speedup vs sklearn: 1.24x
    3. Testing mlsauce Ridge (GPU)...
       Time: 0.095s
       Speedup vs sklearn: 1.22x
       Speedup vs mlsauce CPU: 0.99x
       ✓ GPU IS FASTER!
    
    ================================================================================
    Medium: 5K samples × 100 features
    ================================================================================
    Generating 5,000 samples with 100 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 0.687s
    2. Testing mlsauce Ridge (CPU)...
       Time: 1.099s
       Speedup vs sklearn: 0.63x
    3. Testing mlsauce Ridge (GPU)...
       Time: 1.391s
       Speedup vs sklearn: 0.49x
       Speedup vs mlsauce CPU: 0.79x
       ✗ GPU overhead still dominates
    
    ================================================================================
    Large: 10K samples × 100 features (PDF example)
    ================================================================================
    Generating 10,000 samples with 100 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 2.107s
    2. Testing mlsauce Ridge (CPU)...
       Time: 1.991s
       Speedup vs sklearn: 1.06x
    3. Testing mlsauce Ridge (GPU)...
       Time: 2.817s
       Speedup vs sklearn: 0.75x
       Speedup vs mlsauce CPU: 0.71x
       ✗ GPU overhead still dominates
    
    ================================================================================
    XLarge: 20K samples × 150 features
    ================================================================================
    Generating 20,000 samples with 150 features...
    
    1. Testing sklearn Ridge (CPU)...
       Time: 5.251s
    2. Testing mlsauce Ridge (CPU)...
       Time: 3.100s
       Speedup vs sklearn: 1.69x
    3. Testing mlsauce Ridge (GPU)...
       Time: 3.101s
       Speedup vs sklearn: 1.69x
       Speedup vs mlsauce CPU: 1.00x
       ✓ GPU IS FASTER!
    
    
    Saved: large_scale_gpu_benchmark.png
    
    ====================================================================================================
    BENCHMARK SUMMARY
    ====================================================================================================
    
    Configuration                            sklearn CPU  mlsauce CPU  mlsauce GPU  GPU Speedup 
    ----------------------------------------------------------------------------------------------------
    Small: 1K samples × 50 features          0.116       s 0.094       s 0.095s       1.22x       
    Medium: 5K samples × 100 features        0.687       s 1.099       s 1.391s       0.49x       
    Large: 10K samples × 100 features (PDF example) 2.107       s 1.991       s 2.817s       0.75x       
    XLarge: 20K samples × 150 features       5.251       s 3.100       s 3.101s       1.69x       
    ----------------------------------------------------------------------------------------------------
    
    ====================================================================================================
    KEY INSIGHTS
    ====================================================================================================
    
    ✓ GPU becomes advantageous at:
      - Small: 1K samples × 50 features: 1.22x speedup
      - XLarge: 20K samples × 150 features: 1.69x speedup
    
    💡 For GPU to be beneficial, you typically need:
      1. Dataset: 50,000+ samples (PDF showed 1M+ data points)
      2. Multiple iterations (cross-validation, hyperparameter tuning)
      3. Batch predictions (forecasting 100+ time series simultaneously)
      4. High-dimensional features (200+)
      5. Deep architectures (multiple hidden layers)
    
    ====================================================================================================
    
![image-title-here]({{base}}/images/2026-01-29/2026-01-29-Overfitting-CustomRegressor_13_1.png){:class="img-responsive"}
    

