---
layout: post
title: "Overfitting Random Fourier Features: Universal Approximation Property"
description: "A simple example of overfitting Random Fourier Features"
date: 2025-12-13
categories: [R, Python]
comments: true
---


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Test different numbers of Random Fourier Features
n_components_list = [5, 10, 25, 50, 100, 200]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

train_errors = []
test_errors = []

for idx, n_components in enumerate(n_components_list):
    # Create Random Fourier Features transformer
    rff = RBFSampler(n_components=n_components, gamma=1.0, random_state=42)

    # Transform training data
    X_train_rff = rff.fit_transform(X_train)
    X_test_rff = rff.transform(X_test)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train_rff, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_rff)
    y_test_pred = model.predict(X_test_rff)

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
    ax.set_title(f'RFF Components: {n_components}\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rff_universal_approximation.png', dpi=150, bbox_inches='tight')
print("Saved: rff_universal_approximation.png")

# Create a second figure showing error vs model capacity
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot MSE vs number of components
ax1.plot(n_components_list, train_errors, 'o-', linewidth=2, markersize=8, label='Training MSE')
ax1.plot(n_components_list, test_errors, 's-', linewidth=2, markersize=8, label='Test MSE')
ax1.set_xlabel('Number of RFF Components (Model Capacity)', fontsize=12)
ax1.set_ylabel('Mean Squared Error', fontsize=12)
ax1.set_title('Universal Approximation: Error vs Model Capacity', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Demonstrate overfitting with very high capacity
n_overfit = 500
rff_overfit = RBFSampler(n_components=n_overfit, gamma=1.0, random_state=42)
X_train_overfit = rff_overfit.fit_transform(X_train)
X_test_overfit = rff_overfit.transform(X_test)

model_overfit = LinearRegression()
model_overfit.fit(X_train_overfit, y_train)
y_train_overfit = model_overfit.predict(X_train_overfit)
y_test_overfit = model_overfit.predict(X_test_overfit)

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
plt.savefig('rff_error_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: rff_error_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("UNIVERSAL APPROXIMATION PROPERTY DEMONSTRATION")
print("="*60)
print("\nModel: Random Fourier Features + Linear Regression")
print(f"Training samples: {n_train}")
print(f"Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)")
print("\n" + "-"*60)
print(f"{'Components':<12} {'Train MSE':<15} {'Test MSE':<15} {'Ratio':<10}")
print("-"*60)

for n_comp, train_err, test_err in zip(n_components_list, train_errors, test_errors):
    ratio = test_err / train_err if train_err > 0 else float('inf')
    print(f"{n_comp:<12} {train_err:<15.6f} {test_err:<15.6f} {ratio:<10.2f}")

print("-"*60)
print(f"\n✓ As model capacity increases, training error decreases")
print(f"✓ With sufficient capacity, the model can approximate any continuous function")
print(f"✓ Training MSE improved from {train_errors[0]:.4f} to {train_errors[-1]:.4f}")
print(f"✓ This demonstrates the universal approximation property empirically")
print("="*60)

plt.show()
```

    Saved: rff_universal_approximation.png
    Saved: rff_error_analysis.png
    
    ============================================================
    UNIVERSAL APPROXIMATION PROPERTY DEMONSTRATION
    ============================================================
    
    Model: Random Fourier Features + Linear Regression
    Training samples: 50
    Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)
    
    ------------------------------------------------------------
    Components   Train MSE       Test MSE        Ratio     
    ------------------------------------------------------------
    5            0.185395        0.218596        1.18      
    10           0.054406        0.156741        2.88      
    25           0.013543        0.196464        14.51     
    50           0.013379        0.201987        15.10     
    100          0.013115        0.209879        16.00     
    200          0.012373        0.308260        24.91     
    ------------------------------------------------------------
    
    ✓ As model capacity increases, training error decreases
    ✓ With sufficient capacity, the model can approximate any continuous function
    ✓ Training MSE improved from 0.1854 to 0.0124
    ✓ This demonstrates the universal approximation property empirically
    ============================================================



    
![image-title-here]({{base}}/images/2025-12-13/2025-12-13-Overfitting-Random-Fourier-Features_0_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-12-13/2025-12-13-Overfitting-Random-Fourier-Features_0_2.png){:class="img-responsive"}
    



```python

```


```python

```


```python
%load_ext rpy2.ipython
```

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython



```r
%%R

#install.packages("ggplot2")
install.packages("gridExtra")
```


    Installing package into ‘/usr/local/lib/R/site-library’
    (as ‘lib’ is unspecified)
    trying URL 'https://cran.rstudio.com/src/contrib/gridExtra_2.3.tar.gz'
    Content type 'application/x-gzip' length 1062844 bytes (1.0 MB)
    ==================================================
    downloaded 1.0 MB
    
    
    The downloaded source packages are in
    	‘/tmp/Rtmpc3H0eO/downloaded_packages’




```r
%%R

# Load required libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
  library(dplyr)
})

# Set random seed for reproducibility
set.seed(42)

# Define a complex target function
target_function <- function(x) {
  sin(2 * pi * x) + 0.5 * sin(8 * pi * x) + 0.3 * cos(5 * pi * x)
}

# Generate training and test data
n_train <- 50
n_test <- 200

X_train <- matrix(runif(n_train, 0, 1), ncol = 1)
y_train <- target_function(X_train) + rnorm(n_train, 0, 0.1)
X_test <- matrix(seq(0, 1, length.out = n_test), ncol = 1)
y_test <- target_function(X_test)

# Test different numbers of Random Fourier Features
n_components_list <- c(5, 10, 25, 50, 100, 200)

# Function to create Random Fourier Features
create_rff <- function(X, n_components, gamma = 1.0) {
  n_features <- ncol(X)
  n_samples <- nrow(X)

  # Sample random weights from normal distribution
  W <- matrix(rnorm(n_features * n_components, 0, sqrt(2 * gamma)),
              nrow = n_features, ncol = n_components)
  b <- matrix(runif(n_components, 0, 2 * pi), nrow = 1, ncol = n_components)

  # Transform features (using both sin and cos like sklearn's RBFSampler)
  Z_cos <- sqrt(2 / n_components) * cos(X %*% W + matrix(1, nrow = n_samples) %*% b)
  Z_sin <- sqrt(2 / n_components) * sin(X %*% W + matrix(1, nrow = n_samples) %*% b)

  # Combine cos and sin features
  list(features = cbind(Z_cos, Z_sin), W = W, b = b)
}

# Initialize storage
train_errors <- numeric(length(n_components_list))
test_errors <- numeric(length(n_components_list))
plots_list <- list()

# Create plots for different numbers of RFF components
for (idx in seq_along(n_components_list)) {
  n_components <- n_components_list[idx]

  # Create Random Fourier Features
  rff_train <- create_rff(X_train, n_components, gamma = 1.0)
  X_train_rff <- rff_train$features

  # Transform test data
  n_samples_test <- nrow(X_test)
  Z_cos_test <- sqrt(2 / n_components) * cos(X_test %*% rff_train$W + matrix(1, nrow = n_samples_test) %*% rff_train$b)
  Z_sin_test <- sqrt(2 / n_components) * sin(X_test %*% rff_train$W + matrix(1, nrow = n_samples_test) %*% rff_train$b)
  X_test_rff <- cbind(Z_cos_test, Z_sin_test)

  # Train Linear Regression
  model <- lm(y_train ~ ., data = data.frame(X_train_rff))

  # Make predictions
  y_train_pred <- predict(model, newdata = data.frame(X_train_rff))
  y_test_pred <- predict(model, newdata = data.frame(X_test_rff))

  # Calculate errors
  train_mse <- mean((y_train - y_train_pred)^2)
  test_mse <- mean((y_test - y_test_pred)^2)

  train_errors[idx] <- train_mse
  test_errors[idx] <- test_mse

  # Create plot data
  train_df <- data.frame(x = X_train, y = y_train, type = "Training data")
  test_df <- data.frame(x = X_test, y_true = y_test, y_pred = y_test_pred)

  # Create plot
  p <- ggplot() +
    geom_point(data = train_df, aes(x = x, y = y), color = "red", size = 2, alpha = 0.6) +
    geom_line(data = test_df, aes(x = x, y = y_true), color = "blue", linewidth = 1) +
    geom_line(data = test_df, aes(x = x, y = y_pred), color = "green", linewidth = 1, linetype = "dashed") +
    labs(title = sprintf("RFF Components: %d\nTrain MSE: %.4f, Test MSE: %.4f",
                         n_components, train_mse, test_mse),
         x = "x", y = "y") +
    theme_minimal() +
    theme(plot.title = element_text(size = 10))

  plots_list[[idx]] <- p
}

# Arrange plots in grid and save
grid_plot <- grid.arrange(grobs = plots_list, nrow = 2, ncol = 3)
ggsave("rff_universal_approximation.png", grid_plot, width = 15, height = 10, dpi = 150)
cat("Saved: rff_universal_approximation.png\n")

# Create error analysis plot
error_df <- data.frame(
  n_components = rep(n_components_list, 2),
  mse = c(train_errors, test_errors),
  type = rep(c("Training MSE", "Test MSE"), each = length(n_components_list))
)

# Plot 1: Error vs Model Capacity
p1 <- ggplot(error_df, aes(x = n_components, y = mse, color = type, shape = type)) +
  geom_line(linewidth = 1) +  # Updated from size to linewidth
  geom_point(size = 3) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Number of RFF Components (Model Capacity)",
       y = "Mean Squared Error",
       title = "Universal Approximation: Error vs Model Capacity") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

# Demonstrate overfitting with very high capacity
n_overfit <- 500
rff_overfit <- create_rff(X_train, n_overfit, gamma = 1.0)
X_train_overfit <- rff_overfit$features

# Transform test data
Z_cos_test_overfit <- sqrt(2 / n_overfit) * cos(X_test %*% rff_overfit$W + matrix(1, nrow = n_test) %*% rff_overfit$b)
Z_sin_test_overfit <- sqrt(2 / n_overfit) * sin(X_test %*% rff_overfit$W + matrix(1, nrow = n_test) %*% rff_overfit$b)
X_test_overfit <- cbind(Z_cos_test_overfit, Z_sin_test_overfit)

# Train model
model_overfit <- lm(y_train ~ ., data = data.frame(X_train_overfit))

# Make predictions
y_train_overfit <- predict(model_overfit, newdata = data.frame(X_train_overfit))
y_test_overfit <- predict(model_overfit, newdata = data.frame(X_test_overfit))

# Calculate errors
train_mse_overfit <- mean((y_train - y_train_overfit)^2)
test_mse_overfit <- mean((y_test - y_test_overfit)^2)

# Plot 2: Overfitting example
overfit_df <- data.frame(
  x = X_test,
  y_true = y_test,
  y_pred = y_test_overfit
)
train_df_overfit <- data.frame(x = X_train, y = y_train)

p2 <- ggplot() +
  geom_point(data = train_df_overfit, aes(x = x, y = y),
             color = "red", size = 2, alpha = 0.7) +
  geom_line(data = overfit_df, aes(x = x, y = y_true),
            color = "blue", linewidth = 1.5) +
  geom_line(data = overfit_df, aes(x = x, y = y_pred),
            color = "green", linewidth = 1, linetype = "dashed") +
  labs(title = sprintf("High Capacity Model (Overfitting)\nTrain MSE: %.4f, Test MSE: %.4f",
                       train_mse_overfit, test_mse_overfit),
       x = "x", y = "y") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Combine and save error analysis plot
combined_plot <- grid.arrange(p1, p2, ncol = 2)
ggsave("rff_error_analysis.png", combined_plot, width = 14, height = 5, dpi = 150)
cat("Saved: rff_error_analysis.png\n")

# Print summary statistics
cat("\n", strrep("=", 60), "\n", sep = "")
cat("UNIVERSAL APPROXIMATION PROPERTY DEMONSTRATION\n")
cat(strrep("=", 60), "\n")
cat("\nModel: Random Fourier Features + Linear Regression\n")
cat("Training samples:", n_train, "\n")
cat("Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)\n")
cat("\n", strrep("-", 60), "\n", sep = "")
cat(sprintf("%-12s %-15s %-15s %-10s\n",
            "Components", "Train MSE", "Test MSE", "Ratio"))
cat(strrep("-", 60), "\n")

for (i in seq_along(n_components_list)) {
  n_comp <- n_components_list[i]
  train_err <- train_errors[i]
  test_err <- test_errors[i]
  ratio <- ifelse(train_err > 0, test_err / train_err, Inf)
  cat(sprintf("%-12d %-15.6f %-15.6f %-10.2f\n",
              n_comp, train_err, test_err, ratio))
}

cat(strrep("-", 60), "\n")
cat("\n✓ As model capacity increases, training error decreases\n")
cat("✓ With sufficient capacity, the model can approximate any continuous function\n")
cat(sprintf("✓ Training MSE improved from %.4f to %.4f\n",
            train_errors[1], train_errors[length(train_errors)]))
cat("✓ This demonstrates the universal approximation property empirically\n")
cat(strrep("=", 60), "\n")

# Display plots
invisible(print(grid_plot))
invisible(print(combined_plot))
```

    Saved: rff_universal_approximation.png
    Saved: rff_error_analysis.png
    
    ============================================================
    UNIVERSAL APPROXIMATION PROPERTY DEMONSTRATION
    ============================================================ 
    
    Model: Random Fourier Features + Linear Regression
    Training samples: 50 
    Target function: sin(2πx) + 0.5·sin(8πx) + 0.3·cos(5πx)
    
    ------------------------------------------------------------
    Components   Train MSE       Test MSE        Ratio     
    ------------------------------------------------------------ 
    5            0.058670        0.098736        1.68      
    10           0.058323        0.098800        1.69      
    25           0.053089        0.108782        2.05      
    50           0.052229        0.108501        2.08      
    100          0.040079        0.089596        2.24      
    200          0.039299        0.088857        2.26      
    ------------------------------------------------------------ 
    
    ✓ As model capacity increases, training error decreases
    ✓ With sufficient capacity, the model can approximate any continuous function
    ✓ Training MSE improved from 0.0587 to 0.0393
    ✓ This demonstrates the universal approximation property empirically
    ============================================================ 
    TableGrob (2 x 3) "arrange": 6 grobs
      z     cells    name           grob
    1 1 (1-1,1-1) arrange gtable[layout]
    2 2 (1-1,2-2) arrange gtable[layout]
    3 3 (1-1,3-3) arrange gtable[layout]
    4 4 (2-2,1-1) arrange gtable[layout]
    5 5 (2-2,2-2) arrange gtable[layout]
    6 6 (2-2,3-3) arrange gtable[layout]
    TableGrob (1 x 2) "arrange": 2 grobs
      z     cells    name           grob
    1 1 (1-1,1-1) arrange gtable[layout]
    2 2 (1-1,2-2) arrange gtable[layout]



    
![image-title-here]({{base}}/images/2025-12-13/2025-12-13-Overfitting-Random-Fourier-Features_5_1.png){:class="img-responsive"}
    

