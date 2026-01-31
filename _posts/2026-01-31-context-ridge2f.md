---
layout: post
title: "Enhancing Time Series Forecasting (ahead::ridge2f) with Attention-Based Context Vectors (ahead::contextridge2f)"
description: "Explain the new ahead::contextridge2f function"
date: 2026-01-31
categories: R
comments: true
---

## Introduction

In this post, I'll introduce [`ahead::contextridge2f()`](https://github.com/Techtonique/ahead), a novel forecasting function that combines doubly-constrained Random Vector Functional Link (RVFL) networks with attention-based context vectors with the aim to  improve prediction accuracy.

## The Core Idea

The key insight is simple but powerful: **not all past observations are equally relevant for predicting the future**. An attention mechanism learns to assign different weights to historical values based on their relevance to the current time point.

Instead of treating the time series as a simple sequence, we compute **context vectors**—weighted summaries of the historical data where the weights are determined by an attention mechanism. These context vectors then serve as external regressors in a doubly-constrained Random Vector Functional Link (RVFL) network.

## What is Doubly-Constrained RVFL?

RVFL networks, as implemented in `ridge2f()` (Moudiki et al., 2018), are a type of randomized neural network that:

1. **Use random or quasi-random hidden layer weights** that are not trained (computational efficiency)
2. **Include direct input-to-output connections** (preserves linear relationships)
3. **Apply dual constraints** via ridge penalties on both:
   - Direct connections (λ₁)
   - Hidden layer outputs (λ₂)

This architecture combines the expressiveness of neural networks with the simplicity and speed of linear models, making it particularly well-suited for time series forecasting.

## What Are Context Vectors?

A context vector at time `t` is a weighted sum of all previous observations:

```
context[t] = Σ(attention_weight[t,j] × series[j]) for j ≤ t
```

Where `attention_weight[t,j]` represents how much time point `j` contributes to our understanding of time `t`.

Different attention mechanisms produce different weighting schemes:

- **Exponential**: Recent observations get exponentially higher weights (controlled by `decay_factor`)
- **Gaussian**: Weights decay according to temporal distance with a Gaussian kernel
- **Value-based**: Points with similar values to the current observation get higher weights
- **Hybrid**: Combines temporal proximity and value similarity
- **Cosine**: Uses cosine similarity between local windows
- And several others...

## The Function: `ahead::contextridge2f()`

Here's the implementation:

```r
contextridge2f <- function(y,
                           h = 5L,
                           split_fraction = 0.8,
                           attention_type = "exponential",
                           window_size = 3,
                           decay_factor = 5.0,
                           temperature = 1.0,
                           sigma = 1.0,
                           sensitivity = 1.0,
                           alpha = 0.5,
                           beta = 0.5,
                           ...)
{
  ctx_result <- computeattention(
    series = y,
    attention_type = attention_type,
    window_size = window_size,
    decay_factor = decay_factor,
    temperature = temperature,
    sigma = sigma,
    sensitivity = sensitivity,
    alpha = alpha,
    beta = beta
  )
  
  return(ahead::ridge2f(
    y = y,
    h = h,
    xreg = ctx_result$context_vectors,
    ...
  ))
}
```

The function:
1. Computes attention weights for the entire time series
2. Generates context vectors from these weights
3. Passes them as external regressors (`xreg`) to `ridge2f()`
4. Returns forecasts enhanced by attention-weighted historical information

## Example: AirPassengers Data

Let's see this in action with the classic AirPassengers dataset:

```r
library(ahead)

# Generate forecasts with attention-based context vectors
result <- ahead::contextridge2f(
  AirPassengers, 
  lags = 15L,      # Use 15 lagged values
  h = 15L,         # Forecast 15 steps ahead
  attention_type = "exponential",
  decay_factor = 5.0
)

# Visualize
plot(result)


# Other example
plot(ahead::contextridge2f(fdeaths, h = 20, lags = 15, 
attention_type = "exponential"))
```

### Results

The plot shows:
- **Black line**: Historical AirPassengers data (1949-1960)
- **Blue line**: 15-month ahead forecasts using attention-based context vectors

## What would make this approach effective?

### 1. **Adaptive Weighting**
Unlike fixed lag structures, attention mechanisms adapt the influence of past observations based on the data's characteristics.

### 2. **Captures Long-Range Dependencies**
By computing weighted sums over the entire history, context vectors can capture patterns that extend beyond fixed window sizes.

### 3. **Multiple Perspectives**
Different attention mechanisms capture different aspects of temporal structure:
- Exponential attention: Time-based decay
- Value-based attention: Regime detection
- Hybrid attention: Both temporal and value similarity

### 4. **RVFL Architecture Benefits**
The doubly-constrained RVFL network (Moudiki et al., 2018) provides:
- Fast training (no backpropagation needed)
- Nonlinear modeling through random hidden layers
- Linear components for interpretability
- Dual regularization preventing overfitting

### 5. **Computational Efficiency**
Context vectors are pre-computed once, and RVFL training is much faster than standard neural networks.

## How It Compares to Standard RVFL

Standard doubly-constrained RVFL for time series uses lagged values directly:

```r
# Standard approach
ahead::ridge2f(y, h = 15, lags = 15)
```

Our attention-enhanced version adds context vectors that encode weighted historical information:

```r
# Attention-enhanced approach
ahead::contextridge2f(y, h = 15, lags = 15, attention_type = "exponential")
```

The context vectors provide **additional features** that capture temporal patterns the raw lags might miss. The RVFL network then learns both:
- Direct linear relationships through the input-output connections
- Nonlinear patterns through the random hidden layer
- All while benefiting from the attention-weighted context

## Choosing Attention Types

Different attention mechanisms suit different data patterns:

| Attention Type | Best For | Key Parameter |
|---------------|----------|---------------|
| `exponential` | General use, smooth trends | `decay_factor` |
| `gaussian` | Seasonal patterns | `sigma` |
| `value_based` | Regime changes | `sensitivity` |
| `hybrid` | Complex patterns | `decay_factor`, `sensitivity` |
| `cosine` | Local similarity | `window_size` |
| `linear` | Simple recency bias | None |

For the AirPassengers data, exponential attention works well because recent observations are highly informative for future trends and seasonal patterns.

## Why RVFL Instead of Standard Neural Networks?

The doubly-constrained RVFL approach (Moudiki et al., 2018) offers several advantages over traditional neural networks:

### Speed
- **No backpropagation**: Random hidden weights are never updated
- **Closed-form solution**: Output weights solved via ridge regression
- **Orders of magnitude faster** than gradient-based training

### Simplicity
- **Fewer hyperparameters**: No learning rate, momentum, or complex optimizers
- **No convergence issues**: Direct solution, no local minima problems
- **Reproducible**: Random seed controls all randomness

### Dual Regularization
- **λ₁**: Constrains hidden layer contribution (prevents overfitting from random features)
- **λ₂**: Constrains direct connections (standard ridge penalty)
- Both penalties work together to create robust predictions

### Architecture
```
Input (lags + context) → [Random Hidden Layer] → Output
                    ↘                         ↗
                      [Direct Connections]
```

The direct connections preserve linear relationships while random hidden layers capture nonlinearities—best of both worlds.

## Tuning Parameters

### Decay Factor (for exponential/hybrid)
- **Low values (1-3)**: Strong recency bias
- **Medium values (5-10)**: Balanced influence
- **High values (15+)**: More uniform weighting

### Window Size (for cosine)
- Smaller windows: Capture short-term patterns
- Larger windows: Capture longer-term dependencies

### Sensitivity (for value-based/hybrid)
- Higher values: Stricter matching of similar values
- Lower values: More tolerant matching

## Implementation Details

The underlying `ahead::computeattention()` function is implemented in C++ (via Rcpp) for efficiency, computing:

1. **Attention weights**: An n×n matrix where entry (i,j) represents the attention weight of time j on time i
2. **Context vectors**: Weighted sums using these attention weights

The attention computation enforces **causal constraints**—time point t can only attend to observations at times j ≤ t, ensuring no future information leakage.

## Practical Considerations

### When to Use This Approach

✅ **Good fit:**
- Medium to long time series (n > 50)
- Complex temporal patterns
- When interpretability matters (attention weights are inspectable)
- Nonlinear relationships between past and future

❌ **May not help:**
- Very short series (n < 30)
- Simple random walks
- When simple methods already work well

### Computational Cost

Context vector computation is O(n²) due to the attention matrix, but:
- It's done once per forecast
- C++ implementation is fast
- For typical series (n < 1000), it's negligible

## Extensions and Future Work

Several interesting extensions are possible:

1. **Multi-head attention**: Combine multiple attention types
2. **Learned parameters**: Optimize attention parameters via cross-validation
3. **Multivariate attention**: Extend to multiple time series with cross-series attention
4. **Hierarchical attention**: Different attention at different time scales

## Conclusion

The `ahead::contextridge2f()` function demonstrates how attention mechanisms (widely applied in deep learning) can potentially enhance doubly-constrained RVFL networks for time series forecasting. By computing context vectors that encode weighted historical information, we give the model additional features that capture complex temporal dependencies.

The approach combines:
- **Attention mechanisms** for intelligent temporal weighting
- **RVFL architecture** for fast, nonlinear modeling (Moudiki et al., 2018)
- **Dual regularization** for robust predictions

It is:
- **Simple** to use (single function call)
- **Flexible** (9 different attention mechanisms)
- **Efficient** (C++ attention computation + fast RVFL training)
- **Effective** (additional features improve forecasting)

For the AirPassengers example, the attention-enhanced RVFL forecasts successfully capture both the upward trend and seasonal fluctuations, extending the pattern 15 months into the future.

## Try It Yourself

```r
# Install packages (if needed)
# install.packages("ahead")
# devtools::install_github("Techtonique/ahead")  # for computeattention

library(ahead)

# Basic usage
result <- ahead::contextridge2f(AirPassengers, h = 10)

# With custom attention
result2 <- ahead::contextridge2f(
  AirPassengers,
  h = 15,
  attention_type = "hybrid",
  decay_factor = 7.0,
  sensitivity = 1.5,
  lags = 12
)

plot(result2)
```

## Code Availability

The complete implementation including:
- `computeattention()` - R wrapper function
- C++ attention mechanisms (9 types)
- `contextridge2f()` - Forecasting function

Is available at [Techtonique GitHub repository](https://github.com/Techtonique/ahead).

---

**References:**
- Moudiki, T., Planchet, F., & Cousin, A. (2018). "Multiple Time Series Forecasting Using Quasi-Randomized Functional Link Neural Networks." *Risks*, 6(1), 22.
- Ridge2f implementation: `ahead` package
- Attention mechanisms: Vaswani et al. (2017) "Attention Is All You Need"
- AirPassengers data: Box & Jenkins (1976)

**Keywords:** time series forecasting, attention mechanisms, RVFL networks, doubly-constrained regularization, context vectors, machine learning, R programming