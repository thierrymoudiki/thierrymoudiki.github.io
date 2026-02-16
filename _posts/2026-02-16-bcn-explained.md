---
layout: post
title: "Understanding Boosted Configuration Networks (combined neural networks and boosting): An Intuitive Guide Through Their Hyperparameters"
description: "How BCN combine neural networks and boosting, explained through the knobs you can turn"
date: 2026-02-16
categories: [R, Python]
comments: true
---

**Disclaimer:** This post was written with the help of LLMs, based on: 

- [https://thierrymoudiki.github.io/blog/2022/07/21/r/misc/boosted-configuration-networks](https://thierrymoudiki.github.io/blog/2022/07/21/r/misc/boosted-configuration-networks)
- [https://www.researchgate.net/publication/332291211_Forecasting_multivariate_time_series_with_boosted_configuration_networks](https://www.researchgate.net/publication/332291211_Forecasting_multivariate_time_series_with_boosted_configuration_networks)
- [https://docs.techtonique.net/bcn/articles/bcn-intro.html](https://docs.techtonique.net/bcn/articles/bcn-intro.html)
- [https://github.com/Techtonique/bcn_python](https://github.com/Techtonique/bcn_python)

Potential remaining errors are mine. 

-------------------

What if you could have a model that:
- ✅ Captures non-linear patterns like neural networks
- ✅ Builds iteratively like gradient boosting
- ✅ Provides built-in interpretability through its additive structure
- ✅ Works well on regression, classitication, time series

That's **Boosted Configuration Networks (BCN)**.

**Where BCN fits:** BCN sits between Neural Additive Models (NAMs) and gradient boosting—combining neural flexibility with boosting's greedy refinement. It's particularly effective for:

- Medium-sized tabular datasets (100s to 10,000s of rows)
- Multivariate prediction tasks (multiple outputs that share structure)
- Problems requiring both accuracy and interpretability
- Time series forecasting with multiple related series

In this post, I'll explain BCN's intuition by walking through its hyperparameters. Each parameter reveals something fundamental about how the algorithm works.

---

## The Core Idea: Building Smart Weak Learners

BCN asks a simple question at each iteration:

> "What's the _best_ artificial neural network feature I can add right now to explain what I haven't captured yet?"

Let's break down this sentence:

### 1. artificial **"Neural network feature"** 

At each iteration L, a BCN creates a simple single-layer feedforward neural network:

```
h_L = activation(w_L^T · x)
```
This is just: multiply features by weights, then apply an activation function (tanh or sigmoid; bounded).

### 2. **_Best_**

BCN finds weights `w_L` that **maximize how much this feature explains the residuals**. 

Specifically, it finds the artificial neural network whose output has the **largest regression coefficient** when predicting the residuals. This is captured in the ξ (xi) criterion:

```
ξ = ν(2-ν)·β²_L - penalty
```

where `β_L` is the least-squares coefficient from regressing residuals on `h_L`.

### 3. **"What I haven't captured yet"**

Like all boosting methods, BCN works on **residuals** - the gap between current predictions and truth. Each iteration "carves away" at the error.

### 4. **"Add"**

Once we find the _best_ `h_L`, we add it to our ensemble:

```
new prediction = old prediction + ν · β_L · h_L
```

**Visual mental model:** Imagine starting with the mean prediction (flat surface). Each iteration adds a "bump" (artificial neural network feature) where the residuals are largest, gradually sculpting a complex prediction surface.

Now let's see how the hyperparameters control this process.

---

## Hyperparameter Priority: The Big Three

Before diving deep, here's how parameters rank by impact:

**Tier 1 - Critical (tune these first):**
- `B` (iterations): Model complexity
- `nu` (learning rate): Step size and stability
- `lam` (weight bounds): Feature complexity

**Tier 2 - Regularization (tune for robustness):**
- `r` (convergence rate, most of the times 0.99, 0.999, 0.9999, etc.): Adaptive quality control
- `col_sample` (feature sampling): Regularization via randomness

**Tier 3 - Technical (usually keep defaults):**
- `type_optim` (optimizer): Computational trade-offs
- `activation` (nonlinearity): Usually tanh because bounded
- `hidden_layer_bias`: Usually TRUE
- `tol` (tolerance): Early stopping

---

## Hyperparameter 1: `B` (Number of Iterations)

**Default:** No universal default (typically 100-500)

**What it controls:** How many weak learners to train

**Intuition:** BCN builds your model piece by piece. Each iteration adds one artificial neural network feature that explains some of what you haven't captured.

**Trade-offs:**

- **Small B (10-50):** 
  - ✅ Fast training
  - ✅ Less risk of overfitting
  - ❌ May underfit complex relationships
  
- **Large B (100-1000):**
  - ✅ Can capture subtle patterns
  - ✅ Better accuracy on complex tasks
  - ❌ Slower training
  - ❌ Risk of overfitting without other regularization

**Rule of thumb:** Start with B=100. If using early stopping (`tol` > 0), set B high (500-1000) and let the algorithm stop when improvement plateaus.

**What's happening internally:**
Each iteration finds weights `w_L` that maximize:
```
ξ_L = ν(2-ν) · β²_L - penalty
```
where β²_L measures how much the neural network feature correlates with residuals.

---

## Hyperparameter 2: `nu` (Learning Rate)

**Default:** 0.1 (conservative)  
**Typical range:** 0.1-0.8  
**Sweet spot:** 0.3-0.5

**What it controls:** How aggressively to use each weak learner

**Intuition:** Even if you find a great neural network feature, you might not want to use it at full strength. The learning rate controls the step size.


When BCN finds a good feature h_L with coefficient β_L, it updates predictions by:
```
prediction += nu · β_L · h_L
```

**Trade-offs:**
- **Small ν (0.1-0.3):**
  - ✅ More stable training
  - ✅ Better generalization (smooths out noise)
  - ✅ Less sensitive to individual weak learners
  - ❌ Need more iterations (larger B)
  - ❌ Slower convergence
  
- **Large ν (0.5-1.0):**
  - ✅ Faster convergence
  - ✅ Fewer iterations needed
  - ❌ Risk of overfitting
  - ❌ Can be unstable

**Why ν(2-ν) appears in the math:**

This factor arises when we want to prove the convergence of residuals' L2-norm towards 0. It's **maximized at ν=1** (full gradient step):

```
f(ν) = 2ν - ν² 
f'(ν) = 2 - 2ν = 0  ⟹  ν = 1
```
This ensures stability for ν ∈ (0,2) and explains why ν=1 is the "natural" full step.

**Think of it like:**
- ν=0.1: "I trust each feature a little, build slowly" (like learning rate 0.01 in SGD)
- ν=0.5: "I trust each feature moderately, build steadily"  
- ν=1.0: "I trust each feature fully, build quickly" (can be unstable)

---

## Hyperparameter 3: `lam` (λ - Weight Bounds)

**Default:** 0.1  
**Typical range:** 0.1-100 (often on log scale: 10^0 to 10^2)  
**Sweet spot:** 10^(0.5 to 1.0) ≈ 3-10

**What it controls:** How large the neural network weights can be

**Intuition:** This constrains the weights `w_L` at each iteration to the range [-λ, λ]. It's a form of regularization through box constraints.

```r
# Tight constraints: simpler features
fit_simple <- bcn(x, y, lam = 0.5)

# Loose constraints: more complex features
fit_complex <- bcn(x, y, lam = 10.0)
```

**Why this matters:**

**Small λ (0.1-1.0):**

- Neural network features are "gentle" (bounded outputs)
- Less risk of overfitting
- May miss complex interactions
- ✅ Use for: Small datasets, high interpretability needs

**Large λ (5-100):**

- Neural network features can be more "extreme"
- Can capture stronger non-linearities
- Risk of overfitting if not balanced with other regularization
- ✅ Use for: Complex patterns, large datasets

**What's happening mathematically:**

At each iteration, we solve:

```
maximize ξ(w_L)
subject to: -λ ≤ w_L,j ≤ λ for all features j
```

This is a **constrained optimization** - we're finding the _best_ weights within a box.

**Think of it like:**

- Small λ: "Keep the weak learners simple" (like L∞ regularization)
- Large λ: "Allow complex weak learners"

**Note on consistency:** In the code, this parameter is `lam` (avoiding the Greek letter for R compatibility).

---

## Hyperparameter 4: `r` (Convergence Rate)

**Default:** 0.3  
**Typical range:** 0.3-0.99  
**Sweet spot:** 0.9-0.99999

**What it controls:** How the acceptance threshold changes over iterations

**Intuition:** This is the **most subtle** hyperparameter. It controls how picky BCN is about accepting new weak learners, and this pickiness *decreases* as training progresses. Think of `r` as the "patience" or "quality control" officer: high r means "Only the best features get through the door early on."

**The acceptance criterion:**

BCN only accepts a weak learner if:

```
ξ_L = ν(2-ν)·β²_L - [1 - r - (1-r)/(L+1)]·||residuals||² ≥ 0
```

The penalty term `[1 - r - (1-r)/(L+1)]` **decreases** as L increases:

| Iteration L | r = 0.95 | r = 0.70 | r = 0.50 |
|------------|---------|---------|---------|
| L = 1 | 0.075 | 0.45 | 0.75 |
| L = 10 | 0.055 | 0.33 | 0.55 |
| L = 100 | 0.050 | 0.30 | 0.50 |
| L → ∞ | 0.050 | 0.30 | 0.50 |

**Interpretation:** The penalty starts higher and converges to `(1-r)` as training progresses.

**Trade-offs:**

**Large r (0.9-0.99):**

- Early iterations: very picky (high penalty)
- Later iterations: more permissive
- ✅ Prevents premature commitment to poor features
- ✅ Allows fine-tuning in later iterations
- ✅ Better generalization
- ✅ Use for: Production models, complex tasks

**Small r (0.3-0.7):**

- Less selective throughout training
- ✅ Accepts more weak learners
- ✅ Faster initial progress
- ❌ May accept noisy features early
- ✅ Use for: Quick prototyping, exploratory work

**The dynamic threshold:**

Rearranging the acceptance criterion:

```
Required R² > [1 - r - (1-r)/(L+1)] / [ν(2-ν)]
```

This creates an **adaptive** selection criterion that evolves during training.

**Think of it like:**

- High r: "Be very careful early on (we have lots of iterations left), but allow refinements later"
- Low r: "Accept good-enough features throughout training"

---

## Hyperparameter 5: `col_sample` (Feature Sampling)

**Default:** 1.0 (no sampling)  
**Typical range:** 0.3-1.0  
**Sweet spot:** 0.5-0.7 for high-dimensional data

**What it controls:** What fraction of features to consider at each iteration

**Intuition:** Like Random Forests, BCN can use only a random subset of features at each iteration. This reduces overfitting, adds diversity, and speeds up computation.

```r
# Use all features (no sampling)
fit_full <- bcn(x, y, col_sample = 1.0)

# Use 50% of features at each iteration
fit_sampled <- bcn(x, y, col_sample = 0.5)
```

**How it works:**
At iteration L, randomly sample `col_sample × d` features and optimize only over those:
```
w_L ∈ R^d_reduced    (instead of R^d)
```

Different features are sampled at each iteration, creating diversity like Random Forests but for neural network features.

**Trade-offs:**

**col_sample = 1.0 (no sampling):**

- ✅ Can use all information
- ✅ Potentially better accuracy
- ❌ Slower training (larger optimization)
- ❌ Higher overfitting risk
- ✅ Use for: Small datasets (N < 1000), few features (d < 50)

**col_sample = 0.3-0.7:**

- ✅ Faster training (smaller optimization)
- ✅ Regularization effect (like Random Forests)
- ✅ More diverse weak learners
- ❌ May miss important feature combinations
- ✅ Use for: Large datasets, many features (d > 100)

**Interaction with B:**
Column sampling as implicit regularization means you may need more iterations:

---

## Hyperparameter 6: `activation` (Activation Function)

**Default:** "tanh"  
**Options:** "tanh", "sigmoid"  

**What it controls:** The non-linearity in each weak learner

**Intuition:** This determines the shape of transformations each neural network can create.

**Characteristics:**

**tanh (hyperbolic tangent):**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- Range: [-1, 1]
- Symmetric around 0
- Gradient: 1 - tanh²(z)
- **Good for:** Most tasks, especially when features are centered
- ✅ **Recommended default**

**sigmoid:**
```
sigmoid(z) = 1 / (1 + e^(-z))
```
- Range: [0, 1]  
- Asymmetric
- Gradient: sigmoid(z) · (1 - sigmoid(z))
- **Good for:** When outputs are probabilities or rates

**Why bounded activations?**

BCN requires bounded activations for theoretical guarantees and stability of the ξ criterion. Unbounded activations like ReLU are **not recommended** because:
1. Theoretical issues: The ξ optimization assumes bounded activation *outputs*
2. Stability: Unbounded outputs can destabilize the ensemble
3. While ReLU could *theoretically* work with very tight weight constraints (small λ), tanh/sigmoid provide stronger guarantees

**Rule of thumb:** Use **tanh** as default. It's more balanced, bounded and zero-centered.

---

## Hyperparameter 7: `tol` (Early Stopping Tolerance)

**Default:** 0 (no early stopping)  
**Typical range:** 1e-7 to 1e-3  
**Recommended:** 1e-7 for most tasks

**What it controls:** When to stop training before reaching B iterations

**Intuition:** If the model stops improving (residual norm isn't decreasing much), stop early to avoid overfitting and save computation.

**How it works (corrected):**
BCN tracks the relative improvement in residual norm and stops if progress is too slow:
```
if (relative_decrease_in_residuals < tol):
    stop training
```

**Important clarification:** Early stopping is based on **improvement rate**, not absolute residual magnitude. This means BCN can stop even when residuals are still large (on a hard problem) if adding more weak learners doesn't help.

**Trade-offs:**

**tol = 0 (no early stopping):**

- Always trains for exactly B iterations
- May overfit if B is too large
- ✅ Use for: Quick experiments with small B

**tol = 1e-7 to 1e-5:**

- Stops when improvement becomes negligible
- Prevents overfitting
- Can save significant computation
- ✅ Use for: Production models with large B

**Practical tip:** Set B large (e.g., 500-1000) and tol small (e.g., 1e-7) to let the algorithm decide when to stop. The actual number of iterations used will be stored in `fit$maxL`.

---

## Hyperparameter 8: `type_optim` (Optimization Method)

Gradient-based. 

**Default:** "nlminb"  
**Options:** "nlminb", "adam", "sgd", "nmkb", "hjkb", "randomsearch"

**What it controls:** How to solve the optimization problem at each iteration

**Intuition:** Finding the best weights w_L is a **non-convex optimization** problem. Different solvers have different trade-offs.

**Available optimizers:**

**nlminb** (default):

- Uses gradient and Hessian approximations
- ✅ Robust
- ✅ Well-tested in R
- ✅ Works well in most cases
- ⚠️ Medium speed
- ✅ Use for: General purpose, production

**adam / sgd:**

- Gradient-based optimizers from deep learning
- ✅ Fast, especially for high-dimensional problems
- ✅ Good for large d (many features)
- ⚠️ May need tuning (learning rate, iterations)
- ✅ Use for: d > 100, speed-critical applications

**nmkb / hjkb:**

- Derivative-free Nelder-Mead / Hooke-Jeeves
- ✅ Very robust (no gradient needed)
- ❌ Slow
- ✅ Use when: Other optimizers fail or diverge

**randomsearch:**

- Random sampling + local search
- ✅ Can escape local minima
- ❌ Slower
- ✅ Use when: Problem is very non-convex

**Rule of thumb:** 

- Start with `"nlminb"` 
- If training is slow and d > 100, try `"adam"`
- Can pass additional arguments via `...` (e.g., max iterations, tolerance)

**Important insight:**
Because BCN uses an ensemble, **local optima are OK**! Even if we don't find the globally optimal w_L, the next iteration can compensate. This is why BCN is robust despite non-convex optimization at each step.

---

## Hyperparameter 9: `hidden_layer_bias` (Include Bias Term)

**Default:** TRUE  
**Options:** TRUE, FALSE

**What it controls:** Whether neural networks have a bias/intercept term

**Intuition:** Without bias, h_L = activation(w^T x). With bias, h_L = activation(w^T x + b).

**Trade-offs:**

**hidden_layer_bias = FALSE:**

- Simpler optimization (one less parameter per iteration)
- Faster training
- Assumes data is centered
- ✅ Use when: Features are already centered, want pure multiplicative effects

**hidden_layer_bias = TRUE:**

- More expressive (can handle shifts)
- Can handle non-centered data better
- One additional parameter to optimize per iteration
- ✅ **Recommended default** - safer choice

**Typical choice:** Use TRUE unless you have a specific reason not to (e.g., theoretical interest in purely multiplicative models).

---

## Hyperparameter 10: `n_clusters` (Optional Clustering Features)

**Default:** NULL (no clustering)  
**Typical range:** 2-10

**What it controls:** Whether to add cluster membership features

**Intuition:** BCN can automatically perform k-means clustering on your inputs and add cluster memberships as additional features. This can help capture local patterns.

**When to use:**

- ✅ Data has natural groupings or modes
- ✅ Local patterns differ across regions of feature space
- ❌ Not needed for most standard regression/classification

**Note:** This is an advanced feature - start without it and add only if needed.

---

## Putting It All Together: Hyperparameter Recipes

### Recipe 1: Fast Prototyping (Small Dataset, N < 1000)

```r
fit <- bcn(
  x = X_train, 
  y = y_train,
  B = 50,              # Few iterations for speed
  nu = 0.5,            # Moderate learning rate
  col_sample = 1.0,    # Use all features (dataset is small)
  lam = 10^0.5,        # ~3.16, moderate regularization
  r = 0.9,             # Adaptive threshold
  tol = 1e-5,          # Early stopping
  activation = "tanh",
  type_optim = "nlminb",
  hidden_layer_bias = TRUE
)
```

**Why these choices:**
- Small B for speed
- High nu for faster convergence
- No column sampling (dataset is small)
- Standard other parameters

**Expected performance:** Quick baseline in minutes

---

### Recipe 2: Production Model (Medium Dataset, N ~ 10,000)

```r
fit <- bcn(
  x = X_train,
  y = y_train,
  B = 200,             # Enough iterations with early stopping
  nu = 0.3,            # Conservative for stability
  col_sample = 0.6,    # Some regularization
  lam = 10^0.8,        # ~6.31, allow some complexity
  r = 0.95,            # Very selective early on
  tol = 1e-7,          # Train until converged
  activation = "tanh",
  type_optim = "nlminb",
  hidden_layer_bias = TRUE
)
```

**Why these choices:**
- Moderate B with early stopping safety
- Conservative nu for stability
- Column sampling for regularization
- High r for careful feature selection

**Expected performance:** Robust model, may train 100-150 iterations before stopping

---

### Recipe 3: Complex Task (Large Dataset, High-Dimensional)

```r
fit <- bcn(
  x = X_train,
  y = y_train,
  B = 500,             # Many iterations (will stop early if needed)
  nu = 0.4,            # Balanced
  col_sample = 0.5,    # Strong regularization for high d
  lam = 10^1.0,        # 10, higher complexity allowed
  r = 0.95,            # Adaptive
  tol = 1e-7,          # Early stopping safety
  activation = "tanh",
  type_optim = "adam",  # Fast optimizer for high d
  hidden_layer_bias = TRUE
)
```

**Why these choices:**
- Large B to capture complexity
- Column sampling crucial for high dimensions (d > 100)
- Adam optimizer for speed with many features
- High r to prevent early overfitting

**Expected performance:** May use 200-400 iterations, handles d > 500 well

---

### Recipe 4: Multivariate Time Series / Multi-Output

```r
fit <- bcn(
  x = X_train,
  y = Y_train,         # Matrix with multiple outputs (e.g., N x m)
  B = 300,
  nu = 0.5,            # Can be higher for shared structure
  col_sample = 0.7,
  lam = 10^0.7,
  r = 0.95,            # Critical: enforces shared structure
  tol = 1e-7,
  activation = "tanh",
  type_optim = "nlminb"
)
```

**Why these choices:**
- **High r is critical**: In multivariate mode, BCN computes ξ_k for each output k and requires min_k(ξ_k) ≥ 0 for acceptance. This ensures each weak learner contributes meaningfully across **all** time series/outputs, creating shared representations.
- Higher nu because shared structure is more stable
- Standard B with early stopping

**Note on multivariate:** BCN handles multiple outputs naturally through one-hot encoding (classification) or matrix targets (regression). The min(ξ) criterion prevents sacrificing one output to improve another.

**Expected performance:** Strong on related time series or multi-task learning

---

## Understanding Hyperparameter Interactions

### Interaction 1: `nu` × `B` ≈ Constant

**Trade-off:** Small nu needs large B

```r
# Approximately equivalent final predictions:
fit1 <- bcn(B = 100, nu = 0.5)
fit2 <- bcn(B = 200, nu = 0.25)
```

**Why:** Smaller steps need more iterations to reach similar places.

**Rule:** For similar model complexity, `nu × B ≈ constant` (approximately).

**In practice:**
- Production (stability priority): nu = 0.3, B = 300
- Prototyping (speed priority): nu = 0.5, B = 100

---

### Interaction 2: `lam` ↔ `r` (Complexity Control)

**Both control complexity:**
- `lam`: How complex each weak learner can be
- `r`: How selective we are about accepting weak learners

```r
# More regularization
fit_reg <- bcn(lam = 1.0, r = 0.95)   # Simple features, selective

# Less regularization  
fit_complex <- bcn(lam = 10.0, r = 0.7)  # Complex features, permissive
```

**Balance principle:** If you allow complex features (high lam), be selective (high r) to avoid noise.

**Typical combinations:**
- **High quality**: lam = 10, r = 0.95 → Complex but carefully selected features
- **Moderate**: lam = 5, r = 0.90 → Balanced
- **Fast/loose**: lam = 3, r = 0.80 → Simple features, permissive

---

### Interaction 3: `col_sample` ↔ `B` (Coverage)

**Column sampling as implicit regularization:**

```r
# Fewer features per iteration → need more iterations for coverage
fit1 <- bcn(col_sample = 1.0, B = 100)
fit2 <- bcn(col_sample = 0.5, B = 200)
```

**Rough guideline:**
```
B_needed ≈ B_baseline / col_sample
```

**In practice:**
- col_sample = 1.0 → B = 100-200
- col_sample = 0.5 → B = 200-400
- col_sample = 0.3 → B = 300-500

---

## The Mathematical Connection: How Hyperparameters Appear in ξ

The core optimization criterion ties everything together:

```
ξ_L = ν(2-ν) · β²_L - [1 - r - (1-r)/(L+1)] · ||residuals||²
      └─┬──┘   └┬─┘   └──────────┬──────────┘
       nu    optimized over        r
             w ∈ [-lam, lam]
```

**Reading the formula:**
1. Find w_L (constrained by `lam`) that maximizes β²_L  
   - β_L is the OLS coefficient: β_L = (h_L^T · residuals) / ||h_L||²
2. Scale by ν(2-ν) (controlled by `nu`)
3. Subtract penalty (controlled by `r`)
4. Accept only if ξ ≥ 0 for all outputs
5. Repeat for `B` iterations (or until `tol` reached)
6. At each step, sample `col_sample` fraction of features

This unified view shows how all hyperparameters work together to control the greedy feature selection process.

---

## Practical Tips for Hyperparameter Tuning

### Start Simple, Add Complexity

1. **Begin with defaults:**
   ```r
   fit <- bcn(x, y, B = 100, nu = 0.3, lam = 10^0.7, r = 0.9)
   ```

2. **If underfitting (train error too high):**
   - ↑ Increase B (more capacity)
   - ↑ Increase lam (allow more complex features)
   - ↑ Increase nu (use features more aggressively)
   - ↓ Decrease r (be less selective)

3. **If overfitting (train << test error):**
   - ↓ Decrease nu (smaller, more careful steps)
   - ↓ Decrease lam (simpler features)
   - Add column sampling (col_sample = 0.5-0.7)
   - ↑ Increase r (be more selective)
   - Use early stopping (tol = 1e-7)

### Use Cross-Validation Wisely

**Most important to tune:** `B`, `nu`, `lam`

**Moderately important:** `r`, `col_sample`

**Usually fixed:** `hidden_layer_bias = TRUE`, `type_optim = "nlminb"`, `activation = "tanh"`

**Example CV strategy:**
```r
library(caret)

# Grid search on log-scale for lam
grid <- expand.grid(
  B = c(100, 200, 500),
  nu = c(0.1, 0.3, 0.5),
  lam = 10^seq(0, 1.5, by = 0.5)  # 1, 3.16, 10, 31.6
)

# Use caret, mlr3, or tidymodels for CV
```

### Monitor Training

```r
# Enable verbose output
fit <- bcn(x, y, verbose = 1, show_progress = TRUE)
```

**Watch for:**
- How fast ||residuals||_F decreases (convergence rate)
- Whether ξ stays positive (quality of weak learners)
- If training stops early and at what iteration (capacity needs)

**Diagnostic patterns:**
- Residuals plateau early → Increase B or lam
- ξ often negative → Decrease r or increase lam
- Training very slow → Try adam optimizer or increase col_sample

---

## Quick Reference: Hyperparameter Cheat Sheet

| Hyperparameter | Low Value Effect | High Value Effect | Typical Range | Default |
|---------------|-----------------|-------------------|---------------|---------|
| **B** | Simple, fast, may underfit | Complex, slow, may overfit | 50-1000 | 100 |
| **nu** | Stable, slow convergence | Fast, potentially unstable | 0.1-0.8 | 0.1 |
| **lam** | Linear-ish, simple features | Nonlinear, complex features | 1-100 | 0.1 |
| **r** | Permissive, accepts more | Selective, high quality | 0.3-0.99 | 0.3 |
| **col_sample** | No regularization | Strong regularization | 0.3-1.0 | 1.0 |
| **tol** | No early stop | Aggressive early stop | 0-1e-3 | 0 |
| **activation** | tanh (symmetric) | sigmoid (asymmetric) | - | tanh |
| **type_optim** | nlminb (robust) | adam (fast) | - | nlminb |
| **hidden_layer_bias** | Simpler, through origin | More flexible | - | TRUE |

---

## When NOT to Use BCN

While BCN is versatile, it's not always the best choice:

❌ **Ultra-high-dimensional sparse data (d > 10,000)**
   - Tree-based boosting (XGBoost/LightGBM) may be faster
   - Column sampling helps, but trees handle sparsity natively

❌ **Very large datasets (N > 1,000,000)**
   - Training time scales roughly O(B × N × d)
   - Consider subsampling or streaming methods

❌ **Deep sequential/temporal structure**
   - BCN is static (no recurrence)
   - Use RNNs/Transformers for complex time dependencies

❌ **Image/text/audio from scratch**
   - Convolutional/attention architectures more suitable
   - BCN works on extracted features (embeddings, tabular)

✅ **BCN shines at:**
- Tabular data (100s to 10,000s of rows)
- Multivariate prediction (shared structure across outputs)
- Needing both accuracy AND interpretability
- Time series with extracted features
- When XGBoost works but you want gradient-based explanations

---

## Debugging BCN Training

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| ξ frequently negative early | r too high or lam too low | Decrease r to 0.8 or increase lam to 5-10 |
| Residuals plateau quickly | nu too small or B too low | Increase nu to 0.4-0.5 or B to 300+ |
| Training very slow | col_sample=1 on wide data | Set col_sample=0.5 and try type_optim="adam" |
| High train accuracy, poor test | Overfitting | Decrease nu, increase r, add col_sample < 1 |
| Poor train accuracy | Underfitting | Increase B, increase lam, try different activation |
| Optimizer not converging | Bad initialization or scaling | Check feature scaling, try different type_optim |

---

## Interpretability Example

One of BCN's unique advantages is **gradient-based interpretability**. 

**What makes this special:**
- ✅ Exact analytic gradients (no approximation)
- ✅ Same O(B × m × d) cost as prediction
- ✅ Shows direction of influence (positive/negative)
- ✅ Works for both regression and classification
- ✅ Much faster than SHAP on tree ensembles

---

## Conclusion: The Philosophy of BCN

BCN's hyperparameters reveal its design philosophy:

**1. Iterative Refinement** (via `B`)  
Build the model piece by piece, adding one well-chosen feature at a time.

**2. Conservative Steps** (via `nu`)  
Don't trust any single feature too much - combine many weak learners.

**3. Bounded Complexity** (via `lam`)  
Keep individual weak learners simple to ensure stability and interpretability.

**4. Adaptive Selection** (via `r`)  
Start picky (prevent early mistakes), become permissive (allow refinement).

**5. Randomization** (via `col_sample`)  
Like Random Forests, diversity through randomness helps generalization.

**6. Early Stopping** (via `tol`)  
Know when to stop - more iterations aren't always better.

**7. Explicit Optimization for Interpretability**  
Unlike methods that require post-hoc explanations, BCN is designed with interpretability in mind through its additive structure and differentiable components.

Together, these create a model that's:
- ✅ **Expressive** (neural network features capture non-linearity)
- ✅ **Interpretable** (additive structure + gradients)
- ✅ **Robust** (ensemble of bounded weak learners)
- ✅ **Efficient** (sparse structure, early stopping, column sampling)

---

## Next Steps

**To learn more:**
- 📦 [BCN R Package on GitHub](https://github.com/Techtonique/bcn)
- 📦 [BCN Python Package on GitHub](https://github.com/Techtonique/bcn_python)
- 📝 [Research preprint on BCN](https://www.researchgate.net/publication/332291211_Forecasting_multivariate_time_series_with_boosted_configuration_networks)


**To contribute:**
BCN is open source! Contributions welcome for:
- New activation functions
- Additional optimization methods  
- Interpretability visualizations
- Benchmark studies and applications

