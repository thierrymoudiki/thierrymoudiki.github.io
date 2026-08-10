---
layout: post
title: "'PCARVFLSimulator': a GAN-like tabular data synthesizer built from PCA scores, a Random Vector Functional-Link network, and bootstrap residuals"
description: "A lightweight, GAN-like tabular data synthesizer that swaps the adversarial training loop for PCA, a Random Vector Functional-Link network, and bootstrap residuals."
date: 2026-08-10
categories: Python
comments: true
---

Generative Adversarial Networks are the default reach for synthetic
tabular data, but they are not the only route to the same destination.
`PCARVFLSimulator` (from Python package [synthe](https://github.com/Techtonique/synthe)) is a lightweight, GAN-*like* synthesizer that swaps
the adversarial training loop for three ingredients that are each
individually well understood: Principal Component Analysis (PCA) for a
compact latent representation, a Random Vector Functional-Link (RVFL)
network for a closed-form (Ridge-regression) mapping from latent codes
back to feature space, and a bootstrap of the residuals to reinject the
noise that a purely deterministic reconstruction would otherwise
discard. [Optuna](https://optuna.org) tunes the two RVFL
hyperparameters — the number of random hidden nodes and the ridge
penalty — against a distributional-distance objective, so no adversarial
discriminator, and no gradient descent, is required anywhere in the
pipeline.

Below, the algorithm is described step by step together with the
formulas behind each stage, then illustrated on four classic
`scikit-learn` datasets (`iris`, `wine`, `breast_cancer`, `digits`),
with a full adequacy report — distributional distances, per-feature
goodness-of-fit tests, moment matching, and a random-projection sweep —
for each one.


# 1 - How it works

## 1.1 Standardize, then find a latent space with PCA

Given real data $Y \in \mathbb{R}^{n \times d}$, the simulator first
(optionally, and by default) standardizes it,

$$
Y_s = \frac{Y - \mu}{\sigma},
$$

column-wise, so that PCA and the residual model both operate on
z-scored features and the simulator stays robust to raw, unscaled
inputs. PCA is then fit on $Y_s$ to obtain the latent scores

$$
Z = Y_s V_{1:k},
$$

where $V_{1:k}$ collects the top $k$ principal directions. The number
of components $k$ is chosen automatically as the smallest $k$ such that
the cumulative explained variance ratio exceeds a threshold (95% by
default), or can be set manually.

## 1.2 Learn $Z \to Y_s$ with an RVFL network

An RVFL network maps the low-dimensional latent code back to the
(standardized) feature space using a *fixed*, randomly drawn hidden
layer plus a closed-form output layer:

$$
H = \phi(Z W + b), \qquad
\Phi = [\,Z \mid H\,], \qquad
\hat Y_s = \Phi \beta,
$$

with $W$ and $b$ drawn once from a standard normal distribution and
frozen, $\phi$ an activation (`tanh` by default), $[\,Z \mid H\,]$ the
direct-link augmentation (skip connection) concatenating the raw input
with the random features, and $\beta$ obtained in closed form by ridge
regression:

$$
\beta = (\Phi^\top \Phi + \alpha I)^{-1} \Phi^\top Y_s .
$$

Because $\beta$ solves a linear system, there is no backpropagation and
no training loop — fitting the RVFL layer is a single matrix solve.

## 1.3 Bootstrap the residuals to restore variability

A purely deterministic $\hat Y_s = \Phi\beta$ would collapse every
sample sharing a latent code $Z$ onto the same point, so the training
residuals

$$
\varepsilon = Y_{s,\text{train}} - \hat Y_{s,\text{train}}
$$

are stored and bootstrapped back in at sampling time. To draw a
synthetic row: pick a training latent code $z_i$ at random (with
replacement), predict $\hat y_i = \phi(z_i W + b)\beta$-style output
through the fitted RVFL, add a bootstrapped residual $\varepsilon_j$,
and invert the standardization:

$$
y^{\text{syn}} = \sigma \odot \big(\hat y_i + \varepsilon_j\big) + \mu .
$$

This is the "GAN-like" part of the design: the RVFL plays the role of a
generator conditioned on a latent code, and the residual bootstrap
plays the role of injected noise — but both are fit in closed form
instead of adversarially.

## 1.4 Tuning with Optuna

Optuna searches over the number of random hidden nodes
($n_{\text{nodes}} \in [50, 1000]$, log-scale) and the ridge penalty
($\alpha \in [10^{-5}, 10]$, log-scale), minimizing a distributional
distance — biased MMD² by default, or the energy distance — between a
held-out slice of real data and a batch of samples generated the same
way `sample()` would generate them.

## 1.5 Measuring adequacy

`adequacy_report()` compares real and synthetic data along four axes:

| Category | Metrics | Direction |
|---|---|---|
| Distributional distance (standardized space) | MMD² (biased RBF-kernel estimator), Energy distance | lower is better |
| Per-feature marginals (original scale) | KS statistic / Bonferroni p-value / reject rate, Anderson–Darling statistic / Bonferroni p-value / reject rate | lower stat, higher p, lower reject rate |
| Moments (original scale) | Mean MAE, Std MAE, Std ratio, Frobenius norm of the correlation-matrix difference | lower is better, std ratio ≈ 1 |
| Dependency structure | Mean KS statistic over 50 random 1-D projections | lower is better |


# 2 - Results across four datasets

The tables below summarize `adequacy_report()` on 400 synthetic rows
generated after fitting `PCARVFLSimulator` (default settings, 50 Optuna
trials) on four `scikit-learn` datasets of increasing dimensionality.

A few things stand out.

- **Low-to-moderate dimensional, continuous data (iris, wine,
  breast_cancer)** fare well: MMD² and energy distance stay small,
  Bonferroni-combined KS and AD tests fail to reject the null of equal
  distributions, and the mean/std/correlation-structure gaps are minor.
- **`digits`** is the outlier: KS reject rate jumps to 95%, driven by
  the fact that pixel-intensity features are highly discrete
  (0–16 integer counts with many exact zeros) rather than smoothly
  continuous, which is a much harder target for a Gaussian-residual
  bootstrap on 64 correlated dimensions. Tellingly, the
  **random-projection KS sweep stays low (0.053) even here** — the
  *aggregate* multivariate structure the simulator is actually
  optimized for (via MMD) is captured reasonably well, even though
  several individual discrete-valued pixel marginals are not.


```python
!pip install synthe
```


```python
from synthe import PCARVFLSimulator, adequacy_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

datasets = {
    "iris":          load_iris(return_X_y=True)[0],
    "wine":          load_wine(return_X_y=True)[0],
    "breast_cancer": load_breast_cancer(return_X_y=True)[0],
    "digits":        load_digits(return_X_y=True)[0],
}

for name, X in datasets.items():
    print(f"\n{'=' * 56}")
    print(f"  {name.upper()}  (n={X.shape[0]}, d={X.shape[1]})")
    print(f"{'=' * 56}")
    sim   = PCARVFLSimulator(random_state=42)
    sim.fit(X, n_trials=30)
    X_syn = sim.sample(400)
    adequacy_report(X, X_syn)
```

    
    ========================================================
      IRIS  (n=150, d=4)
    ========================================================
      [PCARVFL] 2 components (95.8% var)
      [PCARVFL] nodes=153 α=5.06e+00 mmd=0.00000
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=150, n_syn=400, d=4)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00198  
      Energy distance             0.01997  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.08875  
      KS p Bonferroni ↑           0.51083  ✓ ok
      KS reject rate ↓              0.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            0.04227  
      AD p Bonferroni ↑           0.48362  ✓ ok
      AD reject rate ↓              0.000  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.07715  
      Std  MAE ↓                  0.02283  
      Std  ratio                   1.0018  (want ≈ 1.00)
      Corr Frobenius ↓             0.1251  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.07285  over 50 directions
    ────────────────────────────────────────────────────────
    
    ========================================================
      WINE  (n=178, d=13)
    ========================================================
      [PCARVFL] 10 components (96.2% var)
      [PCARVFL] nodes=153 α=5.06e+00 mmd=0.00000
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=178, n_syn=400, d=13)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00168  
      Energy distance             0.03782  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.08130  
      KS p Bonferroni ↑           0.87284  ✓ ok
      KS reject rate ↓              0.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            0.06182  
      AD p Bonferroni ↑           0.57427  ✓ ok
      AD reject rate ↓              0.077  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.28633  
      Std  MAE ↓                  0.64890  
      Std  ratio                   0.9760  (want ≈ 1.00)
      Corr Frobenius ↓             0.8654  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.07512  over 50 directions
    ────────────────────────────────────────────────────────
    
    ========================================================
      BREAST_CANCER  (n=569, d=30)
    ========================================================
      [PCARVFL] 10 components (95.2% var)
      [PCARVFL] nodes=79 α=8.63e-05 mmd=0.00000
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=569, n_syn=400, d=30)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00000  
      Energy distance             0.03584  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.05846  
      KS p Bonferroni ↑           0.30133  ✓ ok
      KS reject rate ↓              0.133  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            0.28453  
      AD p Bonferroni ↑           0.58155  ✓ ok
      AD reject rate ↓              0.100  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  1.23670  
      Std  MAE ↓                  1.72216  
      Std  ratio                   0.9997  (want ≈ 1.00)
      Corr Frobenius ↓             1.9026  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.05573  over 50 directions
    ────────────────────────────────────────────────────────
    
    ========================================================
      DIGITS  (n=1797, d=64)
    ========================================================
      [PCARVFL] 40 components (95.1% var)
      [PCARVFL] nodes=181 α=5.59e-04 mmd=0.00062
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=1797, n_syn=400, d=64)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00143  
      Energy distance             0.04743  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.25411  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.953  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)           33.36410  
      AD p Bonferroni ↑           0.06400  ✓ ok
      AD reject rate ↓              0.967  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.18309  
      Std  MAE ↓                  0.12268  
      Std  ratio                   0.9774  (want ≈ 1.00)
      Corr Frobenius ↓             4.5537  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.05309  over 50 directions
    ────────────────────────────────────────────────────────


# 3 - How does an actual GAN do on the same benchmark?

`digits` was the one case above where `PCARVFLSimulator` visibly
strained, so it's a fair place to check it against a real adversarially-
trained synthesizer: [CTGAN](https://github.com/sdv-dev/CTGAN), a GAN
architecture purpose-built for tabular data (mode-specific normalization
per column, a conditional generator, PacGAN-style discriminator). Same
data, same train/sample sizes, same `adequacy_report()`:


```python
!pip install ctgan
```


```python
import pandas as pd
from sklearn.datasets import load_digits
from ctgan import CTGAN

digits = load_digits()
Y_digits = digits.data.astype(float)
df = pd.DataFrame(Y_digits, columns=[f"px{i}" for i in range(64)])

model = CTGAN(epochs=300, cuda=False)
model.fit(df)                     # ~3 minutes on CPU, 1,797 rows

Y_sim_ctgan = model.sample(400).values

report = adequacy_report(Y_digits, Y_sim_ctgan)
```

    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=1797, n_syn=400, d=64)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00946  
      Energy distance             0.59573  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.40585  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              1.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)          188.34046  
      AD p Bonferroni ↑           0.06400  ✓ ok
      AD reject rate ↓              1.000  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.93301  
      Std  MAE ↓                  0.52533  
      Std  ratio                   1.0687  (want ≈ 1.00)
      Corr Frobenius ↓            11.4486  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.14507  over 50 directions
    ────────────────────────────────────────────────────────





| Metric | Direction | PCARVFLSimulator | CTGAN (300 epochs) |
|---|---|---|---|
| MMD (biased) | ↓ better | **0.00143** | 0.00946 |
| Energy distance | ↓ better | **0.04743** | 0.59573 |
| KS stat (mean) | ↓ better | **0.25411** | 0.40585 |
| KS p (Bonferroni) | ↑ better | 0.00000 | 0.00000 |
| KS reject rate | ↓ better | **0.953** | 1.000 |
| AD stat (mean) | ↓ better | **33.36410** | 188.34046 |
| AD p (Bonferroni) | ↑ better | 0.06400 | 0.06400 |
| AD reject rate | ↓ better | **0.967** | 1.000 |
| Mean MAE | ↓ better | **0.18309** | 0.93301 |
| Std MAE | ↓ better | **0.12268** | 0.52533 |
| Std ratio (want ≈ 1) | closer to 1 | **0.9774** | 1.0687 |
| Corr Frobenius | ↓ better | **4.5537** | 11.4486 |
| KS proj (50 dirs) | ↓ better | **0.05309** | 0.14507 |
