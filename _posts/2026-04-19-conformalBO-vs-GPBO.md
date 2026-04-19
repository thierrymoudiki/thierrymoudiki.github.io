---
layout: post
title: "Conformal Optimization Beats Bayesian Optimization, Optuna and Random Search on 72 classification Datasets"
description: "Conformal Bayesian Optimization Beats Gaussian Processes on 72 Datasets"
date: 2026-04-19
categories: Python
comments: true
---

Link to the notebook at the end of this post. The 72 datasets come from subsampled openml-cc18. 

# Conformal Optimization (GPopt-style, Lower Confidence Bounds via Conformal Intervals)

We consider the problem of minimizing an unknown function:
$$
f(x), \quad x \in \mathcal{X}
$$
accessible only through evaluations $$ y = f(x) + \varepsilon $$.

---

## **Inputs**
- Search space: $$ \mathcal{X} $$
- Initial design size: $$ n_{\text{init}} $$
- Number of iterations: $$ T $$
- Conformal miscoverage level: $$ \alpha $$
- Conformal regression model (e.g. any model wrapped to produce prediction intervals)

---

## **Step 0 — Initial Design**

Sample initial points:
$$
x_1, \dots, x_{n_{\text{init}}} \sim \text{Uniform}(\mathcal{X})
$$

Evaluate:
$$
y_i = f(x_i)
$$

Form dataset:
$$
\mathcal{D}_{n_{\text{init}}} = \{(x_i, y_i)\}_{i=1}^{n_{\text{init}}}
$$

---

## **Step 1 — Sequential Optimization Loop**

For $$ t = 1, \dots, T $$:

### **1. Fit surrogate model**

Train a regression model on all observed data:
$$
\hat{f}_t = \text{fit}(\mathcal{D})
$$

---

### **2. Conformal prediction**

Using conformal prediction, compute for any $$ x \in \mathcal{X} $$:

- Mean prediction:
$$
\mu_t(x) = \hat{f}_t(x)
$$

- Prediction interval:
$$
C_t(x) = [\ell_t(x), u_t(x)]
$$

where:
- $$ \ell_t(x) $$: lower conformal bound  
- $$ u_t(x) $$: upper conformal bound  

---

### **3. Acquisition function (LCB)**

Select the next point by minimizing the lower bound:
$$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \ell_t(x)
$$

---

### **4. Evaluate objective**

$$
y_{t+1} = f(x_{t+1})
$$

---

### **5. Update dataset**

$$
\mathcal{D} \leftarrow \mathcal{D} \cup \{(x_{t+1}, y_{t+1})\}
$$

---

## **Step 2 — Output**

Return the best observed point:
$$
x^* = \arg\min_{(x_i, y_i) \in \mathcal{D}} y_i
$$

---

## **Key Properties**

- **Surrogate model:** fully flexible (any regression model)
- **Uncertainty quantification:** via conformal prediction (distribution-free)
- **Acquisition function:** Lower Confidence Bound (LCB), implemented as:
$$
\mathrm{LCB}(x) = \ell(x)
$$

---

## **Remarks**

- No assumption of Gaussianity is required
- Prediction intervals may be:
  - asymmetric  
  - heteroscedastic  
  - nonparametric  
- Exploration is driven by **interval width**, not explicit variance
    

**Link to the notebook:**
[https://github.com/thierrymoudiki/2025-04-15-conformal-optimization-vs-bayesian-optimization/blob/main/2026-03-26-confbo-benchmark.ipynb](https://github.com/thierrymoudiki/2025-04-15-conformal-optimization-vs-bayesian-optimization/blob/main/2026-03-26-confbo-benchmark.ipynb)    

![xxx]({{base}}/images/2024-12-09/2024-12-09-image1.gif){:class="img-responsive"}          