---
layout: post
title: "Simulating Stochastic Scenarios with Diffusion Models: A Guide to Using techtonique.net's API for the purpose"
description: "How to make API calls to techtonique.net for stochastic simulation using diffusion models"
date: 2025-05-29
categories: [R, Python, Techtonique]
comments: true
---

This blog post demonstrates how to use the (work in progress) stochastic simulation API provided by [techtonique.net](https://www.techtonique.net) to generate scenarios using various diffusion models. We'll explore how to simulate paths using:

1. Geometric Brownian Motion (GBM)
2. Cox-Ingersoll-Ross (CIR) process
3. Ornstein-Uhlenbeck (OU) process
4. Gaussian _Shocks_ scenarios

These models are particularly useful for:
- Financial simulation
- Risk assessment
- Portfolio stress testing
- Economic scenario analysis

We'll walk through examples showing how to:
- Make API calls with proper authentication
- Generate multiple scenarios with different parameters

The API supports (for now) various parameters including (also, [read the docs](https://www.techtonique.net/docs)):
- Number of scenarios
- Time horizon
- Frequency (daily, weekly, monthly, quarterly, yearly)
- Initial values
- Model-specific parameters
- Random seed for reproducibility

Let's get started!

First, get a token from: [https://www.techtonique.net/token](https://www.techtonique.net/token).

Now, here's how to use the token in API requests for stochatic simulations. The JSON response contains a key "sims". Each list is a future scenario, as envisaged by the chosen model, and based on the initial value 100.

```bash
# Replace YOUR_TOKEN_HERE with your actual token
curl -X GET "https://www.techtonique.net/scenarios/simulate/?model=GBM&n=6&horizon=5&frequency=quarterly&x0=100&theta1=0.1&theta2=0.2&theta3=0.3&seed=123" \
     -H "accept: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

```bash
curl -X GET "https://www.techtonique.net/scenarios/simulate/?model=CIR&n=6&horizon=5&frequency=quarterly&x0=100&theta1=0.1&theta2=0.2&theta3=0.3&seed=123" \
     -H "accept: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

```bash
curl -X GET "https://www.techtonique.net/scenarios/simulate/?model=OU&n=6&horizon=5&frequency=quarterly&x0=100&theta1=0.1&theta2=0.2&theta3=0.3&seed=123" \
     -H "accept: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

```bash
curl -X GET "https://www.techtonique.net/scenarios/simulate/?model=shocks&n=6&horizon=5&frequency=quarterly&seed=123" \
     -H "accept: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

![image-title-here]({{base}}/images/2023-10-09/2023-10-09-image1.png){:class="img-responsive"}
