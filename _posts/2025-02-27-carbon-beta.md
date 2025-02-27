---
layout: post
title: "Presenting 'Online Probabilistic Estimation of Carbon Beta and Carbon Shapley Values for Financial and Climate Risk' at Institut Louis Bachelier" 
description: "Presenting 'Online Probabilistic Estimation of Carbon Beta and Carbon Shapley Values for Financial and Climate Risk' at Institut Louis Bachelier for the 18th FINANCIAL RISKS INTERNATIONAL FORUM"
date: 2025-02-27
categories: [Python, R, Forecasting]
comments: true
---

Link to the preprint at the end of the post.

As climate change becomes a key financial risk factor, investors seek reliable ways to measure the exposure of stocks to climate transition risks. This paper introduces methods to estimate **Carbon Beta** and **Carbon Shapley values** dynamically and probabilistically.  

- **Carbon Beta** measures how stock returns react to a *Brown Minus Green (BMG)* portfolio, which holds long positions in carbon-intensive (brown) stocks and short positions in climate-friendly (green) stocks.  
- **Carbon Shapley values**, inspired by game theory, quantify the contribution of input factors to model predictions, helping explain stock return sensitivities.  

Unlike traditional methods that assume a fixed, linear relationship between stock and market returns, this approach is **adaptive, nonparametric, and uncertainty-aware**.  

## Context: From CAPM to Carbon Beta  

The **Capital Asset Pricing Model (CAPM)** (Sharpe, 1964) introduced **Beta** as a measure of stock risk relative to the market. Over time, more sophisticated approaches—machine learning (ML), neural networks, and game-theoretic Shapley values—have emerged.  

**Carbon Beta** extends this concept, capturing **climate risk** by analyzing how a stock's returns move with a BMG portfolio. A **high Carbon Beta** means a stock is highly exposed to the risks of transitioning to a low-carbon economy.  

## Proposed Methodology  

This study introduces a **machine learning (ML)-based, online estimation** of Carbon Beta and Carbon Shapley values. Key innovations include:  

1. **No assumption of a “true” Carbon Beta**  
   - Uses numerical derivatives instead of a fixed linear model.  

2. **Uncertainty quantification**  
   - Employs **conformal prediction** to provide confidence intervals around estimates.  

3. **Model-agnostic Shapley values**  
   - Computes dynamic Shapley values to understand the influence of climate risk factors on stock returns.  

The core model is a **Random Vector Functional Link (RVFL) neural network**, trained online to adjust continuously to new market data.  

**Link to the preprint:**

[https://www.researchgate.net/publication/387577137_Online_probabilistic_estimation_of_carbon_beta_and_carbon_Shapley_values_for_financial_and_climate_risk](https://www.researchgate.net/publication/387577137_Online_probabilistic_estimation_of_carbon_beta_and_carbon_Shapley_values_for_financial_and_climate_risk)

![image-title-here]({{base}}/images/2025-02-27/2025-02-27-image1.png)
