---
layout: post
title: "Transfer Learning using ahead::ridge2f on synthetic stocks returns Pt.2: synthetic data generation"
description: "Synthetic data generation."
date: 2025-09-09
categories: [R, Python]
comments: true
---

In [https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f](https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f), I pretrain [`ahead::ridge2f`](https://docs.techtonique.net/ahead/index.html) (also [available Python](https://docs.techtonique.net/ahead_python/ahead.html#Ridge2Regressor)) on 1000 synthetic stock returns using Bayesian Optimization, and test its performance (coverage rate and [Winkler score](https://www.otexts.com/fpp3/distaccuracy.html#winkler-score) for now) on real market data. 

Here's how I obtained the synthetic stock returns:

The overall process in the model simulates asset returns over time by integrating stochastic volatility, regime switching, jumps, and microstructure noise. At each time step $$ t $$, the return $$ r_t $$ is given by:

$$
r_t = \mu + \sqrt{V_t} \cdot \epsilon_t + J_t + \epsilon_{\text{noise}}
$$

where $$ \mu $$ is the drift term, $$ \sqrt{V_t} $$ is the volatility at time $$ t $$, $$ \epsilon_t \sim \mathcal{N}(0, 1) $$ is a random shock from a standard normal distribution, $$ J_t $$ is the jump component, and $$ \epsilon_{\text{noise}} $$ represents microstructure noise. The volatility process $$ V_t $$ follows a mean-reverting SDE inspired by the Heston model:

$$
dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} dW_t
$$

where $$ \kappa $$ is the mean reversion speed, $$ \theta $$ is the long-term mean volatility, $$ \sigma_v $$ is the volatility of volatility, and $$ W_t $$ is a Wiener process. The jumps are modeled using a Poisson process with intensity $$ \lambda_{\text{jump}} $$, and the size of the jump $$ J_t $$ is drawn from one of three distributions: normal $$ \mathcal{N}(0, \sigma_{\text{jump}}) $$, log-normal $$ \log J_t \sim \mathcal{N}(-\frac{1}{2} \sigma_{\text{jump}}^2, \sigma_{\text{jump}}) $$, or exponential $$ J_t \sim \text{Exp}(\sigma_{\text{jump}}) $$. Regime switching is modeled as a two-state Markov process, where the volatility parameters $$ \kappa $$ and $$ \theta $$ are different for each state, and transitions between regimes are governed by a transition matrix $$ \mathbf{P} $$, where:
$$
\mathbf{P} = \begin{bmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{bmatrix}
$$
with $$ p_{11} $$ and $$ p_{22} $$ being the probabilities of staying in the current state, and $$ p_{12} $$ and $$ p_{21} $$ being the probabilities of switching regimes. Finally, the microstructure noise $$ \epsilon_{\text{noise}} $$ is modeled as either normal $$ \mathcal{N}(0, \text{noise scale}) $$ or Student's t-distribution $$ t_{\nu} $$, capturing small-scale market effects. Together, these components combine to simulate realistic asset return dynamics, reflecting continuous volatility evolution, discrete jumps, regime shifts, and market microstructure noise.


More details about this model (actually used in an industrial setting):

- [https://thierrymoudiki.github.io/blog/2025/07/01/r/python/ridge2-bayesian](https://thierrymoudiki.github.io/blog/2025/07/01/r/python/ridge2-bayesian)
- [https://www.mdpi.com/2227-9091/6/1/22](https://www.mdpi.com/2227-9091/6/1/22)
- [https://thierrymoudiki.github.io/blog/2024/02/26/python/r/julia/ahead-v0100](https://thierrymoudiki.github.io/blog/2024/02/26/python/r/julia/ahead-v0100)
- [https://thierrymoudiki.github.io/blog/2025/09/08/r/python/pretraining-ridge2f](https://thierrymoudiki.github.io/blog/2025/09/08/r/python/pretraining-ridge2f)
- [Doc for R](https://docs.techtonique.net/ahead/index.html)
- [Doc for Python](https://docs.techtonique.net/ahead_python/ahead.html#Ridge2Regressor)
