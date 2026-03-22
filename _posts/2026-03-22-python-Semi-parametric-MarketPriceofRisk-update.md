---
layout: post
title: "Python version of 'Option pricing using time series models as market price of risk Pt.3'"
description: "Python version of 'Option pricing using time series models as market price of risk' and  resampling"
date: 2026-03-22
categories: Python
comments: true
---

This post is the Python version of [https://thierrymoudiki.github.io/blog/2026/03/16/r/Semi-parametric-MarketPriceofRisk-update](https://thierrymoudiki.github.io/blog/2026/03/16/r/Semi-parametric-MarketPriceofRisk-update), and 3rd part of [https://thierrymoudiki.github.io/blog/2025/12/07/r/forecasting/ARIMA-Pricing](https://thierrymoudiki.github.io/blog/2025/12/07/r/forecasting/ARIMA-Pricing) and [https://thierrymoudiki.github.io/blog/2026/02/01/r/Semi-parametric-MarketPriceofRisk-Theta](https://thierrymoudiki.github.io/blog/2026/02/01/r/Semi-parametric-MarketPriceofRisk-Theta). These posts showed how to use ARIMA and Theta as market price of risk, to then price options under a risk-neutral measure by resampling _martingale_ innovations. 

After thinking about it more, here's a condensed version of the previous posts, with some formulas and  Python code examples.

## 1. Market setting

Let

- $$S_t$$ = asset price  
- $$r$$ = risk-free rate  
- $$T$$ = maturity  

Define the **discounted price process**

$$
D_t = e^{-rt} S_t
$$

Under the no-arbitrage principle (Fundamental Theorem of Asset Pricing), there exists a probability measure $$Q$$ such that

$$
E_Q[D_t \mid \mathcal{F}_{t-1}] = D_{t-1}
$$

so $$D_t$$ is a **martingale**.

## 2. Empirical innovation extraction

Given simulated or observed price paths $$S_t$$, compute

$$
D_t = e^{-rt} S_t
$$

Define increments

$$
\Delta D_t = D_t - D_{t-1}
$$

Fit a time-series filter

$$
\Delta D_t = f(\Delta D_{t-1}, \ldots, \Delta D_{t-p}) + \varepsilon_t
$$

where

$$
E[\varepsilon_t] = 0
$$

## 3. Bootstrap innovation distribution

Let

$$
\{\varepsilon_1, \ldots, \varepsilon_T\}
$$

be the empirical innovations.

Generate bootstrap resamples

$$
\varepsilon_t^{(i)}, \quad i = 1, \ldots, N
$$

using stationary bootstrap. These sequences define the **innovation law**.

## 4. Martingale reconstruction

Define the discounted process recursively:

$$
D_0 = S_0
$$

$$
D_t = D_{t-1} + \varepsilon_t
$$

which implies

$$
D_t = S_0 + \sum_{i=1}^{t} \varepsilon_i
$$

Since

$$
E[\varepsilon_t] = 0
$$

we obtain

$$
E[D_t] = E[S_0 + \sum_{i=1}^{t} \varepsilon_i] = S_0 + \sum_{i=1}^{t} E[\varepsilon_i] = S_0
$$


## 5. Risk-neutral price process

Recover the price process

$$
S_t = e^{rt} D_t
$$

Then

$$
E[e^{-rt} S_t] = S_0
$$

which satisfies the **risk-neutral condition**.

## 6. Monte Carlo pricing

For payoff $$H(S_T)$$, the derivative price is

$$
V_0 = e^{-rT} E_Q[H(S_T)]
$$

Estimated by Monte Carlo:

$$
V_0 \approx e^{-rT} \frac{1}{N}
\sum_{i=1}^{N} H(S_T^{(i)})
$$

Example (European call):

$$
C_0 = e^{-rT} E_Q[\max(S_T - K, 0)]
$$

Here's the Python code for the whole process:

```python
"""
Option pricing using time series models as market price of risk (Pt. 3)
Python translation of: https://thierrymoudiki.github.io/blog/2026/03/16/r/Semi-parametric-MarketPriceofRisk-update

Methodology
-----------
1. Simulate asset price paths under the physical measure (GBM / Heston / SVJD).
2. Compute discounted prices  D_t = exp(-r·t) · S_t.
3. Fit an AR(1) or auto-ARIMA filter to ΔD_t; extract residuals.
4. Keep only "stationary" residuals (Ljung-Box p-value > 0.05).
5. Centre and stationary-block-bootstrap resample → innovation law.
6. Reconstruct risk-neutral paths:  D_t = D_{t-1} + ε_t,  S_t = exp(r·t)·D_t.
7. Price European calls & puts via Monte Carlo; compare with Black-Scholes.

Dependencies
------------
    pip install numpy pandas scipy statsmodels pmdarima matplotlib
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")          # comment out if running interactively
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Global parameters
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED       = 123
N_PATHS        = 250        # simulated paths (physical measure)
H              = 5          # horizon in years
FREQ           = 252        # trading days per year
N_STEPS        = H * FREQ   # total time steps
R              = 0.05       # risk-free rate
S0             = 100.0      # initial asset price
MU             = 0.08       # physical drift
SIGMA          = 0.04       # (base) volatility
N_SIMS         = 5_000      # Monte Carlo paths for pricing

CHOICE_PROCESS = "GBM"      # "GBM" | "Heston" | "SVJD"
CHOICE_FILTER  = "AR(1)"    # "AR(1)" | "auto.arima"
                            # NOTE: auto.arima is more accurate but slower

dt    = 1.0 / FREQ
t     = np.arange(0, N_STEPS + 1) * dt   # shape (N_STEPS+1,)
rng   = np.random.default_rng(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Simulation under the physical measure
# ─────────────────────────────────────────────────────────────────────────────

def sim_gbm(S0, mu, sigma, t, dt, n_paths, rng):
    """Geometric Brownian Motion — shape (T+1, n_paths)."""
    Z    = rng.standard_normal((len(t) - 1, n_paths))
    lr   = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + np.vstack([np.zeros(n_paths), np.cumsum(lr, axis=0)])
    return np.exp(logS)


def sim_heston(S0, mu, kappa=2.0, theta=SIGMA**2, xi=0.1,
               rho=-0.7, v0=SIGMA**2,
               dt=dt, n_steps=N_STEPS, n_paths=N_PATHS, rng=None):
    """Heston stochastic-volatility (no jumps), Euler-Maruyama."""
    S, v = np.zeros((n_steps + 1, n_paths)), np.zeros((n_steps + 1, n_paths))
    S[0] = S0;  v[0] = v0
    for i in range(n_steps):
        Z1  = rng.standard_normal(n_paths)
        Z2  = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp  = np.maximum(v[i], 0)
        v[i+1] = v[i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1
        S[i+1] = S[i] * np.exp((mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * Z2)
    return S


def sim_svjd(S0, mu, kappa=2.0, theta=SIGMA**2, xi=0.1,
             rho=-0.7, v0=SIGMA**2,
             lam=0.1, mu_J=0.0, sigma_J=0.02,
             dt=dt, n_steps=N_STEPS, n_paths=N_PATHS, rng=None):
    """Bates / SVJD: Heston + compound-Poisson jumps in log-price."""
    S, v = np.zeros((n_steps + 1, n_paths)), np.zeros((n_steps + 1, n_paths))
    S[0] = S0;  v[0] = v0
    for i in range(n_steps):
        Z1  = rng.standard_normal(n_paths)
        Z2  = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp  = np.maximum(v[i], 0)
        v[i+1] = v[i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1
        Nj  = rng.poisson(lam * dt, n_paths)
        J   = Nj * (mu_J + sigma_J * rng.standard_normal(n_paths))
        drift = (mu - 0.5 * vp - lam * (np.exp(mu_J + 0.5 * sigma_J**2) - 1)) * dt
        S[i+1] = S[i] * np.exp(drift + np.sqrt(vp * dt) * Z2 + J)
    return S


print("Simulating price paths …")
sim_GBM    = sim_gbm(S0, MU, SIGMA, t, dt, N_PATHS, rng)
sim_Heston = sim_heston(S0=S0, mu=MU, rng=rng)
sim_SVJD   = sim_svjd(S0=S0, mu=MU, rng=rng)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Discounted prices  D_t = exp(-r·t)·S_t
# ─────────────────────────────────────────────────────────────────────────────
discount    = np.exp(-R * t)[:, None]           # (T+1, 1)
disc_prices = {"GBM"   : discount * sim_GBM,
               "Heston": discount * sim_Heston,
               "SVJD"  : discount * sim_SVJD}[CHOICE_PROCESS]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  First differences  ΔD_t
# ─────────────────────────────────────────────────────────────────────────────
diff_mart  = np.diff(disc_prices, axis=0)       # (T, N_PATHS)
n_dates    = diff_mart.shape[0]
n_dates_1  = n_dates - 1

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Time-series filter → residuals
# ─────────────────────────────────────────────────────────────────────────────

def fit_ar1(y):
    """Fit AR(1) without intercept; return residuals (length T-1)."""
    X    = y[:-1].reshape(-1, 1)
    yy   = y[1:]
    coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    return yy - X @ coef


def fit_auto_arima(y):
    """Fit automatic ARIMA (no intercept); return residuals."""
    import pmdarima as pm
    model = pm.auto_arima(y, with_intercept=False, seasonal=False,
                          information_criterion="aic", stepwise=True,
                          error_action="ignore", suppress_warnings=True)
    return model.resid()


print(f"Fitting {CHOICE_FILTER} filter on {N_PATHS} paths …")
resids_matrix = np.zeros((n_dates_1, N_PATHS))

for j in range(N_PATHS):
    if j % 50 == 0:
        print(f"  path {j}/{N_PATHS}", end="\r")
    y = diff_mart[:, j]
    if CHOICE_FILTER == "AR(1)":
        resids_matrix[:, j] = fit_ar1(y)
    else:
        try:
            r_auto = fit_auto_arima(y)
            resids_matrix[-len(r_auto):, j] = r_auto
        except Exception:
            resids_matrix[:, j] = fit_ar1(y)   # fallback

print("\nFilter done.")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Keep stationary columns  (Ljung-Box p-value > 0.05)
# ─────────────────────────────────────────────────────────────────────────────
pvals = np.array([
    acorr_ljungbox(resids_matrix[:, j], lags=[10], return_df=True)
    ["lb_pvalue"].iloc[0]
    for j in range(N_PATHS)
])
stationary_cols = np.where(pvals > 0.05)[0]
print(f"Stationary paths kept: {len(stationary_cols)}/{N_PATHS}")

resids_stat  = resids_matrix[:, stationary_cols]
centered_res = resids_stat - resids_stat.mean(axis=0)   # centre each column

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Stationary block bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def stationary_block_bootstrap(series, n_out, rng, mean_block=10):
    """
    Geometric block-length stationary bootstrap.
    Wraps around the series circularly; returns array of length n_out.
    """
    T   = len(series)
    out = np.empty(n_out)
    i   = 0
    while i < n_out:
        start  = rng.integers(0, T)
        length = rng.geometric(1.0 / mean_block)
        block  = series[np.arange(start, start + length) % T]
        take   = min(length, n_out - i)
        out[i:i + take] = block[:take]
        i += take
    return out


print(f"Bootstrapping {N_SIMS} innovation sequences …")
M          = centered_res.shape[1]
n_rows     = centered_res.shape[0]
resampled  = np.zeros((n_rows, N_SIMS))
rng_boot   = np.random.default_rng(RNG_SEED + 7)

for s in range(N_SIMS):
    resampled[:, s] = stationary_block_bootstrap(
        centered_res[:, s % M], n_rows, rng_boot
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Reconstruct discounted paths → risk-neutral prices
#      D_0 = S_0,  D_t = D_0 + Σ_{i=1}^t ε_i,  S_t^Q = exp(r·t)·D_t
# ─────────────────────────────────────────────────────────────────────────────
D0 = S0
discounted_paths    = np.vstack([np.full(N_SIMS, D0),
                                 D0 + np.cumsum(resampled, axis=0)])
t_full              = np.arange(0, n_rows + 1) * dt     # (T+1,)
risk_neutral_prices = discounted_paths * np.exp(R * t_full)[:, None]

print(f"Risk-neutral price matrix shape: {risk_neutral_prices.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("I. Basic diagnostics")
print("="*60)

n_negative = (risk_neutral_prices < 0).sum()
print(f"Negative prices : {n_negative}")
assert n_negative == 0, "Negative prices detected — check parameters."

S_T = risk_neutral_prices[-1, :]
print("\nTerminal price S_T summary:")
print(pd.Series(S_T).describe().round(4).to_string())
print(f"Skewness        : {stats.skew(S_T):.4f}")
print(f"Excess kurtosis : {stats.kurtosis(S_T):.4f}")

print("\n" + "="*60)
print("II. Martingale check")
print("="*60)

T_years    = float(H)
discount_T = np.exp(-R * T_years)
mc_price   = discount_T * S_T.mean()
print(f"S_0                    : {S0}")
print(f"E[exp(-rT)·S_T]        : {mc_price:.4f}")
print(f"|error|                : {abs(mc_price - S0):.4f}")

disc_check = risk_neutral_prices * np.exp(-R * t_full)[:, None]
mean_disc  = disc_check.mean(axis=1)
print("\nMean discounted price – first 6 time points:")
print(np.round(mean_disc[:6], 4))
print("Mean discounted price – last 6 time points:")
print(np.round(mean_disc[-6:], 4))

tstat, pval = stats.ttest_1samp(discount_T * S_T - S0, popmean=0)
print(f"\nt-test H0: E[exp(-rT)·S_T] = S0  →  t={tstat:.4f}, p={pval:.4f}")

print("\n" + "="*60)
print("III. Log-return distribution")
print("="*60)

log_ret = np.diff(np.log(risk_neutral_prices), axis=0)
lr_T    = log_ret[-1, :]

sw_stat, sw_p = stats.shapiro(lr_T[:5000])
ks_stat, ks_p = stats.kstest(lr_T, "norm", args=(lr_T.mean(), lr_T.std()))
print(f"Shapiro-Wilk  : W={sw_stat:.6f}, p={sw_p:.4e}")
print(f"KS vs Normal  : D={ks_stat:.6f}, p={ks_p:.4e}")

mean_lr     = lr_T.mean()
theoretical = (R - 0.5 * SIGMA**2) * dt
print(f"\nMean terminal log-return  : {mean_lr:.6f}")
print(f"Theoretical (r-σ²/2)·dt  : {theoretical:.6f}")

emp_var  = float(np.var(np.log(S_T)))
theo_var = SIGMA**2 * T_years
print(f"\nVar(log S_T) empirical  : {emp_var:.4f}")
print(f"Var(log S_T) GBM theory : {theo_var:.4f}")
print(f"Ratio                   : {emp_var/theo_var:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  European option pricing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("IV. European option prices: MC vs Black-Scholes")
print("="*60)

strikes = [80, 90, 95, 100, 105, 110, 120]

mc_call = [np.exp(-R * T_years) * np.mean(np.maximum(S_T - K, 0)) for K in strikes]
mc_put  = [np.exp(-R * T_years) * np.mean(np.maximum(K - S_T, 0)) for K in strikes]


def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


bsc = [bs_call(S0, K, R, SIGMA, T_years) for K in strikes]
bsp = [bs_put(S0,  K, R, SIGMA, T_years) for K in strikes]

pcp_mc  = np.array(mc_call) - np.array(mc_put)
pcp_th  = S0 - np.array(strikes) * np.exp(-R * T_years)
pcp_err = np.abs(pcp_mc - pcp_th)

results = pd.DataFrame({
    "K"        : strikes,
    "BS_call"  : np.round(bsc, 4),
    "MC_call"  : np.round(mc_call, 4),
    "err_call" : np.round(np.abs(np.array(mc_call) - np.array(bsc)), 4),
    "BS_put"   : np.round(bsp, 4),
    "MC_put"   : np.round(mc_put, 4),
    "err_put"  : np.round(np.abs(np.array(mc_put) - np.array(bsp)), 4),
    "PCP_error": np.round(pcp_err, 4),
})
print(results.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Plots
# ─────────────────────────────────────────────────────────────────────────────
plt.close("all")   # ← add this line
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Semi-parametric option pricing  |  process={CHOICE_PROCESS}, filter={CHOICE_FILTER}",
    fontsize=13, fontweight="bold"
)

# Fan chart of risk-neutral paths
ax     = axes[0]
t_plot = np.linspace(0, T_years, risk_neutral_prices.shape[0])
q5, q50, q95 = (np.percentile(risk_neutral_prices, q, axis=1) for q in (5, 50, 95))
ax.fill_between(t_plot, q5, q95, alpha=0.30, color="steelblue", label="5–95 %")
ax.plot(t_plot, q50, color="steelblue", lw=2, label="Median")
ax.axhline(S0, color="black", ls="--", lw=1, label="$S_0$")
ax.set_title("Risk-neutral price paths (fan chart)")
ax.set_xlabel("Time (years)"); ax.set_ylabel("$S_t^Q$"); ax.legend()

# European call
ax = axes[1]
ax.plot(strikes, bsc,     "o-",  color="steelblue", label="Black-Scholes")
ax.plot(strikes, mc_call, "^--", color="coral",     label="Monte Carlo")
ax.set_title("European call prices")
ax.set_xlabel("Strike $K$"); ax.set_ylabel("Price"); ax.legend()

# European put
ax = axes[2]
ax.plot(strikes, bsp,    "o-",  color="steelblue", label="Black-Scholes")
ax.plot(strikes, mc_put, "^--", color="coral",     label="Monte Carlo")
ax.set_title("European put prices")
ax.set_xlabel("Strike $K$"); ax.set_ylabel("Price"); ax.legend()

plt.tight_layout()
plt.show()
```

```python
 Simulating price paths …
Fitting AR(1) filter on 250 paths …
  path 200/250
Filter done.
Stationary paths kept: 242/250
Bootstrapping 5000 innovation sequences …
Risk-neutral price matrix shape: (1260, 5000)

============================================================
I. Basic diagnostics
============================================================
Negative prices : 0

Terminal price S_T summary:
count    5000.0000
mean      128.5454
std        12.4298
min        78.0502
25%       120.1690
50%       128.4840
75%       136.8843
max       170.6847
Skewness        : 0.0190
Excess kurtosis : 0.0060

============================================================
II. Martingale check
============================================================
S_0                    : 100.0
E[exp(-rT)·S_T]        : 100.1112
|error|                : 0.1112

Mean discounted price – first 6 time points:
[100.      99.9947  99.994   99.9932  99.9927  99.997 ]
Mean discounted price – last 6 time points:
[100.1233 100.1241 100.1319 100.1295 100.1303 100.1311]

t-test H0: E[exp(-rT)·S_T] = S0  →  t=0.8125, p=0.4166

============================================================
III. Log-return distribution
============================================================
Shapiro-Wilk  : W=0.999111, p=1.0526e-02
KS vs Normal  : D=0.009983, p=6.9750e-01

Mean terminal log-return  : 0.000209
Theoretical (r-σ²/2)·dt  : 0.000195

Var(log S_T) empirical  : 0.0096
Var(log S_T) GBM theory : 0.0080
Ratio                   : 1.1949

============================================================
IV. European option prices: MC vs Black-Scholes
============================================================
  K  BS_call  MC_call  err_call  BS_put  MC_put  err_put  PCP_error
 80  37.6959  37.8075    0.1115  0.0000  0.0003   0.0003     0.1112
 90  29.9080  30.0226    0.1146  0.0001  0.0034   0.0034     0.1112
 95  26.0147  26.1352    0.1206  0.0008  0.0101   0.0093     0.1112
100  22.1260  22.2665    0.1405  0.0061  0.0353   0.0292     0.1112
105  18.2602  18.4462    0.1860  0.0343  0.1090   0.0748     0.1112
110  14.4727  14.7258    0.2531  0.1407  0.2826   0.1419     0.1112
120   7.6644   8.0484    0.3840  1.1205  1.3932   0.2727     0.1112
```

![image-title-here]({{base}}/images/2026-03-22/2026-03-22-image1.png){:class="img-responsive"}
