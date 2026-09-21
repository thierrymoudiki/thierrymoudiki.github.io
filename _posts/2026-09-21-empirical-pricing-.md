---
layout: post
title: "Python version of Semi-parametric option pricing based on underlying's historical data (accepted at the osQF 2026 (ex R/Finance) conference)"
description: "This post is a Python implementation of the study on semi-parametric option pricing based on underlying's historical data, accepted for presentation at the osQF 2026 conference."
date: 2026-09-21
categories: Python
comments: true
---

Python implementation of *"Semi-parametric option pricing based on underlying's historical data"* (accepted as osQF (ex R/Finance) 2026, [https://www.researchgate.net/publication/414513095_Semi-parametric_option_pricing_based_on_underlying's_historical_data](https://www.researchgate.net/publication/414513095_Semi-parametric_option_pricing_based_on_underlying's_historical_data)). I’m now interested in constructive remarks and feedback on the study, that will allow to enrich, improve and robustify the methodology.

**Idea:** build an empirical pricing _measure_ straight from the underlying's own historical dynamics — no option-market quotes needed anywhere.

1. Discount log-prices at rate `r`, take first differences.
2. Fit a no-intercept **AR(1)** to the differenced series and take residuals.
3. Resample the residuals with a **stationary block bootstrap** (Politis & Romano, 1994) — preserves dependence beyond the mean (e.g. volatility clustering) that an i.i.d. bootstrap would erase.
4. Cumulate resampled residuals to each horizon, exponentiate, then apply a single scalar **Duan–Simonato (1998)** correction so the discounted price is an *exact* martingale under the resampled measure — this is the minimal adjustment consistent with the Fundamental Theorem of Asset Pricing (FTAP).
5. Price European calls/puts, or arithmetic Asian options, as the discounted Monte Carlo average payoff.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Sequence, Literal

%matplotlib inline
```

## 1. Pricing module




```python
# --------------------------------------------------------------------
# Stationary block bootstrap (Politis & Romano, 1994)
# --------------------------------------------------------------------

def stationary_bootstrap(eps, n_sims, mean_block_length=None, rng=None):
    """
    Draw `n_sims` resampled paths of length len(eps) from `eps` using the
    stationary bootstrap of Politis & Romano (1994): blocks have i.i.d.
    Geometric(p) lengths with p = 1 / mean_block_length, so the resampled
    series stays (weakly) stationary, unlike a fixed-block bootstrap.
    """
    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    if rng is None:
        rng = np.random.default_rng()
    if mean_block_length is None:
        mean_block_length = max(1.0, n ** (1.0 / 3.0))
    p = 1.0 / mean_block_length

    out = np.empty((n, n_sims), dtype=float)
    for i in range(n_sims):
        idx = np.empty(n, dtype=int)
        pos = rng.integers(0, n)
        for t in range(n):
            if t > 0 and rng.random() >= p:
                pos = (pos + 1) % n          # continue current block
            elif t > 0:
                pos = rng.integers(0, n)      # geometric restart -> new block
            idx[t] = pos
        out[:, i] = eps[idx]
    return out
```


```python
# --------------------------------------------------------------------
# AR(1)-filtered, bootstrapped residual pool
# --------------------------------------------------------------------

@dataclass
class ResidualPool:
    """Everything needed to price options at a single risk-free rate r."""
    resampled_resids: np.ndarray   # (n_w, n_sims)
    n_w: int
    phi: float
    r: float
    dt: float
    S0: float
    ann_vol: float = 0.0
    skew: float = 0.0
    excess_kurt: float = 0.0


def eq_pool(prices, dates=None, r=0.0, dt=1.0/252.0, window=None,
            n_sims=10_000, seed=123, S0=None):
    """
    Build the AR(1)-filtered, stationary-block-bootstrapped residual pool
    for a given risk-free rate. Re-estimate per rate, as the paper does,
    since the no-intercept AR(1) is sensitive to the mean level of the
    discounted series being fit.
    """
    prices = np.asarray(prices, dtype=float)
    if window is not None:
        prices = prices[-(window + 1):]

    log_p = np.log(prices)
    t_index = np.arange(len(log_p)) * dt
    log_disc = log_p - r * t_index
    x = np.diff(log_disc)

    x_lag, x_lead = x[:-1], x[1:]
    phi = float(np.sum(x_lag * x_lead) / np.sum(x_lag ** 2))

    eps = np.empty_like(x)
    eps[1:] = x_lead - phi * x_lag
    eps[0] = x[0] - x.mean()
    eps = eps - eps.mean()

    rng = np.random.default_rng(seed)
    resampled = stationary_bootstrap(eps, n_sims=n_sims, rng=rng)
    resampled = resampled - resampled.mean(axis=0, keepdims=True)

    return ResidualPool(
        resampled_resids=resampled, n_w=len(eps), phi=phi, r=r, dt=dt,
        S0=float(S0 if S0 is not None else prices[-1]),
        ann_vol=float(np.std(x) * np.sqrt(1.0 / dt)),
        skew=float(np.mean(eps ** 3) / np.std(eps) ** 3),
        excess_kurt=float(np.mean(eps ** 4) / np.std(eps) ** 4 - 3.0),
    )
```


```python
# --------------------------------------------------------------------
# Terminal-price simulation with the Duan-Simonato martingale correction
# --------------------------------------------------------------------

def _simulate_terminal(pool, T, S0=None):
    """
    Cumulate bootstrapped residuals to horizon T, exponentiate, and apply
    the scalar Duan & Simonato (1998) correction so that
        S0 == E[ e^{-rT} * S_T ]
    holds *exactly* under the empirical measure (the FTAP condition).
    """
    S0 = pool.S0 if S0 is None else S0
    step_idx = max(1, min(round(T / pool.dt), pool.n_w))
    cum_resids = pool.resampled_resids[:step_idx, :].sum(axis=0)

    logD_T = np.log(S0) + cum_resids
    S_T_raw = np.exp(logD_T + pool.r * T)

    disc_T = np.exp(-pool.r * T)
    c_ds = S0 / (disc_T * S_T_raw.mean())
    S_T = S_T_raw * c_ds
    return S_T, c_ds, disc_T


def eq_price(pool, T, K, S0=None, type="call"):
    """Price a vector of European strikes K at maturity T from a pool."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    S_T, c_ds, disc_T = _simulate_terminal(pool, T, S0)

    if type == "call":
        payoff = lambda k: np.maximum(S_T - k, 0.0)
    elif type == "put":
        payoff = lambda k: np.maximum(k - S_T, 0.0)
    else:
        raise ValueError("type must be 'call' or 'put'")

    return np.array([disc_T * payoff(k).mean() for k in K])


def eq_price_asian(pool, T, K, n_fix, S0=None):
    """
    Price arithmetic-average Asian calls on n_fix equally spaced fixing
    dates up to T. Applies the Duan-Simonato correction date-by-date,
    since the payoff depends on the whole path.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    S0 = pool.S0 if S0 is None else S0

    fixing_times = np.linspace(T / n_fix, T, n_fix)
    fixing_idx = np.unique(np.clip(np.round(fixing_times / pool.dt).astype(int), 1, pool.n_w))
    max_idx = fixing_idx.max()

    cum_path = np.cumsum(pool.resampled_resids[:max_idx, :], axis=0)

    S_fix_cols = []
    for idx in fixing_idx:
        t_k = idx * pool.dt
        cum_resids = cum_path[idx - 1, :]
        S_raw = np.exp(np.log(S0) + cum_resids + pool.r * t_k)
        disc_k = np.exp(-pool.r * t_k)
        c_k = S0 / (disc_k * S_raw.mean())
        S_fix_cols.append(S_raw * c_k)

    S_fix = np.column_stack(S_fix_cols)
    avgA = S_fix.mean(axis=1)
    disc_T = np.exp(-pool.r * T)
    return np.array([disc_T * np.maximum(avgA - k, 0.0).mean() for k in K])
```


```python
# --------------------------------------------------------------------
# Black-Scholes reference prices (for comparison / validation only)
# --------------------------------------------------------------------

def bs_call(S, K, r, sigma, T):
    S, K, sigma, T = map(np.asarray, (S, K, sigma, T))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    return bs_call(S, K, r, sigma, T) - S + K * np.exp(-r * T)

def kv_geometric_call(S0, K, r, sigma, T, n_fix):
    """Closed-form Kemna-Vorst (1990) geometric-average Asian call."""
    m = np.log(S0) + (r - 0.5 * sigma ** 2) * (n_fix + 1) * T / (2 * n_fix)
    v2 = sigma ** 2 * (n_fix + 1) * (2 * n_fix + 1) / (6 * n_fix ** 2) * T
    v = np.sqrt(v2)
    d1 = (m - np.log(K)) / v + v
    d2 = d1 - v
    return np.exp(-r * T) * (np.exp(m + v2 / 2) * norm.cdf(d1) - K * norm.cdf(d2))
```

## 2. A historical price series

This pulls real daily closes with **`yfinance`** and falls back to a synthetic Student-t path if the fetch fails (e.g. no internet access, ticker unavailable, or rate-limited). Swap `TICKER` for any Yahoo Finance symbol — an index, a single stock, an FX pair, a private/illiquid name you track yourself, etc. The rest of the notebook is unchanged either way.


```python
try:
    import yfinance as yf
except ImportError:
    %pip install -q yfinance
    import yfinance as yf
```


```python
import yfinance as yf

TICKER = "^GDAXI"          # DAX index -- change to any Yahoo Finance ticker
START, END = "1998-07-05", "2002-07-05"

try:
    raw = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no rows")
    px = raw["Close"].dropna()
    if hasattr(px, "squeeze"):
        px = px.squeeze()          # yfinance can return a 1-col DataFrame
    prices = px.to_numpy().astype(float)
    dates = px.index.to_numpy()
    S0 = float(prices[-1])
    dt = 1 / 252
    print(f"Loaded {len(prices)} daily closes for {TICKER} from yfinance, "
          f"{str(dates[0])[:10]} to {str(dates[-1])[:10]}.")
except Exception as e:
    print(f"yfinance fetch failed ({e!r}); falling back to a synthetic price path.")
    rng = np.random.default_rng(123)
    n_days = 1045                  # ~4y of daily data
    dt = 1 / 252
    mu, sigma_true = 0.04, 0.28
    t_innov = rng.standard_t(df=5, size=n_days) / np.sqrt(5 / 3)
    log_rets = (mu - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * t_innov
    S0 = 4468.17
    prices = S0 * np.exp(np.cumsum(log_rets))
    prices = np.insert(prices, 0, S0)[:-1]

plt.figure(figsize=(9, 3))
plt.plot(prices)
plt.title(f"{TICKER} price path")
plt.xlabel("Trading day")
plt.ylabel("Price")
plt.tight_layout()
```

    Loaded 1015 daily closes for ^GDAXI from yfinance, 1998-07-06 to 2002-07-04.



    
![image-title-here]({{base}}/images/2026-09-21/2026-09-21-empirical-pricing-_9_1.png){:class="img-responsive"}
    


## 3. European calls & puts vs. Black-Scholes

**A note on the "systematic bias" you may see if you compare against an arbitrary volatility:** the model's only promise is *internal* no-arbitrage consistency (put-call parity, the martingale condition).


```python
from tqdm import tqdm

T_grid = [0.0389, 0.1139, 0.2083, 0.4583, 0.7111, 0.9583, 1.4556, 1.9528]
r_grid = [0.0357, 0.0349, 0.0341, 0.0355, 0.0359, 0.0368, 0.0386, 0.0401]
strikes = np.array([3400, 3600, 3800, 4000, 4200, 4400, 4500, 4600,
                     4800, 5000, 5200, 5400, 5600])

results = []
pools = {}
for T, r in tqdm(zip(T_grid, r_grid)):
    pool = eq_pool(prices, r=r, dt=dt, n_sims=10_000, seed=123, S0=S0)
    pools[r] = pool

    iv = pool.ann_vol   # <- use the pool's own realized vol, not an arbitrary one
    bs_c = bs_call(S0, strikes, r, iv, T)
    bs_p = bs_put(S0, strikes, r, iv, T)
    model_c = eq_price(pool, T, strikes, S0=S0, type="call")
    model_p = eq_price(pool, T, strikes, S0=S0, type="put")

    for k, bc, bp, mc, mp in zip(strikes, bs_c, bs_p, model_c, model_p):
        results.append(dict(T=T, r=r, K=k, bs_call=bc, model_call=mc,
                             bs_put=bp, model_put=mp))

print(f"Priced {len(results)} (T, K) European call/put pairs across {len(T_grid)} maturities.")
```

    8it [01:12,  9.12s/it]

    Priced 104 (T, K) European call/put pairs across 8 maturities.


    



```python
# Put-call parity check: model_call - model_put == S0 - K e^{-rT}
max_err = 0.0
for row in results:
    lhs = row["model_call"] - row["model_put"]
    rhs = S0 - row["K"] * np.exp(-row["r"] * row["T"])
    max_err = max(max_err, abs(lhs - rhs))
print(f"Max abs model put-call parity error: {max_err:.6e} ({100 * max_err / S0:.2e}% of S0)")
```

    Max abs model put-call parity error: 9.094947e-13 (2.14e-14% of S0)



```python
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, T in zip(axes.flat, T_grid):
    d = [r for r in results if r["T"] == T]
    d.sort(key=lambda r: r["K"])
    ks = [r["K"] for r in d]
    ax.plot(ks, [r["bs_call"] for r in d], "o--", color="steelblue",
            label="BS @ pool's realized vol", markerfacecolor="none")
    ax.plot(ks, [r["model_call"] for r in d], "o-", color="coral",
            label="Empirical model")
    ax.axvline(S0, color="gray", ls=":", lw=1)
    ax.set_title(f"T={T:.2f}")
    ax.set_xlabel("Strike")
    if ax is axes.flat[0]:
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
fig.suptitle("Model price vs Black-Scholes (matched vol), by maturity")
fig.tight_layout()
```


    
![image-title-here]({{base}}/images/2026-09-21/2026-09-21-empirical-pricing-_13_0.png){:class="img-responsive"}
    


## 4. Arithmetic Asian calls vs. a Black-Scholes Monte Carlo benchmark

The BS-MC engine is itself checked against the closed-form Kemna-Vorst geometric-average price before use, exactly as in the paper.


```python
def bs_mc_asian(T, r, sigma, K_vec, n_fix, n_sims=10_000, rng=None):
    rng = rng or np.random.default_rng(7)
    fixing_times = np.linspace(T / n_fix, T, n_fix)
    dts = np.diff(np.concatenate(([0.0], fixing_times)))
    Z = rng.standard_normal((n_fix, n_sims))
    log_incr = (r - 0.5 * sigma ** 2) * dts[:, None] + sigma * np.sqrt(dts)[:, None] * Z
    logS_path = np.log(S0) + np.cumsum(log_incr, axis=0)
    avgA = np.exp(logS_path).mean(axis=0)
    avgG = np.exp(logS_path.mean(axis=0))
    disc = np.exp(-r * T)
    price_arith = np.array([disc * np.maximum(avgA - k, 0.0).mean() for k in K_vec])
    price_geom = np.array([disc * np.maximum(avgG - k, 0.0).mean() for k in K_vec])
    return price_arith, price_geom

# sanity check the BS-MC engine against the closed-form KV geometric price
r_chk, sigma_chk, n_fix_chk = 0.036, 0.28, 12
strikes_asian = np.arange(4000, 5001, 200)
_, geom_mc = bs_mc_asian(1.0, r_chk, sigma_chk, strikes_asian, n_fix_chk, n_sims=200_000)
geom_cf = kv_geometric_call(S0, strikes_asian, r_chk, sigma_chk, 1.0, n_fix_chk)
print("BS-MC vs closed-form Kemna-Vorst, max abs diff:", np.max(np.abs(geom_mc - geom_cf)))
```

    BS-MC vs closed-form Kemna-Vorst, max abs diff: 0.5563267884775769



```python
T_asian = [0.25, 0.50, 0.75, 0.9583]

fig2, axes2 = plt.subplots(1, len(T_asian), figsize=(16, 4))
for ax, T in zip(axes2, T_asian):
    r = float(np.interp(T, T_grid, r_grid))
    pool = pools.get(r) or eq_pool(prices, r=r, dt=dt, n_sims=10_000, seed=123, S0=S0)
    iv = pool.ann_vol
    n_fix = max(1, round(T * 12))

    bsmc, _ = bs_mc_asian(T, r, iv, strikes_asian, n_fix)
    emp = eq_price_asian(pool, T, strikes_asian, n_fix, S0=S0)

    ax.plot(strikes_asian, bsmc, "o--", color="steelblue", markerfacecolor="none",
            label="BS Monte Carlo")
    ax.plot(strikes_asian, emp, "o-", color="coral", label="Empirical model")
    ax.set_title(f"T={T:.2f}")
    ax.set_xlabel("Strike")
    if ax is axes2[0]:
        ax.set_ylabel("Asian call price")
        ax.legend(fontsize=8)
fig2.suptitle("Empirical bootstrap vs Black-Scholes Monte Carlo, arithmetic Asian calls")
fig2.tight_layout()
```


    
![image-title-here]({{base}}/images/2026-09-21/2026-09-21-empirical-pricing-_16_0.png){:class="img-responsive"}
    

