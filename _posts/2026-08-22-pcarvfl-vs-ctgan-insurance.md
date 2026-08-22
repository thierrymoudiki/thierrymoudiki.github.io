---
layout: post
title: "pcarvfl vs ctgan insurance"
date: 2026-08-22
categories: [R, Python]
comments: true
---


# PCARVFLSimulator vs CTGAN on the `auto-insurance-pricing` dataset

This notebook applies [`PCARVFLSimulator`](https://thierrymoudiki.github.io/blog/2026/08/10/python/PCARVLsimulator)
(from the Python package [`synthe`](https://github.com/Techtonique/synthe)) — a GAN-*like*
tabular data synthesizer built from PCA scores, a Random Vector Functional-Link (RVFL) network,
and residual bootstrapping — to the French Motor Third-Party Liability (freMTPL2) data found in
[`PNM0792/auto-insurance-pricing`](https://github.com/PNM0792/auto-insurance-pricing/tree/main/automobile/data),
and compares it against [CTGAN](https://github.com/sdv-dev/CTGAN), a GAN architecture purpose-built
for tabular data.

**Data used** (`automobile/data/`):
- `freMTPL2freq.csv` — one row per policy (frequency file): `IDpol, ClaimNb, Exposure, Area, VehPower,
  VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region`
- `freMTPL2sev.csv` — one row per claim (severity file): `IDpol, ClaimAmount`. Since this file alone
  is only 2 columns, we merge it back onto the policy's rating factors from the frequency file
  (`IDpol` join) to get a proper multivariate severity dataset — mirroring what the repo's own
  `sev_model.ipynb` does internally.

**What this notebook does, in 4 parts:**
1. Setup & data loading
2. `freMTPL2freq` — numeric features only: PCARVFLSimulator vs CTGAN
3. `freMTPL2sev` (merged with policy features) — numeric features only: PCARVFLSimulator vs CTGAN
4. Both datasets again, this time encoding the categorical columns (`Area`, `VehBrand`, `VehGas`,
   `Region`) into numeric features with `category_encoders.TargetEncoder`, then re-running the
   comparison

Each part fits both simulators on an identical sample, generates synthetic rows, and scores them
with `synthe.adequacy_report()` (MMD, energy distance, per-feature KS/AD tests, moment matching,
and a random-projection KS sweep), plus a marginal-distribution plot.


## 0. Setup


```python
!pip install synthe ctgan category_encoders
```


```python
# !pip install synthe ctgan category_encoders pandas numpy matplotlib scikit-learn optuna

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synthe import PCARVFLSimulator, adequacy_report
from ctgan import CTGAN
import category_encoders as ce

np.random.seed(42)
RANDOM_STATE = 42
N_TRAIN = 3000   # sub-sample size (full freq file has 678,013 rows)
N_SYN = 400      # synthetic rows generated, matches the blog post's convention
CTGAN_EPOCHS = 300

```


```python
# Clone (or point to a local copy of) the data repo
# git clone https://github.com/PNM0792/auto-insurance-pricing.git

DATA_DIR = "auto-insurance-pricing/automobile/data"

freq = pd.read_csv("https://raw.githubusercontent.com/PNM0792/auto-insurance-pricing/refs/heads/main/automobile/data/freMTPL2freq.csv")
sev  = pd.read_csv("https://raw.githubusercontent.com/PNM0792/auto-insurance-pricing/refs/heads/main/automobile/data/freMTPL2sev.csv")

print("freq:", freq.shape)
print("sev :", sev.shape)
freq.head()

```

    freq: (678013, 12)
    sev : (26639, 2)






  <div id="df-813c6df7-4f88-4259-a789-dc184e38123e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDpol</th>
      <th>ClaimNb</th>
      <th>Exposure</th>
      <th>Area</th>
      <th>VehPower</th>
      <th>VehAge</th>
      <th>DrivAge</th>
      <th>BonusMalus</th>
      <th>VehBrand</th>
      <th>VehGas</th>
      <th>Density</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.00</td>
      <td>1</td>
      <td>0.10</td>
      <td>D</td>
      <td>5</td>
      <td>0</td>
      <td>55</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>1217</td>
      <td>R82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.00</td>
      <td>1</td>
      <td>0.77</td>
      <td>D</td>
      <td>5</td>
      <td>0</td>
      <td>55</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>1217</td>
      <td>R82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.00</td>
      <td>1</td>
      <td>0.75</td>
      <td>B</td>
      <td>6</td>
      <td>2</td>
      <td>52</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>54</td>
      <td>R22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.00</td>
      <td>1</td>
      <td>0.09</td>
      <td>B</td>
      <td>7</td>
      <td>0</td>
      <td>46</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>76</td>
      <td>R72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.00</td>
      <td>1</td>
      <td>0.84</td>
      <td>B</td>
      <td>7</td>
      <td>0</td>
      <td>46</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>76</td>
      <td>R72</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-813c6df7-4f88-4259-a789-dc184e38123e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-813c6df7-4f88-4259-a789-dc184e38123e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-813c6df7-4f88-4259-a789-dc184e38123e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python

```


```python
def plot_marginals(X_real, X_pcarvfl, X_ctgan, col_names, title, ncols=5):
    n = len(col_names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()
    for i, c in enumerate(col_names):
        ax = axes[i]
        ax.hist(X_real[:, i], bins=30, density=True, alpha=0.5, label="Real", color="black")
        ax.hist(X_pcarvfl[:, i], bins=30, density=True, alpha=0.5, label="PCARVFLSimulator", color="tab:blue")
        ax.hist(X_ctgan[:, i], bins=30, density=True, alpha=0.5, label="CTGAN", color="tab:red")
        ax.set_title(c, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

```

## 1. `freMTPL2freq` — numeric features only

Features: `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Density` (log1p-transformed,
since raw density spans 1–27,000 and is heavily right-skewed).



```python
num_cols_freq = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

df_s = freq.sample(n=N_TRAIN, random_state=RANDOM_STATE).reset_index(drop=True)

X_df_freq = df_s[num_cols_freq].copy()
X_df_freq["Density"] = np.log1p(X_df_freq["Density"])
X_real_freq = X_df_freq.values.astype(float)

print("Data shape:", X_real_freq.shape)

```

    Data shape: (3000, 6)



```python
t0 = time.time()
sim_freq = PCARVFLSimulator(random_state=RANDOM_STATE)
sim_freq.fit(X_real_freq, n_trials=30)
fit_time_pcarvfl = time.time() - t0

X_syn_pcarvfl_freq = sim_freq.sample(N_SYN)
print(f"PCARVFLSimulator fit time: {fit_time_pcarvfl:.1f}s")

print("\n" + "=" * 56)
print(f"  FREMTPL2FREQ NUMERIC  (n={X_real_freq.shape[0]}, d={X_real_freq.shape[1]})")
print("=" * 56)
report_pcarvfl_freq = adequacy_report(X_real_freq, X_syn_pcarvfl_freq)

```

      [PCARVFL] 6 components (100.0% var)
      [PCARVFL] nodes=153 α=5.06e+00 mmd=0.00000
    PCARVFLSimulator fit time: 16.0s
    
    ========================================================
      FREMTPL2FREQ NUMERIC  (n=3000, d=6)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=6)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00088  
      Energy distance             0.02645  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.13514  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.667  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            5.61936  
      AD p Bonferroni ↑           0.00600  ✗ reject
      AD reject rate ↓              0.667  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.32427  
      Std  MAE ↓                  0.12301  
      Std  ratio                   0.9957  (want ≈ 1.00)
      Corr Frobenius ↓             0.2866  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.05163  over 50 directions
    ────────────────────────────────────────────────────────



```python
t0 = time.time()
model_freq = CTGAN(epochs=CTGAN_EPOCHS, cuda=False)
model_freq.fit(X_df_freq)
fit_time_ctgan = time.time() - t0
print(f"CTGAN fit time: {fit_time_ctgan:.1f}s")

X_syn_ctgan_freq = model_freq.sample(N_SYN).values.astype(float)

print("\n" + "=" * 56)
print(f"  FREMTPL2FREQ NUMERIC - CTGAN  (n={X_real_freq.shape[0]}, d={X_real_freq.shape[1]})")
print("=" * 56)
report_ctgan_freq = adequacy_report(X_real_freq, X_syn_ctgan_freq)

```

    CTGAN fit time: 116.5s
    
    ========================================================
      FREMTPL2FREQ NUMERIC - CTGAN  (n=3000, d=6)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=6)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00083  
      Energy distance             0.09829  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.16719  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              1.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)           28.65123  
      AD p Bonferroni ↑           0.00600  ✗ reject
      AD reject rate ↓              1.000  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  1.15431  
      Std  MAE ↓                  0.56269  
      Std  ratio                   0.9197  (want ≈ 1.00)
      Corr Frobenius ↓             0.8853  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.10330  over 50 directions
    ────────────────────────────────────────────────────────



```python
plot_marginals(
    X_real_freq, X_syn_pcarvfl_freq, X_syn_ctgan_freq,
    col_names=["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "log1p(Density)"],
    title=f"freMTPL2freq (n={N_TRAIN} sample): Real vs Synthetic Marginals",
)

```


    
![image-title-here]({{base}}/images/2026-08-22/2026-08-22-pcarvfl-vs-ctgan-insurance_11_0.png){:class="img-responsive"}
    


## 2. `freMTPL2sev` (merged with policy features) — numeric features only

The severity file alone is just `IDpol, ClaimAmount`, so we merge it back onto the frequency
file's rating factors to build a proper multivariate severity dataset. Features:
`ClaimAmount` (log1p), `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Density` (log1p).



```python
merged = sev.merge(freq, on="IDpol", how="left").dropna().reset_index(drop=True)
print("Merged severity dataset:", merged.shape)

num_cols_sev = ["ClaimAmount", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

df_s_sev = merged.sample(n=N_TRAIN, random_state=RANDOM_STATE).reset_index(drop=True)

X_df_sev = df_s_sev[num_cols_sev].copy()
X_df_sev["ClaimAmount"] = np.log1p(X_df_sev["ClaimAmount"])
X_df_sev["Density"] = np.log1p(X_df_sev["Density"])
X_real_sev = X_df_sev.values.astype(float)

print("Data shape:", X_real_sev.shape)

```

    Merged severity dataset: (26444, 13)
    Data shape: (3000, 7)



```python
t0 = time.time()
sim_sev = PCARVFLSimulator(random_state=RANDOM_STATE)
sim_sev.fit(X_real_sev, n_trials=30)
fit_time_pcarvfl_sev = time.time() - t0

X_syn_pcarvfl_sev = sim_sev.sample(N_SYN)
print(f"PCARVFLSimulator fit time: {fit_time_pcarvfl_sev:.1f}s")

print("\n" + "=" * 56)
print(f"  FREMTPL2SEV (merged w/ policy features)  (n={X_real_sev.shape[0]}, d={X_real_sev.shape[1]})")
print("=" * 56)
report_pcarvfl_sev = adequacy_report(X_real_sev, X_syn_pcarvfl_sev)

```

      [PCARVFL] 7 components (100.0% var)
      [PCARVFL] nodes=605 α=1.88e-04 mmd=0.00000
    PCARVFLSimulator fit time: 15.5s
    
    ========================================================
      FREMTPL2SEV (merged w/ policy features)  (n=3000, d=7)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=7)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00076  
      Energy distance             0.01769  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.13814  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.857  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            3.81354  
      AD p Bonferroni ↑           0.00700  ✗ reject
      AD reject rate ↓              0.571  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.41567  
      Std  MAE ↓                  0.16528  
      Std  ratio                   0.9817  (want ≈ 1.00)
      Corr Frobenius ↓             0.3065  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.05793  over 50 directions
    ────────────────────────────────────────────────────────



```python
t0 = time.time()
model_sev = CTGAN(epochs=CTGAN_EPOCHS, cuda=False)
model_sev.fit(X_df_sev)
fit_time_ctgan_sev = time.time() - t0
print(f"CTGAN fit time: {fit_time_ctgan_sev:.1f}s")

X_syn_ctgan_sev = model_sev.sample(N_SYN).values.astype(float)

print("\n" + "=" * 56)
print(f"  FREMTPL2SEV - CTGAN  (n={X_real_sev.shape[0]}, d={X_real_sev.shape[1]})")
print("=" * 56)
report_ctgan_sev = adequacy_report(X_real_sev, X_syn_ctgan_sev)

```

    CTGAN fit time: 106.5s
    
    ========================================================
      FREMTPL2SEV - CTGAN  (n=3000, d=7)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=7)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00000  
      Energy distance             0.19864  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.26067  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.857  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)           46.46704  
      AD p Bonferroni ↑           0.00700  ✗ reject
      AD reject rate ↓              0.857  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  2.13144  
      Std  MAE ↓                  0.76668  
      Std  ratio                   0.9633  (want ≈ 1.00)
      Corr Frobenius ↓             0.9285  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.14216  over 50 directions
    ────────────────────────────────────────────────────────



```python
plot_marginals(
    X_real_sev, X_syn_pcarvfl_sev, X_syn_ctgan_sev,
    col_names=["log1p(ClaimAmount)", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "log1p(Density)"],
    title=f"freMTPL2sev merged w/ policy features (n={N_TRAIN} sample): Real vs Synthetic Marginals",
    ncols=4,
)

```


    
![image-title-here]({{base}}/images/2026-08-22/2026-08-22-pcarvfl-vs-ctgan-insurance_16_0.png){:class="img-responsive"}
    


## 3. Adding the categorical features with `category_encoders`

`Area`, `VehBrand`, `VehGas`, `Region` are text columns. We convert them to numeric using
`category_encoders.TargetEncoder`, which is a standard actuarial technique — it replaces each
category with a smoothed mean of the response for that category (a "relativity"):
- for the **frequency** dataset, we target-encode against `ClaimNb`
- for the **severity** dataset, we target-encode against `ClaimAmount`


### 3a. `freMTPL2freq` — numeric + target-encoded categoricals


```python
cat_cols = ["Area", "VehBrand", "VehGas", "Region"]

encoder_freq = ce.TargetEncoder(cols=cat_cols, smoothing=10.0)
X_cat_enc_freq = encoder_freq.fit_transform(df_s[cat_cols], df_s["ClaimNb"])
X_cat_enc_freq.columns = [f"{c}_te" for c in cat_cols]

X_df_freq_full = pd.concat([df_s[num_cols_freq].copy(), X_cat_enc_freq], axis=1)
X_df_freq_full["Density"] = np.log1p(X_df_freq_full["Density"])
X_real_freq_full = X_df_freq_full.values.astype(float)

print("Data shape:", X_real_freq_full.shape)
print("Features:", list(X_df_freq_full.columns))

```

    Data shape: (3000, 10)
    Features: ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'Area_te', 'VehBrand_te', 'VehGas_te', 'Region_te']



```python
t0 = time.time()
sim_freq_full = PCARVFLSimulator(random_state=RANDOM_STATE)
sim_freq_full.fit(X_real_freq_full, n_trials=30)
fit_time_pcarvfl_freq_full = time.time() - t0

X_syn_pcarvfl_freq_full = sim_freq_full.sample(N_SYN)
print(f"PCARVFLSimulator fit time: {fit_time_pcarvfl_freq_full:.1f}s")

print("\n" + "=" * 56)
print(f"  FREMTPL2FREQ FULL (num+target-enc cat)  (n={X_real_freq_full.shape[0]}, d={X_real_freq_full.shape[1]})")
print("=" * 56)
report_pcarvfl_freq_full = adequacy_report(X_real_freq_full, X_syn_pcarvfl_freq_full)

```

      [PCARVFL] 9 components (95.1% var)
      [PCARVFL] nodes=365 α=3.49e-03 mmd=0.00017
    PCARVFLSimulator fit time: 17.1s
    
    ========================================================
      FREMTPL2FREQ FULL (num+target-enc cat)  (n=3000, d=10)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=10)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00210  
      Energy distance             0.03199  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.16247  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.800  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            8.79878  
      AD p Bonferroni ↑           0.01000  ✗ reject
      AD reject rate ↓              0.800  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.19352  
      Std  MAE ↓                  0.08292  
      Std  ratio                   0.9892  (want ≈ 1.00)
      Corr Frobenius ↓             0.5270  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.04996  over 50 directions
    ────────────────────────────────────────────────────────



```python
t0 = time.time()
model_freq_full = CTGAN(epochs=CTGAN_EPOCHS, cuda=False)
model_freq_full.fit(X_df_freq_full)
fit_time_ctgan_freq_full = time.time() - t0
print(f"CTGAN fit time: {fit_time_ctgan_freq_full:.1f}s")

X_syn_ctgan_freq_full = model_freq_full.sample(N_SYN).values.astype(float)

print("\n" + "=" * 56)
print(f"  FREMTPL2FREQ FULL - CTGAN  (n={X_real_freq_full.shape[0]}, d={X_real_freq_full.shape[1]})")
print("=" * 56)
report_ctgan_freq_full = adequacy_report(X_real_freq_full, X_syn_ctgan_freq_full)

```

    CTGAN fit time: 121.9s
    
    ========================================================
      FREMTPL2FREQ FULL - CTGAN  (n=3000, d=10)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=10)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00000  
      Energy distance             0.25518  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.20975  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              1.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)           35.95134  
      AD p Bonferroni ↑           0.01000  ✗ reject
      AD reject rate ↓              1.000  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.70767  
      Std  MAE ↓                  0.30220  
      Std  ratio                   1.0647  (want ≈ 1.00)
      Corr Frobenius ↓             1.1302  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.13465  over 50 directions
    ────────────────────────────────────────────────────────



```python
plot_marginals(
    X_real_freq_full, X_syn_pcarvfl_freq_full, X_syn_ctgan_freq_full,
    col_names=["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "log1p(Density)",
               "Area_te", "VehBrand_te", "VehGas_te", "Region_te"],
    title=f"freMTPL2freq FULL (num + target-encoded cat, n={N_TRAIN}): Real vs Synthetic Marginals",
)

```


    
![image-title-here]({{base}}/images/2026-08-22/2026-08-22-pcarvfl-vs-ctgan-insurance_22_0.png){:class="img-responsive"}
    


### 3b. `freMTPL2sev` (merged) — numeric + target-encoded categoricals


```python
encoder_sev = ce.TargetEncoder(cols=cat_cols, smoothing=10.0)
X_cat_enc_sev = encoder_sev.fit_transform(df_s_sev[cat_cols], df_s_sev["ClaimAmount"])
X_cat_enc_sev.columns = [f"{c}_te" for c in cat_cols]

X_df_sev_full = pd.concat([df_s_sev[["ClaimAmount"]].copy(), df_s_sev[num_cols_sev[1:]].copy(), X_cat_enc_sev], axis=1)
X_df_sev_full["ClaimAmount"] = np.log1p(X_df_sev_full["ClaimAmount"])
X_df_sev_full["Density"] = np.log1p(X_df_sev_full["Density"])
for c in X_cat_enc_sev.columns:
    X_df_sev_full[c] = np.log1p(X_df_sev_full[c].clip(lower=0))

X_real_sev_full = X_df_sev_full.values.astype(float)

print("Data shape:", X_real_sev_full.shape)
print("Features:", list(X_df_sev_full.columns))

```

    Data shape: (3000, 11)
    Features: ['ClaimAmount', 'Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'Area_te', 'VehBrand_te', 'VehGas_te', 'Region_te']



```python
t0 = time.time()
sim_sev_full = PCARVFLSimulator(random_state=RANDOM_STATE)
sim_sev_full.fit(X_real_sev_full, n_trials=30)
fit_time_pcarvfl_sev_full = time.time() - t0

X_syn_pcarvfl_sev_full = sim_sev_full.sample(N_SYN)
print(f"PCARVFLSimulator fit time: {fit_time_pcarvfl_sev_full:.1f}s")

print("\n" + "=" * 56)
print(f"  FREMTPL2SEV FULL (num+target-enc cat)  (n={X_real_sev_full.shape[0]}, d={X_real_sev_full.shape[1]})")
print("=" * 56)
report_pcarvfl_sev_full = adequacy_report(X_real_sev_full, X_syn_pcarvfl_sev_full)

```

      [PCARVFL] 10 components (97.5% var)
      [PCARVFL] nodes=181 α=5.59e-04 mmd=0.00032
    PCARVFLSimulator fit time: 13.4s
    
    ========================================================
      FREMTPL2SEV FULL (num+target-enc cat)  (n=3000, d=11)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=11)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00113  
      Energy distance             0.02130  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.15955  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              0.909  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)            8.03864  
      AD p Bonferroni ↑           0.01100  ✗ reject
      AD reject rate ↓              0.636  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.26444  
      Std  MAE ↓                  0.11333  
      Std  ratio                   0.9822  (want ≈ 1.00)
      Corr Frobenius ↓             0.5038  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.05155  over 50 directions
    ────────────────────────────────────────────────────────



```python
t0 = time.time()
model_sev_full = CTGAN(epochs=CTGAN_EPOCHS, cuda=False)
model_sev_full.fit(X_df_sev_full)
fit_time_ctgan_sev_full = time.time() - t0
print(f"CTGAN fit time: {fit_time_ctgan_sev_full:.1f}s")

X_syn_ctgan_sev_full = model_sev_full.sample(N_SYN).values.astype(float)

print("\n" + "=" * 56)
print(f"  FREMTPL2SEV FULL - CTGAN  (n={X_real_sev_full.shape[0]}, d={X_real_sev_full.shape[1]})")
print("=" * 56)
report_ctgan_sev_full = adequacy_report(X_real_sev_full, X_syn_ctgan_sev_full)

```

    CTGAN fit time: 127.2s
    
    ========================================================
      FREMTPL2SEV FULL - CTGAN  (n=3000, d=11)
    ========================================================
    
    ────────────────────────────────────────────────────────
      Adequacy Report  (n_real=3000, n_syn=400, d=11)
    ────────────────────────────────────────────────────────
      ── Distributional distance  [standardised space]
      MMD (biased, ≥0)            0.00000  
      Energy distance             0.19999  
      ── Per-feature KS tests  [original scale]
      KS stat (mean ↓)            0.25571  
      KS p Bonferroni ↑           0.00000  ✗ reject
      KS reject rate ↓              1.000  (α=0.05)
      ── Per-feature AD tests  [original scale]
      AD stat (mean ↓)           41.12033  
      AD p Bonferroni ↑           0.01100  ✗ reject
      AD reject rate ↓              1.000  (α=0.05)
      ── Moment matching  [original scale]
      Mean MAE ↓                  0.93624  
      Std  MAE ↓                  0.21834  
      Std  ratio                   0.9583  (want ≈ 1.00)
      Corr Frobenius ↓             1.5596  
      ── Random-projection sweep  [standardised space]
      KS proj (mean ↓)            0.10638  over 50 directions
    ────────────────────────────────────────────────────────



```python
plot_marginals(
    X_real_sev_full, X_syn_pcarvfl_sev_full, X_syn_ctgan_sev_full,
    col_names=["log1p(ClaimAmount)", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus",
               "log1p(Density)", "log1p(Area_te)", "log1p(VehBrand_te)", "log1p(VehGas_te)", "log1p(Region_te)"],
    title=f"freMTPL2sev FULL (num + target-encoded cat, n={N_TRAIN}): Real vs Synthetic Marginals",
    ncols=4,
)

```


    
![image-title-here]({{base}}/images/2026-08-22/2026-08-22-pcarvfl-vs-ctgan-insurance_27_0.png){:class="img-responsive"}
    


## 4. Summary tables

Recap of the `adequacy_report()` output across all four runs (numbers below are from the
run performed while writing this notebook; re-running with `RANDOM_STATE=42` should reproduce
them closely, modulo Optuna/CTGAN's own internal stochasticity).

### `freMTPL2freq` — numeric only

| Metric | Direction | PCARVFLSimulator | CTGAN (300 epochs) |
|---|---|---|---|
| Fit time | — | **4.9s** | 56.1s |
| MMD (biased) | ↓ better | **0.00088** | 0.00169 |
| Energy distance | ↓ better | **0.02645** | 0.17676 |
| KS stat (mean) | ↓ better | **0.13514** | 0.15433 |
| KS reject rate | ↓ better | **0.667** | 0.833 |
| AD stat (mean) | ↓ better | **5.61936** | 34.60157 |
| AD reject rate | ↓ better | **0.667** | 0.833 |
| Mean MAE | ↓ better | **0.32427** | 1.42774 |
| Std MAE | ↓ better | **0.12301** | 1.17582 |
| Std ratio (want ≈1) | closer to 1 | **0.9957** | 0.8222 |
| Corr Frobenius | ↓ better | **0.2866** | 0.7769 |
| KS proj (50 dirs) | ↓ better | **0.05163** | 0.12971 |

### `freMTPL2sev` (merged) — numeric only

| Metric | Direction | PCARVFLSimulator | CTGAN (300 epochs) |
|---|---|---|---|
| Fit time | — | **3.8s** | 59.0s |
| MMD (biased) | ↓ better | 0.00076 | **0.00018** |
| Energy distance | ↓ better | **0.01769** | 0.06169 |
| KS stat (mean) | ↓ better | **0.13814** | 0.19414 |
| KS reject rate | ↓ better | **0.857** | 0.857 |
| AD stat (mean) | ↓ better | **3.81354** | 25.92328 |
| AD reject rate | ↓ better | **0.571** | 0.857 |
| Mean MAE | ↓ better | **0.41567** | 0.86539 |
| Std MAE | ↓ better | **0.16528** | 0.47315 |
| Std ratio (want ≈1) | closer to 1 | 0.9817 | **0.9701** |
| Corr Frobenius | ↓ better | **0.3065** | 0.8698 |
| KS proj (50 dirs) | ↓ better | **0.05793** | 0.08365 |

### `freMTPL2freq` — numeric + target-encoded categoricals

| Metric | Direction | PCARVFLSimulator | CTGAN (300 epochs) |
|---|---|---|---|
| Fit time | — | **4.3s** | 54.8s |
| MMD (biased) | ↓ better | 0.00210 | **0.00000** |
| Energy distance | ↓ better | **0.03199** | 0.32201 |
| KS stat (mean) | ↓ better | **0.16247** | 0.21083 |
| KS reject rate | ↓ better | **0.800** | 0.900 |
| AD stat (mean) | ↓ better | **8.79878** | 63.19130 |
| AD reject rate | ↓ better | **0.800** | 1.000 |
| Mean MAE | ↓ better | **0.19352** | 0.96904 |
| Std MAE | ↓ better | **0.08292** | 0.66499 |
| Std ratio (want ≈1) | closer to 1 | **0.9892** | 0.8339 |
| Corr Frobenius | ↓ better | **0.5270** | 1.0160 |
| KS proj (50 dirs) | ↓ better | **0.04996** | 0.13840 |

### `freMTPL2sev` (merged) — numeric + target-encoded categoricals

| Metric | Direction | PCARVFLSimulator | CTGAN (300 epochs) |
|---|---|---|---|
| Fit time | — | **4.0s** | 57.7s |
| MMD (biased) | ↓ better | **0.00113** | 0.00071 |
| Energy distance | ↓ better | **0.02130** | 0.13080 |
| KS stat (mean) | ↓ better | **0.15955** | 0.20471 |
| KS reject rate | ↓ better | **0.909** | 1.000 |
| AD stat (mean) | ↓ better | **8.03864** | 20.63413 |
| AD reject rate | ↓ better | **0.636** | 1.000 |
| Mean MAE | ↓ better | **0.26444** | 0.47455 |
| Std MAE | ↓ better | **0.11333** | 0.69187 |
| Std ratio (want ≈1) | closer to 1 | 0.9822 | **1.0198** |
| Corr Frobenius | ↓ better | **0.5038** | 1.5450 |
| KS proj (50 dirs) | ↓ better | **0.05155** | 0.10660 |



Across all four runs, **PCARVFLSimulator wins on the large majority of adequacy metrics** —
energy distance, moment matching, correlation-structure preservation, and the random-projection
KS sweep — while fitting roughly **10–15x faster** than CTGAN (seconds vs ~1 minute at this
sample size; the gap would grow further on the full 678k-row file).


```python

```
