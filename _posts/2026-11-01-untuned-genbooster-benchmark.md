---
layout: post
title: "genbooster, mlsauce, XGBoost, LightGBM — untuned small benchmark"
description: "A comparison of different boosting libraries on various datasets."
date: 2026-11-01
categories: Python
comments: true
---

Every scikit-learn regressor (minus an exclusion list) is used as the base
learner of [`genbooster`](https://github.com/Techtonique/genbooster) and of [`mlsauce.GenericBoosting*`](https://github.com/Techtonique/mlsauce), with identical settings and **no tuning**, on scikit-learn's bundled `load_*` datasets:

- **Regression** (`BoosterRegressor` vs `GenericBoostingRegressor`): `load_diabetes`
  — test RMSE, MAE, time.
- **Classification** (`BoosterClassifier` vs `GenericBoostingClassifier`; both boost
  regressors on one-hot targets): `load_iris`, `load_wine`, `load_breast_cancer`
  — test balanced accuracy, time.

Untuned `XGBoost` and `LightGBM` are reference points. Time = fit + predict.
(`load_linnerud` is left out: 20 samples, multi-output. `load_digits` is left out: 10
classes x 100 stages = 1,000 base learners per model, too slow and memory-hungry.)


```python
!pip install -U pip
```

    Requirement already satisfied: pip in /usr/local/lib/python3.13/dist-packages (26.2.2)



```python
!pip install git+https://github.com/Techtonique/genbooster.git

!pip install mlsauce
```

```python
import multiprocessing as mp
import resource
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlsauce as ms
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import (load_breast_cancer, load_diabetes, load_iris,
                              load_wine, make_friedman1)
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import all_estimators
from tqdm.auto import tqdm

from genbooster import BoosterClassifier, BoosterRegressor

warnings.filterwarnings("ignore")

# Shared, untuned settings (genbooster defaults, mirrored in mlsauce)
SEED = 42
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
N_HIDDEN_FEATURES = 5
TOLERANCE = 1e-4
TIMEOUT_S = 120     # per run, wall clock; slower runs are killed and recorded as failures
MEM_BUDGET_GB = 3   # per run; guards against base learners that store large fitted state
TEST_SIZE = 0.2

EXCLUDED = [
    "MultiOutputRegressor", "MultiOutputClassifier", "StackingRegressor", "StackingClassifier",
    "VotingRegressor", "VotingClassifier", "TransformedTargetRegressor", "RegressorChain",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor", "RandomForestRegressor",
    "ExtraTreesRegressor", "MLPRegressor",
]

REG_DATASETS = {"diabetes": load_diabetes(return_X_y=True)}
CLF_DATASETS = {
    "iris": load_iris(return_X_y=True),
    "wine": load_wine(return_X_y=True),
    "breast_cancer": load_breast_cancer(return_X_y=True),
}
```

## Models and helpers


```python
COMMON = dict(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
              n_hidden_features=N_HIDDEN_FEATURES, direct_link=True,
              dropout=0.0, tolerance=TOLERANCE)
GB_ONLY = dict(weights_distribution="uniform", random_state=SEED)
MS_ONLY = dict(weights_distr="uniform", activation="relu", row_sample=1.0, col_sample=1.0,
               seed=SEED, verbose=0)

BOOSTERS = {
    "regression": {
        "genbooster": lambda base: BoosterRegressor(base_estimator=base, **COMMON, **GB_ONLY),
        "mlsauce": lambda base: ms.GenericBoostingRegressor(base_model=base, **COMMON, **MS_ONLY),
    },
    "classification": {
        "genbooster": lambda base: BoosterClassifier(base_estimator=base, **COMMON, **GB_ONLY),
        "mlsauce": lambda base: ms.GenericBoostingClassifier(base_model=base, **COMMON, **MS_ONLY),
    },
}

# Untuned references: library defaults, only seed and logging set
REFERENCES = {
    "regression": {
        "XGBoost": lambda: xgb.XGBRegressor(random_state=SEED, verbosity=0),
        "LightGBM": lambda: lgb.LGBMRegressor(random_state=SEED, verbose=-1),
    },
    "classification": {
        "XGBoost": lambda: xgb.XGBClassifier(random_state=SEED, verbosity=0),
        "LightGBM": lambda: lgb.LGBMClassifier(random_state=SEED, verbose=-1),
    },
}


def seeded(base):
    """Same base-learner seed for both boosters (genbooster would otherwise set it itself)."""
    base = clone(base)
    if "random_state" in base.get_params():
        base.set_params(random_state=SEED)
    return base


def scores(task, y_te, pred):
    if task == "regression":
        if not np.isfinite(pred).all():
            raise ValueError("non-finite predictions")
        return {"rmse": mean_squared_error(y_te, pred) ** 0.5,
                "mae": mean_absolute_error(y_te, pred)}
    return {"balanced_accuracy": balanced_accuracy_score(y_te, pred)}


def evaluate(task, model, X_tr, X_te, y_tr, y_te):
    """Fit + predict; return the task's metrics and elapsed seconds."""
    t0 = time.perf_counter()
    pred = np.asarray(model.fit(X_tr, y_tr).predict(X_te)).ravel()
    elapsed = time.perf_counter() - t0
    if pred.shape != y_te.shape:
        raise ValueError(f"predictions have shape {pred.shape}, expected {y_te.shape}")
    return {**scores(task, y_te, pred), "time_s": elapsed}


def _vm_data_bytes():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) * 1024 for l in f if l.startswith("VmData"))


def _die_with_parent():
    """Linux: kill this child if the notebook kernel dies or is interrupted, so orphans
    don't keep using the CPU and skew later timings."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def _child(queue, task, model_fn, data):
    _die_with_parent()
    try:
        # Cap new allocations so one runaway base learner raises MemoryError instead of
        # getting the whole kernel OOM-killed
        limit = _vm_data_bytes() + int(MEM_BUDGET_GB * 2**30)
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
        queue.put(evaluate(task, model_fn(), *data))
    except MemoryError:
        queue.put({"error": f"memory >{MEM_BUDGET_GB} GB"})
    except Exception as e:
        queue.put({"error": f"{type(e).__name__}: {e}"[:80]})


# Each run happens in a forked child: a hard timeout, a memory cap, and memory returned
# after every run. Without fork (Windows, macOS default) runs happen in-process, unguarded.
FORK = "fork" in mp.get_all_start_methods() and hasattr(resource, "RLIMIT_DATA")


def safe_evaluate(row, model_fn, *data):
    if not FORK:
        try:
            row.update(evaluate(row["task"], model_fn(), *data))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:80]
        return row
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    p = ctx.Process(target=_child, args=(queue, row["task"], model_fn, data))
    p.start()
    try:
        row.update(queue.get(timeout=TIMEOUT_S))
    except Exception:  # queue.Empty: still running, or the child died
        row["error"] = f"timeout >{TIMEOUT_S}s" if p.is_alive() else f"crashed (exit {p.exitcode})"
    p.kill()
    p.join()
    return row


def split(task, X, y):
    if task == "classification":
        # Both packages' predict() returns class indices, so encode labels as 0..k-1
        y = LabelEncoder().fit_transform(y)
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
```

## Run the benchmark

Each dataset takes a few minutes. Runs slower than `TIMEOUT_S` or using more than `MEM_BUDGET_GB` are
recorded as failures.


```python
base_learners = [(n, c) for n, c in all_estimators(type_filter="regressor") if n not in EXCLUDED]

# Untimed warm-up so one-off import costs don't land on the first timed model
Xw, yw = make_friedman1(n_samples=60, random_state=0)
for task in BOOSTERS:
    yt = yw if task == "regression" else (yw > np.median(yw)).astype(int)
    for make in BOOSTERS[task].values():
        make(Ridge()).fit(Xw, yt).predict(Xw)

rows = []
for task, datasets in [("regression", REG_DATASETS), ("classification", CLF_DATASETS)]:
    for dname, (X, y) in datasets.items():
        data = split(task, X, y)
        for ref_name, ref_fn in REFERENCES[task].items():
            rows.append(safe_evaluate(dict(task=task, dataset=dname, base="-", model=ref_name),
                                      ref_fn, *data))
        for name, RegressorClass in tqdm(base_learners, desc=f"{task}: {dname}"):
            try:
                base = RegressorClass()
            except Exception as e:  # needs required constructor args
                rows.append(dict(task=task, dataset=dname, base=name, model="-",
                                 error=f"init: {e}"[:80]))
                continue
            for bname, make in BOOSTERS[task].items():
                rows.append(safe_evaluate(dict(task=task, dataset=dname, base=name, model=bname),
                                          lambda: make(seeded(base)), *data))

results = pd.DataFrame(rows)
results.to_csv("benchmark_vs_mlsauce.csv", index=False)
ok_mask = results["time_s"].notna()
print(f"{ok_mask.sum()} successful runs, {(~ok_mask).sum()} failed")
```


    regression: diabetes:   0%|          | 0/45 [00:00<?, ?it/s]



    classification: iris:   0%|          | 0/45 [00:00<?, ?it/s]



    classification: wine:   0%|          | 0/45 [00:00<?, ?it/s]



    classification: breast_cancer:   0%|          | 0/45 [00:00<?, ?it/s]


    285 successful runs, 83 failed


## Results

Untuned references:


```python
ok = results[results["time_s"].notna()]
METRICS = {"regression": ["rmse", "mae"], "classification": ["balanced_accuracy"]}
refs = {task: ok[(ok["task"] == task) & ok["model"].isin(REFERENCES[task])]
                .set_index(["dataset", "model"])[METRICS[task] + ["time_s"]]
        for task in METRICS}
pd.concat(refs, names=["task"]).round(3)
```





  <div id="df-1fb825e2-1652-4694-ac32-9fc63c9e3a7e" class="colab-df-container">
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
      <th></th>
      <th></th>
      <th>rmse</th>
      <th>mae</th>
      <th>time_s</th>
      <th>balanced_accuracy</th>
    </tr>
    <tr>
      <th>task</th>
      <th>dataset</th>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">regression</th>
      <th rowspan="2" valign="top">diabetes</th>
      <th>XGBoost</th>
      <td>57.46</td>
      <td>46.02</td>
      <td>0.19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>56.60</td>
      <td>44.85</td>
      <td>0.11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">classification</th>
      <th rowspan="2" valign="top">iris</th>
      <th>XGBoost</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.06</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.04</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">wine</th>
      <th>XGBoost</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.07</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.06</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">breast_cancer</th>
      <th>XGBoost</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.11</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.15</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1fb825e2-1652-4694-ac32-9fc63c9e3a7e')"
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
        document.querySelector('#df-1fb825e2-1652-4694-ac32-9fc63c9e3a7e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1fb825e2-1652-4694-ac32-9fc63c9e3a7e');
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




genbooster vs mlsauce, over the base learners on which both succeeded:


```python
PRIMARY = {"regression": ("rmse", False), "classification": ("balanced_accuracy", True)}
wide = {task: (ok[(ok["task"] == task) & ok["model"].isin(BOOSTERS[task])]
               .pivot_table(index=["dataset", "base"], columns="model",
                            values=METRICS[task] + ["time_s"])
               .dropna())
        for task in METRICS}

summary = []
for task, datasets in [("regression", REG_DATASETS), ("classification", CLF_DATASETS)]:
    metric, higher_better = PRIMARY[task]
    for dname in datasets:
        t = wide[task].loc[dname]
        g, m = t[metric]["genbooster"], t[metric]["mlsauce"]
        tie = np.isclose(g, m, rtol=1e-3, atol=1e-9)
        g_better = (g > m) if higher_better else (g < m)
        r = refs[task]
        summary.append({
            "task": task, "dataset": dname, "metric": metric,
            "base learners": len(t),
            "genbooster better": int((g_better & ~tie).sum()),
            "mlsauce better": int((~g_better & ~tie).sum()),
            "ties": int(tie.sum()),
            "best genbooster": f"{(g.max() if higher_better else g.min()):.3f} "
                               f"({g.idxmax() if higher_better else g.idxmin()})",
            "best mlsauce": f"{(m.max() if higher_better else m.min()):.3f} "
                            f"({m.idxmax() if higher_better else m.idxmin()})",
            "XGBoost": round(r.loc[(dname, "XGBoost"), metric], 3),
            "LightGBM": round(r.loc[(dname, "LightGBM"), metric], 3),
            "total time gb / ms (s)": f"{t['time_s']['genbooster'].sum():.0f} / "
                                      f"{t['time_s']['mlsauce'].sum():.0f}",
        })
summary = pd.DataFrame(summary).set_index(["task", "dataset"]).T
summary
```





  <div id="df-9299e82f-5f25-4ade-abfa-bcb8643567cf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>task</th>
      <th>regression</th>
      <th colspan="3" halign="left">classification</th>
    </tr>
    <tr>
      <th>dataset</th>
      <th>diabetes</th>
      <th>iris</th>
      <th>wine</th>
      <th>breast_cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>metric</th>
      <td>rmse</td>
      <td>balanced_accuracy</td>
      <td>balanced_accuracy</td>
      <td>balanced_accuracy</td>
    </tr>
    <tr>
      <th>base learners</th>
      <td>35</td>
      <td>34</td>
      <td>35</td>
      <td>34</td>
    </tr>
    <tr>
      <th>genbooster better</th>
      <td>10</td>
      <td>6</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>mlsauce better</th>
      <td>20</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ties</th>
      <td>5</td>
      <td>23</td>
      <td>29</td>
      <td>16</td>
    </tr>
    <tr>
      <th>best genbooster</th>
      <td>49.278 (SVR)</td>
      <td>1.000 (HuberRegressor)</td>
      <td>1.000 (BayesianRidge)</td>
      <td>0.981 (TweedieRegressor)</td>
    </tr>
    <tr>
      <th>best mlsauce</th>
      <td>49.217 (SVR)</td>
      <td>1.000 (HuberRegressor)</td>
      <td>1.000 (BayesianRidge)</td>
      <td>0.981 (TweedieRegressor)</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>57.46</td>
      <td>0.93</td>
      <td>1.00</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>56.60</td>
      <td>0.90</td>
      <td>1.00</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>total time gb / ms (s)</th>
      <td>143 / 148</td>
      <td>112 / 140</td>
      <td>123 / 154</td>
      <td>355 / 359</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9299e82f-5f25-4ade-abfa-bcb8643567cf')"
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
        document.querySelector('#df-9299e82f-5f25-4ade-abfa-bcb8643567cf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9299e82f-5f25-4ade-abfa-bcb8643567cf');
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


  <div id="id_0f686db7-9bbe-41ee-8444-c5ef2ded4981">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('summary')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0f686db7-9bbe-41ee-8444-c5ef2ded4981 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary');
      }
      })();
    </script>
  </div>

    </div>
  </div>




## Charts

One row per base learner; the two markers are the two boosters fitted on it (a
square inside a ring means they tie). Dashed / dotted vertical lines are untuned
XGBoost / LightGBM. Rows are sorted by the better of the two boosters, best on top.


```python
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, AXIS = "#e1e0d9", "#fcfcfb", "#c3c2b7"
STYLE = {  # color + marker shape: identity never rests on color alone
    # genbooster: larger ring drawn first, so a tie shows as a square inside a ring
    "genbooster": dict(color="#2a78d6", marker="o", s=90, zorder=3),
    "mlsauce": dict(color="#eb6834", marker="s", s=34, zorder=4),
}
REF_STYLE = {"XGBoost": "--", "LightGBM": ":"}


def dumbbell(ax, t, metric, higher_better, ref_x, title, top_n):
    r = t[metric]
    best = r.max(axis=1) if higher_better else r.min(axis=1)
    r = r.assign(best=best).sort_values("best", ascending=not higher_better).head(top_n)[::-1]
    ypos = np.arange(len(r))
    ax.set_facecolor(SURFACE)
    ax.hlines(ypos, r["genbooster"], r["mlsauce"], color=GRID, lw=2, zorder=1)
    for bname, st in STYLE.items():
        ax.scatter(r[bname], ypos, label=bname, edgecolor=SURFACE, linewidth=1.2, **st)
    for ref, ls in REF_STYLE.items():
        ax.axvline(ref_x[ref], color=INK_2, ls=ls, lw=1.2, zorder=2)
    # Reference labels sit above the plot area, stacked so they never hit the data
    for k, (ref, x) in enumerate(sorted(ref_x.items(), key=lambda kv: kv[1])):
        ax.annotate(f"{ref} {x:.3g}", (x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4 + 12 * k), textcoords="offset points",
                    fontsize=8.5, color=INK_2, ha="center", va="bottom")
    ax.margins(y=0.02)
    ax.set_yticks(ypos, r.index, fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=34)
    label = metric.replace("_", " ")
    ax.set_xlabel(f"test {label} ({'higher' if higher_better else 'lower'} is better)",
                  fontsize=9.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot_task(task, datasets, ncols, top_n, suptitle):
    metric, higher_better = PRIMARY[task]
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, nrows * (0.3 * top_n + 1.6) + 0.6),
                             facecolor=SURFACE, layout="constrained", squeeze=False)
    for ax, dname in zip(axes.flat, datasets):
        t = wide[task].loc[dname]
        ref_x = {ref: refs[task].loc[(dname, ref), metric] for ref in REF_STYLE}
        dumbbell(ax, t, metric, higher_better, ref_x,
                 f"{dname} — top {min(top_n, len(t))} base learners", top_n)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    ref_handles = [plt.Line2D([], [], color=INK_2, ls=ls, lw=1.2) for ls in REF_STYLE.values()]
    fig.legend(handles + ref_handles, labels + [f"{r} (untuned)" for r in REF_STYLE],
               loc="outside lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13.5, color=INK)
    return fig


fig = plot_task("regression", REG_DATASETS, ncols=1, top_n=25,
                suptitle="Regression (load_diabetes): untuned boosting of scikit-learn regressors")
fig.savefig("benchmark_regression.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.show()
```


    
![image-title-here]({{base}}/images/2026-11-01/2026-11-01-untuned-genbooster-benchmark_14_0.png){:class="img-responsive"}
    



```python
fig = plot_task("classification", CLF_DATASETS, ncols=3, top_n=15,
                suptitle="Classification: untuned boosting of scikit-learn regressors "
                         "(balanced accuracy)")
fig.savefig("benchmark_classification.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.show()
```


    
![image-title-here]({{base}}/images/2026-11-01/2026-11-01-untuned-genbooster-benchmark_15_0.png){:class="img-responsive"}
    

