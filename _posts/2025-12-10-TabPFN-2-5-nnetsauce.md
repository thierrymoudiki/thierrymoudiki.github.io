---
layout: post
title: "Zero-Shot Probabilistic Time Series Forecasting with TabPFN 2.5 and nnetsauce"
description: "Univariate and multivariate probabilistic time series zero-shot forecasting with TabPFN 2.5 and nnetsauce"
date: 2025-12-10
categories: Python
comments: true
---

Probabilistic time series forecasting has seen major progress in the last few years, with foundation models beginning to reshape what’s possible even for small datasets. In this post, we explore how **TabPFN 2.5**, a powerful pretrained transformer for tabular regression, can be combined with **nnetsauce**’s `MTS` framework to produce **fast, distribution-aware forecasts** for both **univariate** and **multivariate** series.

Using classic datasets such as **AirPassengers**, **a10**, and **USAccDeaths**, as well as a multivariate example linking **ice cream sales and heater usage**, we illustrate how TabPFN can be turned into a plug-and-play probabilistic forecaster. With just a few lines of code, we obtain **prediction intervals, sample paths, and full predictive distributions**, all while leveraging TabPFN’s zero-shot capabilities.

Pros: If you're looking for a lightweight yet highly flexible approach to probabilistic forecasting—without training complex deep learning models—this workflow offers a powerful option.

Cons: The process can be slow.


```python
!pip install --upgrade tabpfn-client
```


```python
!pip install nnetsauce
```


```python
import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import RidgeCV
from time import time
from tabpfn_client import init, TabPFNRegressor
```


```python


url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/AirPassengers.csv"
df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date)
df.drop(columns=['date'], inplace=True)
df_air = df.copy()

regr_tabpfn = ns.MTS(obj=TabPFNRegressor(model_path="v2.5_real"),
                     type_pi="scp2-kde",
                      replications=250,
                      kernel='gaussian',
                      lags=25, verbose=False,
                      show_progress=False)
start = time()
regr_tabpfn.fit(df_air)
regr_tabpfn.predict(h=40)
print("time: ", time() - start)

regr_tabpfn.plot(type_plot="pi")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">########  ########   ###  #########  #########       ###         #####     ########  ########</span>
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">     ###        ##   ###  ###   ###        ###       ###        ###  ###   ##   ###  ###     </span>
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">########  #######    ###  ###   ###  #######         ###        ########   ######    ########</span>
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">###       ###   ##   ###  ###   ###  ###   ###       ###        ###  ###   ##   ###       ###</span>
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">###       ###   ##   ###  #########  ###   ###       ########   ###  ###   ########  ########                      </span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Thanks for being part of the journey</span>

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  TabPFN is under active development, please help us improve and report any bugs/ideas you find.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  <span style="color: #000080; text-decoration-color: #000080">Report issues: </span><span style="color: #000080; text-decoration-color: #000080; text-decoration: underline">https://github.com/priorlabs/tabpfn-client/issues</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  <span style="color: #000080; text-decoration-color: #000080">Press Ctrl+C anytime to exit</span>
</pre>



    
    Opening browser for login. Please complete the login/registration process in your browser and return here.
    
    
    Could not open browser automatically. Falling back to command-line login...
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[1]    </span> Create a TabPFN account     
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[2]    </span> Login to your TabPFN account
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">[q]    </span> Quit                        
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #000080; text-decoration-color: #000080; font-weight: bold">→</span> Choose <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/q<span style="font-weight: bold">)</span>: </pre>



    2



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="font-weight: bold">Login</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Email: </pre>

    Password: ··········



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">Login successful!</span>
</pre>



    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]


    time:  408.57407093048096



    
![image-title-here]({{base}}/images/2025-12-05/2025-12-05-TabPFN-2-5-nnetsauce_3_18.png){:class="img-responsive"}
    



```python
url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/a10.csv"
df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date)
df.drop(columns=['date'], inplace=True)
df_a10 = df.diff().dropna().copy()

regr_tabpfn = ns.MTS(obj=TabPFNRegressor(model_path="v2.5_real"),
                     type_pi="scp2-kde",
                      replications=250,
                      kernel='gaussian',
                      lags=25, verbose=False,
                      show_progress=False)
start = time()
regr_tabpfn.fit(df_a10)
regr_tabpfn.predict(h=40)
print("time: ", time() - start)

regr_tabpfn.plot(type_plot="pi")
```

    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing:   0%|          | [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]


    time:  68.24992322921753



    
![image-title-here]({{base}}/images/2025-12-05/2025-12-05-TabPFN-2-5-nnetsauce_4_2.png){:class="img-responsive"}
    



```python
import pandas as pd

url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/USAccDeaths.csv"
df2 = pd.read_csv(url)
df2.index = pd.DatetimeIndex(df2.date)
df2.drop(columns=['date'], inplace=True)

df_usacc = df2.copy()

regr_tabpfn = ns.MTS(obj=TabPFNRegressor(model_path="v2.5_real"),
                     type_pi="scp2-kde",
                      replications=250,
                      kernel='gaussian',
                      lags=25, verbose=False,
                      show_progress=False)
start = time()
regr_tabpfn.fit(df_usacc)
regr_tabpfn.predict(h=40)
print("time: ", time() - start)

regr_tabpfn.plot(type_plot="pi")
```

    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing:   0%|          | [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing:   0%|          | [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]


    time:  67.44699144363403



    
![image-title-here]({{base}}/images/2025-12-05/2025-12-05-TabPFN-2-5-nnetsauce_5_2.png){:class="img-responsive"}
    



```python

```


```python
url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/multivariate/ice_cream_vs_heater.csv"
df_temp = pd.read_csv(url)
df_temp.index = pd.DatetimeIndex(df_temp.date)
df = df_temp.drop(columns=['date']).diff().dropna()

df_heat = df.copy()

regr_tabpfn = ns.MTS(obj=TabPFNRegressor(model_path="v2.5_real"),
                     type_pi="scp2-kde",
                      replications=250,
                      kernel='gaussian',
                      lags=25, verbose=False,
                      show_progress=False)
start = time()
regr_tabpfn.fit(df_heat)
regr_tabpfn.predict(h=40)
print("time: ", time() - start)

regr_tabpfn.plot("heater", type_plot="spaghetti")
```

    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:00<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]
    Processing: 100%|██████████| [00:01<00:00]


    time:  145.63059449195862



    
![image-title-here]({{base}}/images/2025-12-05/2025-12-05-TabPFN-2-5-nnetsauce_7_2.png){:class="img-responsive"}
    



```python
regr_tabpfn.plot("icecream", type_plot="spaghetti")
```


    
![image-title-here]({{base}}/images/2025-12-05/2025-12-05-TabPFN-2-5-nnetsauce_8_0.png){:class="img-responsive"}
    

