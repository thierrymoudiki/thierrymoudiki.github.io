---
layout: post
title: "Diffusion models in Python with esgtoolkit"
description: Diffusion models in Python with esgtoolkit.
date: 2023-12-11
categories: [R, Python]
---

A few weeks ago, on 2023-10-02, I announced **esgtoolkit** `v1.0.0` for Python. 

Well, `v1.0.0` for Python is more like a _proof of concept_ as of today, as there isn't an exact mapping with the [R API](https://techtonique.r-universe.dev/esgtoolkit) so far. Next week, most likely in a `v1.1.0` for both, the Python version will be aligned with the R version -- **as much as possible**. 

For those who aren't familiar with **esgtoolkit** yet, I've been developing and maintaining it (with a lot of roller coasters, still not sure why) for R since 2014. e.s.g here, stands for _Economic Scenarios Generators_, but the name has become less relevant since diffusion models are widely used in Physics and -- more recently -- in Generative AI (for images). 

You can read this document (focusing on quantitative finance) for an introductory review: [https://www.researchgate.net/publication/338549100_ESGtoolkit_a_tool_for_stochastic_simulation_v020](https://www.researchgate.net/publication/338549100_ESGtoolkit_a_tool_for_stochastic_simulation_v020). 


# Examples of use of `esgtoolkit` in Python 

## 1 - Install and import packages


```python
!pip install matplotlib numpy pandas esgtoolkit
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esgtoolkit import simdiff
```

## 2 - Code examples

### 2 - 1 Ornstein-Uhlenbeck process 


```python
kappa = 1.5
V0 = theta = 0.04
sigma_v = 0.2
theta1 = kappa * theta
theta2 = kappa
theta3 = sigma_v

sims_OU = simdiff(
    n=7,
    horizon=5,
    frequency="quarterly",
    model="OU",
    x0=V0,
    theta1=theta1,
    theta2=theta2,
    theta3=theta3,
)

print(sims_OU)
```

          Series 1  Series 2  Series 3  Series 4  Series 5  Series 6  Series 7
    0.00  0.040000  0.040000  0.040000  0.040000  0.040000  0.040000  0.040000
    0.25 -0.007010 -0.049564 -0.018269  0.071842  0.040483 -0.019586  0.049868
    0.50 -0.011616 -0.039839 -0.017487  0.019752  0.072648  0.020594 -0.032688
    0.75  0.135263 -0.100929 -0.105646 -0.001864  0.031349  0.005971 -0.051103
    1.00  0.111387 -0.117995  0.121822 -0.074206  0.088102 -0.012538 -0.044094
    1.25  0.099907 -0.121014  0.197554 -0.128390  0.054566 -0.075927  0.136858
    1.50  0.225026 -0.212136  0.054084 -0.050274  0.077840 -0.043452  0.051887
    1.75  0.205826 -0.063020  0.015887  0.015550  0.158005 -0.083190  0.067913
    2.00  0.047863 -0.017940 -0.015713  0.027641  0.157605 -0.184567  0.065723
    2.25 -0.012206 -0.095284  0.067129  0.108862  0.093491 -0.146234 -0.022997
    2.50 -0.033261  0.052185  0.051653  0.259280  0.173120 -0.010915 -0.009278
    2.75  0.092319  0.084145  0.069256  0.149523  0.214823 -0.043251  0.127294
    3.00  0.106138  0.045591  0.057713 -0.078409  0.206151  0.033776  0.137867
    3.25  0.119071  0.118922  0.048578  0.042976  0.174218 -0.099979  0.110721
    3.50  0.103628  0.167896  0.160688 -0.017439  0.079580 -0.060866  0.053169
    3.75  0.037109  0.196812  0.104011 -0.057185  0.181329  0.014241 -0.123167
    4.00  0.187892  0.205535  0.211189  0.059226  0.086787  0.047556  0.022749
    4.25  0.183402  0.200231  0.027754  0.029329  0.255620  0.054057 -0.094369
    4.50 -0.026393  0.144932  0.080618 -0.069723  0.316742 -0.004079  0.009713
    4.75  0.053196  0.086456  0.078305 -0.020204  0.210432 -0.061564  0.179312
    5.00  0.009414  0.040016  0.084439 -0.013027  0.071045 -0.115703  0.014640


### 2 - 2 Geometric Brownian motion


```python
sims_GBM = simdiff(
    n=10,
    horizon=5,
    frequency="semi-annual",
    model="GBM",
    x0=V0,
    theta1=theta1,
    theta2=theta2,
    theta3=theta3,
)

print(sims_GBM)
```

         Series 1  Series 2      Series 3  Series 4  Series 5  Series 6  Series 7  \
    0.0  0.040000  0.040000  4.000000e-02  0.040000  0.040000  0.040000  0.040000   
    0.5  0.012960  0.086032  7.566884e-03  0.036919  0.011241  0.030725  0.035130   
    1.0  0.005961  0.073984  3.525718e-03  0.015851  0.005294  0.017501  0.012107   
    1.5  0.018284  0.066449  6.972174e-04  0.024051  0.000812  0.009819  0.004992   
    2.0  0.011569  0.043875  1.889517e-04  0.035840  0.004758  0.024617  0.000995   
    2.5  0.007791  0.014286  5.717045e-05  0.050299  0.010060  0.011376  0.000187   
    3.0  0.028206  0.055817  5.609877e-06  0.061307  0.001795  0.033362  0.000152   
    3.5  0.027002  0.055569  8.009562e-06  0.064775  0.000687  0.003789  0.000143   
    4.0  0.004144  0.004052  5.533457e-06  0.035614  0.000246  0.004136  0.000089   
    4.5  0.001174  0.005006  9.715489e-07  0.015116  0.000330  0.002770  0.000139   
    5.0  0.000430  0.001780  2.156546e-06  0.005928  0.000178  0.002045  0.000718   
    
         Series 8  Series 9  Series 10  
    0.0  0.040000  0.040000   0.040000  
    0.5  0.013951  0.023629   0.067367  
    1.0  0.000707  0.020877   0.070761  
    1.5  0.001207  0.008273   0.053518  
    2.0  0.000334  0.009621   0.016144  
    2.5  0.000095  0.004471   0.040134  
    3.0  0.000165  0.003732   0.012467  
    3.5  0.000071  0.007014   0.074483  
    4.0  0.000011  0.006534   0.222215  
    4.5  0.000008  0.002715   0.101611  
    5.0  0.000004  0.005391   0.020085  


## 3 - Spaghetti plot


```python
#plt.style.use('seaborn-darkgrid')
 
palette = plt.get_cmap('Set1')
 
for num, column in enumerate(sims_GBM):
    
    plt.plot(sims_GBM.index, sims_GBM[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=1, ncol=2)
 
# Add titles
plt.title("esgtoolkit.simdiff's result for Geometric Brownian Motion", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Series")

# Show the graph
plt.show()
```
    
![Geometric Brownian motion simulations]({{base}}/images/2023-12-11/2023-12-11-image1.png){:class="img-responsive"}
    