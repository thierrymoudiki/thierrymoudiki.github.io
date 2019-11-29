---
layout: post
title: "Benchmarking the querier's verbs"
description: Benchmarking the verbs from Python package querier
date: 2019-11-29
---

The [querier]({% post_url 2019-10-25-the-querier-1 %}) is a query language for Python pandas Data Frames, inspired by relational databases querying. There are also new ways of using pandas Data Frames for optimizing performance, such as `Dask` or `modin`. I'm considering an  integration of the `querier`with them, and the first step in this direction, was for me to understand the `querier`'s perfomance itself.


__Average timings in seconds:__

![image-title-here]({{base}}/images/2019-11-29/2019-11-29-image1.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2019-11-29/2019-11-29-image2.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2019-11-29/2019-11-29-image3.png){:class="img-responsive"}

![image-title-here]({{base}}/images/2019-11-29/2019-11-29-image4.png){:class="img-responsive"}

__The full code for these benchmarks is:__


From Terminal, install the package:

```bash
pip install git+https://github.com/thierrymoudiki/querier.git
```

In Python: 

```python
import pandas as pd
import numpy as np
from time import time
from  tqdm import tqdm

import querier as qr

# Ex1: summarize

# n: 100 to 1000000
# p: 10 to 100
n_range = [int(x) for x in np.linspace(start=100, stop=1e6, num=10)]
p_range = [int(x) for x in np.linspace(start=10, stop=1e2, num=10)]
res = np.zeros((len(n_range), len(p_range)))

for idx, n in tqdm(enumerate(n_range)):
    
    print(f" idx = {idx} -----")
    print("\n")
    
    for idy, p in tqdm(enumerate(p_range)):
    
        np.random.seed(123)
        df1 = pd.DataFrame(np.random.randint(0, 10, size=(n, p)), 
                      columns=['v' + str(i) for i in range(p)])
        col_group1 = 'v' + str(p)
        col_group2 = 'v' + str(p-1)
        df1[col_group1] = np.random.choice(a=("choice1", "choice2"), size=n)
        df1[col_group2] = np.random.choice(a=("choice3", "choice4"), size=n)
        
        start = time()
        
        [qr.summarize(df1, req = "avg(v1), avg(v2),"+col_group1+","+col_group2, 
              group_by = col_group1+","+col_group2) for _ in range(10)]                
        
        res[idx, idy] = time() - start
        
        qr.summarize.cache.clear()

np.min(res)
np.max(res)
np.savetxt("summarize.csv", res/10, delimiter=",")


# Ex2: filter

res2 = np.zeros((len(n_range), len(p_range)))

for idx, n in tqdm(enumerate(n_range)):
    
    print(f" idx = {idx} -----")
    print("\n")
    
    for idy, p in tqdm(enumerate(p_range)):
    
        np.random.seed(123)
        df1 = pd.DataFrame(np.random.randint(0, 10, size=(n, p)), 
                      columns=['v' + str(i) for i in range(p)])
        col_group1 = 'v' + str(p)
        col_group2 = 'v' + str(p-1)
        df1[col_group1] = np.random.choice(a=("choice1", "choice2"), size=n)
        df1[col_group2] = np.random.choice(a=("choice3", "choice4"), size=n)
        
        start = time()
        
        [qr.filtr(df1, req = "(" + col_group1 + "== 'choice1')" + " & " + "(" + col_group2 + "== 'choice4')") for _ in range(10)]                
        
        res2[idx, idy] = time() - start
        
        qr.filtr.cache.clear()


np.min(res2)
np.max(res2)
np.savetxt("filtr.csv", res2/10, delimiter=",")


# Ex3: select

res3 = np.zeros((len(n_range), len(p_range)))

for idx, n in tqdm(enumerate(n_range)):
    
    print(f" idx = {idx} -----")
    print("\n")
    
    for idy, p in tqdm(enumerate(p_range)):
    
        np.random.seed(123)
        df1 = pd.DataFrame(np.random.randint(0, 10, size=(n, p)), 
                      columns=['v' + str(i) for i in range(p)])
        col_group1 = 'v' + str(p)
        col_group2 = 'v' + str(p-1)
        df1[col_group1] = np.random.choice(a=("choice1", "choice2"), size=n)
        df1[col_group2] = np.random.choice(a=("choice3", "choice4"), size=n)
        
        start = time()
        
        [qr.select(df1, req = col_group1 + ", " + col_group2) for _ in range(10)]                
        
        res3[idx, idy] = time() - start
        
        qr.select.cache.clear()


np.min(res3)
np.max(res3)
np.savetxt("select.csv", res3/10, delimiter=",")


# Ex4: join

n_range = [int(x) for x in np.linspace(start=100, stop=1e4, num=10)]
p_range = [int(x) for x in np.linspace(start=10, stop=1e2, num=10)]

res4 = np.zeros((len(n_range), len(p_range)))

for idx, n in tqdm(enumerate(n_range)):
    
    print(f" idx = {idx} -----")
    print("\n")
    
    for idy, p in tqdm(enumerate(p_range)):
    
        np.random.seed(123)
        df1 = pd.DataFrame(np.random.randint(0, 10, size=(n, p)), 
                      columns=['v' + str(i) for i in range(p)])
        col_group1 = 'v' + str(p)
        col_group2 = 'v' + str(p-1)
        df1[col_group1] = np.random.choice(a=("choice1", "choice2"), size=n)
        df1[col_group2] = np.random.choice(a=("choice3", "choice4"), size=n)
        
        np.random.seed(234)
        df2 = pd.DataFrame(np.random.randint(0, 10, size=(n, p)), 
                      columns=['v' + str(i) for i in range(p)])
        col_group1 = 'v' + str(p)
        col_group2 = 'v' + str(p-1)
        df2[col_group1] = np.random.choice(a=("choice1", "choice2"), size=n)
        df2[col_group2] = np.random.choice(a=("choice3", "choice4"), size=n)
        
        start = time()
        
        [qr.join(df1, df2, on=col_group1 + ", " + col_group2) for _ in range(10)]                
        
        res4[idx, idy] = time() - start
        
        qr.join.cache.clear()


np.min(res4)
np.max(res4)
np.savetxt("join.csv", res4/10, delimiter=",")
```

You may have noticed the instruction `qr.join.cache.clear()` in this code. It's there because each `querier` verb has a cache, a dictionary (Python `dict`) that you can _manipulate_ accordingly. Only the first function call might be time-consuming (or not!), but __subsequent calls will be much, much faster__. 

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!





