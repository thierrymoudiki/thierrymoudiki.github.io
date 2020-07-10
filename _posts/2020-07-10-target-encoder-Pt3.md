---
layout: post
title: "Maximizing your tip as a waiter (Part 2)"
description: A target-based categorical encoder for Statistical/Machine Learning (based on correlations) Part 3.
date: 2020-07-10
categories: [Python, R, Misc]
---



In [Part 1]({% post_url 2020-06-05-target-encoder-correlation-2 %}) of "Maximizing your tip as a waiter", I talked about a __target-based categorical encoder__ for Statistical/Machine Learning, firstly introduced [in this post]({% post_url 2020-04-24-target-encoder-correlation %}). An example dataset of [`tips`](https://github.com/thierrymoudiki/querier/tree/master/querier/tests/data/tips.csv) was used for the purpose, and we'll use the __same dataset__ today. Here is a snippet of `tips`:

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image0.png){:class="img-responsive"}

Based on these informations, how would you maximize your __tip__ as a waiter working in this restaurant? 

# 1 - Descriptive analysis

The tips (available in variable `tip` in `tips`) range from 0 to 10€, and are __mostly comprised between 2 and 4€__: 

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image2.png){:class="img-responsive"}

Another interesting information is the __amount of total bills__, which is comprised between 3 and 50€, and mostly between 10 and 20€: 

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image3.png){:class="img-responsive"}

Both distributions -- of tips and total bill amounts -- are __left-skewed__. We could fit a probability distribution to each one of them, such as lognormal or Weibull, but this would not be extremely informative. We would be to derive some confidence intervals or things like the __probability of having a total bill higher than 40€__ though. Generally, in addition to `tip` and `total_bill`, we have the following raw information on the __marginal distributions of  other variables__:

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image3.5.png){:class="img-responsive"}


A transformation of `tips` dataset using a one-hot encoder (cf. the [beginning of this post]({% post_url 2020-04-24-target-encoder-correlation %}) to understand what this means) allows to obtain a dataset with numerical columns at the expense of creating a larger dataset, and to __derive correlations__: 

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image1.png){:class="img-responsive"}

Some correlations mean nothing at all. For example, the correlation between `daySat` and `dayThur` or `sexMale` and `timeLunch`. The most interesting ones are those between `tip` and the other variables. Tips in € are more positively correlated with total bills amounts, and with the number of people dining at a table. Here, contrary to the [previous post]({% post_url 2020-06-05-target-encoder-correlation-2 %}) and for a learning purpose presented later, we will categorize our tips in __four classes__: 

- __Class 0__: tip in a ]0; 2] € range -- __Low__
- __Class 1__: tip in a ]2; 3] € range -- __Medium__
- __Class 2__: tip in a ]3; 4] € range -- __High__
- __Class 3__: tip in a ]4; 10] € range -- __Very high__

We'll hence be considering a __classification problem__: how to be in class 2 or 3 given the explanatory variables?

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image4.png){:class="img-responsive"}

__Class 0__, __low tip__ contains 78 observations. __Class 1__, __medium tip__ contains 68 observations.  __Class 2__, __high tip__ contains 57 observations. __Class 3__, __very high tip__ contains 41 observations. Below, as an __additional descriptive information related to these classes__, we present a distribution of tips (in four classes) as a function of explanatory variables __smoker__, __sex__, __time__, __day__, __size__ and __total bill__ (with the total bill being segmented according to its histogram breaks): 

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image5.png){:class="img-responsive"}

According to this figure, the fact that the table is reserved for smokers or not, doesn't highly affect the __median tip__. The same remark holds for the __waiter's sex__ and the __time of the day__ when the meals are served (dinner or lunch), which both don't seem to have a substantial effect on  median amounts of tips. 

Conversely, __Sunday seems to be the best day for you to work__ if you want to maximize your tip. The __number of people dining at a table, and total bills amounts are other influential explanatory variables for the tip__: the higher, the better. But unless you can choose the table you'll be assigned to (you're the boss, or his friend!), or are great at embellishing and advertising the menu, your influence on these variables -- **size** and **total_bill** -- will be limited.

In section 2 of this post, we'll study these effects more systematically by using a statistical learning procedure; a procedure designed for accurately classifying tips within the four classes we've just defined (low, medium, high, very high), given our explanatory variables. More precisely, we'll study the effects of the [numerical target encoder]({% post_url 2020-06-05-target-encoder-correlation-2 %}) on a Random Forest's accuracy.

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image6.png){:class="img-responsive"}

# 2 - Encoding using [mlsauce](https://github.com/thierrymoudiki/mlsauce); cross-validation

**Import Python packages**

```python
import requests
import nnetsauce as ns
import mlsauce as ms
import numpy as np
import pandas as pd
import querier as qr
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
```

**Import `tips`**

```python
url = 'https://github.com/thierrymoudiki/querier/tree/master/querier/tests/data/tips.csv'

f = requests.get(url)

df = qr.select(pd.read_html(f.text)[0], 
          'total_bill, tip, sex, smoker, day, time, size')
```

**Create the response (for classification)**

```python
# tips' classes = response variable

y_int = np.asarray([0, 0, 2, 2, 2, 3, 0, 2, 0, 2, 0, 3, 0, 1, 2, 2, 0, 2, 2, 2, 3, 
	1, 1, 3, 2, 1, 0, 0, 3, 1, 0, 1, 1, 1, 2, 2, 0,
2, 1, 3, 1, 1, 2, 0, 3, 1, 3, 3, 1, 1, 1, 1, 3, 0, 3, 2, 1, 0, 0, 3, 2, 0, 0, 2, 1, 2, 1, 0, 1, 1, 0, 1, 2, 3,
1, 0, 2, 2, 1, 1, 1, 2, 0, 3, 1, 3, 0, 2, 3, 1, 1, 2, 0, 3, 2, 3, 2, 0, 1, 0, 1, 1, 1, 2, 3, 0, 3, 3, 2, 2, 1,
0, 2, 1, 2, 2, 3, 0, 0, 1, 1, 0, 1, 0, 1, 3, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 2, 3, 3, 3, 1, 0, 0, 0,
0, 0, 1, 0, 1, 0, 0, 3, 3, 2, 1, 0, 2, 1, 0, 0, 1, 2, 1, 3, 0, 0, 3, 2, 3, 2, 2, 2, 0, 0, 2, 2, 2, 3, 2, 3, 1,
3, 2, 0, 2, 2, 0, 3, 1, 1, 2, 0, 0, 3, 0, 0, 2, 1, 0, 1, 2, 2, 2, 1, 1, 1, 0, 3, 3, 1, 3, 0, 1, 0, 0, 2, 1, 2,
0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 0, 1, 0, 0, 0, 3, 3, 0, 0, 0, 1])

```

**Obtain a distribution of scores, using encoding**

Here, we use `corrtarget_encoder` from the [mlsauce](https://github.com/thierrymoudiki/mlsauce) to __convert categorical variables (containing character strings) to numerical variables__: 

```python
n_cors = 15
n_repeats = 10
scores_rf = {k: [] for k in range(n_cors)} # accuracy scores

for i, rho in enumerate(np.linspace(-0.9, 0.9, num=n_cors)):

  print("\n")

  for j in range(n_repeats):

  # Use the encoder
    df_temp = ms.corrtarget_encoder(df, target='tip', 
                                    rho=rho, 
                                    seed=i*10+j*10)[0]

    X = qr.select(df_temp, 'total_bill, sex, smoker, day, time, size').values    

    regr = RandomForestClassifier(n_estimators=250) 

    scores_rf[i].append(cross_val_score(regr, X, y_int, cv=3).mean()) 
```

From these accuracy scores `scores_rf`, we obtain the following figure:

![image-title-here]({{base}}/images/2020-07-10/2020-07-10-image7.png){:class="img-responsive"}

__Quite low accuracies... Why is that?__ With that said, the best scores are still obtained for high correlations between response and pseudo response. In Part 3 of "Maximizing your tip as a waiter", __here are the options that we'll investigate__: 

- Compare the correlation-based encoder with one-hot's accuracy
- Further decorrelate the numerically encoded variables by using a new *trick* (summing different, independent pseudo targets instead of one currently)
- Consider the use a different dataset if classification results remain poor on `tips`. Maybe `tips` is just random?
- Use the [teller](https://github.com/thierrymoudiki/teller) to understand what drives the probability of a given class higher (well, that's definitely the laaaaast, last step)


Your remarks are welcome as usual, **stay tuned!**