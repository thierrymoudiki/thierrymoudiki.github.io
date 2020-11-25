---
layout: post
title: "Composing the querier's verbs for data wrangling"
description: Data wrangling by composing operations with Python package querier
date: 2019-11-22
categories: DataBases
---



The [querier]({% post_url 2019-10-25-the-querier-1 %}) is a query language for Python pandas Data Frames, inspired by relational databases querying. If you like SQL, Structured Query Language, you'll like the `querier`. If you haven't had a taste of SQL yet, no problem: the `querier`'s language is __intuitive, and contains 9 verbs__ in its current form. You can see how these verbs work individually  in the following notebooks: 

- [`concat`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_251019_concat.ipynb): __concatenates two Data Frames__, either horizontally or vertically
- [`delete`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_241019_delete.ipynb): __deletes rows__ from a Data Frame based on given criteria
- [`drop`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_241019_drop.ipynb): __drops columns__ from a Data Frame
- [`filtr`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_filtr.ipynb): __filters rows__ of the Data Frame based on given criteria
- [`join`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_join.ipynb): __joins two Data Frames__ based on given criteria 
- [`select`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_select.ipynb): __selects columns__ from the Data Frame
- [`summarize`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_summarize.ipynb): obtains __summaries of data__ based on grouping columns
- [`update`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_251019_update.ipynb): __updates a column__, using an operation given by the user
- [`request`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_request.ipynb): for operations more complex than the previous 8 ones, makes it possible to use a __SQL query on the Data Frame__


It is now possible to __compose the `querier`'s verbs__, to construct  more powerful queries for your Data Frames. Here is how to do it: 

### Installing the package

From command line: 

```bash
!pip install git+https://github.com/Techtonique/querier.git
```

### Import packages and dataset

```python
import pandas as pd
import querier as qr


# Import data -----

url = ('https://raw.github.com/pandas-dev'
   '/pandas/master/pandas/tests/data/tips.csv')
df = pd.read_csv(url)
print(df.head())
```
```
total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4

```

### Example1: 

- select columns `tip, sex, smoker, time` from tips dataset
- filter rows in which `smoker == No` only
- obtain cumulated tips by `sex` and `time` of the day

```python
# Example 1 -----

qrobj = qr.Querier(df)

df1 = qrobj\
.select(req="tip, sex, smoker, time")\
.filtr(req="smoker == 'No'")\
.summarize(req="sum(tip), sex, time", group_by="sex, time")\
.df

print(df1)
```
```
sum_tip     sex    time
0    88.28  Female  Dinner
1    61.49  Female   Lunch
2   243.17    Male  Dinner
3    58.83    Male   Lunch

```

The query could be written in one line, but it would be less readable (hence the "\\" for line continuation).

### Example2: 

- select columns `tip, sex, day, size` from tips dataset
- filter rows corresponding to weekends only
- obtain average tips by `sex` and `day`

```python
# Example 2 -----

df2 = qr.Querier(df)\
.select(req='tip, sex, day, size')\
.filtr(req="(day == 'Sun') | (day == 'Sat')")\
.summarize(req="avg(tip), sex, day", group_by="sex, day")\
.df

print(df2)
```
```
    avg_tip     sex  day
0  2.801786  Female  Sat
1  3.367222  Female  Sun
2  3.083898    Male  Sat
3  3.220345    Male  Sun
```


A notebook containing these results can be found [here](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_221119_chaining.ipynb). Contributions/remarks are welcome as usual, you can submit a pull request [on Github](https://github.com/Techtonique/querier).


__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!



