---
layout: post
title: "Benchmarking 30 statistical/Machine Learning models on the VN1 Forecasting -- Accuracy challenge"
description: "Benchmarking 30 statistical/Machine Learning models on a few products, based on the validation set (phase 2 of the VN1 Forecasting -- Accuracy challenge)"
date: 2024-10-04
categories: Python
comments: true
---

[This post](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_20241003_vn1_competition.ipynb) is about the [VN1 Forecasting -- Accuracy challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description). The aim is to accurately forecast future sales for various products across different
clients and warehouses, using historical sales and pricing data. 

Phase 1 was a warmup to get an idea of what works and what wouldn't (and... for overfitting the validation set, so that the leaderboard is almost  meaningless). It's safe to say, based on empirical observations, that an advanced artillery would be useless here. In phase 2 (people are still welcome to enter the challenge, no pre-requisites from phase 1 needed), the validation set is provided, and there's no leaderboard for the test set (which is great, basically: "real life"; no overfitting).

My definition of "winning" the challenge will be to have an accuracy close to the winning solution by a factor of 1% (2 decimals). Indeed, the focus on accuracy means: we are litterally targetting a point on the real line (well, the "real line", the interval is probably bounded but still contains an  infinite number of points). If the metric was a metric for quantifying uncertainty... there would be too much winners :) 

In the examples, I show you how you can start the competition by benchmarking 30 statistical/Machine Learning models on a few products, based on the validation set provided yesterday. No tuning, no overfitting. Only hold-out set validation. You can notice, on some examples, that a model can be the most accurate on point forecasting, but completely off-track when trying to capture the uncertainty aroung the point forecast. Food for thought. 

## 0 - Functions and packages


```python
!pip uninstall nnetsauce --yes
!pip install nnetsauce --upgrade --no-cache-dir
```

```python
import numpy as np
import pandas as pd
```


```python
def rm_leading_zeros(df):
    if 'y' in df.columns and (df['y'] == 0).any():
        first_non_zero_index_y = (df['y'] != 0).idxmax()
        df = df.loc[first_non_zero_index_y:].reset_index(drop=True)
    return df.dropna().reset_index(drop=True)
```


```python
# Read price data
price = pd.read_csv("/kaggle/input/2024-10-02-vn1-forecasting/Phase 0 - Price.csv", na_values=np.nan)
price["Value"] = "Price"
price = price.set_index(["Client", "Warehouse","Product", "Value"]).stack()

# Read sales data
sales = pd.read_csv("/kaggle/input/2024-10-02-vn1-forecasting/Phase 0 - Sales.csv", na_values=np.nan)
sales["Value"] = "Sales"
sales = sales.set_index(["Client", "Warehouse","Product", "Value"]).stack()

# Read price validation data
price_test = pd.read_csv("/kaggle/input/2024-10-02-vn1-forecasting/Phase 1 - Price.csv", na_values=np.nan)
price_test["Value"] = "Price"
price_test = price_test.set_index(["Client", "Warehouse","Product", "Value"]).stack()

# Read sales validation data
sales_test = pd.read_csv("/kaggle/input/2024-10-02-vn1-forecasting/Phase 1 - Sales.csv", na_values=np.nan)
sales_test["Value"] = "Sales"
sales_test = sales_test.set_index(["Client", "Warehouse","Product", "Value"]).stack()

# Create single dataframe
df = pd.concat([price, sales]).unstack("Value").reset_index()
df.columns = ["Client", "Warehouse", "Product", "ds", "Price", "y"]
df["ds"] = pd.to_datetime(df["ds"])
df = df.astype({"Price": np.float32,
                "y": np.float32,
                "Client": "category",
                "Warehouse": "category",
                "Product": "category",
                })

df_test = pd.concat([price_test, sales_test]).unstack("Value").reset_index()
df_test.columns = ["Client", "Warehouse", "Product", "ds", "Price", "y"]
df_test["ds"] = pd.to_datetime(df_test["ds"])
df_test = df_test.astype({"Price": np.float32,
                "y": np.float32,
                "Client": "category",
                "Warehouse": "category",
                "Product": "category",
                })
```


```python
display(df.head())
display(df_test.head())
```


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
      <th>Client</th>
      <th>Warehouse</th>
      <th>Product</th>
      <th>ds</th>
      <th>Price</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2020-07-06</td>
      <td>10.90</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2020-07-13</td>
      <td>10.90</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2020-07-20</td>
      <td>10.90</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2020-07-27</td>
      <td>15.58</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2020-08-03</td>
      <td>27.29</td>
      <td>7.00</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Client</th>
      <th>Warehouse</th>
      <th>Product</th>
      <th>ds</th>
      <th>Price</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2023-10-09</td>
      <td>51.86</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2023-10-16</td>
      <td>51.86</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2023-10-23</td>
      <td>51.86</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2023-10-30</td>
      <td>51.23</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>367</td>
      <td>2023-11-06</td>
      <td>51.23</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.describe()
df_test.describe()
```




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
      <th>ds</th>
      <th>Price</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195689</td>
      <td>85630.00</td>
      <td>195689.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2023-11-20 00:00:00</td>
      <td>63.43</td>
      <td>19.96</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2023-10-09 00:00:00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2023-10-30 00:00:00</td>
      <td>17.97</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2023-11-20 00:00:00</td>
      <td>28.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2023-12-11 00:00:00</td>
      <td>48.27</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2024-01-01 00:00:00</td>
      <td>5916.04</td>
      <td>15236.00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>210.48</td>
      <td>128.98</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(df.info())
display(df_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2559010 entries, 0 to 2559009
    Data columns (total 6 columns):
     #   Column     Dtype         
    ---  ------     -----         
     0   Client     category      
     1   Warehouse  category      
     2   Product    category      
     3   ds         datetime64[ns]
     4   Price      float32       
     5   y          float32       
    dtypes: category(3), datetime64[ns](1), float32(2)
    memory usage: 51.6 MB



    None


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 195689 entries, 0 to 195688
    Data columns (total 6 columns):
     #   Column     Non-Null Count   Dtype         
    ---  ------     --------------   -----         
     0   Client     195689 non-null  category      
     1   Warehouse  195689 non-null  category      
     2   Product    195689 non-null  category      
     3   ds         195689 non-null  datetime64[ns]
     4   Price      85630 non-null   float32       
     5   y          195689 non-null  float32       
    dtypes: category(3), datetime64[ns](1), float32(2)
    memory usage: 4.3 MB



    None


## 1 - AutoML for a few products


```python

```

### 1 - 1 Select a product


```python
np.random.seed(413)
#np.random.seed(13) # uncomment to select a different product
#np.random.seed(1413) # uncomment to select a different product
#np.random.seed(71413) # uncomment to select a different product
random_series = df.sample(1).loc[:, ['Client', 'Warehouse', 'Product']]
client = random_series.iloc[0]['Client']
warehouse = random_series.iloc[0]['Warehouse']
product = random_series.iloc[0]['Product']
df_filtered = df[(df.Client == client) & (df.Warehouse == warehouse) & (df.Product == product)]
df_filtered = rm_leading_zeros(df_filtered)
display(df_filtered)
df_filtered_test = df_test[(df_test.Client == client) & (df_test.Warehouse == warehouse) & (df_test.Product == product)]
display(df_filtered_test)
```


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
      <th>Client</th>
      <th>Warehouse</th>
      <th>Product</th>
      <th>ds</th>
      <th>Price</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-11-15</td>
      <td>54.95</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-11-22</td>
      <td>54.95</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-11-29</td>
      <td>54.95</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-12-06</td>
      <td>54.95</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-12-13</td>
      <td>54.95</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-12-20</td>
      <td>54.95</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2021-12-27</td>
      <td>54.95</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-01-03</td>
      <td>54.95</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-01-10</td>
      <td>54.95</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-01-17</td>
      <td>52.84</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-01-24</td>
      <td>54.95</td>
      <td>19.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-01-31</td>
      <td>49.45</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-02-07</td>
      <td>54.95</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-02-14</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-02-21</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-02-28</td>
      <td>54.95</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-03-07</td>
      <td>54.95</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-03-14</td>
      <td>54.95</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-03-21</td>
      <td>54.95</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-03-28</td>
      <td>54.95</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-04-18</td>
      <td>54.95</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-12-12</td>
      <td>54.95</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-12-19</td>
      <td>54.95</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2022-12-26</td>
      <td>54.95</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-01-02</td>
      <td>54.26</td>
      <td>16.00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-01-09</td>
      <td>54.95</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-01-16</td>
      <td>54.95</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-01-23</td>
      <td>54.95</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-01-30</td>
      <td>54.95</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-02-06</td>
      <td>54.95</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>30</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-02-13</td>
      <td>54.95</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-02-20</td>
      <td>54.95</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>32</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-02-27</td>
      <td>54.95</td>
      <td>18.00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-03-06</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>34</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-03-13</td>
      <td>54.95</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-03-20</td>
      <td>54.64</td>
      <td>18.00</td>
    </tr>
    <tr>
      <th>36</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-03-27</td>
      <td>54.95</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>37</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-04-03</td>
      <td>54.43</td>
      <td>21.00</td>
    </tr>
    <tr>
      <th>38</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-04-10</td>
      <td>54.49</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>39</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-04-17</td>
      <td>54.95</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-04-24</td>
      <td>54.95</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>41</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-05-01</td>
      <td>54.95</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>42</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-05-08</td>
      <td>54.45</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>43</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-05-15</td>
      <td>54.95</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>44</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-06-26</td>
      <td>54.95</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>45</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-07-03</td>
      <td>54.95</td>
      <td>21.00</td>
    </tr>
    <tr>
      <th>46</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-07-10</td>
      <td>47.37</td>
      <td>29.00</td>
    </tr>
    <tr>
      <th>47</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-07-17</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>48</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-07-24</td>
      <td>54.95</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>49</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-07-31</td>
      <td>54.95</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>50</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-08-07</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>51</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-08-14</td>
      <td>54.95</td>
      <td>18.00</td>
    </tr>
    <tr>
      <th>52</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-08-21</td>
      <td>54.95</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>53</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-09-04</td>
      <td>43.96</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>54</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-09-11</td>
      <td>54.95</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>55</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-09-18</td>
      <td>53.73</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>56</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-09-25</td>
      <td>51.29</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>57</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-10-02</td>
      <td>54.95</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Client</th>
      <th>Warehouse</th>
      <th>Product</th>
      <th>ds</th>
      <th>Price</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>174213</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-10-09</td>
      <td>54.95</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>174214</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-10-16</td>
      <td>54.45</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>174215</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-10-23</td>
      <td>54.95</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>174216</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-10-30</td>
      <td>54.95</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>174217</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-11-06</td>
      <td>54.95</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>174218</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-11-13</td>
      <td>54.03</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>174219</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-11-20</td>
      <td>54.95</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>174220</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-11-27</td>
      <td>54.95</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>174221</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-12-04</td>
      <td>54.95</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>174222</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-12-11</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>174223</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-12-18</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>174224</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2023-12-25</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>174225</th>
      <td>41</td>
      <td>88</td>
      <td>8498</td>
      <td>2024-01-01</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_selected = df_filtered[['y', 'ds']].set_index('ds')
df_selected.index = pd.to_datetime(df_selected.index)
display(df_selected)
```


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
      <th>y</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-11-15</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2021-11-22</th>
      <td>5.00</td>
    </tr>
    <tr>
      <th>2021-11-29</th>
      <td>9.00</td>
    </tr>
    <tr>
      <th>2021-12-06</th>
      <td>20.00</td>
    </tr>
    <tr>
      <th>2021-12-13</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2021-12-20</th>
      <td>8.00</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2022-01-10</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2022-01-17</th>
      <td>26.00</td>
    </tr>
    <tr>
      <th>2022-01-24</th>
      <td>19.00</td>
    </tr>
    <tr>
      <th>2022-01-31</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2022-02-07</th>
      <td>15.00</td>
    </tr>
    <tr>
      <th>2022-02-14</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2022-02-21</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2022-02-28</th>
      <td>17.00</td>
    </tr>
    <tr>
      <th>2022-03-07</th>
      <td>31.00</td>
    </tr>
    <tr>
      <th>2022-03-14</th>
      <td>20.00</td>
    </tr>
    <tr>
      <th>2022-03-21</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2022-03-28</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2022-04-18</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2022-12-12</th>
      <td>4.00</td>
    </tr>
    <tr>
      <th>2022-12-19</th>
      <td>2.00</td>
    </tr>
    <tr>
      <th>2022-12-26</th>
      <td>5.00</td>
    </tr>
    <tr>
      <th>2023-01-02</th>
      <td>16.00</td>
    </tr>
    <tr>
      <th>2023-01-09</th>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2023-01-16</th>
      <td>4.00</td>
    </tr>
    <tr>
      <th>2023-01-23</th>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2023-01-30</th>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2023-02-06</th>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2023-02-13</th>
      <td>9.00</td>
    </tr>
    <tr>
      <th>2023-02-20</th>
      <td>6.00</td>
    </tr>
    <tr>
      <th>2023-02-27</th>
      <td>18.00</td>
    </tr>
    <tr>
      <th>2023-03-06</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2023-03-13</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>2023-03-20</th>
      <td>18.00</td>
    </tr>
    <tr>
      <th>2023-03-27</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2023-04-03</th>
      <td>21.00</td>
    </tr>
    <tr>
      <th>2023-04-10</th>
      <td>24.00</td>
    </tr>
    <tr>
      <th>2023-04-17</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2023-04-24</th>
      <td>8.00</td>
    </tr>
    <tr>
      <th>2023-05-01</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2023-05-08</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2023-05-15</th>
      <td>6.00</td>
    </tr>
    <tr>
      <th>2023-06-26</th>
      <td>26.00</td>
    </tr>
    <tr>
      <th>2023-07-03</th>
      <td>21.00</td>
    </tr>
    <tr>
      <th>2023-07-10</th>
      <td>29.00</td>
    </tr>
    <tr>
      <th>2023-07-17</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2023-07-24</th>
      <td>15.00</td>
    </tr>
    <tr>
      <th>2023-07-31</th>
      <td>17.00</td>
    </tr>
    <tr>
      <th>2023-08-07</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2023-08-14</th>
      <td>18.00</td>
    </tr>
    <tr>
      <th>2023-08-21</th>
      <td>5.00</td>
    </tr>
    <tr>
      <th>2023-09-04</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2023-09-11</th>
      <td>2.00</td>
    </tr>
    <tr>
      <th>2023-09-18</th>
      <td>9.00</td>
    </tr>
    <tr>
      <th>2023-09-25</th>
      <td>6.00</td>
    </tr>
    <tr>
      <th>2023-10-02</th>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_selected_test = df_filtered_test[['y', 'ds']].set_index('ds')
df_selected_test.index = pd.to_datetime(df_selected_test.index)
display(df_selected_test)
```


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
      <th>y</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-10-09</th>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2023-10-16</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2023-10-23</th>
      <td>8.00</td>
    </tr>
    <tr>
      <th>2023-10-30</th>
      <td>15.00</td>
    </tr>
    <tr>
      <th>2023-11-06</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2023-11-13</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>2023-11-20</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2023-11-27</th>
      <td>15.00</td>
    </tr>
    <tr>
      <th>2023-12-04</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2023-12-11</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2023-12-18</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2023-12-25</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2024-01-01</th>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


### 1 - 2 AutoML (Hold-out set)


```python
import nnetsauce as ns
import numpy as np
from time import time
```


```python
# Custom error metric 
def custom_error(objective, submission):
    try: 
        pred = submission.mean.values.ravel()
        true = objective.values.ravel()
        abs_err = np.nansum(np.abs(pred - true))
        err = np.nansum((pred - true))
        score = abs_err + abs(err)
        score /= true.sum().sum()
    except Exception:
        score = 1000
    return score
```


```python
regr_mts = ns.LazyMTS(verbose=0, ignore_warnings=True, 
                          custom_metric=custom_error,                      
                          type_pi = "scp2-kde", # sequential split conformal prediction
                          lags = 1, n_hidden_features = 0,
                          sort_by = "Custom metric",
                          replications=250, kernel="tophat",
                          show_progress=False, preprocess=False)
models, predictions = regr_mts.fit(X_train=df_selected.values.ravel(), 
                                   X_test=df_selected_test.values.ravel())

```

    100%|██████████| 32/32 [00:24<00:00,  1.28it/s]


### 1 - 3 models leaderboard


```python
display(models)
```


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
      <th>RMSE</th>
      <th>MAE</th>
      <th>MPL</th>
      <th>WINKLERSCORE</th>
      <th>COVERAGE</th>
      <th>Time Taken</th>
      <th>Custom metric</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MTS(RANSACRegressor)</th>
      <td>5.85</td>
      <td>5.20</td>
      <td>2.60</td>
      <td>30.43</td>
      <td>100.00</td>
      <td>0.83</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>ETS</th>
      <td>5.83</td>
      <td>5.45</td>
      <td>2.72</td>
      <td>27.99</td>
      <td>100.00</td>
      <td>0.02</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>MTS(TweedieRegressor)</th>
      <td>6.18</td>
      <td>4.50</td>
      <td>2.25</td>
      <td>32.86</td>
      <td>100.00</td>
      <td>0.78</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>MTS(LassoLars)</th>
      <td>6.23</td>
      <td>4.48</td>
      <td>2.24</td>
      <td>34.33</td>
      <td>100.00</td>
      <td>0.79</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>MTS(Lasso)</th>
      <td>6.23</td>
      <td>4.48</td>
      <td>2.24</td>
      <td>34.33</td>
      <td>100.00</td>
      <td>0.78</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>MTS(RandomForestRegressor)</th>
      <td>5.69</td>
      <td>5.15</td>
      <td>2.58</td>
      <td>38.53</td>
      <td>100.00</td>
      <td>1.10</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>MTS(ElasticNet)</th>
      <td>6.24</td>
      <td>4.47</td>
      <td>2.24</td>
      <td>32.58</td>
      <td>100.00</td>
      <td>0.78</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>MTS(DummyRegressor)</th>
      <td>6.20</td>
      <td>4.49</td>
      <td>2.25</td>
      <td>32.97</td>
      <td>100.00</td>
      <td>0.79</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>MTS(HuberRegressor)</th>
      <td>5.91</td>
      <td>4.46</td>
      <td>2.23</td>
      <td>29.41</td>
      <td>92.31</td>
      <td>0.80</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>ARIMA</th>
      <td>6.67</td>
      <td>4.76</td>
      <td>2.38</td>
      <td>28.68</td>
      <td>100.00</td>
      <td>0.04</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>MTS(BayesianRidge)</th>
      <td>6.87</td>
      <td>4.85</td>
      <td>2.42</td>
      <td>33.29</td>
      <td>100.00</td>
      <td>0.79</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>MTS(PassiveAggressiveRegressor)</th>
      <td>7.61</td>
      <td>5.87</td>
      <td>2.94</td>
      <td>42.12</td>
      <td>84.62</td>
      <td>0.79</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>MTS(LinearSVR)</th>
      <td>7.23</td>
      <td>4.82</td>
      <td>2.41</td>
      <td>34.22</td>
      <td>100.00</td>
      <td>1.02</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>MTS(DecisionTreeRegressor)</th>
      <td>8.66</td>
      <td>7.28</td>
      <td>3.64</td>
      <td>57.11</td>
      <td>92.31</td>
      <td>0.81</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>MTS(RidgeCV)</th>
      <td>7.55</td>
      <td>5.80</td>
      <td>2.90</td>
      <td>34.95</td>
      <td>92.31</td>
      <td>0.77</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>MTS(Ridge)</th>
      <td>7.64</td>
      <td>5.86</td>
      <td>2.93</td>
      <td>38.08</td>
      <td>84.62</td>
      <td>0.80</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>MTS(ElasticNetCV)</th>
      <td>7.34</td>
      <td>5.77</td>
      <td>2.89</td>
      <td>32.61</td>
      <td>84.62</td>
      <td>0.88</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>MTS(TransformedTargetRegressor)</th>
      <td>7.74</td>
      <td>6.03</td>
      <td>3.02</td>
      <td>34.57</td>
      <td>84.62</td>
      <td>0.77</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>MTS(LinearRegression)</th>
      <td>7.74</td>
      <td>6.03</td>
      <td>3.02</td>
      <td>34.57</td>
      <td>84.62</td>
      <td>0.81</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>MTS(Lars)</th>
      <td>7.74</td>
      <td>6.03</td>
      <td>3.02</td>
      <td>34.57</td>
      <td>84.62</td>
      <td>0.80</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>MTS(MLPRegressor)</th>
      <td>7.58</td>
      <td>5.14</td>
      <td>2.57</td>
      <td>31.32</td>
      <td>92.31</td>
      <td>1.33</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>MTS(LassoLarsIC)</th>
      <td>7.77</td>
      <td>6.03</td>
      <td>3.02</td>
      <td>36.19</td>
      <td>84.62</td>
      <td>0.78</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>MTS(LassoCV)</th>
      <td>7.77</td>
      <td>6.03</td>
      <td>3.02</td>
      <td>38.92</td>
      <td>84.62</td>
      <td>0.90</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>MTS(LarsCV)</th>
      <td>7.79</td>
      <td>6.05</td>
      <td>3.02</td>
      <td>39.09</td>
      <td>84.62</td>
      <td>0.79</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>MTS(LassoLarsCV)</th>
      <td>7.79</td>
      <td>6.05</td>
      <td>3.02</td>
      <td>39.09</td>
      <td>84.62</td>
      <td>0.79</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>MTS(SGDRegressor)</th>
      <td>7.84</td>
      <td>6.04</td>
      <td>3.02</td>
      <td>40.52</td>
      <td>84.62</td>
      <td>0.80</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>MTS(KNeighborsRegressor)</th>
      <td>8.00</td>
      <td>5.65</td>
      <td>2.82</td>
      <td>32.97</td>
      <td>100.00</td>
      <td>0.80</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>MTS(AdaBoostRegressor)</th>
      <td>8.61</td>
      <td>6.35</td>
      <td>3.18</td>
      <td>37.34</td>
      <td>100.00</td>
      <td>0.97</td>
      <td>1.56</td>
    </tr>
    <tr>
      <th>MTS(ExtraTreesRegressor)</th>
      <td>11.47</td>
      <td>9.26</td>
      <td>4.63</td>
      <td>55.96</td>
      <td>84.62</td>
      <td>1.03</td>
      <td>2.12</td>
    </tr>
    <tr>
      <th>MTS(ExtraTreeRegressor)</th>
      <td>13.72</td>
      <td>10.88</td>
      <td>5.44</td>
      <td>85.45</td>
      <td>84.62</td>
      <td>0.79</td>
      <td>2.52</td>
    </tr>
    <tr>
      <th>MTS(BaggingRegressor)</th>
      <td>15.49</td>
      <td>13.10</td>
      <td>6.55</td>
      <td>95.70</td>
      <td>76.92</td>
      <td>0.99</td>
      <td>3.21</td>
    </tr>
  </tbody>
</table>
</div>


## 2 - *Best* model


```python
best_model = regr_mts.get_best_model()
display(best_model)
```


<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DeepMTS(kernel=&#x27;tophat&#x27;, n_hidden_features=0, n_layers=1,
        obj=RANSACRegressor(random_state=42), replications=250,
        show_progress=False, type_pi=&#x27;scp2-kde&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">DeepMTS</label><div class="sk-toggleable__content"><pre>DeepMTS(kernel=&#x27;tophat&#x27;, n_hidden_features=0, n_layers=1,
        obj=RANSACRegressor(random_state=42), replications=250,
        show_progress=False, type_pi=&#x27;scp2-kde&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">obj: RANSACRegressor</label><div class="sk-toggleable__content"><pre>RANSACRegressor(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">RANSACRegressor</label><div class="sk-toggleable__content"><pre>RANSACRegressor(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>



```python
display(best_model.get_params())
```


    {'a': 0.01,
     'activation_name': 'relu',
     'agg': 'mean',
     'backend': 'cpu',
     'bias': True,
     'block_size': None,
     'cluster_encode': True,
     'direct_link': True,
     'dropout': 0,
     'kernel': 'tophat',
     'lags': 1,
     'n_clusters': 2,
     'n_hidden_features': 0,
     'n_layers': 1,
     'nodes_sim': 'sobol',
     'obj__base_estimator': 'deprecated',
     'obj__estimator': None,
     'obj__is_data_valid': None,
     'obj__is_model_valid': None,
     'obj__loss': 'absolute_error',
     'obj__max_skips': inf,
     'obj__max_trials': 100,
     'obj__min_samples': None,
     'obj__random_state': 42,
     'obj__residual_threshold': None,
     'obj__stop_n_inliers': inf,
     'obj__stop_probability': 0.99,
     'obj__stop_score': inf,
     'obj': RANSACRegressor(random_state=42),
     'replications': 250,
     'seed': 123,
     'show_progress': False,
     'type_clust': 'kmeans',
     'type_pi': 'scp2-kde',
     'type_scaling': ('std', 'std', 'std'),
     'verbose': 0}

