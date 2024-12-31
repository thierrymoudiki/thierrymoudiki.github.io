---
layout: post
title: "Python and Interactive dashboard version of Stock price forecasting with Deep Learning: throwing power at the problem (and why it won't make you rich)"
description: "Python and JavaScript version of Stock price forecasting with Deep Learning: throwing power at the problem (and why it won't make you rich)"
date: 2024-12-31
categories: Python
comments: true
---


This is the Python and JavaScript version of the [Stock price forecasting with Deep Learning: throwing power at the problem (and why it won't make you rich)](https://thierrymoudiki.github.io/blog/2024/12/29/r/stock-forecasting) post. Read the post for more details.

**Contents:**

- [Interactive dashboard](#interactive-dashboard)
- [Python code](#python-code)


# Interactive dashboard

{% include 2024-12-31-stockanalysis.html %}

# Python code

```python
!pip install nnetsauce
```


```python
import matplotlib.pyplot as plt
import nnetsauce as ns
import pandas as pd
```


```python
tscv = ns.utils.model_selection.TimeSeriesSplit()
```


```python
stocks = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/multivariate/EuStockMarkets.csv")
```


```python
stocks.head()
```





  <div id="df-8458b7ef-6d78-4fd8-8633-2c55b04c03b4" class="colab-df-container">
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
      <th>DAX</th>
      <th>SMI</th>
      <th>CAC</th>
      <th>FTSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1628.75</td>
      <td>1678.10</td>
      <td>1772.80</td>
      <td>2443.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1613.63</td>
      <td>1688.50</td>
      <td>1750.50</td>
      <td>2460.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1606.51</td>
      <td>1678.60</td>
      <td>1718.00</td>
      <td>2448.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1621.04</td>
      <td>1684.10</td>
      <td>1708.10</td>
      <td>2470.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1618.16</td>
      <td>1686.60</td>
      <td>1723.10</td>
      <td>2484.70</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8458b7ef-6d78-4fd8-8633-2c55b04c03b4')"
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
        document.querySelector('#df-8458b7ef-6d78-4fd8-8633-2c55b04c03b4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8458b7ef-6d78-4fd8-8633-2c55b04c03b4');
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


<div id="df-6387c4b8-e2e7-4e1f-91cc-d57fdf64b6a9">
  <button class="colab-df-quickchart" onclick="quickchart('df-6387c4b8-e2e7-4e1f-91cc-d57fdf64b6a9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6387c4b8-e2e7-4e1f-91cc-d57fdf64b6a9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
from tqdm import tqdm

n = stocks.shape[0]
half_n = n//2

for stock_index in range(stocks.shape[1]):

  tscv_obj = tscv.split(stocks,
                        initial_window=half_n,
                        horizon=1,
                        fixed_window=False)

  iterator = tqdm(tscv_obj, total=tscv.n_splits)
  observed = [] # observed stock prices for the next day
  forecasts = [] # random walk forecasts
  correct_guesses = [] # correctly guessing the direction of stock price?

  for i, (train_index, test_index) in enumerate(iterator):
      observed.append(stocks.iloc[test_index[0], stock_index]) # observed stock price for the next day
      forecasts.append(stocks.iloc[train_index[-1], stock_index]) # random walk forecast
      if i == 0:
          continue
      correct_guesses.append(1 if ((observed[-1]-observed[-2])*(forecasts[-1]-forecasts[-2]) > 0) else 0)

  fig, axes = plt.subplots(2, 2, figsize=(15, 10))

  # Plot 1: Observed vs. Forecast Line Plot
  axes[0, 0].plot(observed, label='Observed')
  axes[0, 0].plot(forecasts, label='Forecast')
  axes[0, 0].set_xlabel('Time')
  axes[0, 0].set_ylabel('Stock Price')
  axes[0, 0].set_title('Observed vs. Forecast')
  axes[0, 0].legend()


  # Plot 2: Observed vs. Forecast Scatter Plot
  axes[0, 1].scatter(observed, forecasts, alpha=0.5)
  axes[0, 1].plot([min(observed), max(observed)], [min(observed), max(observed)], color='red', linestyle='--', label='x=y')
  axes[0, 1].set_xlabel('Observed Values')
  axes[0, 1].set_ylabel('Forecast Values')
  axes[0, 1].set_title('Observed vs. Forecast Scatterplot')
  axes[0, 1].legend()


  # Plot 3: Residuals Plot
  residuals = [observed[i] - forecasts[i] for i in range(len(observed))]
  axes[1, 0].plot(residuals)
  axes[1, 0].axhline(y=0, color='r', linestyle='--')
  axes[1, 0].set_xlabel('Time')
  axes[1, 0].set_ylabel('Residuals (Observed - Forecast)')
  axes[1, 0].set_title('Observed - Forecast Residuals')


  # Plot 4: Percentage of Correct/Incorrect Direction Guesses
  percentage_1 = (sum(correct_guesses) / len(correct_guesses)) * 100 if correct_guesses else 0
  percentage_0 = 100 - percentage_1
  categories = ['Correct Direction', 'Incorrect Direction']
  percentages = [percentage_1, percentage_0]

  axes[1, 1].bar(categories, percentages, color=['green', 'red'])
  axes[1, 1].set_xlabel('Prediction Accuracy')
  axes[1, 1].set_ylabel('Percentage')
  axes[1, 1].set_title('Percentage of Correct and Incorrect Direction Guesses')
  axes[1, 1].set_ylim(0, 100)

  for i, v in enumerate(percentages):
      axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')

  plt.tight_layout()  # Adjust layout to prevent overlapping
  plt.show()
```

    930it [00:00, 10694.50it/s]          



    
![png](2024_12_31_stock_random_walk_files/2024_12_31_stock_random_walk_5_1.png)
    


    100%|██████████| 930/930 [00:00<00:00, 12536.57it/s]



    
![png](2024_12_31_stock_random_walk_files/2024_12_31_stock_random_walk_5_3.png)
    


    100%|██████████| 930/930 [00:00<00:00, 6134.14it/s]



    
![png](2024_12_31_stock_random_walk_files/2024_12_31_stock_random_walk_5_5.png)
    


    100%|██████████| 930/930 [00:00<00:00, 8359.88it/s]



    
![png](2024_12_31_stock_random_walk_files/2024_12_31_stock_random_walk_5_7.png)
    

