---
layout: post
title: "No-code Machine Learning Cross-validation and Interpretability in techtonique.net"
description: "No-code Machine Learning Cross-validation and Interpretability in techtonique.net: what are the variables that drive an increase in the price of an apartment, or the probability of a patient having cancer? Is the effect of the number of rooms on the price of an apartment statistically significant? To what extent does the number of rooms affects the price of an apartment?"
date: 2024-12-23
categories: [Python, R, techtonique]
comments: true
---

I've added a new feature to [techtonique.net](https://www.techtonique.net/): **No-code Machine Learning Cross-validation and Interpretability** for tabular data (supervised learning,regression and classification). [techtonique.net](https://www.techtonique.net/) will remain free to use until ~~December 24~~, December 30, 2024. **As many others have already done, give it a try!**

To use this new feature, you'll need to navigate to [https://www.techtonique.net/mlcvexplain](https://www.techtonique.net/mlcvexplain), and select a file from [Techtonique/datasets](https://github.com/Techtonique/datasets). The files to be used are those with names ending with a "2", in folders /classification or /regression, i.e tables with a training set index. This means: you can create similar files with your own data, and add a training set index as an additional column for predictive purposes.

![xxx]({{base}}/images/2024-12-23/2024-12-23-image1.gif){:class="img-responsive"}

Once the file is uploaded, you can click on the "Submit" button to see (work in progress): 

- **5-fold cross-validation results**: Balanced accuracy for classification, or Root Mean Squared Error (RMSE) for regression (how well the model performs on the training set)
- **Out-of-sample errors**: Balanced accuracy for classification, or Root Mean Squared Error (RMSE) for regression (how well the model performs on the test set of interest)
- **Sensitivity analysis summary**: impact of each feature on the response variable (e.g **what are the variables that drive an increase in the price of an apartment, or the probability of a patient having cancer**)
- **Parametric tests of significance (for now)**. Identify variables for which there's no significant ambiguity (given the assumptions hold) regarding the fact that their effect on the response variable is different from zero (e.g **is the effect of the number of rooms on the price of an apartment statistically significant?**)
- **Feature effects' heterogeneity**: the distribution of the effects of each feature on the response variable (e.g **to what extent does the number of rooms affects the price of an apartment?**)

As a reminder of [techtonique.net](https://www.techtonique.net/) features: 

You can simulate predictive scenarios using R, Python, and Excel, by using [Techtonique API](https://www.techtonique.net/docs), available at [https://www.techtonique.net](https://www.techtonique.net). Input csv files used in the examples are available in [Techtonique/datasets](https://github.com/Techtonique/datasets). 

The **Excel version** can be found in the Excel file [https://github.com/thierrymoudiki/techtonique-excel-server/VBA-Web.xlsm](https://github.com/thierrymoudiki/techtonique-excel-server) (in 'Sheet3'). Behind the scenes, I'm using Visual Basic for Applications (VBA) to send requests to the API. All you need to do to see it in action is [get a token](https://www.techtonique.net/token) and press a button. Remember to enable macros in Excel when asked to do so (this is safe).

![xxx]({{base}}/images/2024-11-03/2024-11-03-image1.png){:class="img-responsive"}  

Here's the **Python version**, which relies on [`forecastingapi`](https://techtonique.github.io/techtonique_api_py/forecastingapi/forecastingapi.html) Python package: 

```Python
import forecastingapi as fapi
import numpy as np
import pandas as pd 
from time import time
import matplotlib.pyplot as plt
import ast 

# examples in https://github.com/Techtonique/datasets/tree/main/time_series        
path_to_file = '/Users/t/Documents/datasets/time_series/univariate/AirPassengers.csv' 
    
start = time() 
res_get_forecast = fapi.get_forecast(path_to_file,     
base_model="RidgeCV",
n_hidden_features=5,
lags=25,
type_pi='scp2-kde',
replications=10,
h=10)
print(f"Elapsed: {time() - start} seconds \n")

print(res_get_forecast)

# Convert lists to numpy arrays for easier handling
mean = np.asarray(ast.literal_eval(res_get_forecast['mean'])).ravel()
lower = np.asarray(ast.literal_eval(res_get_forecast['lower'])).ravel()
upper = np.asarray(ast.literal_eval(res_get_forecast['upper'])).ravel()
sims = np.asarray(ast.literal_eval(res_get_forecast['sims']))

# Plotting
plt.figure(figsize=(10, 6))

# Plot the simulated lines
for sim in sims:
    plt.plot(sim, color='gray', linestyle='--', alpha=0.6, label='Simulations' if 'Simulations' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot the mean line
plt.plot(mean, color='blue', linewidth=2, label='Mean')

# Plot the lower and upper bounds as shaded areas
plt.fill_between(range(len(mean)), lower, upper, color='lightblue', alpha=0.2, label='Confidence Interval')

# Labels and title
plt.xlabel('Time Point')
plt.ylabel('Value')
plt.title('Spaghetti Plot of Mean, Bounds, and Simulated Paths')
plt.legend()
plt.show()
```

![xxx]({{base}}/images/2024-11-03/2024-11-03-image2.png){:class="img-responsive"}


The **R version** relies on [`forecastingapi`](https://techtonique.github.io/techtonique_api_r/index.html) R package: 

```R
path_to_file <- "/Users/t/Documents/datasets/time_series/univariate/AirPassengers.csv"
forecastingapi::get_forecast(path_to_file)
forecastingapi::get_forecast(path_to_file, type_pi='scp2-kde', h=10L, replications=10L)
sims <- forecastingapi::get_forecast(path_to_file, type_pi="scp2-kde", replications=10L)$sims
matplot(sims, type='l', lwd=2)
```

![xxx]({{base}}/images/2024-11-03/2024-11-03-image3.png){:class="img-responsive"}


In addition, you can obtain insights from your tabular data by chatting with it in [techtonique.net](https://www.techtonique.net). No plotting yet (coming soon), but you can already ask questions like:

- What is the average of column `A`?
- Show me the first 5 rows of data
- Show me 5 random rows of data
- What is the sum of column `B`?
- What is the average of column `A` grouped by column `B`?
- ...

You can also run R or Python code interactively in your browser, on [www.techtonique.net/consoles](https://www.techtonique.net/consoles). 

[Techtonique web app](https://www.techtonique.net/) is a tool designed to help you make informed, data-driven decisions using Mathematics, Statistics, Machine Learning, and Data Visualization. As of September 2024, the tool is in its beta phase (subject to crashes) and will remain completely free to use until December 30, 2024. 
After registering, you will receive an email. CHECK THE SPAMS.

The tool is built on [Techtonique](https://github.com/Techtonique) and the powerful Python ecosystem. Both clickable web interfaces and Application Programming Interfaces (APIs, see below) are available.

Currently, the available functionalities include:

- [Data visualization](https://en.wikipedia.org/wiki/Data_and_information_visualization). **Example:** Which variables are correlated, and to what extent?
- [Probabilistic forecasting](https://en.wikipedia.org/wiki/Probabilistic_forecasting). **Example:** What are my projected sales for next year, including lower and upper bounds?
- [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) (regression or classification) for tabular datasets. **Example:** What is the price range of an apartment based on its age and number of rooms?
- [Survival analysis](https://en.wikipedia.org/wiki/Survival_analysis), analyzing *time-to-event* data. **Example:** How long might a patient live after being diagnosed with Hodgkin's lymphoma (cancer), and how accurate is this prediction?
- [Reserving](https://en.wikipedia.org/wiki/Chain-ladder_method) based on insurance claims data. **Example:** How much should I set aside today to cover potential accidents that may occur in the next few years?

As mentioned earlier, this tool includes both clickable web interfaces and Application Programming Interfaces (APIs).  
APIs allow you to send requests from your computer to perform specific tasks on given **resources**. APIs are [programming language-agnostic](https://curlconverter.com/) (supporting Python, R, JavaScript, etc.), relatively fast, and require no additional package installation before use. This means you can keep using your preferred programming language or legacy code/tool, as long as it can *speak* to the internet.  What are **requests** and **resources**?

In Techtonique/APIs, **resources** are **Statistical/Machine Learning** (ML) model predictions or forecasts.  
A common type of [request](https://en.wikipedia.org/wiki/Representational_state_transfer) might be to obtain sales, weather, or revenue **forecasts** for the next five weeks. In general, requests for tasks are short, typically involving a **verb** and a **URL path** — which leads to a **response**.

Below is an example. In this case, the **resource** we want to manage is a list of **users**.

<p>- Request type (verb): <strong>GET</strong></p>
<ul>
    <li><strong>URL Path:</strong> <code>http://users</code> &nbsp;|&nbsp; Endpoint: users &nbsp;|&nbsp; <strong>API Response:</strong> Displays a list of all users</li>
    <li><strong>URL Path:</strong> <code>http://users/:id</code> &nbsp;|&nbsp; Endpoint: users/:id &nbsp;|&nbsp; <strong>API Response:</strong> Displays a specific user</li>
</ul>

<p>- Request type (verb): <strong>POST</strong></p>
<ul>
    <li><strong>URL Path:</strong> <code>http://users</code> &nbsp;|&nbsp; Endpoint: users &nbsp;|&nbsp; <strong>API Response:</strong> Creates a new user</li>
</ul>  

<p>- Request type (verb): <strong>PUT</strong></p>
<ul>
    <li><strong>URL Path:</strong> <code>http://users/:id</code> &nbsp;|&nbsp; Endpoint: users/:id &nbsp;|&nbsp; <strong>API Response:</strong> Updates a specific user</li>
</ul>

<p>- Request type (verb): <strong>DELETE</strong></p>
<ul>
    <li><strong>URL Path:</strong> <code>http://users/:id</code> &nbsp;|&nbsp; Endpoint: users/:id &nbsp;|&nbsp; <strong>API Response:</strong> Deletes a specific user</li>
</ul>

In Techtonique/APIs, a typical resource endpoint would be `/MLmodel`. Since the resources are predefined and do not need to be updated (PUT) or deleted (DELETE), **every request will be a [POST](https://en.wikipedia.org/wiki/Representational_state_transfer) request** to a `/MLmodel`, with additional parameters for the ML model.  
After reading this, you can proceed to the [/howtoapi](https://www.techtonique.net/howtoapi) page.



